# Step 2b — Design notes for the `Qwen35Weights` vindex adapter

Investigation from session 2026-05-16 (post PR #149) so a fresh session
can pick up Step 2b cold.

## Target API

```rust
// In crates/larql-inference/src/attention/qwen35_load_vindex.rs (NEW)
pub fn load_qwen35_weights_from_vindex(
    vindex_dir: &std::path::Path,
) -> Result<crate::attention::qwen35_forward::Qwen35Weights, LoadError>;
```

## Field map: Qwen35Weights → vindex artifacts

`Qwen35Weights` is defined at
`crates/larql-inference/src/attention/qwen35_forward.rs:98`. Fields:

| Field | Source | Approach |
|---|---|---|
| `embed: ArcArray2<f32>` | `embeddings.bin` | f32 reader (existing `load_vindex_embeddings`) |
| `embed_quant: Option<QuantTensor>` | none in vindex today | leave `None` for first cut |
| `layers: Vec<Qwen35FullLayerWeights>` | per-layer assembly — see below | new code |
| `final_norm: Arc<[f32]>` | `norms.bin` → `norm.weight` | existing norms reader |
| `lm_head: ArcArray2<f32>` | 0×0 placeholder | placeholder (lazy path) |
| `lm_head_quant: Option<QuantTensor>` | `lm_head_q4.bin` | `QuantTensor::from_raw(read_to_vec, TYPE_Q6_K, vocab, hidden)` |
| `ffn_dim: usize` | `index.json::model_config::moe_intermediate_size` | direct |
| `backend: Option<Arc<dyn ComputeBackend>>` | none | `None` |

### Qwen35FullLayerWeights (per-layer assembly)

```rust
pub struct Qwen35FullLayerWeights {
    pub block: Qwen35LayerWeights,   // Linear(DeltaNetLayerWeights) | Attention(Qwen35AttentionLayerWeights)
    pub attn_post_norm: Arc<[f32]>,
    // Dense SwiGLU FFN slots — 0×0 placeholders for MoE layers.
    pub ffn_gate: ArcArray2<f32>,
    pub ffn_up: ArcArray2<f32>,
    pub ffn_down: ArcArray2<f32>,
    pub ffn_gate_quant: Option<QuantTensor>,
    pub ffn_up_quant: Option<QuantTensor>,
    pub ffn_down_quant: Option<QuantTensor>,
    // MoE FFN — populated for every layer on qwen35moe.
    pub moe: Option<Qwen35MoeFfnWeights>,
}
```

For Qwen 3.6 35B-A3B (every layer is MoE):
- Set all 6 dense/quant FFN slots to placeholders (`Array2::zeros((0,0))` / `None`).
- Populate `moe: Some(Qwen35MoeFfnWeights { ... })` from
  `layers/layer_{LL}.weights`.

For block (per layer):

```rust
if arch.is_linear_attention_layer(layer) {
    Qwen35LayerWeights::Linear(DeltaNetLayerWeights { ... })
} else {
    Qwen35LayerWeights::Attention(Qwen35AttentionLayerWeights { ... })
}
```

### DeltaNetLayerWeights (linear layers — 30 of 40)

`crates/larql-inference/src/attention/deltanet_block.rs:53`. Fields:

| Field | Vindex source | Format | Note |
|---|---|---|---|
| `attn_norm: Arc<[f32]>` | `norms.bin` `layers.{L}.attn_norm.weight` | f32 vector | hidden=2048 |
| `attn_qkv: ArcArray2<f32>` | 0×0 placeholder | — | use lazy path |
| `attn_gate: ArcArray2<f32>` | 0×0 placeholder | — | use lazy path |
| `ssm_conv1d: ArcArray2<f32>` | `norms.bin` `layers.{L}.ssm_conv1d.weight` | f32 2-D flattened | reshape `[d_conv=4, conv_dim]` |
| `ssm_dt: Arc<[f32]>` | `norms.bin` `layers.{L}.ssm_dt.weight` | f32 vector | `n_v_heads` |
| `ssm_a: Arc<[f32]>` | `norms.bin` `layers.{L}.ssm_a.weight` | f32 vector | `n_v_heads` |
| `ssm_beta: ArcArray2<f32>` | `deltanet_weights_q4k.bin` | Q4_K → f32 | dequant; no lazy slot |
| `ssm_alpha: ArcArray2<f32>` | `deltanet_weights_q4k.bin` | Q4_K → f32 | dequant; no lazy slot |
| `ssm_norm: Arc<[f32]>` | `norms.bin` `layers.{L}.ssm_norm.weight` | f32 vector | `head_v_dim` |
| `ssm_out: ArcArray2<f32>` | 0×0 placeholder | — | use lazy path |
| `attn_qkv_quant: Option<QuantTensor>` | `deltanet_weights_q4k.bin` | Q4_K mmap | **lazy path** |
| `attn_gate_quant: Option<QuantTensor>` | `deltanet_weights_q4k.bin` | Q4_K mmap | **lazy path** |
| `ssm_out_quant: Option<QuantTensor>` | `deltanet_weights_q4k.bin` | Q4_K mmap | **lazy path** |

Key insight: `ssm_alpha` and `ssm_beta` have **no `*_quant` variant** in
the struct. Either dequant them at load (cheap — [32, 2048] = 65K f32
each × 30 linear layers = 7.5 MB total) or add `_quant` variants. For
the first cut, **dequant at load**.

### QuantTensor construction from vindex bytes

`crates/larql-models/src/quant/lazy.rs` exposes:

```rust
pub fn from_raw(data: Vec<u8>, tensor_type: u32, rows: usize, cols: usize)
    -> Result<Self, ModelError>;

pub fn from_mmap_region(mmap: Arc<memmap2::Mmap>, byte_offset: usize,
    byte_len: usize, tensor_type: u32, rows: usize, cols: usize)
    -> Result<Self, ModelError>;
```

Vindex storage today holds `bytes::Bytes`, NOT `Arc<Mmap>`. Two
options for zero-copy:

**Option A (recommended for first cut):** use `from_raw` with a
`view.as_slice().to_vec()` copy at load time. ~542 MB of DeltaNet
copy on the Qwen 3.6 35B-A3B vindex; one-time cost. Simple.

**Option B (RAM-optimal, defer):** extend `QuantBacking` (currently
`Heap(Arc<[u8]>) | Mmap(Arc<memmap2::Mmap>)`) with a new
`Bytes(bytes::Bytes)` variant. Or expose the underlying
`Arc<Mmap>` from `VindexStorage` via a new method. This is a
follow-up RAM optimisation, not blocking Step 2b's correctness.

Use `from_raw` for Step 2b. File a follow-up arc for the Bytes-
backed `QuantTensor`.

### Q6_K vs Q4_K formats

`deltanet_q4k_layer_data` returns a `format` tag string (e.g.
`"Q4_K"`). Map to the GGML tensor type constants via
`larql_models::quant::ggml::{TYPE_Q4_K, TYPE_Q6_K}` (see
`crates/larql-vindex/src/quant/registry.rs::lookup`).

### MoE PerExpert from `layers/layer_{LL}.weights`

`crates/larql-vindex/src/format/weights/write_layers.rs`:
- `parse_layer_weights_header(bytes)` returns
  `(format, num_entries, inter, hidden, offsets)` where each
  offset entry is `(gate_up_off, gate_up_bytes, down_off, down_bytes)`.
- Per-expert layout: `gate_up` is `[2*inter, hidden]` Q4_K
  (interleaved `[gate rows | up rows]`); `down` is
  `[hidden, padded_inter]` Q4_K.

For `Qwen35MoeFfnWeights`:

```rust
pub struct Qwen35MoeFfnWeights {
    pub router: QuantTensor,        // [num_experts, hidden] — from norms.bin? gate_vectors.bin?
    pub gate_exps: QuantTensor,     // [num_experts * expert_ffn_dim, hidden]
    pub up_exps: QuantTensor,       // [num_experts * expert_ffn_dim, hidden]
    pub down_exps: QuantTensor,     // [num_experts * hidden, expert_ffn_dim]
    pub shexp_gate: Option<QuantTensor>,
    pub shexp_up: Option<QuantTensor>,
    pub shexp_down: Option<QuantTensor>,
    pub num_experts: usize,
    pub top_k: usize,
}
```

This struct expects **packed** per-projection tensors (`gate_exps`
concatenating all 256 expert gate rows into one big
`[256 * inter, hidden]` matrix), not 256 separate `gate_proj`
QuantTensors. Two options:

1. Build a **packed Vec<u8>** by concatenating the 256 expert
   gate bytes (and similarly for up + down), then call
   `QuantTensor::from_raw(packed, TYPE_Q4_K, num_experts * inter, hidden)`.
   This matches the consumer's expectation.

2. Add a `Vec<QuantTensor>` per-expert alternate path. More
   invasive on the forward side.

**Recommended:** option 1. The vindex `layer_LL.weights` already
stores experts contiguously (entry table → entry data → next entry).
The packed Vec is essentially `gate_up_bytes` from each expert in
order. Verify the stride math vs `quantize_dense_entry`'s
`[2 * inter, hidden]` layout — gate rows must come before up rows
per expert.

The **router** tensor:
`crates/larql-vindex/src/format/weights/write_q4k/norms.rs:73-89`
writes it into `norms.bin` (as f32, flattened) under
`arch.moe_router_key(layer)` only for `is_hybrid_moe`. Qwen35moe
returns `false` for is_hybrid_moe so this isn't fired — **bug:
the router is currently NOT being written for qwen35moe**.

**Step 2b prerequisite:** fix the router write. Either:
- Loosen the `is_hybrid_moe()` guard in `norms.rs:73` to
  `is_moe() && expert_format() != PackedMxfp4` so qwen35moe
  PerExpert also writes the router.
- OR write the router into the per-layer file.

Recommend the first — keeps `norms.bin` as the single source for
small-scalars-per-layer.

This is a **separate small PR** before Step 2b can complete.

## Phase outline (refined)

| Phase | Scope | LoC |
|---|---|---:|
| 2b.0 | Fix router write for `qwen35moe` PerExpert in `write_q4k/norms.rs` | ~10 |
| 2b.1 | Norms reader: extract DeltaNet small tensors from `norms.bin` per layer | ~80 |
| 2b.2 | Per-layer assembly: `DeltaNetLayerWeights` from vindex bytes | ~120 |
| 2b.3 | Per-layer assembly: `Qwen35AttentionLayerWeights` for full-attn layers | ~80 |
| 2b.4 | MoE per-layer: parse `layer_LL.weights` → pack 256 experts into `Qwen35MoeFfnWeights` | ~150 |
| 2b.5 | Top-level: `load_qwen35_weights_from_vindex` orchestrator | ~50 |
| 2b.6 | Unit test: load `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex`, dimension-check the produced struct | ~50 |

Total ~540 LoC. (Previous ~300 estimate was wishful.) Single
focused PR target.

## Out of scope (still)

- Step 2c — server dispatch routing `qwen35*` arches.
- Forward parity vs llama.cpp on vindex-loaded weights.
- 40 GB `gate_vectors.bin` size optimisation.
- `QuantBacking::Bytes` zero-copy variant (option B above).
- DeepSeek V4 Flash MLA — still parked.

## Reference paths

- Live vindex: `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex/`
- Source GGUF: `/tank/ai/Qwen/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- `Qwen35Weights` struct: `crates/larql-inference/src/attention/qwen35_forward.rs:98`
- `DeltaNetLayerWeights` struct: `crates/larql-inference/src/attention/deltanet_block.rs:53`
- `Qwen35MoeFfnWeights` struct: `crates/larql-inference/src/attention/qwen35_forward.rs:51`
- `QuantTensor::from_raw`: `crates/larql-models/src/quant/lazy.rs:81`
- `parse_layer_weights_header`: `crates/larql-vindex/src/format/weights/write_layers.rs:241`
- DeltaNet storage reader: `crates/larql-vindex/src/index/storage/deltanet.rs`
- Existing GGUF loader: `crates/larql-inference/src/attention/qwen35_load.rs:62`
  (`load_qwen35_weights`) — use as semantic reference for how each
  field is populated from a GGUF mmap; the vindex loader should
  produce structurally equivalent output.
