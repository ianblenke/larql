## Why

Path D from `cuda-decode-perf-results`: concatenate Q/K/V's Q4_K
weights into one `[q_dim + 2 * kv_dim] × hidden` matrix per layer,
do ONE fused mmvq instead of three separate calls. Predicted
gains: 2 saved graph nodes per layer × 34 layers = 68 fewer node
launches per token, plus 2 saved redundant reads of the shared
`h_attn_q8_1` input. Estimated savings: 0.07–0.15 ms.

This change implements Path D and **records the empirical
regression**: Q/K/V fusion is **0.13 ms SLOWER** on Gemma 3 4B
than the unfused baseline (8.17 ms vs 8.04 ms). Reason: the K
and V projections (1024-row shapes) were benefiting from the
shape-aware warp-cooperative kernel from
`cuda-q4k-mmvq-warp-cooperative`. Fusing into a 4096-row matrix
puts it past the `rows ≤ 1024` coop threshold, forcing the
legacy 1-warp-per-row path for what was previously two coop
calls.

## Empirical results

`larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens
20 --warmup 3` (5-run avg unless noted):

| Variant | decode ms/tok | tok/s | Notes |
|---|---:|---:|---|
| Baseline (f16-kv tip) | **8.04** | **124.4** | unfused, K/V on coop kernel |
| Path D fused (default coop) | 8.30 | ~120 | bimodal 8.04 / 8.46 |
| Path D fused + `LARQL_CUDA_Q4K_COOP=1` | 8.27 | ~121 | coop forced even on 4096-row weight |
| Path D fused (10-run avg) | **8.17** | ~122 | range 8.04–8.46 (variance regression) |

Forcing coop on the fused 4096-row weight is *worse* than letting
it fall back to legacy — confirming that 4096 rows is past the
size where coop wins.

## Mechanism

For Gemma 3 4B's projection shapes (hidden = 2560, n_super_blocks
= 10):

| Original | rows | dispatcher choice | per-call time |
|---|---:|---|---:|
| Q | 2048 | legacy (rows > 1024) | ~5.4 µs |
| K | 1024 | **coop** (rows ≤ 1024) | **3.8 µs** |
| V | 1024 | **coop** (rows ≤ 1024) | **3.8 µs** |
| Total Q+K+V | — | mixed | **~13 µs / layer** |

| Fused | rows | dispatcher choice | per-call time |
|---|---:|---|---:|
| QKV concat | 4096 | legacy (rows > 1024) | ~17 µs |

That's a 4 µs/layer regression × 34 layers = **136 µs per token**
of structural cost from giving up coop on K/V. The 2 saved
launches per layer (~68 µs / token in graph mode) and saved
input reads (~50 µs) don't cover it. Net: **~0.13 ms regression**
on average, plus a meaningful **variance regression** (run-to-run
range 0.42 ms vs 0.02 ms baseline) — likely from L1 cache
contention on the larger fused weight working set per SM.

## Why this is worth shipping (as a negative result)

The fusion-vs-shape-aware-dispatch interaction is non-obvious:
intuitively "fewer launches must be faster", but `cuda-q4k-mmvq-
warp-cooperative` had already extracted a real win from the
1024-row shapes via a different mechanism. Documenting this
prevents a future contributor from re-implementing fusion under
the same intuition, plus narrows the search space:

- Path D fusion does NOT help GQA models with the existing coop
  dispatcher.
- Removing the coop optimization first would let fusion potentially
  win, but at the cost of regressing the un-fused K/V calls. Net
  worse.

## What This Change Ships

The change is **kept on the branch** (compiles, parity-passes,
implements all infrastructure) but is **NOT merged** — production
stays on the unfused path.

- `q4k_qkv_concat_device_cache` cache field on `CudaBackend`.
- `arc_q4k_qkv_concat_device_buf(wq, wk, wv)` helper that
  byte-concatenates the three packed Q4_K streams.
- `LayerArcs::qkv_concat: Option<Arc<CudaSlice<u8>>>` populated
  for layers with all-Q4_K Q/K/V weights.
- `DecodeScratch::qkv` buffer of size `q_dim + 2 * kv_dim`.
- `fused_decode_attention_device_kv_into` signature change:
  `q_dev / k_new_dev / v_new_dev` from `&CudaSlice<f32>` to
  `&CudaView<'_, f32>`. Lets the captured pipeline pass slice
  views of `scratch.qkv` without an intermediate copy.
- Captured-decode pipeline branches on `arcs.qkv_concat.is_some()`:
  if yes, one fused mmvq into `scratch.qkv` + slice views;
  otherwise fall back to the 3-mmvq path.

The path is fully implemented and parity-verified. It just
doesn't beat the baseline on this particular model shape.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the Q/K/V mmvq fusion path and
  documents its empirical regression on GQA-shaped Q4_K models.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/backend.rs` —
    `q4k_qkv_concat_device_cache` + `arc_q4k_qkv_concat_device_buf`.
  - `crates/larql-compute/src/cuda/scratch.rs` — `qkv` buffer.
  - `crates/larql-compute/src/cuda/decode.rs` — `LayerArcs::
    qkv_concat`, `run_decode_pipeline_into_scratch` fused branch.
  - `crates/larql-compute/src/cuda/attn.rs` —
    `fused_decode_attention_device_kv_into` accepts CudaView.
- **Affected systems**: GPU only.

## Risks and back-out

- The branch is shippable in code but produces a regression; the
  recommendation is to NOT merge it.
- If merged: revert by reverting this branch (no env-var back-out
  added because the regression is across-the-board).

## Acceptance bar

- All 200+ existing tests pass with the fused path.
- Bench documents the empirical regression vs the unfused
  baseline (5–10 run avg).
- Future contributors see this proposal first when considering
  Q/K/V mmvq fusion.
