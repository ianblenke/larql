# larql resume prompt — feed this to a fresh session

## Where we are (state at end of session 2026-05-16)

Seven PRs landed end-to-end during 2026-05-16:

- **PR #139** (Q4kDirectFfn) — direct Q4_K × Q8_K matvec FFN for decode.
- **PR #140** (f16 subnormal fix) — pre-existing latent bug in
  `larql_compute::f16_to_f32` decoded every subnormal 2× too large. Found
  while testing Q6_K matvec on Gemma 3 4B V/FFN_DOWN (small weights →
  subnormal `d`). Masked by a shadow `fn f16_to_f32` in the test module
  that decoded subnormals correctly while the production fn was buggy.
- **PR #142** (Q4kDirectAttention) — direct Q4_K × Q8_K matvec for the
  decode-step Q/K/V/O projections. Same pattern as Q4kDirectFfn.
- **PR #143** (rayon-parallel matvec) — extract per-row AVX2 dot
  products and dispatch row chunks via `rayon::par_chunks_mut`. Small
  matvecs (rows < 16) keep the sequential path so existing bit-exact
  oracles still hold.
- **PR #144** (direct Q4_K lm_head matvec) — populate
  `weights.lm_head_quant` from `lm_head_q4.bin` and dispatch the final
  vocab projection through `QuantTensor::matvec` (rayon-parallel AVX2
  Q4_K × Q8_K) instead of f32 BLAS GEMV.
- **PR #146** (drop f32 dequant cache) — `insert_q4k_layer_tensors`'s
  10 GB FFN+attn dequant cache and the 2.6 GB f32 lm_head form are no
  longer populated when every layer can run direct. New
  `run_attention_block_prefill_q4k_direct` provides the multi-row
  prefill direct path that closed the dependency on `weights.tensors`.
  RSS 24.6 → 10.3 GB; prefill 14-tok 2.7 → 1.0 s; decode flat at 9.7
  tok/s. Closes the bulk of larql's RAM gap vs llama.cpp.
- **PR #145** (Q8K cache hazard + BLAS thread pin) — two coupled fixes:
  (a) `with_q8k_for` cache used `(ptr, len)` as key; #144's per-step
  lm_head `to_owned()` allocator reuse hit the cache and returned stale
  Q8K bytes → silent gibberish at long decode lengths. (b) OpenBLAS
  default thread count was contending with rayon on small per-head
  attention dots, adding ~160ms/step at 150-token cache. Cache key now
  fingerprints `x[0]`+`x[len-1]`; main pins `OPENBLAS_NUM_THREADS=1`.

### Arcs C-I — CPU decode-step direct matvec + parallel + lm_head + threading + RAM

`predict_q4k_hidden_with_cache` now dispatches direct-matvec backends
for both FFN (`Q4kDirectFfn`) and attention
(`run_attention_block_decode_step_q4k_direct`) on single-row, non-MoE,
Q8_K-aligned layers. Skips f32 materialisation of weights entirely on
the decode path; routes through `q4k_q8k_*` / `q6k_q8k_*` kernels from
PRs #102–#119. The f16 fix (#140) was a prerequisite for Q6_K V/O to
match the dequant reference within Q8_K activation noise. PR #143
then parallelises the kernel row loop across rayon threads. PR #144
extends the direct path to lm_head — the dominant remaining cost
after matvec parallelism (`vocab × hidden = 671M MACs per step` at
Gemma 3 4B's shape).

Gemma 3 1B (`hidden=1152`) falls back to the f32 BLAS path — not
Q8_K-aligned. Gemma 3 4B (`hidden=2560`) engages every direct path.

| Model | Path | Before | After | Speedup |
|---|---|---:|---:|---:|
| Gemma 3 4B Q4_K_M | `predict_q4k_hidden_with_cache` (pure decode, no lm_head) | 193 ms/step | **89 ms/step** | **2.16×** |
| Gemma 3 4B Q4_K_M | larql CPU `/v1/chat/completions`, 150 tok | 0.117 tok/s | **9.81 tok/s** | **83.9×** |

llama.cpp CPU 14.1 tok/s reference unchanged. Remaining gap is **~1.44×**
on 4B end-to-end (70% of llama.cpp performance). FFN, attention, and
lm_head are all on the same rayon-parallel AVX2 Q4_K × Q8_K path with
BLAS pinned to 1 thread for the residual gqa per-head dots.

⚠ The bench number reported in PR #144 (3.96 tok/s) was actually
subtle-gibberish output due to the cache hazard fixed in #145 —
throughput-only benchmark missed the correctness regression. PR #145
both fixed correctness and brought the corrected post-#144 baseline
(4.18 tok/s coherent) up to 9.81 tok/s.

## Where we were (state at end of session 2026-05-15)

Two arcs landed end-to-end this session:

### Arc A — CPU Q4K KV cache (PRs #132 / #134 / #135)

Three-step arc that wired a persistent `KvCache` through the CPU Q4K
forward path. All merged to `main`.

- **#132** plumbing — add `predict_q4k_hidden_with_cache`,
  `run_layer_with_ffn_with_cache`, `run_attention_block_with_kv_out_with_cache`
  as additive variants. Cache accepted but not consumed.
- **#134** consume — three operating modes (no cache / prefill+snapshot /
  decode-step). Headline correctness invariant proven by
  `cached_prefill_then_decode_matches_uncached_full_prefill` (1e-4 abs
  match against uncached `[N+1, hidden]` prefill).
- **#135** driver — `generate_via_cpu_q4k` and
  `generate_q4k_cpu_constrained_streaming_sampled_with_eos` now allocate
  a `KvCache` once per request and feed only the newly-sampled token on
  each decode step. Falls back to full-replay on hybrid MoE / cross-layer
  K/V share archs.

### Arc B — Gemma 3 CPU inference correctness (PRs #136 / #137)

Diagnosed and fixed six distinct bugs that combined to make Gemma 3 CPU
inference produce multilingual gibberish on every GGUF-extracted vindex.
Built llama.cpp's `eval-callback` locally and did a per-layer/per-stage
side-by-side diff to isolate each issue.

**#136** consolidates the first five fixes:

1. **Q4_K dequant row-stride** — writer pads each row to next
   256-element block; loader was reading `rows*cols` assuming unpadded
   layout, drifting 72 bytes/row from row 1 onwards. Affected every
   Q4_K weight whose cols aren't a multiple of 256 (Gemma 3 hidden=1152
   for 1B, 2560 for 4B is aligned but K/V are 256-wide).
2. **Gemma 3 GGUF norm key remap** — global `ffn_norm →
   post_attention_layernorm` mapping is right for Llama but WRONG for
   Gemma 3, which has 4 layer norms. Added arch-aware `remap_gemma_norms`
   that places `post_attention_norm` / `ffn_norm` / `post_ffw_norm` /
   `attn_q_norm` / `attn_k_norm` into the canonical HF slots.
3. **GGUF norm offset** — HF stores `w - 1.0`; GGUF stores `w` directly.
   Subtract 1.0 from all Gemma layer + QK + final norms on load so the
   runtime's `+1.0` recovers `w_eff`.
4. **Q5_0 dequant layout** — larql interleaved low/high nibbles
   (`[lo0, hi0, lo1, hi1, …]`); llama.cpp groups them in sequential
   halves (`[lo0..lo15, hi0..hi15]`). Same bug pattern as the C.5j Q6_K
   layout fix from PR #83 but for Q5_0. Affects every `attn_q` / `attn_k`
   / `ffn_gate` / `ffn_up` on Q4_K_M unsloth GGUFs.
5. **lm_head row-stride padding** — same writer pads lm_head; loader was
   missing the matching read-side handling.

**#137** adds the sixth:

6. **Vocab padding truncation** — Gemma 3 4B unsloth GGUF ships
   `token_embd` shape (262208, 2560) but `gemma3.vocab_size = 262144`
   (the extra 64 rows are SIMD alignment). The writer passed through
   the full GGUF shape while `index.json` recorded the logical vocab,
   so embed.bin and lm_head_q4.bin were sized for vocab=262208 while
   the config said 262144 → loader `ShapeError`. Truncate to logical
   vocab on write.

## Current bench

After Arc B fixes, on the same 48-core host (no CUDA available):

| Model | Path | Output | Decode tok/s |
|---|---|---|---:|
| Gemma 3 1B Q4_K_M | larql CPU `/v1/chat/completions` | coherent ("Cats are fascinating creatures…") | **0.257** |
| Gemma 3 4B Q4_K_M | larql CPU `/v1/chat/completions` | coherent ("Cats, with their independent spirits…") | **0.117** |
| Gemma 3 4B Q4_K_M | llama.cpp CPU (`-ngl 0`, full threads) | coherent | **14.1** |
| Qwen 3.6 27B Q4_K_S | larql `qwen35_forward` (parity oracle) | matches llama-eval-callback gt_rank=0 at every step | n/a |

**Remaining speed gap on Gemma 3 4B is ~120×.** This is **not a correctness
issue any more** — both implementations now produce sensible English. The
gap is the FFN-dequant-per-step bottleneck on larql's CPU path: `insert_q4k_layer_tensors`
re-dequantises every layer's gate/up/down Q4_K weights to f32 on every
decode step. Per-step FFN dequant time dominates; the KV cache savings on
attention compute are a small fraction of total wall clock.

The KV cache infra is correct (proven by integration tests against the
real Gemma 3 4B vindex — `tests/test_kv_cache_real_gemma3.rs`). It just
isn't where the bottleneck is on this path.

## Active arc — Qwen 3.6 35B-A3B vindex bridge (2026-05-16)

User picked Qwen3.6-35B-A3B as the new showcase model (replaces
Gemma 3 4B as the primary demo target). Memory:
[showcase MoE models](~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/project_showcase_moe_models.md),
[model storage](~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/reference_model_storage.md).

### Step 1 — Writer ✅ MERGED (PR #147, commit a7aed4f)

`vindex-qwen35moe-extraction` shipped:
- New `write_q4k/deltanet.rs` — Q4_K matmul tensors per linear layer
  (attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out).
- `write_q4k/moe_layers.rs` — extended for `ExpertFormat::PerExpert`
  (256 experts × 3 projections × 40 layers).
- `write_q4k/attn.rs` — skips linear layers; standard Q/K/V/O only
  fires on the 10 full-attention layers.
- `write_q4k/norms.rs` — DeltaNet small tensors (ssm_norm/dt/a/conv1d)
  per linear layer.
- `convert_cmd.rs:515` guard removed.

Live smoke conversion on `/tank/ai/Qwen/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
(21 GB) → `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex/` (60 GB structurally
complete, 542 MB deltanet bytes, 148 MB attn bytes, 40 × 256-expert
layer files, ssm_*small tensors in norms.bin, index.json reflects
`qwen35moe` + `full_attention_interval=4`).

### Step 2a — Storage reader ✅ MERGED (PR #149, commit 3e83070)

`vindex-qwen35moe-reader` **Phase 1** shipped:
- `crates/larql-vindex/src/index/storage/deltanet.rs` (NEW) —
  `VectorIndex::load_deltanet_q4k(dir)` + `deltanet_q4k_layer_data(layer)`
  + `has_deltanet_q4k()`.
- `MmapStorage`: `deltanet_q4k` + `deltanet_q4k_manifest` fields,
  `set_deltanet_q4k`, `has_deltanet_q4k` inherent methods.
- `VindexStorage` trait: `deltanet_q4k_layer_data` method,
  `DELTANET_TENSORS_PER_LAYER = 5` constant.
- `DeltanetManifestEntry { key, offset, length, format }` struct.
- Sparse-manifest awareness: accessor resolves by
  `layers.{layer}.` prefix + tensor-name suffix (not `layer * 5`
  arithmetic), so it tolerates the 10-of-40 attn / 30-of-40
  deltanet split.
- 3 new unit tests (load no-op when absent; 5-entry layer +
  partial-layer None; missing-format rejection).
- 925/925 unit tests pass.

### Step 2b.0 — Router write fixes ✅ MERGED (PRs #152 + #155)

Two coupled prerequisites for the reader's `Qwen35MoeFfnWeights::router`
field:
- **2b.0a (PR #152)**: widened the router-write gate in
  `write_q4k/norms.rs:114` from `is_hybrid_moe()` to
  `is_moe() && expert_format() != PackedMxfp4` so qwen35moe
  PerExpert also writes the router.
- **2b.0b (PR #155)**: fixed `Qwen35MoeArch::moe_router_key` to
  return the loader-canonical name `layers.{L}.ffn_gate_inp.weight`
  (was returning HF-style `layers.{L}.mlp.gate.weight` that no GGUF
  tensor matches because `normalize_gguf_key` has no remap rule
  for `ffn_gate_inp.`). Mixtral uses the same per-arch convention.

Smoke validated after #155: live conversion of the 21 GB GGUF
produced a vindex at `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v3/` whose
`weight_manifest.json` carries exactly **40
`layers.{L}.ffn_gate_inp.weight` entries** (one per MoE layer).
Writer arc structurally complete.

### Step 2b.1 — Norms reader ✅ NO-OP (confirmed via #154)

`crates/larql-vindex/src/format/weights/load.rs:534` already loads
every `kind::VECTOR` entry from `norms.bin` (via
`weight_manifest.json`) into a `HashMap<String, Vec<f32>>` keyed by
the full tensor name. DeltaNet small tensors + the post-#155
router land there automatically. No new reader code needed for
Step 2b's consumers — they read by key.

### Step 2b — adapter complete ✅ SMOKE-VALIDATED on v4 vindex

Live smoke result against
`/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v4/`:

```
smoke OK — 40 layers (30 linear + 10 full-attn), 40 MoE blocks,
           final_norm + lm_head_quant present
```

All Step 2b PRs merged:
- #157 stub
- #161 DeltaNet bridge (2b.2)
- #163 sparse attn accessor (2b.3a)
- #164 full-attn bridge (2b.3b)
- #165 MoE 256-expert packed bridge (2b.4)
- #166 orchestrator body (2b.5)
- #167 smoke test + 3 surface bugs (writer per-head norms, arch
  metadata drop, ssm_conv1d shape reshape) — 2b.6

`load_qwen35_weights_from_vindex(dir)` produces a structurally
complete `Qwen35Weights` from any qwen35moe vindex built by
PR #147+ with the router fixes. Load time: ~77 s on the 60 GB
vindex.

### Step 2b — adapter stub (historical reference)

`crates/larql-inference/src/attention/qwen35_load_vindex.rs` (NEW)
lands the module skeleton + `VindexLoadError` enum + public
signature `pub fn load_qwen35_weights_from_vindex(vindex_dir) ->
Result<Qwen35Weights, VindexLoadError>`. Arch sanity-check in the
body refuses non-qwen35 vindexes cleanly. Per-layer assembly is
`todo!()` until 2b.2..2b.4 land.

### Step 2b.2..2b.6 — adapter body (NEXT)

The hard part. Phase 2 of `vindex-qwen35moe-reader`.

The stub from PR #157 has the right module wiring + error type +
signature. The remaining work fills in the `todo!()` body with
per-layer assembly. Per `qwen35_forward.rs:98`, the struct has:
- `embed`: `ArcArray2<f32>` + `embed_quant: Option<QuantTensor>`
- `layers: Vec<Qwen35FullLayerWeights>` — each carries
  `block: Qwen35LayerWeights` (`Linear(DeltaNetLayerWeights)` or
  `Attention(Qwen35AttentionLayerWeights)`), `attn_post_norm`,
  dense + lazy SwiGLU FFN slots, and `moe: Option<Qwen35MoeFfnWeights>`.
- `final_norm`, `lm_head` + `lm_head_quant`, `ffn_dim`.

The forward dispatches based on which slot is populated (`*_quant`
takes precedence). Cleanest path: populate the `*_quant` slots from
vindex bytes by constructing `QuantTensor` instances over the
vindex shape + format tags. RAM-cheap; matches the GGUF lazy path.

Key dependency: `QuantTensor` constructor from raw bytes + shape +
tensor type. Need to verify that path exists in `larql-models` —
the GGUF mmap loader uses it; the vindex bytes should fit the same
contract (Q4_K block layout is identical between GGUF and vindex
after PR #137 / PR #136 stride fixes).

For DeltaNet small tensors (ssm_norm, ssm_dt, ssm_a, ssm_conv1d):
read from `norms.bin` via the existing `load_vindex_norms` path,
then plug into the matching `DeltaNetLayerWeights` fields.

For 256-expert MoE: parse `layers/layer_{LL}.weights` headers via
`larql_vindex::format::weights::write_layers::parse_layer_weights_header`,
then construct per-expert `QuantTensor` slices over the byte
ranges.

Estimated ~455 LoC across:
- 2b.2 DeltaNet layer assembly (~120)
- 2b.3 Full-attn layer assembly (~80)
- 2b.4 MoE 256-expert packed assembly (~150)
- 2b.5 Top-level orchestrator (~50)
- 2b.6 Smoke test against `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex` (~50)

Detailed design in `openspec/changes/vindex-qwen35moe-reader/step-2b-design.md`.

### Step 2c — server dispatch

Phase 3 of `vindex-qwen35moe-reader`. ~100 LoC.

When the server loads a vindex with `arch_family ∈ {qwen35,
qwen35moe}`, route `/v1/chat/completions` decoding through a
`qwen35_forward_step`-based helper instead of
`predict_q4k_hidden_with_cache`.

### Smoke gate

Live `/v1/chat/completions` against
`/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex` returns 200 with
non-degenerate text. Full parity vs llama.cpp is a separate change.

See `openspec/changes/vindex-qwen35moe-reader/tasks.md` for the
full breakdown.

### Out of scope on this arc

- DeepSeek V4 Flash MLA path — still parked.
- 40 GB `gate_vectors.bin` size optimisation — separate writer
  emitting MoE router weights in f32; structural correctness is
  achieved either way.
- Arc #1 (batched multi-row Q4_K × Q8_K matmul kernel) — still
  open as a general perf lever on top of either Gemma 3 4B or the
  new qwen35moe path once it reaches the server.
- Arc #2 (mmap-backed embed) — same.

---

## Open levers — pick one

After PRs #138 (dequant cache) + #139 (Q4kDirectFfn), the next ~10×
gap to llama.cpp is no longer FFN. Likely candidates:

### 1. Batched multi-row Q4_K × Q8_K kernel for prefill (largest remaining CPU lever)

After #146, prefill on long prompts (~189 tokens) still runs at 18
tok/s vs llama.cpp's 274 tok/s — the per-row direct-matvec loop pays a
rayon dispatch per row, which doesn't amortise. A batched kernel
(`q4k_q8k_matmul_into` over [N, hidden] × [out, hidden]^T) processing
all N rows in one rayon dispatch would close most of this 15× gap and
make larql competitive with llama.cpp CPU on long-context prefill.
Short-prompt chat (≤50 tokens) is already fast post-#146, so this is
specifically a long-context lever.

### 2. mmap-backed embed (~2.5 GB RAM win)

`load_model_weights_q4k` reads `embeddings.bin` and calls `to_vec()`
on the mmap → 2.5 GB heap allocation on top of the mmap'd pages.
Keeping the f32 embed as a memory-mapped view (via an
`ArcArray`-compatible wrapper that holds the mmap) would eliminate
the Vec copy. Lookup semantics unchanged. Closes most of the
remaining RAM gap to llama.cpp's 3.85 GB.

### 3. Heterogeneous deploy showcase (FFN on CPU, attention on GPU)

The original design point. Infrastructure exists (`--ffn-only`
service mode, `RemoteWalkBackend`, `/v1/walk-ffn` HTTP, gRPC walk_ffn,
`--join` grid coordination) but isn't proven end-to-end on a model
that doesn't fit in 24 GB VRAM. Qwen 3.5 122B-A10B Q4_K_M (~68 GB on
disk) is in `/tank/ai/qwen3.5-122b/gguf/Q4_K_M/` and would be the
right showcase — llama.cpp on -ngl 99 can't load it, and -ngl partial
is bottlenecked on PCIe. larql's per-component split should win here.

### 4. Wire CUDA into single-binary chat path

`larql_compute::default_backend()` doesn't return CUDA even with
`--features cuda` built. After #146 most matvecs went off f32 BLAS,
so the leverage of CUDA dispatch in the single-binary chat completion
is smaller than it would have been pre-#146, but it'd still help
prefill (still BLAS-bound) and unlock testing CUDA paths locally
without a multi-process deploy.

### 5. Padded-col QuantTensor variant (unblocks 1B direct lm_head)

Gemma 3 1B (`hidden=1152`) keeps using the f32 BLAS path because
`QuantTensor::from_raw` doesn't model padded-col layout. Adding a
`from_raw_padded(rows, cols, padded_cols)` constructor + matching
matvec semantics would extend every "direct Q4_K" arc to 1B too.

### 6. Profile-first (always-useful sanity check)

The ~10× gap to llama.cpp is now small enough that intuition isn't
reliable. Run a flamegraph / perf-annotate on 4B decode and identify
where the time actually goes. Likely suspects: K/V concat memcpy,
softmax in `gqa_attention_decode_step`, per-head `ndarray.dot` for
small matrices, missing thread parallelism (`q4k_q8k_matvec_into` may
be single-threaded — checking would inform a rayon arc that's
potentially bigger than #1).

### 7. Qwen 3.6 hybrid SSM Q4_K writer (option 2 of prior resume)

Unblock `larql convert gguf-to-vindex --quant q4k` on Qwen 3.6 35B-A3B
(currently rejected because the Q4_K attn writer doesn't handle DeltaNet
layers). Extends PR #125.

### 8. walk_path_audit resurrection (option 3 of prior resume)

`crates/larql-inference/examples/walk_path_audit.rs` is gated behind
`#[cfg(any())]` (PR #129). Split `MaskedGateIndex`'s `impl GateIndex`
block into separate `impl GateLookup` / `impl PatchOverrides` /
`impl FfnRowAccess` blocks. ~1-2 hour focused session.

### 9. Smaller drive-bys

- **Compute-crate clippy errors** — `larql-compute` has 4 pre-existing
  clippy errors (`identity_op`, `needless_range_loop`) that block any
  workspace-wide `cargo clippy --workspace --tests`. None caused by
  today's work; would unblock CI's `make lint`.
- **Q5_0 / Q8_0 round-trip tests** — the Q5_0 layout fix in #136 didn't
  add a unit test against `gguf` python lib's reference. A focused unit
  test would catch any regression. Same for Q8_0 (which is currently
  fine).
- **`larql-models` GGUF Gemma 4 norm remap** — the `remap_gemma_norms`
  helper in #136 dispatches on `family() == "gemma3" | "gemma4"` but I
  only validated against Gemma 3. Gemma 4 has additional features (PLE,
  layer scalar, cross-layer KV share) that interact differently; might
  need extra handling.
- **Bounded-time `/v1/chat/completions` integration test** — `tasks.md`
  open follow-up that would catch any future server hang of the form #123
  fixed.
- **Q4K bitwise passthrough** in convert — the GGUF Q4_K → f32 → vindex
  Q4_K roundtrip is currently lossy. With the row-stride and layout fixes
  consolidated, a direct byte-passthrough writer would skip the
  requantisation noise and double extract speed.

## Critical environment notes

- **No CUDA / Metal on this host** — only CPU available for benches.
- **Linux x86_64**, glibc 2.31+ (chat-completion `pick_template`
  deadlock fix from PR #123 stays applied).
- **`make ci` blocker**: workspace-wide `cargo clippy --workspace --tests`
  hits the pre-existing compute clippy errors (drive-by 4 above). Open
  PRs land via admin merge with all per-crate CI green; ` workspace-wide
  clippy is currently red on `main` independently of any feature work.
- **llama.cpp is built** at `/tmp/llama.cpp/build/bin/llama-{cli,gguf,eval-callback}`.
  Reusable for any future per-layer diff investigation.
- **Diagnostic tests committed** that you'll want for any further Gemma
  work:
  - `tests/test_kv_cache_real_gemma3.rs` — cache infra correctness against real vindex.
  - `tests/test_gemma3_layer_health.rs` — per-layer hidden-state stats.
  - `tests/test_gemma3_wv_dump.rs` — W_V tensor inspection.
  - `tests/test_gemma3_v_proj_source_compare.rs` — GGUF vs vindex diff (parameterised by `LARQL_TARGET_KEY`).
  - `tests/test_q6k_roundtrip.rs` / `tests/test_v_proj_writer_roundtrip.rs` — quant fidelity checks.

## Standing rules

- **Don't self-merge PRs unattended.** User authorises each merge
  individually via `merge and continue`. Standing auth is per-PR, not
  session-wide. Admin-merge is acceptable when CI's only failures are
  pre-existing main-branch issues unrelated to the PR.
  See `~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/feedback_unattended_merging.md`.
- **OpenSpec workflow** — every code change references a capability under
  `openspec/specs/<name>/spec.md`. Run `make traceability` after any
  source file's test line numbers shift.
- **Q4K vindex is the production fast-decode path.** Re-extracting a
  vindex picks up writer fixes — old vindexes on disk may need to be
  re-extracted after any writer-side fix lands.

## Quick start prompt for a fresh session

> Read RESUME_PROMPT.md. We're sitting at 9.81 tok/s on Gemma 3 4B
> Q4_K_M after #139 / #140 / #142 / #143 / #144 / #145 — 83.9× the
> 0.117 baseline, ~1.44× from llama.cpp's 14.1 tok/s (70% of llama.cpp).
> FFN, attention, and lm_head all flow through the same Q4_K × Q8_K
> rayon-parallel AVX2 path; BLAS pinned to 1 thread. Don't regress 1B
> coherence (hidden=1152, not Q8_K aligned, must keep falling back to
> WeightFfn / WeightAttention / f32 BLAS lm_head). Watch for the
> `with_q8k_for` allocator-reuse hazard fixed in #145 — any new caller
> that drops `x` and reallocates needs the content-fingerprint cache
> key to stay correct.
