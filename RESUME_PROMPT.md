# larql resume prompt — feed this to a fresh session

## Where we are (state at end of session 2026-05-12)

Three lines of work have landed end-to-end. All merged to `main`.

### 1. Qwen 3.6 (Qwen3-Next) forward-pass correctness — DONE

Per-token greedy decode matches `llama-eval-callback` exactly on
Qwen3.6-27B Q4_K_S for the first 5 generated tokens, GT rank 0 at
every step. Decoded sequence: `<think>\n\n</think>\n\nHello` (logits
[28.18, 24.78, 25.47, 30.39, 21.66]).

Key bug fixes (newest first):
- **C.5j** (PR #83): Q6_K dequant layout was sequential; llama.cpp uses
  interleaved `y[l]/y[l+32]/y[l+64]/y[l+96]` with different scales.
  Hit `output.weight` (lm_head). Also flipped DeltaNet recurrence from
  paper-order to **decay-first** matching
  `ggml_compute_forward_gated_delta_net_one_chunk`.
- **C.5i** (PR #82): CYCLE GQA (`kh = h % h_k`) is correct. Token-rank
  hid the bug; the **elementwise binary tensor parity oracle**
  exposed it (pearson 0.9999 at layer 0 with CYCLE vs 0.77 with BLOCK).
- Earlier C-phase fixes:
  `openspec/changes/inference-qwen35-deltanet/C4-investigation-summary.md`.

### 2. Lazy-quantised matmul — Phases 1 → 2d (RAM 105 → 20 GiB)

Stop dequantising 27 B params to f32 at load time. PRs #86 through
#91 progressively lazy-load every big tensor. Opt-in via
`LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1`.

### 3. GPU dispatch — Phase E.1/E.2 (PR #92, just landed)

Wires `larql_compute::cuda::CudaBackend` (existing `q4k_direct` /
`q6k_direct` kernels) into `qwen35_forward_step`'s lm_head and 192
FFN matvecs/token. Opt-in via `--features cuda` +
`LARQL_QWEN35_GPU=1`. **The DeltaNet recurrence still runs on CPU**
and that's now the dominant bottleneck.

## Current bench (Qwen3.6-27B Q4_K_S, RTX 4090, prefill 16 / decode 4)

| Config | Decode (t/s) | RSS / VRAM |
|---|---:|---:|
| llama.cpp CUDA GPU | **50.60** | 14.76 GiB VRAM |
| llama.cpp CPU (-ngl 0) | 2.60 | ~16 GiB |
| larql baseline (full dequant + BLAS) | 0.49 | 105.25 GiB |
| larql Phase 2 (lazy FFN, scalar) | 0.06 | 46.65 GiB |
| larql Phase 3 (+AVX2 +rayon) | 0.20 | 46.65 GiB |
| larql Phase 2b (+DN projs lazy) | 0.20 | 29.62 GiB |
| larql Phase 2c (+full-attn lazy) | 0.23 | 24.07 GiB |
| larql Phase 2d (+embed lazy) | 0.23 | 19.99 GiB |
| **larql Phase E.1/E.2 (+ GPU FFN & lm_head)** | **0.28** | 21 GiB host |

**RAM** : 105 → 20 GiB (−81 %), within ~4 GiB of llama.cpp CPU's
target.

**Speed** : still 180× off llama.cpp GPU. The current Phase E.1/E.2
gain is modest because **DeltaNet recurrence remains on CPU** —
48 layers × per-head state update per token is now the steady-state
bottleneck (~3.6 s/token at decode).

Reproduce:

```bash
# All-on-GPU lazy bench (current state, ~0.28 t/s)
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=16 LARQL_QWEN35_BENCH_DECODE=4 \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_bench -- --nocapture

# Parity test (must show GT rank 0 every step)
LARQL_QWEN35_GGUF=... LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 \
LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_token_diff_vs_llama_cpp -- --nocapture

# llama.cpp baseline (for comparison)
~/3rd-party/llama.cpp/build/bin/llama-bench \
  -m output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf -p 128 -n 64 -r 2
```

## Open levers — pick one to continue

User's previous "next?" answer pointed at GPU work. Phase E.1/E.2
landed but the recurrence bottleneck still caps us at 0.28 t/s.

1. **Phase E.4 — CUDA DeltaNet recurrence + Conv1D kernels.** The
   real unlock. Per-head state matrix is 128×128 f32 (64 KB), fits
   in shared memory on Ampere/Ada SM-89. Mirrors llama.cpp's
   `ggml_compute_forward_gated_delta_net_one_chunk` which we already
   diffed bit-exact in Phase C.5j (decay-first algorithm). One CUDA
   block per (batch, head). Conv1D-with-state is straightforward
   depthwise 4-tap × 10240 channels. Estimated ~600 LoC + cudarc
   PTX plumbing. Expected: ~10 t/s decode.
2. **Phase E.3 — Plumb DeltaNet projections + full-attn q/k/v/o
   through the GPU backend.** Pure plumbing now that E.1/E.2 is in
   — same `matvec_with_backend` wrapper just at more dispatch sites.
   Marginal win without E.4 (recurrence still serial CPU).
3. **Phase E.6 — Device-resident weights + KV cache + CUDA Graphs.**
   Stops the per-matvec host↔device round-tripping (~1.5 ms/token
   overhead). Estimated ~30 t/s after this lands (≈ 60 % of
   llama.cpp GPU). The remaining gap is fused-attention / SwiGLU
   fusion / Flash-Attention.
4. **Pivot to paged KV + SSD cache (oMLX-style).** Different axis —
   multi-turn reuse for OpenAI API serving. Independent of perf.
   Scope notes in
   `~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/reference_omlx_cache.md`.
5. **Qwen3.6-35B-A3B MoE validation.** Memory wins compound; likely
   uncovers MoE-specific architecture-handler bugs.

## Pointers (where the work lives)

- **GPU openspec change**: `openspec/changes/qwen35-gpu-forward/`
  (proposal, tasks E.0–E.6, spec delta). E.0/E.1/E.2 complete;
  E.3–E.6 queued.
- **Lazy-quant openspec change**:
  `openspec/changes/qwen35-lazy-quant-matmul/`. Phases 1 / 2 / 2b /
  2c / 2d shipped.
- **Bench numbers + protocol**:
  `openspec/changes/inference-qwen35-deltanet/bench-baseline.md`.
  Full evolution baseline → Phase E.1/E.2.
- **Investigation summary**:
  `openspec/changes/inference-qwen35-deltanet/C4-investigation-summary.md`.
- **Core impl files**:
  - `crates/larql-models/src/quant/lazy.rs` — `QuantTensor` + matvec
    dispatch (rayon row-parallel), `row_to_f32` for embed lookup.
  - `crates/larql-models/src/quant/ggml/{q4_k,q6_k}.rs` — scalar +
    NEON + AVX2 row dots. NEON Q6_K forced to scalar after C.5j
    (TODO to port interleaved layout).
  - `crates/larql-models/src/loading/gguf.rs` — `load_gguf`,
    `load_gguf_lazy_lm_head`, `load_gguf_lazy_tensors`. The lazy
    loader handles the embed special case (own struct field).
  - `crates/larql-inference/src/attention/quant_dispatch.rs` — GPU
    bridge between `QuantTensor` and `QuantMatVec`.
  - `crates/larql-inference/src/attention/{qwen35_forward, qwen35_load,
    qwen35_block, deltanet_block, deltanet_recurrence}.rs` — Qwen 3.6
    forward, layer bridge, DeltaNet kernel. The CPU recurrence to
    port to CUDA in E.4 lives in `deltanet_recurrence.rs::delta_net_step`.
- **Existing CUDA infrastructure**:
  - `crates/larql-compute/src/cuda/q4k_mmvq.rs` (893 LoC), `q6k_mmvq.rs`
    (440 LoC), `q4k_direct.rs` (218 LoC), `matmul.rs`, `attn.rs`,
    `cache.rs`, `dequant.rs`, `backend.rs`. The `QuantMatVec` trait
    is the dispatch surface (lives at
    `crates/larql-compute/src/backend/quant_matvec.rs:40`).
- **llama.cpp parity oracle** (local clone, NOT this repo):
  `/home/ianblenke/3rd-party/llama.cpp/`. Modified `common/debug.cpp`
  to add `LLAMA_DUMP_BIN_DIR` env var that writes full f32 tensors
  to a directory. This established elementwise parity in C.5i and
  is the right oracle for any GPU-vs-CPU debugging.
- **GGUF cache**:
  `output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf` (14.76 GiB).
  `output/gguf-cache/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
  also downloaded for MoE follow-up.

## Key conventions the next session must respect

- **Spec-first workflow.** Every code change references an OpenSpec
  capability under `openspec/specs/<capability>/spec.md` or, for
  in-flight work,
  `openspec/changes/<id>/specs/<capability>/spec.md`.
- **`make ci` before push.** Chains fmt, clippy, tests, traceability,
  openspec validate. The traceability gate will fail if
  `openspec/coverage/traceability.{md,json}` are stale — run `make
  traceability` and commit.
- **Workflow is feature-branch → PR → squash-merge to `main`.** Do
  not push directly to main. GitHub repo: `ianblenke/larql`. The
  `upstream` remote points at `chrishayuk/larql` and should NOT be
  the PR base — `gh repo set-default ianblenke/larql` if PRs
  auto-target the wrong base.
- **Token rank is a misleading metric** for parity work. C.5h's
  reversion of CYCLE GQA was driven by rank and was wrong. **Trust
  the elementwise binary tensor parity oracle**
  (`LLAMA_DUMP_BIN_DIR` on llama.cpp side,
  `LARQL_QWEN35_DUMP_BIN_DIR` on ours).
- **GPU work is more valuable than CPU work.** The user called this
  out and was right. CPU AVX2 / rayon plateaus around 2-3 t/s at
  best (matching llama.cpp CPU); the 4090 is sitting idle until
  Phase E.4 / E.6 land. Default the planning to GPU.

## How to ask the fresh session to pick up

Generic resume:

> Read RESUME_PROMPT.md in this repo. Pick one of the open levers
> and continue. The user's most recent direction was Phase E (GPU
> forward); E.1/E.2 landed in PR #92 but the headline 0.28 t/s is
> still 180× off llama.cpp GPU because the DeltaNet recurrence is
> still CPU.

Targeted (recommended):

> Read RESUME_PROMPT.md. Tackle **Phase E.4 — CUDA DeltaNet
> recurrence + Conv1D kernels**. The CPU reference is
> `crates/larql-inference/src/attention/deltanet_recurrence.rs::
> delta_net_step` (decay-first, verified bit-exact in C.5j against
> llama.cpp's `ggml_compute_forward_gated_delta_net_one_chunk`).
> Per-head state matrix is 128×128 f32, fits in shared memory.
> One CUDA block per (batch, head). Reuse the cudarc PTX plumbing
> from `crates/larql-compute/src/cuda/q4k_mmvq.rs`. Bench delta
> goes in `openspec/changes/inference-qwen35-deltanet/bench-baseline.md`.
> Parity check: `real_gguf_qwen35_token_diff_vs_llama_cpp` under
> `LARQL_QWEN35_GPU=1 LARQL_QWEN35_LAZY_FFN=1` must still emit
> `[<think>, \n\n, </think>, \n\n, Hello]` with GT rank 0.

Or for the long arc:

> Read RESUME_PROMPT.md. Work through `openspec/changes/qwen35-gpu-forward/tasks.md`
> sequentially: E.3 (plumb attn projections through GPU backend)
> first, then E.4 (CUDA recurrence kernel), E.5 (full softmax-attn
> on GPU via existing cuda/attn.rs), then E.6 (device-resident
> weights + CUDA Graphs). Each phase its own PR. Parity oracle
> stays green every step.
