# larql resume prompt — feed this to a fresh session

## Where we are (state at the end of 2026-05-12)

Two large lines of work have landed end-to-end. Both are merged to `main`.

### 1. Qwen 3.6 (Qwen3-Next) forward-pass correctness — DONE

Per-token greedy decode matches `llama-eval-callback` exactly on
Qwen3.6-27B Q4_K_S for the first 5 generated tokens, GT rank 0 at every
step. The decoded sequence is `<think>\n\n</think>\n\nHello` (logits
[28.18, 24.78, 25.47, 30.39, 21.66]).

The key fixes that got us here (newest first):
- **C.5j** (PR #83): Q6_K dequant layout was sequential; llama.cpp uses
  interleaved `y[l]/y[l+32]/y[l+64]/y[l+96]` per `l in 0..32` of each
  half with different scales. Hit `output.weight` (lm_head). Also flipped
  DeltaNet recurrence from paper-order (sk-before-decay) to **decay-first**
  matching llama.cpp's `ggml_compute_forward_gated_delta_net_one_chunk`.
- **C.5i** (PR #82): CYCLE GQA (`kh = h % h_k`) is correct, not BLOCK.
  C.5h had reverted to BLOCK based on token-rank, but token-rank is
  contaminated by downstream bugs; the **elementwise binary tensor
  parity oracle** showed pearson 0.9999 at layer-0 with CYCLE vs 0.77
  with BLOCK.
- Earlier C-phase fixes are listed in
  `openspec/changes/inference-qwen35-deltanet/C4-investigation-summary.md`.

### 2. Lazy-quantised matmul (RAM reduction) — Phase 1 → 2c shipped

Goal: stop dequantising 27 B params to f32 at GGUF load (was ~100 GiB
resident). PRs #86, #87, #88, #89, #90 land progressively more lazy
paths. All opt-in via env vars; default behaviour unchanged.

Current numbers on the same RTX 4090 host:

| Config | Prefill (t/s) | Decode (t/s) | RSS / VRAM |
|---|---:|---:|---:|
| llama.cpp CUDA GPU | 2097 | 50.60 | 14.76 GiB VRAM |
| llama.cpp CPU (-ngl 0) | 37.3 | 2.60 | ~16 GiB |
| larql baseline (dequant + BLAS) | 0.48 | 0.49 | 105.25 GiB |
| larql Phase 2 (lazy FFN, scalar) | 0.06 | 0.06 | 46.65 GiB |
| larql Phase 3 (+AVX2 +rayon) | 0.21 | 0.20 | 46.65 GiB |
| larql Phase 2b (+attn_qkv/gate/ssm_out) | 0.21 | 0.20 | 29.62 GiB |
| **larql Phase 2c (+full-attn q/k/v/o)** | **0.21** | **0.23** | **24.07 GiB** |

**RAM: 105 → 24 GiB (−77 %)** with argmax bit-exact. llama.cpp's CPU
~16 GiB target is ~8 GiB away.

Reproduce the bench:

```bash
# llama.cpp baseline
~/3rd-party/llama.cpp/build/bin/llama-bench \
  -m output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
  -p 128 -n 64 -r 2           # GPU
~/3rd-party/llama.cpp/build/bin/llama-bench -m ... -p 32 -n 8 -r 2 -ngl 0  # CPU

# larql lazy-quant bench (all our gains stacked)
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=16 LARQL_QWEN35_BENCH_DECODE=4 \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 \
cargo test -p larql-inference --release --lib real_gguf_qwen35_bench -- --nocapture

# parity test (must show GT rank 0 every step)
LARQL_QWEN35_GGUF=... LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 \
cargo test -p larql-inference --release --lib \
  real_gguf_qwen35_token_diff_vs_llama_cpp -- --nocapture
```

The bench / parity test live in
`crates/larql-inference/src/attention/qwen35_load.rs` (search for `fn
real_gguf_qwen35_bench` and `fn real_gguf_qwen35_token_diff_vs_llama_cpp`).

## Open levers — pick one to continue

The user's last "next?" answer pointed at **lazifying embed and the
remaining per-head SSM tensors** to close to ~16 GiB RAM (llama.cpp
parity on memory). The natural follow-ups, in priority:

1. **Phase 2d — embed lazy lookup.** The embed `{vocab=248320,
   hidden=5120}` Q4_K is ~5 GiB f32. Needs a new `QuantTensor::row_to_f32(token_id)`
   method that dequants one row on demand at embed-lookup time (NOT a
   matvec). Hot path is `weights.embed.row(token_id as usize).to_owned()`
   in `qwen35_forward_step` — patch that to use the lazy path when
   present. Add `LARQL_QWEN35_LAZY_EMBED=1` opt-in.
2. **Phase 3b — cache-tile batched Q4_K matvec.** Port llama.cpp's
   `mul_mat_q4k_q8k` style: tile rows for L1 reuse, batch quantize
   activation to Q8_K once per matvec, do the dot with `vpmaddubsw`.
   The existing `crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs` has
   the AVX2 plumbing already. This is the next big speed lever after
   rayon — should close most of the remaining 13× gap to llama.cpp CPU.
3. **oMLX-style paged KV + SSD cache.** Different axis; user flagged
   it earlier. Scope in `openspec/changes/qwen35-lazy-quant-matmul/` is
   not the home for this — would be a fresh openspec change. See
   `~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/reference_omlx_cache.md`
   for the design notes.
4. **Qwen3.6-35B-A3B MoE validation.** Memory wins compound on a 35 B
   model; likely needs MoE-specific architecture-handler tweaks.

## Pointers (where the work lives)

- **openspec change**: `openspec/changes/qwen35-lazy-quant-matmul/`
  (proposal, tasks, spec delta). Spec for the inference path:
  `openspec/changes/inference-qwen35-deltanet/specs/inference-gated-deltanet/spec.md`.
- **Bench numbers + protocol**:
  `openspec/changes/inference-qwen35-deltanet/bench-baseline.md`. Has
  the full evolution Phase 1 → 2c.
- **Investigation summary**:
  `openspec/changes/inference-qwen35-deltanet/C4-investigation-summary.md`.
- **Core impl files**:
  - `crates/larql-models/src/quant/lazy.rs` — `QuantTensor` + matvec
    dispatch (rayon row-parallel).
  - `crates/larql-models/src/quant/ggml/q4_k.rs` — scalar + NEON + AVX2
    `q4k_row_dot`.
  - `crates/larql-models/src/quant/ggml/q6_k.rs` — scalar + NEON
    `q6k_row_dot` (NEON forced to scalar after C.5j; TODO to port).
  - `crates/larql-models/src/loading/gguf.rs` — `load_gguf`,
    `load_gguf_lazy_lm_head`, `load_gguf_lazy_tensors`.
  - `crates/larql-inference/src/attention/{qwen35_forward,
    qwen35_load, qwen35_block, deltanet_block, deltanet_recurrence}.rs`
    — the Qwen 3.6 forward, layer bridge, and DeltaNet kernel.
- **llama.cpp parity oracle** (local clone, NOT this repo):
  `/home/ianblenke/3rd-party/llama.cpp/`. Modified
  `common/debug.cpp` to add `LLAMA_DUMP_BIN_DIR` env var that writes
  full f32 tensors to a directory. This is how elementwise parity
  was established in C.5i.
- **GGUF cache**:
  `output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf` (14.76 GiB).
  `output/gguf-cache/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` is
  also downloaded for future MoE work.

## Key conventions the next session must respect

- **Spec-first workflow.** Every code change references an OpenSpec
  capability under `openspec/specs/<capability>/spec.md` or, for
  in-flight work, `openspec/changes/<id>/specs/<capability>/spec.md`.
- **`make ci` before push.** Chains fmt, clippy, tests, traceability,
  openspec validate. The traceability gate will fail if
  `openspec/coverage/traceability.{md,json}` are stale — run `make
  traceability` and commit.
- **Workflow is feature-branch → PR → squash-merge to `main`.** Don't
  push directly to main. GitHub repo: `ianblenke/larql`. The `upstream`
  remote points at `chrishayuk/larql` and should NOT be the PR base —
  `gh repo set-default ianblenke/larql` if PRs auto-target the wrong
  base.
- **Token rank is a misleading metric** for parity work. The C.5h
  reversion of CYCLE GQA was driven by rank and was wrong. **Trust the
  elementwise binary tensor parity oracle** (`LLAMA_DUMP_BIN_DIR` on
  llama.cpp side, `LARQL_QWEN35_DUMP_BIN_DIR` on ours).

## How to ask the fresh session to pick up

> "Read RESUME_PROMPT.md in this repo. It summarises the current state.
> Phase 2c lazy-quant has just landed — RAM is at 24 GiB on Qwen3.6-27B
> Q4_K_S, llama.cpp's ~16 GiB target is ~8 GiB away. Pick one of the
> open levers and continue."

Or pick a specific one:

> "Read RESUME_PROMPT.md. Tackle **Phase 2d — embed lazy lookup**. Add
> `QuantTensor::row_to_f32(token_id)` and wire it into the embed
> lookup in `qwen35_forward_step`. Opt-in via `LARQL_QWEN35_LAZY_EMBED=1`.
> Re-bench and verify GT rank 0 still holds."
