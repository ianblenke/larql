# larql resume prompt — feed this to a fresh session

## Where we are (state at end of session 2026-05-16)

PR #139 (Q4kDirectFfn — direct Q4_K × Q8_K matvec FFN for CPU decode)
lands the next-biggest CPU speedup after the dequant cache (PR #138).

### Arc C — CPU decode-step direct Q4_K × Q8_K FFN (PR #139)

`predict_q4k_hidden_with_cache` now dispatches a new `Q4kDirectFfn`
backend on single-row, non-MoE, Q8_K-hidden-aligned layers. Skips f32
materialisation of FFN weights on the decode path; routes gate+up
through `q4k_q8k_gate_up_into` and down through `q4k_q8k_matvec_into` /
`q6k_q8k_matvec_into` (kernels from PRs #102–#119).

Gemma 3 1B (hidden=1152) falls back to WeightFfn — not Q8_K-aligned.
Gemma 3 4B (hidden=2560) engages the direct path.

| Model | Path | Before | After | Speedup |
|---|---|---:|---:|---:|
| Gemma 3 4B Q4_K_M | larql CPU `/v1/chat/completions`, 50 tok | 0.117 tok/s | **1.45 tok/s** | **12.4×** |
| Gemma 3 4B Q4_K_M | larql CPU `/v1/chat/completions`, 150 tok | 0.117 tok/s | **1.36 tok/s** | **11.6×** |

llama.cpp CPU 14.1 tok/s reference unchanged.

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

## Open levers — pick one

After PRs #138 (dequant cache) + #139 (Q4kDirectFfn), the next ~10×
gap to llama.cpp is no longer FFN. Likely candidates:

### 1. Direct Q4_K × Q8_K attention Q/K/V/O (probably moderate)

The decode-step attention Q/K/V/O projections still read f32 from the
dequant cache via BLAS GEMV. Direct path mirrors the Q4kDirectFfn arc:
new `Q4kDirectAttention` adapter that calls `q4k_q8k_matvec_into` on
the vindex's Q4_K Q/K/V/O bytes. Expected ~25% additional speedup —
the projection BW is ~1/5 of FFN BW.

Implementation surface: `crates/larql-inference/src/attention/decode.rs`
`run_attention_block_decode_step_backend` reads `weights.tensors` for
W_Q/K/V/O. Either thread `index: &VectorIndex` through
`run_attention_block_with_kv_out_with_cache` (~8 call sites) or add an
`AttentionBackend` trait. Q8_K hidden alignment guard same as FFN
(gates Gemma 3 1B back to WeightAttention).

### 2. Profile-first (recommended before more arcs)

The ~10× gap to llama.cpp is now small enough that intuition isn't
reliable. Run a flamegraph / perf-annotate on 4B decode and identify
where the time actually goes. Likely suspects: K/V concat memcpy,
softmax in `gqa_attention_decode_step`, per-head `ndarray.dot` for
small matrices, missing thread parallelism (`q4k_q8k_matvec_into` may
be single-threaded — checking would inform a rayon arc that's
potentially bigger than #1).

### 3. Qwen 3.6 hybrid SSM Q4_K writer (option 2 of prior resume)

Unblock `larql convert gguf-to-vindex --quant q4k` on Qwen 3.6 35B-A3B
(currently rejected because the Q4_K attn writer doesn't handle DeltaNet
layers). Extends PR #125.

### 4. walk_path_audit resurrection (option 3 of prior resume)

`crates/larql-inference/examples/walk_path_audit.rs` is gated behind
`#[cfg(any())]` (PR #129). Split `MaskedGateIndex`'s `impl GateIndex`
block into separate `impl GateLookup` / `impl PatchOverrides` /
`impl FfnRowAccess` blocks. ~1-2 hour focused session.

### 5. Smaller drive-bys

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

> Read RESUME_PROMPT.md. We're sitting at 1.36 tok/s on Gemma 3 4B
> Q4_K_M after #138 (dequant cache) + #139 (Q4kDirectFfn) — ~12×
> better than the 0.117 baseline, and ~10× from llama.cpp's 14.1
> tok/s. The bottleneck has moved off FFN. Run lever 2 (profile-first)
> on a 50-token Gemma 3 4B completion to identify where wall-clock
> time actually goes before picking lever 1 (Q4kDirectAttention) or a
> threading arc. Don't regress 1B coherence (hidden=1152, not Q8_K
> aligned, must keep falling back to WeightFfn).
