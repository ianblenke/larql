# Phase C.4–C.5 investigation summary

Status as of 2026-05-11 after ~30 sub-phases on the
`inference-qwen35-deltanet` openspec change.

## Where we are

- **Plumbing complete.** GGUF → Qwen35Weights bridge, tokenizer
  round-trip, prefill + decode all work end-to-end on real
  Qwen3.6 27B Q4_K_S in ~3 s/token CPU. 113 unit tests + 9
  real-GGUF env-gated tests pass.
- **Parity oracle operational.** `llama-eval-callback` is built;
  per-layer tensor comparison vs llama.cpp now drives bug
  bisection. Layer-0 `x_norm`, `qkv_conv`, and `final_out` all
  verified BIT-EXACT or near-bit-exact (f32 precision) at sampled
  positions.
- **4 confirmed bug fixes landed** (PRs #60, #63, #74-reverts-#69,
  #76).
- **1 known remaining issue**: layer-0 `linear_attn_out` is ~3×
  larger than llama.cpp's. Source unidentified — the `ssm_out`
  matmul input is bit-exact but output diverges. Layer 1's
  `attn_norm` also ~3× larger, indicating consistent bounded
  amplification (not compounding per layer).

## Confirmed fixes (in `main`)

1. **PR #60** — DeltaNet `[s_v, h_v]` reshape was scrambling
   head/dim indices. Now: reshape via
   `(n_v_heads, head_v_dim).reversed_axes()`.
2. **PR #63** — `split_q_gate` assumed split-half layout; actual
   Qwen 3.6 layout is interleaved-per-head per llama.cpp
   `qwen35.cpp:220-244`.
3. **PR #69 (reverted in PR #74)** — Tried Gemma-style `(1+w)`
   RMSNorm offset at bridge load. **WRONG** — empirical inspection
   via `llama-eval-callback` showed GGUF stores `(1+w)` baked in
   (stored weights ≈ 1.0). PR #74 reverts the double-application.
4. **PR #76 (C.5c) — HEAD-MAJOR FLATTEN** of DeltaNet recurrence
   output. The naïve `o.into_iter()` flatten produced DIM-MAJOR
   flat layout, scrambling `rms_norm_heads` slices across multiple
   heads and producing 46× over-amplification. Transpose first
   so layout becomes head-major (matching HF Qwen3-Next's
   `out.reshape(B, S, -1)` of an `[..., n_v_heads, head_v_dim]`
   tensor). **Step-1 GT rank: 216,947 → 7,617 (top 3%).**

## Token-rank progression

For ground-truth token at step 1 (` [` = 498):

| State | step-1 GT rank | step-1 GT logit |
|---|---:|---:|
| Pre-C.4 fixes | n/a | n/a |
| After PR #64 (cycle GQA, wrong) | 119,184 | 0.075 |
| After PR #69 ((1+w) + block GQA) | 185,292 | -1.978 |
| After PR #74 (revert 1+w, keep block) | 216,947 | -2.045 |
| **After PR #76 (head-major flatten)** | **7,617** | **+3.505** |

## Parity oracle: verified bit-exact (or near) at layer 0

Using `llama-eval-callback` to dump tensors during a real Qwen3.6-27B
forward, then `LARQL_QWEN35_DUMP_L0=1` to dump ours:

| Tensor | Match status |
|---|---|
| `embed` (input embedding lookup) | implied bit-exact |
| `attn_norm` (x_norm output) | **BIT-EXACT** at first-3 + last-3 |
| `attn_qkv` matmul (qkv_mixed) | ~1-4% per-element Q5_K dequant noise |
| `conv1d + silu` (qkv_conv) | **near-bit-exact at f32 precision** |
| `recurrence` (o) | matches expected (small magnitude) |
| `ssm_norm + silu(z)` (final_out, pre-ssm_out) | **BIT-EXACT** at first-3 + last-3 |
| `ssm_out` matmul (linear_attn_out) | **3× too large** (16.7 vs ~5.7) ← bug |

## Hypotheses ruled out

- **lm_head row outlier** (C.4k) — row norms within population σ.
- **ssm_a sign error** (C.4s) — all 48 values negative as
  expected.
- **GGUF `ssm_a` storage** (C.4v) — pre-computed `-exp(A_log)`
  matches llama.cpp's direct multiplication.
- **HF chunkwise recurrence formula** (C.4w) — paper's recurrent
  form is correct at chunk_size=1.
- **`(1+w)` RMSNorm offset** (C.5a) — GGUF already pre-applies.
- **Embedding scale / softcapping** — verified absent in HF and
  llama.cpp.

## Verified consistent with HF Qwen3-Next + llama.cpp

After reading both references:

- RMSNorm semantics (epsilon, mean, weight broadcast)
- Softplus / sigmoid pointwise math
- Attention scale = `1/sqrt(head_dim)`
- DeltaNet decay: `g_exp = exp(ssm_a * softplus(alpha + dt))`
- Conv1D time direction (weight[0] = oldest token)
- Conv state shift-and-insert pattern
- Residual add pattern (attn block to original x, FFN to residual)
- Pre-attention norm placement (inside block)
- Post-attention norm placement (between residual and FFN input)
- Final RMSNorm + lm_head sequence
- RoPE pairing convention (split-half for NEOX/IMROPE)
- MRoPE-text-only reduces to partial-RoPE at first `rotary_dim` dims
- `ggml_l2_norm` per-head normalization
- Q+gate split convention (PR #63)
- DeltaNet GQA block pattern (PR #76 / `repeat_interleave`)
- DeltaNet recurrence output flatten: HEAD-MAJOR (PR #76)
- Embedding lookup: row by token id, no scale

## Remaining bug investigation paths

The 3× `linear_attn_out` discrepancy with bit-exact `final_out`
input means the `ssm_out` matmul produces 3× larger output than
llama.cpp's. Candidate explanations to investigate:

1. **Q5_K dequant precision differences for `ssm_out.weight`** —
   middle-element noise that doesn't show in abbreviated first/last
   prints but compounds through the matmul.
2. **Matmul precision** — llama.cpp may use f16 intermediate; we
   use full f32 BLAS.
3. **A missing per-head scale factor** between recurrence output
   and final projection.
4. **A possible 3× normalization factor** we're missing.

The 3× ratio is consistent across `linear_attn_out` (l2 16.7 vs
~5.7) and `attn_norm-1` (l2 44.8 vs ~14.3), suggesting a single
multiplicative cause not compounding through layers.

## Diagnostics infrastructure

Env-gated, all checked in:

- `LARQL_QWEN35_GGUF=/path/to/gguf` enables 9 real-GGUF tests
- `LARQL_QWEN35_TRACE=1` per-layer residual stream l2 trace
- `LARQL_QWEN35_DUMP_L0=1` layer-0 tensor first/last-3 dumps
  (x_norm, qkv_mixed, qkv_conv, o(recurrence), final_out,
  linear_attn_out)
- `LARQL_QWEN35_DUMP_FINAL=1` x_final dump pre-lm_head

llama.cpp side:
- `llama-eval-callback` binary built and ready (in
  `~/3rd-party/llama.cpp/build/bin/`); dumps all tensors during
  a forward pass.

## Next session's concrete agenda

The parity oracle has reduced the bug from "somewhere in 64
layers" to "in the `ssm_out` matmul or its immediate inputs."
Pick one:

1. **Dump more positions of `final_out`** (e.g. positions 1024,
   2048, 3072, 5000) to verify bit-exact across the full 6144
   vector. If middle differs, the 3× is from Q5_K input noise
   amplified by matmul.

2. **Dump `ssm_out.weight` row norms** for a few rows; compare
   to magnitudes implied by llama.cpp's `linear_attn_out` values.

3. **Try a HIGHER-precision GGUF** (Q8_0 if available) — if the
   3× discrepancy disappears, the bug is Q5_K dequant precision
   (not a logic bug).

4. **Switch token-diff test to ground-truth feeding** for clean
   per-step parity comparison.
