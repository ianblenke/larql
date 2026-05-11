# Phase C.4 investigation summary

Status as of 2026-05-11 after ~20 sub-phases on the
`inference-qwen35-deltanet` openspec change.

## Where we are

- **Plumbing complete.** GGUF → Qwen35Weights bridge, tokenizer
  round-trip, prefill + decode all work end-to-end on real
  Qwen3.6 27B Q4_K_S in ~3 s/token CPU. 111 unit tests + 7
  real-GGUF env-gated tests pass.
- **3 confirmed layout bug fixes landed** (PRs #60, #63, #64).
- **1 remaining correctness bug.** Per C.4r token-diff vs
  llama-cli ground truth: ground-truth tokens (`|-`, ` [`,
  `Start`, ` thinking`, `]`) all land at rank 100,000+ out of
  248,320 vocab in our logit distribution. Our argmax tokens
  get logits ~10, ground truth gets ~0. **Output is in the
  wrong basis direction.**

## Confirmed fixes (in `main`)

1. **PR #60** — DeltaNet `[s_v, h_v]` reshape was scrambling
   head/dim indices. Now: reshape via
   `(n_v_heads, head_v_dim).reversed_axes()`.
2. **PR #63** — `split_q_gate` assumed split-half layout; actual
   Qwen 3.6 layout is interleaved-per-head per llama.cpp
   `qwen35.cpp:220-244`.
3. **PR #64** — DeltaNet GQA used block pattern (`h / repeat`);
   ggml_repeat is cycle (`h % h_k`).

## Hypotheses ruled out (via diagnostics)

- **lm_head row outlier** (C.4k) — row norms within population σ.
- **ssm_a sign error** (C.4s) — all 48 values negative as
  expected.
- **L00 first-block-output magnitude** (C.4l observation) —
  likely init-state characteristic, not a bug.

## Verified consistent with llama.cpp reference

After reading `llama.cpp/src/models/qwen35.cpp` and related
ggml ops:

- RMSNorm semantics (no offset, no weight bias add)
- Softplus / sigmoid pointwise math
- Attention scale = `1/sqrt(head_dim)`
- DeltaNet decay: `g = exp(ssm_a * softplus(alpha + dt))`
- Conv1D time direction (weight[0] = oldest token)
- Conv state shift-and-insert pattern
- Residual add pattern (attn block to original x, FFN to residual)
- Pre-attention norm placement (inside block)
- Post-attention norm placement (between residual and FFN input)
- Final RMSNorm + lm_head sequence
- RoPE pairing convention (split-half for NEOX/IMROPE)
- MRoPE-text-only reduces to partial-RoPE at first `rotary_dim` dims
- `ggml_l2_norm` per-head normalization
- ssm_norm + silu(z) gating order
- Q+gate split convention (PR #63)
- DeltaNet GQA cycle pattern (PR #64)
- Embedding lookup: row by token id, no scale
- All weights from the GGUF have consumers in our forward

## Still-possible bug locations

Ranked by likelihood, the remaining bug is most likely in:

1. **DeltaNet output flatten layout** (C.4p attempt). A
   `o.t().to_owned()` fix to produce head-major flat was
   empirically WORSE (regressed to single attractor), but the
   theoretical analysis says it should be right. Could be (a)
   the fix is wrong, or (b) the fix is right and unmasks
   another bug. Cannot disambiguate without a per-layer
   parity oracle.

2. **`attn_qkv` post-conv split layout** — within Q/K/V slabs,
   per-head interleaving may differ from simple
   `flat[h * head_dim + d]` head-major.

3. **`attn_post_norm` placement details** — design.md says it
   applies to the residual; double-check against llama.cpp's
   exact tensor flow.

4. **`final_norm` weight offset** — Gemma-style adds 1.0;
   verify Qwen 3.6 doesn't.

5. **Embedding scale / final-logit softcapping** — defaults
   should be 1.0 / None; verify.

## Diagnostics infrastructure that's landed

All env-gated on `LARQL_QWEN35_GGUF=/path/to/gguf` so unit-test
CI passes without the GGUF:

- `real_gguf_qwen35_bridge_smoke` (C.4f) — load + bridge shape
  verification.
- `real_gguf_qwen35_forward_one_token_smoke` (C.4g) — single
  forward, finite logits check.
- `real_gguf_qwen35_multi_token_argmax_decode` (C.4h) —
  multi-token cosine cross-step.
- `real_gguf_qwen35_tokenizer_roundtrip` (C.4i) — text→tokens→
  forward→tokens→text.
- `real_gguf_qwen35_chat_prompt_forward` (C.4j) — chat-template
  prompt.
- `real_gguf_qwen35_lm_head_row_norms_diagnostic` (C.4k) —
  lm_head + embed row norm statistics.
- `real_gguf_qwen35_chat_prompt_temperature_sampling` (C.4m) —
  temperature sampling.
- `real_gguf_qwen35_token_diff_vs_llama_cpp` (C.4r) — rank of
  ground-truth tokens in our logits.
- `real_gguf_qwen35_ssm_a_sign_diagnostic` (C.4s) — ssm_a values.

Tracer: `LARQL_QWEN35_TRACE=1` env var prints per-layer
residual stream l2/max norms.

## Next session's concrete agenda — Phase C.5 parity oracle

The remaining bug requires per-layer hidden-state diff vs
llama.cpp. Suggested approach:

1. **Capture llama.cpp per-layer hidden states.** Options:
   a. Use llama.cpp's tensor callback hook (build with
      `GGML_PERF=1` or similar) to dump named tensors per layer
      to disk.
   b. Patch llama.cpp temporarily to `ggml_print` specific
      tensors (`attn_norm`, `attn_residual`, `attn_post_norm`,
      `ffn_residual`, `post_ffn`, final `cur`) for layer 0
      only, for a one-token prompt. ~50 LoC patch.
   c. Use `llama-perplexity` or write a small standalone
      ggml program that runs the forward and dumps tensors.

2. **Run our forward in the same harness.** Capture the
   equivalent tensors at each step.

3. **Diff layer 0 first.** Compare:
   - `attn_norm(x)` — pre-norm input
   - `block_out` from `qwen35_attention_block_step` or
     `deltanet_block_step`
   - `residual = x + block_out`
   - `attn_post_norm(residual)`
   - `ffn_out`
   - `final = residual + ffn_out`

4. **Bisect.** First diverging op IS the bug. With layer 0
   isolated, the search space drops from "anywhere in 64
   layers × multiple ops" to "this specific op in this
   specific block kind".

5. **Generalize.** Once layer 0 matches, run multi-layer to
   verify the residual stream stays aligned. If layer 0
   matches but later layers diverge, the bug is a cumulative
   numerical drift — harder but bounded.

## Time budget for Phase C.5

Estimate: 4-8 hours of focused work to build the parity
harness, plus 2-4 hours to bisect the remaining bug given the
harness exists. Could go faster if the bug is in one of the
ranked-1 hypotheses and a direct fix attempt works.
