# Tasks

This change is documentation-only. The action items below capture the work each gap implies; they're held outside this change so root-cause investigation can produce focused fix PRs.

## Open follow-ups

- [ ] **Gap 1**: investigate `/v1/chat/completions` hang on Gemma 3 4B vindex (task #127 in the agent task tracker).
  - Inspect `crates/larql-server/src/routes/openai/chat.rs`.
  - Inspect `crates/larql-inference/src/vindex/q4k_forward/walk_ffn.rs` for down-features-missing handling.
  - Check whether `--feature-major-down` extraction is silently required by the chat route.
  - Add an integration test that hits `/v1/chat/completions` on a small fixture vindex with a strict timeout, so this regression doesn't slip past CI again.

- [ ] **Gap 2**: implement MoE branch in `convert gguf-to-vindex`.
  - Pattern-match `crates/larql-vindex/src/format/weights/write_q4k/moe_layers.rs` (the working safetensors-MoE writer).
  - GGUF MoE tensors are 3-D `[expert_count, intermediate, hidden]` per layer — translate to the per-expert 2-D layout the vindex MoE writer expects.
  - Test against the Qwen 3.6-35B-A3B unsloth GGUF; verify expert weight files are non-zero and a `larql serve` warmup completes.

- [ ] Once both gaps are closed, **re-run the head-to-head bench** and update `cpu-kquant-matvec-correctness-avx2/proposal.md` with the end-to-end tok/s + VRAM numbers (replacing the extrapolated estimates).
