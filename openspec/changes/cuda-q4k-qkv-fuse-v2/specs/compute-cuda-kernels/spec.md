## ADDED Requirements

### Requirement: Q/K/V mmvq fusion path SHALL be implemented and benched even when the result is a regression

The captured-decode pipeline SHALL contain a code path that, when
the layer's Q/K/V weights are all Q4_K, fuses them into one
`[q_dim + 2 * kv_dim] × hidden` packed weight stream and issues
one mmvq instead of three. The path SHALL:

- Use `arc_q4k_qkv_concat_device_buf(wq, wk, wv)` to lazy-allocate
  the concatenated device buffer, keyed by the (host-pointer,
  length) triple.
- Write the fused mmvq output into `scratch.qkv`.
- Pass slice views of `scratch.qkv` to
  `fused_decode_attention_device_kv_into` (which now accepts
  `CudaView<'_, f32>` instead of `CudaSlice<f32>` for q/k_new/v_new).
- Fall back to the 3-mmvq path when any of W_q / W_k / W_v is
  not Q4_K (e.g., mixed Q6_K).

The path's empirical performance vs the unfused baseline SHALL be
recorded in `proposal.md`'s acceptance bar so future contributors
don't re-implement the same fusion under the assumption that
"fewer launches must be faster".

#### Scenario: parity is preserved through the fused path

- **WHEN** the existing decode parity tests
  (`decode_token_phase1_matches_host_fallback`,
  `decode_token_graph_matches_per_call_over_5_steps`) are run
  on a layer with all-Q4_K Q/K/V weights, exercising the fused
  path
- **THEN** the per-step max-element absolute difference SHALL
  be ≤ 1e-3 against the legacy host-fallback / per-call
  reference paths
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_graph_matches_per_call_over_5_steps -->
