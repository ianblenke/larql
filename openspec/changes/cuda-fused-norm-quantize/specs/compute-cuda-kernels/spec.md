## ADDED Requirements

### Requirement: rms_norm + Q8_1 quantize fusion path SHALL be implemented and benched

The captured-decode pipeline SHALL contain a code path that fuses
`rms_norm + quantize_q8_1` into a single kernel
`rms_norm_quantize_q8_1_f32`. The kernel SHALL launch as one
block of 1024 threads with shared memory sized
`max(bdim, n) × sizeof(f32)`, executing three phases:
sum-of-squares reduction, write normalised values to smem, and
per-warp Q8_1 quantize strided across `n / 32` blocks. The
captured-decode pipeline's pre-attn (`h → h_attn_q8_1`) and
pre-FFN (`h → h_ffn_q8_1`) sites SHALL invoke the fused wrapper.

The path's empirical performance vs the unfused baseline SHALL
be recorded in `proposal.md` so future contributors don't
re-implement under the assumption "fewer launches must be
faster".

#### Scenario: parity is preserved through the fused path

- **WHEN** the existing decode parity tests
  (`decode_token_phase1_matches_host_fallback`,
  `decode_token_graph_matches_per_call_over_5_steps`) are run
  with the fused norm+quantize path
- **THEN** the per-step max-element absolute difference SHALL
  be ≤ 1e-3 against the legacy reference paths
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_graph_matches_per_call_over_5_steps -->
