## ADDED Requirements

### Requirement: a multi-warp WMMA score-matmul SHALL be benched alongside SIMT and single-warp WMMA

The microbench `cuda_attn_score_simt_vs_wmma_bench` SHALL include
a `score_wmma_mw` kernel that uses 4 cooperating warps per block,
each accumulating an independent `16×16` partial over a stride-4
slice of the head_dim/16 d-tiles, with per-warp partials summed
in shared memory before the score-tile is written. The microbench
SHALL print all three variants' per-iter µs, speedup ratios vs
SIMT, and parity max-diffs vs the SIMT reference. Parity max-diff
MUST be 0 (bit-exact) for both WMMA variants.

#### Scenario: multi-warp WMMA still loses to SIMT on every Gemma 3 4B shape

- **WHEN** the microbench is run on a sm_80+ host with a CUDA
  toolkit available, against the Gemma 3 4B GQA shape
  (num_q = 8, num_kv = 4, head_dim = 256), for n_ctx ∈ {16, 32,
  64, 256, 1024}
- **THEN** the printed `mw vs SIMT` speedup ratio SHALL be
  ≤ 1.0 at every n_ctx (i.e., multi-warp WMMA still does not
  beat SIMT on GQA shapes), AND parity max-diff SHALL be 0.0
<!-- test: larql_compute::cuda::backend::tests::cuda_attn_score_simt_vs_wmma_bench -->
