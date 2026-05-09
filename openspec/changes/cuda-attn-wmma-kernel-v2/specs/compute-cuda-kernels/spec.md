## ADDED Requirements

### Requirement: a microbench SHALL document the WMMA-on-GQA dead-end

The repository SHALL contain a microbench
`cuda_attn_score_simt_vs_wmma_bench` that times the score-matmul
`scores = Q @ K^T` on Gemma 3 4B's GQA shape (num_q = 8,
num_kv = 4, head_dim = 256) via two implementations: a SIMT
1-warp-per-q_head dot product, and a single-warp-per-block
WMMA-fragment matmul (the obvious Phase 2B sketch). The
microbench SHALL be ignored by default (`#[ignore]`), sweep
`n_ctx ∈ {16, 32, 64, 256, 1024}`, print per-iter µs + speedup
ratio + parity max-diff, and produce parity max-diff = 0
(bit-exact).

#### Scenario: SIMT wins on every Gemma 3 4B shape

- **WHEN** the microbench is run on a sm_70+ host with a CUDA
  toolkit available
- **THEN** for every n_ctx in the sweep the SIMT/WMMA speedup
  ratio SHALL be ≤ 1.0 (i.e., WMMA does not beat SIMT on GQA
  shapes), AND the parity max-diff SHALL be ≤ 1e-3
<!-- test: larql_compute::cuda::backend::tests::cuda_attn_score_simt_vs_wmma_bench -->
