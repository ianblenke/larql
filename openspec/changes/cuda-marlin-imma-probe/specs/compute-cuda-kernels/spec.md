## ADDED Requirements

### Requirement: a microbench SHALL document the Marlin INT8 IMMA dead-end

The repository SHALL contain a microbench
`cuda_mmvq_dp4a_vs_imma_bench` that times INT8 mmvq via two
implementations: the existing `__dp4a` SIMT pattern and an INT8
WMMA path using `mma.sync.aligned`-equivalent fragments. The
microbench SHALL be ignored by default (`#[ignore]`), sweep
every Gemma 3 4B Q4_K projection shape (q, kv, wo, gate, up,
down), and report per-call µs + speedup ratio + parity max-diff.
The IMMA path's parity max-diff MUST be 0 (bit-exact); the
speedup ratio MUST be ≤ 1.0 at every shape (i.e., IMMA does not
beat dp4a at batch=1, settling the Marlin Path A dead-end).

#### Scenario: dp4a wins on every Gemma 3 4B projection shape

- **WHEN** the microbench is run on a sm_80+ host with a CUDA
  toolkit available
- **THEN** for every shape in the sweep the printed
  `imma vs dp4a speedup` SHALL be ≤ 1.0, AND the parity max-diff
  SHALL be 0
<!-- test: larql_compute::cuda::backend::tests::cuda_mmvq_dp4a_vs_imma_bench -->
