## ADDED Requirements

### Requirement: A microbench SHALL document the decode-TC dead-end

The repository SHALL contain a microbench that times the existing
`__dp4a` Q4_K mmvq path against cuBLAS `hgemm` with batch-size = 1
on every Gemma 3 4B Q4_K projection shape. The microbench SHALL be
ignored by default (`#[ignore]`) so it doesn't run in CI, and
SHALL print per-call µs for both paths plus the ratio. The
ratio MUST be ≥ 1.0 on every shape (i.e., dp4a wins) on a
representative sm_89 (Ada / RTX 4090) host.

#### Scenario: dp4a wins on every projection shape

- **WHEN** `LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute
  --features cuda --lib q4k_decode_dp4a_vs_hgemm_b1 --release --
  --ignored --nocapture` is run on a sm_89 host with Gemma 3 4B
  Q4_K weights
- **THEN** the printed ratio SHALL be ≥ 1.5 for every
  projection shape (q, kv, wo, gate, up, down) — confirming the
  cuBLAS-based decode-TC path is not viable for batch-1 inference
<!-- test: larql_compute::cuda::q4k_mmvq::tests::q4k_decode_dp4a_vs_hgemm_b1 -->
