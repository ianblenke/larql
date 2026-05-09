## ADDED Requirements

### Requirement: WMMA viability SHALL be verified by an in-tree smoke test

The repository SHALL contain a unit test
`cuda_wmma_mma_sync_smoke_test` that compiles an
`<mma.h>`-based 16×16×16 f16-input / f32-accumulator
kernel via NVRTC, runs it on a fixed `A @ B^T` test
input, and asserts max-element absolute difference ≤ 1e-2
against an f32 host reference. The test SHALL only run when
`LARQL_CUDA_AVAILABLE=1`. NVRTC SHALL be invoked with
`include_paths` pointing to a discovered CUDA toolkit include
directory. The test gates the production WMMA attention kernel
in Phase 2B — without it passing, that kernel cannot land.

#### Scenario: WMMA smoke test passes on the dev box

- **WHEN** `LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute
  --features cuda --lib cuda_wmma_mma_sync_smoke_test --
  --test-threads=1` is run on a sm_70+ host with a CUDA
  toolkit available
- **THEN** the test SHALL pass with max-element absolute
  difference ≤ 1e-2 against the f32 host reference
<!-- test: larql_compute::cuda::backend::tests::cuda_wmma_mma_sync_smoke_test -->
