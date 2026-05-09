## Why

The next move on the decode path's mmvq budget (~3 ms/token across
all projections) was supposed to be Tensor Cores — analogous to the
40% prefill win shipped in `cuda-prefill-tensor-cores`. This change
investigates the obvious-but-cheap variant (cuBLAS `hgemm` with
batch-size = 1 over the f16 weight cache) and **records the negative
result**: hgemm at batch-1 is 2.0–5.0× **slower** than the existing
`__dp4a` mmvq path on every Gemma 3 4B Q4_K projection shape.

## Why hgemm-batch-1 doesn't work for decode

Tensor Cores are matrix-matrix accelerators. The smallest tile on
sm_89 is 16×16×16, so an MMA call always produces a 16×N partial
output. For batch-size = 1 (single-token decode), 15 out of 16
output rows are wasted compute and HBM traffic; effective TC
throughput drops to ~1/16 of peak.

dp4a is, by contrast, a **vector-vector** instruction (4-way INT8
SIMD dot product) with no batched-output requirement. For Q4_K
mmvq with packed weights and Q8_1 inputs, dp4a is the right tool
on every supported card.

## Empirical evidence

`q4k_decode_dp4a_vs_hgemm_b1` microbench (RTX 4090, CUDA 12.5,
200 iterations after warmup):

| Shape (rows × hidden) | dp4a | hgemm_b1 | ratio |
|---|---:|---:|---:|
| q     (2048 × 2560)   |  5.4 µs | 12.7 µs | **2.33×** |
| kv    (1024 × 2560)   |  5.4 µs | 10.6 µs | **1.96×** |
| wo    (2560 × 2048)   |  4.8 µs | 12.2 µs | **2.54×** |
| gate  (10240 × 2560)  | 13.7 µs | 67.7 µs | **4.94×** |
| up    (10240 × 2560)  | 13.5 µs | 67.0 µs | **4.97×** |
| down  (2560 × 10240)  | 17.4 µs | 64.8 µs | **3.72×** |

Every shape is faster on dp4a. The wider `gate`/`up` shapes
(rows = 10240) suffer the most from hgemm's wasted-output-tile
overhead.

## What Changes

- ADD `q4k_decode_dp4a_vs_hgemm_b1` ignored microbench in
  `q4k_mmvq.rs::tests` to prevent regressions if someone tries
  this experiment again. Also serves as documentation: it's the
  ONE-paragraph proof that the obvious cuBLAS-based decode TC
  path doesn't work.
- This change makes **no production code changes**. The decode
  mmvq path stays on dp4a.

## Out of scope (future work)

The negative result rules out the cheapest TC variant. The remaining
candidates are all multi-day refactors with no guarantee of net win:

1. **Custom INT4-IMMA WMMA kernel** — sm_80+ supports INT4 inputs
   on Tensor Cores via `wmma::experimental::col_major::*`. A custom
   kernel that does Q4_K dequant on-load, packs INT4 quants into
   tensor-core fragments, and runs WMMA at batch-1 with explicit
   reuse of the output tile across multiple INT4 chunks could in
   principle approach dp4a's throughput while leaving room for
   future batching. Marlin and related projects in the
   `llm.c` / Triton ecosystem do exactly this. ~3-5 days of CUDA
   work, no guarantee of net speedup over dp4a on Ada.

2. **Speculative decoding** — batch 4-8 candidate decode tokens
   per real token. With batch ≥ 8 the TC paths become viable;
   gate/up's 67 µs / 8 = 8.4 µs amortised per real token, beating
   dp4a's 13.7 µs. But this requires draft-model integration,
   tree attention, and verification logic — orders of magnitude
   bigger than the kernel-level work here.

3. **Persistent-thread / cooperative groups mmvq** — keep more
   work resident per CTA to reduce per-launch overhead. Has
   nothing to do with Tensor Cores; would compete with dp4a on
   different axes (occupancy / cache reuse).

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the dp4a-vs-hgemm-batch-1
  measurement contract.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/q4k_mmvq.rs::tests` — new
    `q4k_decode_dp4a_vs_hgemm_b1` ignored microbench.
- **Affected systems**: GPU only, dev-only (microbench).
- **No production behaviour change.**

## Acceptance bar

- The microbench compiles and runs (`cargo test --release ...
  q4k_decode_dp4a_vs_hgemm_b1 -- --ignored --nocapture`).
- Reported ratios on dev box (RTX 4090) confirm dp4a wins for
  every projection shape — locking in the architectural conclusion.
