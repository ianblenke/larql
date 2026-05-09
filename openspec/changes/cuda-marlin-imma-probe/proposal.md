## Why

Marlin (vLLM's INT4 Tensor-Core kernel) was the headline next-step
in `cuda-decode-perf-results`'s Path A: 5-10 days, est. 1.0-1.5 ms
decode savings. Before committing to that multi-day kernel work,
this change runs a head-to-head INT8 IMMA viability probe against
the existing dp4a path.

**Result: definitive negative.** INT8 IMMA via Tensor Cores is
**3-7× SLOWER** than dp4a on every Gemma 3 4B projection shape at
batch=1. The Marlin path is settled negative for single-stream
decode without speculative decoding to lift the batch dimension.

This change ships the microbench as a permanent record so a future
contributor doesn't repeat the analysis, and updates the
`cuda-decode-perf-results` Path A entry with the empirical result.

## Empirical results

`cuda_mmvq_dp4a_vs_imma_bench` on RTX 4090 / sm_89 / CUDA 12.5,
200 iters after warmup, INT8 surrogates (Marlin's full kernel
would do Q4_K → INT8 staging in shared memory; if the INT8 proxy
loses to dp4a, the full Marlin kernel cannot recover):

| Shape (rows × hidden) | dp4a (µs) | INT8 IMMA (µs) | speedup | parity |
|---|---:|---:|---:|---:|
| q     ( 2048 ×  2560) |  3.36 | 18.22 | **0.18×** | bit-exact |
| kv    ( 1024 ×  2560) |  2.79 | 18.07 | **0.15×** | bit-exact |
| wo    ( 2560 ×  2048) |  3.27 | 15.39 | **0.21×** | bit-exact |
| gate  (10240 ×  2560) |  7.56 | 24.47 | **0.31×** | bit-exact |
| up    (10240 ×  2560) |  7.57 | 24.45 | **0.31×** | bit-exact |
| down  ( 2560 × 10240) |  8.79 | 68.47 | **0.13×** | bit-exact |

INT8 IMMA loses 3-7× at every shape. Parity is bit-exact (max-diff
= 0); this is purely a throughput regression, not a correctness
issue.

## Why IMMA loses at batch=1 (mechanism)

Same root cause as the empirically-settled WMMA-on-GQA dead-end
(`cuda-attn-wmma-multi-warp`). For `mma.sync.aligned.m16n16k16`
INT8:

- Output tile shape: 16 rows × 16 cols.
- Batch=1 matvec means N=1 — only **1 of 16 output columns** is
  real. **15/16 of MMA throughput wasted.**
- Plus: the single input vector x must be replicated into a 16×16
  col-major shared-memory tile per K-tile (N - 1 redundant
  copies that the dp4a path doesn't pay).
- Plus: the 16×16 INT32 output tile must be materialised in
  shared memory, then column 0 extracted into the result vector.

dp4a, by contrast, is a vector-vector instruction with 1×1
"output" per call — 100% useful work per instruction.

## Why Marlin can't recover this

Marlin's published advantage on RTX 4090 is **for batch ≥ 2** (the
vLLM use case: many sequences batched together). The win comes
from:

1. Better HBM bandwidth utilisation via cp.async + ldmatrix.
2. Permuted weight layout that aligns with MMA fragments.
3. Multi-warp cooperation across rows.

But (1) and (2) only give bandwidth efficiency wins; they don't
recover the 15/16 wasted MMA throughput at batch=1. (3) is
already what dp4a does effectively on this shape.

For batch=1 single-stream decode, **dp4a is the right tool**
(both for us and for llama.cpp, which uses essentially the same
mmvq pattern after we ported their warp-cooperative parameterisation
in `cuda-q4k-mmvq-warp-cooperative`).

## Implications for "match llama.cpp"

The empirical sweep of all Tensor-Core variants now covers:

| Path | Variant | Result | Branch |
|---|---|---|---|
| cuBLAS hgemm batch-1 | f16 | -3-5× slower | `cuda-tensor-cores-q4k` |
| WMMA single-warp | f16 attention | -1.3-1.5× slower | `cuda-attn-wmma-kernel-v2` |
| WMMA multi-warp | f16 attention | -1.13-1.27× slower | `cuda-attn-wmma-multi-warp` |
| INT8 IMMA (Marlin proxy) | INT8 mmvq | **-3-7× slower** | this change |

**No Tensor-Core variant beats the existing SIMT/dp4a path at
batch=1 on Gemma's GQA shape.** The matrix dimension wastage from
batch=1 + GQA-2 dominates Tensor Cores' per-MMA throughput
advantage.

To unlock Tensor Cores for decode, the **only known path is
speculative decoding** (lifts effective batch ≥ 8 via draft-token
verification). That's a 2-3 week architectural change.

**To match llama.cpp without speculative decoding, the path is
incremental**: per-shape dp4a tuning, kernel fusion that preserves
across-SM parallelism, and audit-style micro-optimisations like
`cuda-mmvq-hw-f16-cvt` (which gave a one-off 7.5%). Each
contribution is small; closing the remaining 1.71× gap requires
many of them sustained over multiple sessions.

## What This Change Ships

- ADD `cuda_mmvq_dp4a_vs_imma_bench` ignored microbench in
  `cuda::backend::tests`. Embeds two NVRTC kernels:
  `mmvq_int8_dp4a` (existing pattern, just operating on INT8
  surrogates instead of Q4_K) and `mmvq_int8_imma` (16×16×16
  WMMA-INT8 with col-replicated x staging in shared memory).
- The probe is the gating risk for the multi-day Marlin work;
  with this negative result, Marlin is settled-out without that
  investment.
- **No production code change.**

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the INT8 IMMA vs dp4a microbench
  contract.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/backend.rs::tests` — adds the
    microbench.
- **Affected systems**: GPU only, ignored test (manual run).
- **No production behaviour change.**

## Risks and back-out

None — pure investigation.

## Acceptance bar

- The microbench compiles and runs.
- All 6 shape-rows in the sweep show `imma vs dp4a speedup ≤ 1.0`
  on a sm_80+ host with Gemma 3 4B's projection shapes.
- `cuda-decode-perf-results` Path A's effort/risk/reward entry is
  updated to reflect this empirical result.
