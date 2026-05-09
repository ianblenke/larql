## Why

`cuda-attn-wmma-phase2` proved WMMA is technically supported on this
stack and sketched a Phase 2B kernel that uses
`wmma::load_matrix_sync` / `mma_sync` for the score and output
matmuls. This change is the **empirical follow-up**: a head-to-head
microbench between SIMT and WMMA implementations of the score matmul
(`scores = Q @ K^T`), measuring whether WMMA actually wins on
Gemma-shaped GQA decode, before committing to the multi-day Phase 2B
kernel rewrite.

**Result: WMMA loses on every shape tested.** This change ships the
microbench as a permanent record so a future contributor doesn't
repeat the analysis, and updates the `cuda-attn-wmma-phase2` proposal
to point here for the negative result.

## Empirical results

`cuda_attn_score_simt_vs_wmma_bench` on RTX 4090 / sm_89 / CUDA 12.5,
500 iters after warmup, Gemma 3 4B shape (`num_q = 8`, `num_kv = 4`,
`head_dim = 256`, GQA group = 2):

| n_ctx | SIMT (µs) | WMMA (µs) | speedup | parity max-diff |
|---:|---:|---:|---:|---:|
|   16 |   6.55 |   9.65 | **0.68×** | 0 |
|   32 |  11.58 |  16.18 | **0.72×** | 0 |
|   64 |  21.67 |  29.27 | **0.74×** | 0 |
|  256 |  82.12 | 107.83 | **0.76×** | 0 |
| 1024 | 306.48 | 388.51 | **0.79×** | 0 |

WMMA is **20-32% slower** at every context length. Parity is
bit-exact (max-element diff = 0).

## Why WMMA loses on GQA shapes (mechanistic)

The Phase 2B sketch packs all q_heads into a 16-row Q fragment and
runs one MMA per (kvh-group, K-tile, head_dim-tile). For Gemma 3 4B's
2 q_heads per kvh-group, the math is:

- **Fragment row utilization**: 2 of 16 rows are real → 12.5% of MMA
  output is useful. The other 87.5% is wasted compute (still costs
  cycles and energy on the Tensor Cores).

- **Per-block warp utilization**: WMMA 16×16×16 fragments are issued
  by a single warp. With 4 warps per block (128 threads), 3 warps
  are idle during the MMA — 75% of block compute wasted.

- **Block count**: SIMT runs 1 block per q_head (8 blocks for
  Gemma 3 4B); WMMA runs 1 block per kv_head (4 blocks). Half the
  chip occupied.

Combined, WMMA's effective throughput on this shape is roughly
`(1 / 8) × (1/16 × 4) = 1/32` of the SIMT path's parallelism — and
the per-thread Tensor Core throughput advantage doesn't cover that
gap.

## What would have to change to make WMMA win

1. **Multiple warps per block all running MMAs in parallel** — i.e.,
   process multiple K-tiles per block simultaneously, with each warp
   responsible for one tile's MMA chain. Requires careful shared-memory
   layout and warp-level synchronization.
2. **Drop the kvh-grouped layout** — instead treat all q_heads as
   independent rows of a single 16-row fragment, accept that some
   rows will use "wrong" K data for their kv_head, and accumulate
   masked. Or: replicate K data per q_head in shared memory (defeats
   the bandwidth advantage of GQA).
3. **Use raw `mma.sync.aligned` PTX intrinsics** for finer warp-level
   control than the high-level `wmma::*` API allows.

Estimated cost: 5-10 days of focused CUDA work, with no guaranteed
net win on Ada (sm_89). The simple "drop in WMMA" path is dead.

## What This Change Ships

- ADD `cuda_attn_score_simt_vs_wmma_bench` ignored microbench in
  `cuda::backend::tests`. Records:
  - SIMT and WMMA score-matmul implementations side-by-side as
    embedded NVRTC kernels.
  - Sweep over `n_ctx ∈ {16, 32, 64, 256, 1024}`.
  - Per-iter timing + parity check (max-diff vs the SIMT
    reference).
- The microbench compiles with the same `<mma.h>` include-path
  autodiscovery as `cuda-attn-wmma-phase2`'s smoke test.
- **No production code change.** The decode path stays on the SIMT
  attention kernel.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the SIMT-vs-WMMA score-matmul
  comparison contract.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/backend.rs::tests` — adds the
    microbench.
- **Affected systems**: GPU only, ignored test (manual run).
- **No production behaviour change.**

## Risks and back-out

- No production code touched; no back-out needed.

## Acceptance bar

- The microbench compiles and runs (`cargo test --release ...
  cuda_attn_score_simt_vs_wmma_bench -- --ignored --nocapture`).
- Reported ratios on dev box (RTX 4090, sm_89) confirm SIMT wins
  on every Gemma-3-4B-shaped case — locking in the architectural
  conclusion that the **simple Phase 2B WMMA path is not viable
  for GQA models**.
