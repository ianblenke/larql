## Why

`cuda-attn-wmma-kernel-v2` proved the single-warp WMMA score-matmul
loses 20-32% to SIMT on Gemma 3 4B's GQA shape, and listed three
candidate mitigations:

1. **Multi-warp MMA per block** (3-4 warps each issuing in parallel)
2. Drop the kvh-grouped layout
3. Raw `mma.sync.aligned` PTX intrinsics

This change implements and benches mitigation **#1** as the cheapest
of the three. Each block has 4 warps each computing an independent
`16×16` accumulator over a stride-4 slice of the head_dim/16
d-tiles; partials are summed in shared memory before the
score-tile is written. This restores the warp-level parallelism
the single-warp WMMA path was wasting.

## Empirical results

`cuda_attn_score_simt_vs_wmma_bench` extended with a new
`score_wmma_mw` kernel. RTX 4090 / sm_89 / CUDA 12.5, 500 iters
after warmup, Gemma 3 4B shape (`num_q = 8`, `num_kv = 4`,
`head_dim = 256`):

| n_ctx | SIMT (µs) | WMMA single (µs) | speedup | WMMA-mw (µs) | mw vs SIMT | mw vs single |
|---:|---:|---:|---:|---:|---:|---:|
|   16 |   6.55 |   9.65 | 0.68× |  8.93 | **0.73×** | 1.08× |
|   32 |  11.58 |  16.19 | 0.72× | 14.89 | **0.78×** | 1.09× |
|   64 |  21.63 |  29.27 | 0.74× | 26.66 | **0.81×** | 1.10× |
|  256 |  82.12 | 107.82 | 0.76× | 97.32 | **0.84×** | 1.11× |
| 1024 | 305.54 | 390.71 | 0.78× | 351.78 | **0.87×** | 1.11× |

Multi-warp gives a consistent **8-11% lift** over the single-warp
WMMA path. But it is **still slower than SIMT at every n_ctx** —
13-27% slower depending on context length.

Parity is bit-exact (max-element diff = 0) for both WMMA variants.

## Why mitigation #1 alone isn't enough

The structural problem is **GQA fragment row utilization**, which
multi-warp parallelism does not fix:

- Each MMA still uses only **2 of 16** Q-fragment rows (the 2
  q_heads in the kvh-group). The other 14 are zero-padded.
- That's 12.5% useful Q × 100% K-tile work × 100% accumulator
  output → 12.5% MMA utilization per call. No amount of
  warp-level parallelism around the MMA call changes this.

Multi-warp helps by parallelising the **d-tile reduction loop**
across 4 warps, but the per-MMA waste is unchanged. The result is
roughly:

- Single-warp WMMA: ~12.5% × 25% (1 of 4 warps) = ~3% useful
  block compute.
- Multi-warp WMMA: ~12.5% × 100% (all 4 warps) = ~12.5% useful
  block compute.
- 4× lift on block utilisation = the observed 8-11% lift on
  wall-clock (since not all kernel time is in MMAs).

To get past 100% of SIMT, you'd need either mitigation **#2**
(drop kvh-grouped layout — replicate K per q_head, defeating the
GQA bandwidth win) or **#3** (raw `mma.sync.aligned` for finer
warp control, possibly with a non-standard fragment shape). Both
are 5-10 day efforts with no guaranteed net win.

The gap shrinks at long n_ctx (0.73× at 16 → 0.87× at 1024),
hinting that compute-bound long-context decode might cross
zero with further tuning. For our short-context decode bench
(n_ctx ≈ 20-30), the answer is conclusively negative.

## What This Change Ships

- ADD `score_wmma_mw` kernel embedded in
  `cuda_attn_score_simt_vs_wmma_bench`'s NVRTC source. Same
  signature as `score_wmma`, but with per-warp partial
  accumulators + shared-memory reduction.
- EXTEND the microbench to time all three (`simt`, `wmma`,
  `wmma_mw`) and print speedup ratios + parity.
- **No production code change.** Decode attention stays on the
  SIMT kernel.

## Update to Phase 2 outlook

Combined with `cuda-attn-wmma-kernel-v2`'s prior result, **two
of the three Phase 2 mitigations** sketched in
`cuda-attn-wmma-phase2`'s proposal are now empirically settled:

- **#1 (multi-warp MMA)**: locked in here as a real but
  insufficient lift. Tested. Negative.
- **#2 (drop kvh-grouped layout)**: untested. Would replicate K
  per q_head in shared memory; defeats GQA's bandwidth win.
- **#3 (raw `mma.sync.aligned` PTX)**: untested. Multi-day to
  get right; finer warp control is the only knob left to turn.

The straightforward Phase 2B WMMA path is dead for Gemma-shaped
GQA decode. Future Phase 2 work has to make a real bet on #2 or
#3 with the empirical data here as the bar to clear.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the multi-warp WMMA score-matmul
  measurement contract.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/backend.rs::tests` —
    `score_wmma_mw` kernel + extended bench.
- **Affected systems**: GPU only, ignored test (manual run).
- **No production behaviour change.**

## Risks and back-out

- No production code touched.

## Acceptance bar

- The microbench compiles and runs.
- Ratios on dev box (RTX 4090) confirm SIMT still wins on every
  Gemma-3-4B-shaped case — locking in the architectural
  conclusion.
