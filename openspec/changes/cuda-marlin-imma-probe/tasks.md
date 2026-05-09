# cuda-marlin-imma-probe — tasks

## 1. INT8 IMMA viability microbench

- [x] 1.1 `cuda_mmvq_dp4a_vs_imma_bench` ignored test in
      `cuda::backend::tests`. Two NVRTC kernels embedded:
      `mmvq_int8_dp4a` (1 warp/row, scalar inner loop, 4 rows/block)
      and `mmvq_int8_imma` (16×16×16 INT8 WMMA, 1 warp per
      16-row tile, x replicated into 16-col shared-mem tile).
- [x] 1.2 Sweep all six Gemma 3 4B Q4_K projection shapes.
- [x] 1.3 200 iters per kernel after 5-iter warmup. Report
      per-call µs + speedup ratio + parity max-diff.

## 2. Empirical conclusion

- [x] 2.1 INT8 IMMA loses 3-7× to dp4a on every shape.
- [x] 2.2 Parity bit-exact (max-diff = 0) — purely a throughput
      regression, not correctness.
- [x] 2.3 Mechanism: m16n16k16 MMA wastes 15/16 output cols
      at batch=1; col-replication of x adds redundant smem
      staging; output-column extraction adds smem round-trip.

## 3. Cross-link

- [x] 3.1 `cuda-decode-perf-results` Path A entry updated:
      Marlin INT4-IMMA's batch-1 ceiling is dp4a, not 1.0-1.5 ms
      below it.
- [x] 3.2 Empirical Tensor-Core dead-end summary table in
      `proposal.md` (4 variants tested, all negative).

## 4. Recommended next direction

- [x] 4.1 Speculative decoding (batch ≥ 8) is the only known
      path to unlock Tensor Cores for decode. 2-3 weeks
      architectural change.
- [x] 4.2 Without speculative decoding: continue incremental
      dp4a tuning + audit-style micro-opts (analogous to
      `cuda-mmvq-hw-f16-cvt`'s 7.5% win).

## 5. Archive

- [ ] 5.1 Archive when reviewed.
