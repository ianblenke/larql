# cuda-attn-wmma-kernel-v2 — tasks

## 1. SIMT-vs-WMMA score-matmul microbench

- [x] 1.1 `cuda_attn_score_simt_vs_wmma_bench` ignored test in
      `cuda::backend::tests`. Embeds two NVRTC kernels:
      `score_simt` (1 warp per q_head, scalar dot product) and
      `score_wmma` (1 block per kvh-group, single-warp WMMA).
- [x] 1.2 Sweep n_ctx ∈ {16, 32, 64, 256, 1024} on the
      Gemma 3 4B shape (num_q=8, num_kv=4, head_dim=256).
- [x] 1.3 Report per-iter µs + speedup ratio + parity max-diff.

## 2. Empirical conclusion (this change documents)

- [x] 2.1 WMMA loses 20-32% at every n_ctx. Parity bit-exact.
- [x] 2.2 Mechanism analysis: GQA gives 12.5% fragment row
      utilization × 25% warp utilization × 50% block-count
      utilization = SIMT wins.
- [x] 2.3 Documented future-work options (multi-warp MMA, drop
      kvh-grouped layout, raw mma.sync) and their estimated cost
      (5-10 days, no guaranteed net win).

## 3. Cross-link

- [x] 3.1 `cuda-attn-wmma-phase2`'s proposal points here for the
      empirical conclusion. Phase 2B as originally sketched is
      not viable; future Phase 2 work has to use one of the
      mitigations in §2.3.

## 4. Archive

- [ ] 4.1 Archive when reviewed. No production code change to
      revert.
