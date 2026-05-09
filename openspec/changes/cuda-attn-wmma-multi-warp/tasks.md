# cuda-attn-wmma-multi-warp — tasks

## 1. Multi-warp WMMA score-matmul kernel

- [x] 1.1 `score_wmma_mw` kernel embedded in
      `cuda_attn_score_simt_vs_wmma_bench`. 4 warps; each warp
      accumulates over a stride-4 slice of the head_dim/16 d-tiles.
- [x] 1.2 Per-warp partial accumulator stored to shared memory
      `partials[4 × 16 × 16]`; cooperative reduction sums to
      `s_smem[16 × 16]`.
- [x] 1.3 Shared-memory budget: `q_smem (16 × hd × 2B) + k_smem
      (16 × hd × 2B) + partials (4 × 256 × 4B) + s_smem (256 × 4B)`
      = 8 KB + 8 KB + 4 KB + 1 KB = 21 KB for hd=256. Within
      RTX 4090's 100 KB/SM dynamic shmem limit.

## 2. Empirical sweep

- [x] 2.1 Extend the microbench to time all three variants:
      `simt`, `wmma` (single-warp), `wmma_mw` (multi-warp).
      Sweep n_ctx ∈ {16, 32, 64, 256, 1024}.
- [x] 2.2 Report speedup ratios + parity max-diffs.
- [x] 2.3 Run the bench. Multi-warp lifts WMMA by 8-11% over
      single-warp but is **still 13-27% slower than SIMT at
      every n_ctx**. Parity bit-exact.

## 3. Empirical conclusion

- [x] 3.1 Mechanism: GQA fragment-row utilization (12.5%) is
      orthogonal to warp-level parallelism. Multi-warp restores
      the warp utilization but doesn't fix the fragment-level
      waste.
- [x] 3.2 Document remaining mitigations (#2 drop kvh-grouped
      layout, #3 raw mma.sync.aligned) and their estimated
      cost in `proposal.md`.
- [x] 3.3 Cross-reference from `cuda-attn-wmma-phase2`'s
      proposal so the next session has the full negative-
      results record.

## 4. Archive

- [ ] 4.1 Archive when reviewed. No production code change to
      revert.
