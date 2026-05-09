# cuda-tensor-cores-q4k — tasks

## 1. Microbench

- [x] 1.1 `q4k_decode_dp4a_vs_hgemm_b1` ignored test in
      `q4k_mmvq.rs::tests`. For each Gemma 3 4B Q4_K projection
      shape (q, kv, wo, gate, up, down), times 200 iterations of
      both paths and prints per-call µs + ratio.
- [x] 1.2 Manual run: dp4a wins on every shape with ratios 1.96×
      to 4.97×.

## 2. Documentation

- [x] 2.1 `proposal.md` records the empirical result table and
      explains the architectural reason (Tensor Cores are
      matrix-matrix accelerators; batch-1 wastes 15/16 of the
      16×N output tile).
- [x] 2.2 Out-of-scope follow-ups documented:
      INT4-IMMA / WMMA, speculative decoding, persistent-thread.

## 3. Archive

- [ ] 3.1 Archive when reviewed. (No production code change to
      revert — this is a "we measured, here's why we're not
      doing it" entry.)
