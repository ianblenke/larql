# cuda-fused-norm-quantize — tasks

## 1. Fused kernel (kept on branch but not merged)

- [x] 1.1 `RMS_NORM_QUANTIZE_Q8_1_SRC` PTX: 3-phase kernel
      (sum-of-squares reduction → write normalised values to
      smem → per-warp Q8_1 quantize strided across n_blocks).
- [x] 1.2 `RMS_NORM_QUANTIZE_Q8_1_FUNC: OnceLock` cell.
- [x] 1.3 `elem::rms_norm_quantize_q8_1_into(...)` wrapper.
      Single block of 1024 threads, smem = max(bdim, n) × 4 B.

## 2. Pipeline integration

- [x] 2.1 Pre-attn site (h → h_attn_q8_1): replace
      `rms_norm_device_into → quantize_q8_1_device_into` with
      one `rms_norm_quantize_q8_1_into` call.
- [x] 2.2 Pre-FFN site (h → h_ffn_q8_1): same.

## 3. Empirical results

- [x] 3.1 All 200+ tests pass.
- [x] 3.2 Bench (10-run avg, with graph): fused 8.22 ms/tok
      vs baseline 8.04 ms = 0.18 ms regression.
- [x] 3.3 Bench (5-run avg, graph off): fused 9.37 ms vs
      baseline 9.05 ms = 0.32 ms regression. Confirms the
      kernel itself is slower, not graph-capture variance.

## 4. Negative-result documentation

- [x] 4.1 Mechanism: separate `quantize_q8_1` ran as 80
      blocks across 80 SMs; fused phase 3 collapses to 32
      warps on 1 SM with 3 strided iters. Same per-warp
      work, but parallelism collapse.
- [x] 4.2 Pattern empirically established: 3 fusion-style
      negative results in this session (Path D Q/K/V mmvq,
      WMMA-on-GQA, and Path E here). Future fusion ideas
      have to explain how they preserve across-SM
      parallelism.

## 5. Recommendation

- [ ] 5.1 **Do NOT merge**. Production stays on the unfused
      pair.

## 6. Archive

- [ ] 6.1 Archive when reviewed.
