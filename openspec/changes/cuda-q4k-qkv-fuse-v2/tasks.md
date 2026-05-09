# cuda-q4k-qkv-fuse-v2 — tasks

## 1. Infrastructure (kept on branch but not merged)

- [x] 1.1 `q4k_qkv_concat_device_cache` cache field on
      `CudaBackend` + `arc_q4k_qkv_concat_device_buf` helper.
- [x] 1.2 `DecodeScratch::qkv` buffer of size `q_dim + 2 *
      kv_dim`.
- [x] 1.3 `LayerArcs::qkv_concat: Option<Arc<CudaSlice<u8>>>`
      populated for all-Q4_K Q/K/V layers.
- [x] 1.4 `fused_decode_attention_device_kv_into` signature
      change: `q_dev / k_new_dev / v_new_dev` from
      `&CudaSlice<f32>` to `&CudaView<'_, f32>`. Lets the
      pipeline pass slices of `scratch.qkv` without an
      intermediate copy.
- [x] 1.5 `run_decode_pipeline_into_scratch` branches on
      `arcs.qkv_concat.is_some()`: fused path uses one mmvq
      into `scratch.qkv` + slice views; legacy path uses 3
      mmvq calls + `scratch.q/k/v.as_view()`.

## 2. Empirical results

- [x] 2.1 All 200+ tests pass.
- [x] 2.2 Bench: 10-run avg with fused = 8.17 ms/tok vs
      baseline 8.04 ms (range 8.04–8.46).
- [x] 2.3 Bench: forcing `LARQL_CUDA_Q4K_COOP=1` is *worse*
      (8.27 ms avg), confirming 4096 rows is past the coop
      threshold.

## 3. Negative-result documentation

- [x] 3.1 Mechanism analysis: K and V (1024-row shapes)
      benefit from coop kernel (~3.8 µs vs ~5.3 µs legacy).
      Fusing into 4096 rows forces legacy on the combined
      weight, costing more (~4 µs/layer) than the saved
      launches buy (~2 µs/layer in graph mode).
- [x] 3.2 Variance regression noted: 0.42 ms range vs 0.02 ms
      baseline. Likely L1 cache contention on the larger fused
      weight.

## 4. Recommendation

- [ ] 4.1 **Do NOT merge**. Production stays on the unfused
      path. This change records the experiment for future
      contributors so the negative interaction with
      `cuda-q4k-mmvq-warp-cooperative` is empirically locked in.

## 5. Archive

- [ ] 5.1 Archive when reviewed.
