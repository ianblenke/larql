## 1. P0 — Audit & groundwork

- [x] 1.1 Enumerate the GGUF tensor types DSv4-Flash actually uses (dump `tensor_type` per tensor); confirm all large matmul weights are Q4_K/Q5_K/Q6_K/Q8_0, and list any format needing the f32 fallback — `real_gguf_audit_tensor_types` in `dsv4_gguf_reader.rs`. Result: F32 (684, norms/small), Q4_K (598, large matmul weights), Q6_K (43, output/lm_head), I32 (3, routing tables handled by the int reader). **No format needs an unexpected f32 fallback** — every matmul weight is Q4_K/Q6_K, both lazy-quant-supported.
- [x] 1.2 Confirm `QuantTensor::from_raw` accepts each of those formats and that `expert_slice` packing matches DSv4's expert layout — `real_gguf_audit_expert_slice_packing` in `dsv4_gguf_reader.rs`. Result: `ffn_gate_exps` GGUF dims `[in_dim=4096, out_dim=2048, n_expert=256]` → `from_raw` flat `[n_expert*out_dim=524288, in_dim=4096]`; `expert_slice(e, 256)` yields `[2048, 4096]` per expert; `matvec` runs on a slice with finite output (no full dequant). Same `[n_expert*out_dim, in_dim]` packing as qwen35.
- [x] 1.3 Decide the resident-RAM gate: document the host RAM requirement (~161 GB → ≥192 GB host) and how callers opt into resident vs streaming — captured in design D4 (resident forward is a *new* entry point; `dsv4_streaming_model_forward_cached` retained for model-exceeds-RAM) and the spec's "Streaming path retained for oversized models" scenario. Caller opt-in is an explicit resident-forward constructor, not a silent default (design Risks: "gate the resident path on an explicit caller choice").

## 2. P1 — Dual storage + quant-aware loader (no forward change)

- [ ] 2.1 Add `Option<QuantTensor>` fields to `DsV4LayerWeightStorage` (wq_a/wq_b/wkv/wo_a/wo_b) alongside the f32 arrays in `dsv4_storage.rs` — *deferred to a follow-up; attention weights are tiny (negligible RAM), their quant residency only matters for P4 GPU offload*
- [x] 2.2 Add `Option<QuantTensor>` fields to `FfnStorage` — done for the routed experts (`gate_exps_quant`/`up_exps_quant`/`down_exps_quant`), the ~26 GB/layer memory hog, alongside the f32 `Array3`s. Dual-representation contract documented on the struct (quant `Some` ⇒ f32 empty). Footprint proven: `real_gguf_resident_expert_footprint` shows 4.18 GB quant vs 25.77 GB f32 per layer (6.2× smaller). Shared-expert + `gate_inp` stay f32 (small) for now.
- [ ] 2.3 Add `Option<QuantTensor>` for compressor (wkv/wgate) and indexer (wq_b/wproj) and mHC hc_fn — or document they stay f32 (small)
- [x] 2.4 Raw-bytes expert reader — added `read_dsv4_layer_raw_expert_tensors_from_gguf` (returns `RawExpertTensor { bytes, tensor_type, rows, cols }` for `ffn_{gate,up,down}_exps`, `from_raw`-ready `[n_expert*out_dim, in_dim]` shape, no dequant) **alongside** the existing f32 reader rather than replacing it — the f32 reader stays for the streaming path. Verified by `real_gguf_raw_expert_reader_round_trips_to_quant_tensor` (byte length matches the quant shape; bytes build a `QuantTensor` with no f32 expansion).
- [x] 2.5 Resident builder — added `build_layer_storage_resident(tensors, raw_experts, int_tensors, hp, cr)` sharing a `build_layer_storage_inner` core with the f32 `build_layer_storage`. Resident path wraps the routed-expert raw bytes in `QuantTensor`s (moving bytes, no clone, no f32 expansion) and leaves the f32 `*_exps` arrays empty (`0×0×0`); f32 path unchanged (still `take_3d`, still errors on missing experts). New `DsV4BuildError::ResidentQuant` for `from_raw` failures.
- [x] 2.6 Existing call sites compile unchanged: `build_layer_storage` keeps its signature (wrapper → inner with `None`), forward still reads f32. `build_layer_storage_resident_populates_quant_and_empties_f32` asserts the f32 path still produces full f32 experts + `None` quant.
- [x] 2.7 Verified: `cargo test -p larql-inference --lib attention::dsv4` → 231 passed; the resident builder test runs in CI (synthetic, no GGUF); real-GGUF footprint proven by `real_gguf_resident_expert_footprint` (4.18 GB quant vs 25.77 GB f32/layer). Backs spec scenario "Quantized tensor populates the QuantTensor field".

## 3. P2 — Quant-aware forward dispatch

- [ ] 3.1 Add a dispatch helper in larql-inference: `quant ? QuantTensor::matmul/matvec : dot_proj_gpu(x, w, backend)` (single + batch shapes)
- [ ] 3.2 Wire attention Q/KV/O projections (`dsv4_attn_block*.rs`) through the dispatch helper
- [ ] 3.3 Wire compressor wkv/wgate, indexer wq_b/wproj, mHC hc_fn, router gate_inp, shared-expert gate/up/down through the dispatch helper
- [ ] 3.4 Wire routed-MoE dispatch (`dsv4_moe_dispatch.rs`) to use `QuantTensor::expert_slice` + lazy-quant matmul; remove per-expert f32 dequant
- [ ] 3.5 Add tolerance-based parity test: quant path vs f32 path on a few real-GGUF layers (relative tolerance, document the bound)
- [ ] 3.6 Add greedy-token-equality test: quant vs f32 forward produce identical greedy tokens on a real-GGUF prompt
- [ ] 3.7 Verify: existing cached/prefill equivalence tests pass under the quant path (relaxed tolerance where needed)

## 4. P3 — Resident (non-streaming) forward

- [ ] 4.1 Add a resident forward entry point that takes pre-built `&[DsV4LayerWeightStorage]` (all layers' QuantTensors) and runs all decode steps against them
- [ ] 4.2 Add a loader that builds the full resident weight set once (all 43 layers) and reports total RAM
- [ ] 4.3 Keep `dsv4_streaming_model_forward_cached` available + documented for model-exceeds-RAM
- [ ] 4.4 Extend `dsv4_bench_cpu_vs_cuda` (or add a sibling) with a `resident` mode that loads once + decodes N steps; report prefill / steady-state tok/s
- [ ] 4.5 Verify on RTX 4090 host: resident decode tok/s materially exceeds the streaming 0.01 tok/s; record numbers in `project_dsv4_gpu_push` memory

## 5. P4 — CPU-FFN / GPU-attention hybrid

- [ ] 5.1 Thread per-callsite backend selection: attention sites get `Some(cuda)`, FFN/MoE sites get `None` (CPU + resident quant)
- [ ] 5.2 For attention-on-GPU, dequant attention weights to device once (enable PR #368 weight cache via `LARQL_CUDA_WEIGHT_CACHE_MAX_ELEMS`) — verify they're device-resident, not re-uploaded
- [ ] 5.3 Confirm attention device footprint + KV cache fit in 24 GB on the 4090
- [ ] 5.4 Bench the hybrid: compare all-CPU-resident vs hybrid (attn-GPU / FFN-CPU); record tok/s + VRAM
- [ ] 5.5 Update `project_dsv4_gpu_push` + `project_larql_driving_goal` memory with the hybrid result

## 6. Wrap-up

- [ ] 6.1 `make ci` green (fmt, clippy, tests, traceability, openspec validate)
- [ ] 6.2 Link spec scenarios to the new tests via `<!-- test: <fqn> -->` annotations
- [ ] 6.3 Archive the change once P1–P4 land (`openspec archive`)
