## 1. P0 — Audit & groundwork

- [ ] 1.1 Enumerate the GGUF tensor types DSv4-Flash actually uses (dump `tensor_type` per tensor); confirm all large matmul weights are Q4_K/Q5_K/Q6_K/Q8_0, and list any format needing the f32 fallback
- [ ] 1.2 Confirm `QuantTensor::from_raw` accepts each of those formats and that `expert_slice` packing matches DSv4's `[n_expert, n_ff_exp, n_embd]` expert layout (write a tiny unit test in larql-models or larql-inference)
- [ ] 1.3 Decide the resident-RAM gate: document the host RAM requirement (~161 GB → ≥192 GB host) and how callers opt into resident vs streaming

## 2. P1 — Dual storage + quant-aware loader (no forward change)

- [ ] 2.1 Add `Option<QuantTensor>` fields to `DsV4LayerWeightStorage` (wq_a/wq_b/wkv/wo_a/wo_b) alongside the f32 arrays in `dsv4_storage.rs`
- [ ] 2.2 Add `Option<QuantTensor>` fields to `FfnStorage` (gate_inp, gate_exps, up_exps, down_exps, shared gate/up/down) alongside f32
- [ ] 2.3 Add `Option<QuantTensor>` for compressor (wkv/wgate) and indexer (wq_b/wproj) and mHC hc_fn — or document they stay f32 (small)
- [ ] 2.4 Change `dsv4_gguf_reader.rs::read_dsv4_layer_tensors_from_gguf` to return raw bytes + tensor_type for the large weights instead of eagerly dequantizing
- [ ] 2.5 Update `dsv4_storage_build.rs::build_layer_storage` to build `QuantTensor::from_raw` for supported formats, f32 fallback otherwise
- [ ] 2.6 Keep all existing call sites compiling: f32 fields still present, forward still reads f32 (no dispatch yet)
- [ ] 2.7 Verify: `cargo test -p larql-inference --lib attention::dsv4` green; real-GGUF load smoke still passes; measure resident footprint of a few layers to confirm ~Q4_K size

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
