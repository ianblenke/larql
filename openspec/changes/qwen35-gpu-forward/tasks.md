# Tasks — Qwen3.6 GPU forward

## E.1 — single-matvec PoC (~150 LoC)

- [ ] E.1.1 Add `pub backend: Option<Arc<dyn larql_compute::backend::QuantMatVec
        + Send + Sync>>` to `Qwen35Weights`. Default `None` keeps
      the CPU lazy path. Bench harness constructs
      `CudaBackend::new()?` when `LARQL_QWEN35_GPU=1`.
- [ ] E.1.2 Extend `QuantTensor::matvec` (or add
      `matvec_with_backend`) that takes an optional `&dyn
      QuantMatVec`. If `Some`, calls `backend.quant_matvec(format,
      &self.data, x, rows, cols)` and falls back to the existing
      rayon CPU path on `None` return.
- [ ] E.1.3 In `qwen35_forward_step`, when computing the final
      `lm_head` matvec, pass `weights.backend.as_deref()` through.
      Default behaviour unchanged.
- [ ] E.1.4 Map our `tensor_type` (u32 ggml id) to
      `larql_compute::QuantFormat`. Helper in `quant/lazy.rs`.
- [ ] E.1.5 Env-gated test
      `real_gguf_qwen35_gpu_lm_head_diagnostic` — load lazy lm_head,
      construct `CudaBackend`, run prefill + 1 decode step, assert
      argmax matches dequant baseline.
- [ ] E.1.6 Extend `real_gguf_qwen35_bench` to use the GPU backend
      when `LARQL_QWEN35_GPU=1`. Print kernel-vs-fallback dispatch
      counts. Record bench delta in `bench-baseline.md`.

## E.2 — FFN on GPU (~50 LoC plumbing)

Once E.1 lands, the rest is just dispatching the FFN tensors the same
way. Already plumbed via Phase 2's lazy lookup.

- [ ] E.2.1 `swiglu_ffn_lazy` takes `backend: Option<&dyn QuantMatVec>`,
      passes through to `QuantTensor::matvec_with_backend` for each
      of gate / up / down.
- [ ] E.2.2 Bench: expect ≥ 1.0 t/s decode.

## E.3 — Attn projections on GPU (~50 LoC plumbing)

- [ ] E.3.1 Dispatch DeltaNet `attn_qkv`/`attn_gate`/`ssm_out` and
      full-attn `attn_q/k/v/o` matvecs through the same backend.
- [ ] E.3.2 Bench: expect ≥ 3 t/s decode (the matvec-heavy half is
      now all GPU; rest is recurrence + conv1d on CPU).

## E.4 — DeltaNet recurrence + Conv1D CUDA kernels (~600 LoC)

- [ ] E.4.1 `cuda/deltanet_recurrence.cu` — one CUDA block per
      `(batch, head)`. State matrix `S[s_v, s_v]` lives in shared
      memory (16 KB for 128×128 f32 ≈ 64 KB → ok on Ampere). Mirrors
      llama.cpp's `ggml_compute_forward_gated_delta_net_one_chunk`
      decay-first algorithm.
- [ ] E.4.2 `cuda/causal_conv1d.cu` — depthwise Conv1D with state
      shift-and-insert. Trivial: 4-tap × 10240 channels.
- [ ] E.4.3 Per-head RMSNorm + L2-norm on GPU (small reductions
      that don't need their own kernel; can fold into conv or
      delta entry).
- [ ] E.4.4 Bench: expect ≥ 10 t/s decode.

## E.5 — Full softmax-attention on GPU (~200 LoC plumbing)

- [ ] E.5.1 Wire existing `cuda/attn.rs` (or `attn_tree.rs`) into
      the full-attn block forward.
- [ ] E.5.2 Bench: expect ≥ 15 t/s decode.

## E.6 — Device-resident weights + KV cache (~400 LoC)

- [ ] E.6.1 Upload all Q4_K/Q6_K weight bytes to VRAM once at load.
      Keep host bytes only when an explicit `--cpu-fallback` flag is
      set.
- [ ] E.6.2 KV cache buffers live in VRAM.
- [ ] E.6.3 CUDA Graphs for the per-token compute path.
- [ ] E.6.4 Bench: expect ≥ 30 t/s decode (within 2× of llama.cpp
      GPU).

## Validation

- [ ] V.1 `cargo test -p larql-inference --release --lib
      real_gguf_qwen35_token_diff_vs_llama_cpp` under
      `LARQL_QWEN35_GPU=1` passes (GT rank 0 every step).
- [ ] V.2 `openspec validate qwen35-gpu-forward --strict` passes.
- [ ] V.3 Each phase's PR includes its bench delta in
      `openspec/changes/inference-qwen35-deltanet/bench-baseline.md`.
