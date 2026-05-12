# Qwen3.6 27B bench baseline — larql vs llama.cpp

Captured 2026-05-11 on the same host, immediately after C.5k landed
(parity test green, GT rank 0 at every step).

## Setup

- **Model**: Qwen3.6-27B Q4_K_S GGUF, 14.76 GiB on disk, 26.90 B params.
- **Host**: Linux x86_64, NVIDIA RTX 4090 (24 GiB VRAM), llama.cpp build
  b1-389ff61 with CUDA, larql at `c5k` (post PR #84).
- **Workload sizes**: prefill 32 tokens, decode 8 tokens. Small numbers
  because larql is CPU-only — keeps total wall time under 2 minutes.

## Results

| Config | Backend | Prefill (tok/s) | Decode (tok/s) | Load | Memory |
|---|---|---:|---:|---:|---:|
| llama.cpp | CUDA RTX 4090 (`ngl=99`, `pp128`/`tg64`) | 2097.18 | 50.60 | ~2 s | 14.76 GiB VRAM |
| llama.cpp | CPU (`ngl=0`, `pp32`/`tg8`) | 37.33 | 2.60 | ~5 s | ~16 GiB RAM |
| **larql** | CPU (scalar Rust, `pp32`/`tg8`) | **0.48** | **0.49** | 49 s | **~100 GiB RAM** |

`pp` = prefill tok/s, `tg` = decode tok/s. llama.cpp numbers from
`llama-bench`; larql numbers from
`real_gguf_qwen35_bench` test (loops `qwen35_forward_step`).

## Headline ratios

| | larql / llama.cpp (CPU) | larql / llama.cpp (GPU) |
|---|---:|---:|
| Prefill speed | 1/78 (1.3%) | 1/4370 (0.02%) |
| Decode speed | 1/5.3 (19%) | 1/103 (1.0%) |

## Memory blow-up explained

larql currently **dequantizes every Q4_K_S weight to f32 at load time**
and holds the full f32 model in RAM. 26.90 B params × 4 bytes ≈ 107 GiB
matches the observed 100 GiB RSS. llama.cpp keeps the model in its
quantized form (14.76 GiB on disk, ~16 GiB resident with overhead) and
dequantizes per-tile during matmul.

This is the single biggest item on the perf TODO list. Until that lands:
- ~2 GiB/s f32 matmul throughput (BLAS) × ~50 GB per forward pass at
  27 B params → seconds per token, exactly what we see.
- 100 GiB RAM means larql can't actually run a 35-B-MoE host without
  ≥ 128 GiB system memory.

## 2026-05-11 update — Phase 3 AVX2 + rayon for Q4_K matvec

Phase 3 adds (a) an AVX2 inner-loop kernel for `q4k_row_dot` on
x86_64 with fully vectorised dequant + FMA, and (b) rayon
`par_iter_mut` over the rows of every Q4_K and Q6_K matvec in
`QuantTensor::matvec`. Same opt-ins (`LARQL_QWEN35_LAZY_FFN=1
LARQL_QWEN35_LAZY_LM_HEAD=1`).

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| baseline (dequant + BLAS) | 0.48 | 0.49 | 105.25 GiB |
| Phase 2 (lazy, scalar) | 0.06 | 0.06 | 46.65 GiB |
| **Phase 3 (lazy, AVX2 + rayon)** | **0.21** | **0.20** | **46.65 GiB** |
| Δ vs Phase 2 | +250 % | +233 % | same |
| Δ vs baseline | −56 % | −59 % | −58.6 GiB |

The AVX2 kernel on its own barely moved the needle (0.06 → 0.07)
— LLVM auto-vectorises the scalar code well already. **Rayon
across rows** is where the speedup came from: 192 FFN matvecs per
token now fan out 14336 / 5120 row-dots across cores in parallel,
saturating the multi-core machine. Per-row AVX2 is the cherry on
top.

Now only 2.4× slower than the f32 BLAS baseline at less than half
the RAM. The remaining gap is mostly the per-row dispatch overhead
and the fact that BLAS sgemv batches rows in cache-friendly tiles.
Phase 3b (batched-row AVX2 matvec à la llama.cpp's
`mul_mat_q4k_q8k`) is the next perf lever.

Parity preserved: `real_gguf_qwen35_token_diff_vs_llama_cpp` still
emits the same `[<think>, \n\n, </think>, \n\n, Hello]` sequence
with GT rank 0 at every step.

## 2026-05-11 update — Phase 2 lazy-quant FFN

Same harness, smaller workload (prefill 8 / decode 2) because the
all-lazy path is 8× slower per token. Opt-in:
`LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1`.

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| larql CPU, fully dequant (baseline) | 0.48 | 0.49 | **105.25 GiB** |
| larql CPU, lazy lm_head only | 0.31 | 0.31 | 101.30 GiB |
| **larql CPU, lazy lm_head + FFN** | **0.06** | **0.06** | **46.65 GiB** |
| Δ vs baseline | −88 % | −88 % | **−58.6 GiB** |

192 FFN matvecs per token now route through scalar Q4_K
`q4k_row_dot` instead of f32 BLAS — that's where the 8× slowdown
comes from. The RAM win is huge but the trade-off is real.

**Parity preserved**: `real_gguf_qwen35_token_diff_vs_llama_cpp`
still produces argmax `[<think>, \n\n, </think>, \n\n, Hello]`
with logits `[28.18, 24.78, 25.47, 30.39, 21.66]` and GT rank 0
at every step, identical to the dequant baseline. The lazy path
is bit-exact in the matvec results (modulo Q4_K dequant rounding,
which is identical to llama.cpp's).

**Remaining RAM** (~47 GiB) is mostly the embed (5.1 GiB), DeltaNet
SSM tensors (alpha/beta/gate/qkv/out/conv1d/norm), and full-attn
projections. Phase 2b (lazy these) would close most of the gap to
llama.cpp's ~16 GiB resident.

The Phase 3 AVX2 quant kernels are now clearly the next perf lever
— without them, this path is unusable for serving.

## 2026-05-11 update — Phase 1 lazy-quant lm_head (PR follow-up)

The `qwen35-lazy-quant-matmul` Phase 1 change introduces
`load_gguf_lazy_lm_head` and `QuantTensor::matvec` to keep
`output.weight` (Q6_K) in its native form. Opt-in via
`LARQL_QWEN35_LAZY_LM_HEAD=1`. Same bench harness, smaller workload
(prefill 16 / decode 4 to keep total wall time manageable):

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| larql CPU, dequant lm_head | 0.48 | 0.49 | 105.25 GiB |
| **larql CPU, lazy lm_head** | **0.31** | **0.31** | **101.30 GiB** |
| Δ | −36 % | −37 % | **−3.95 GiB** |

Phase 1 trades a 37 % per-step slowdown on lm_head for ~4 GiB RAM
recovery. The slowdown is from the scalar Q6_K `q6k_row_dot` path
beating f32 BLAS — expected per the Phase 1 proposal's non-goals.
Phase 3 (x86 AVX2 quant kernels) will close that. Phase 2 (lazy
FFN tensors) is where the RAM number drops toward llama.cpp's
~16 GiB.

## Implications for Phase E / F roadmap

1. **Quant-aware matmul** is now the bottleneck. Even staying on CPU,
   if our matmul stays in Q4_K_S we'd jump from 0.5 tok/s decode to
   somewhere near llama.cpp's 2.6 tok/s. That's the next big win.
2. **CUDA path** (Phase E in `tasks.md`) gives the 50–100× over CPU
   that llama.cpp shows. Realistically required to be competitive at
   all on 27B+ models.
3. **VRAM headroom story** (the `--ffn` remote-offload pitch from the
   original Phase F design) only becomes meaningful once attention is
   on GPU. Today it's all CPU.
4. **Correctness is unblocked.** Parity is bit-exact (modulo Q5_K/Q6_K
   quant noise) per C.5j/C.5k. The remaining work is purely
   performance — no more semantic surprises expected.

## Reproducibility

```bash
# llama.cpp baseline
~/3rd-party/llama.cpp/build/bin/llama-bench \
  -m output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
  -p 128 -n 64 -r 2          # GPU
~/3rd-party/llama.cpp/build/bin/llama-bench \
  -m ... -p 32 -n 8 -r 2 -ngl 0   # CPU

# larql baseline
LARQL_QWEN35_GGUF=output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=32 LARQL_QWEN35_BENCH_DECODE=8 \
cargo test -p larql-inference --release --lib real_gguf_qwen35_bench \
  -- --nocapture
```
