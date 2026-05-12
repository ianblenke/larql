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
