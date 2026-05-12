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

## 2026-05-12 update — Phase E.1/E.2 GPU dispatch for lm_head + FFN

Pivot off the CPU AVX2 axis: route lm_head Q6_K matvec and all 192
FFN Q4_K matvecs/token through `larql_compute::cuda::CudaBackend`
(the existing `q6k_direct` / `q4k_direct` GPU kernels). Opt-in
behind `--features cuda` and `LARQL_QWEN35_GPU=1`. Weights upload
to VRAM on first dispatch; cache reused thereafter.

| Config | Decode (t/s) | Δ vs CPU lazy | VmRSS |
|---|---:|---:|---:|
| Phase 2d (CPU lazy + AVX2 + rayon) | 0.23 | — | 19.99 GiB |
| **Phase E.1/E.2 (+ GPU lm_head & FFN)** | **0.28** | **+22 %** | 21 GiB (host) |
| llama.cpp CUDA GPU | 50.60 | 220× theirs | 14.76 GiB VRAM |

Modest +22 % gain — much less than expected. The matvec wins are
real, but **the DeltaNet recurrence stays on CPU** and dominates
steady-state decode time at 3.6 s/token. Per-token contributions:

- **DeltaNet recurrence (CPU scalar)** — 48 layers × `delta_net_step`
  with per-head state matrices: this is now the bottleneck.
- **Per-matvec host↔device transfer** — ~480 transfers/token at
  PCIe Gen4 ~25 GB/s ≈ 1.5 ms each, adds up.
- **Non-matvec ops (norms, silu, residual adds)** still CPU.

**The next big perf lever is Phase E.4** — a CUDA kernel for the
DeltaNet recurrence + Conv1D-with-state. Per-head state matrices
fit in shared memory (128×128 f32 = 64 KB per head, ok on Ampere
SM-89). Mirrors llama.cpp's
`ggml_compute_forward_gated_delta_net_one_chunk` which we already
diffed bit-exact in Phase C. Estimated ~600 LoC + the cudarc PTX
plumbing.

E.3 (DeltaNet `attn_qkv`/`attn_gate`/`ssm_out` + full-attn
q/k/v/o through GPU) is now done; it is pure projection plumbing
and remains marginal until E.4 moves the recurrence. E.6
(device-resident weights + KV cache + CUDA Graphs) is the
longer-term arc.

**Parity preserved**: `real_gguf_qwen35_token_diff_vs_llama_cpp`
under `LARQL_QWEN35_GPU=1` still emits the same
`[<think>, \n\n, </think>, \n\n, Hello]` with GT rank 0 every step.

## 2026-05-12 update — Phase E.3 GPU dispatch for DeltaNet + full-attn projections

Extended the Phase E backend path to route the remaining
lazy-quantised projection matvecs through `CudaBackend`: DeltaNet
`attn_qkv`, `attn_gate`, `ssm_out`, and full-attn `attn_q`,
`attn_k`, `attn_v`, `attn_o`. The token-diff parity harness now
attaches the same backend when `LARQL_QWEN35_GPU=1`, so the
documented GPU validation command exercises CUDA dispatch.

Protocol:

```bash
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=16 LARQL_QWEN35_BENCH_DECODE=4 \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_bench -- --nocapture
```

Result:

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.1/E.2 (+ GPU lm_head & FFN) | — | 0.28 | 21 GiB (host) |
| **Phase E.3 (+ DeltaNet/full-attn projections)** | **0.33** | **0.33** | **21.16 GiB** |

E.3 buys another modest +18 % over E.1/E.2. The headline still
confirms the same bottleneck: DeltaNet recurrence and Conv1D remain
CPU-resident, so Phase E.4 is still the real unlock.

## 2026-05-12 update — Phase E.4 first pass: CUDA Conv1D + DeltaNet recurrence kernels

Added CUDA kernels for the Qwen3.6 DeltaNet Conv1D-with-state and
decay-first recurrence. The recurrence kernel keeps the `state[s, s,
h_v]` layout used by ndarray (`h_v` fastest), uses one CUDA block per
V head, and matches the llama.cpp-compatible C.5j decay-first order.
`CudaBackend` now caches the Conv1D and recurrent state buffers by
host state pointer and re-uploads only when `next_position == 0`, so
the device buffer is authoritative during an active sequence.

Validation:

```bash
cargo test -p larql-compute --features cuda cuda::deltanet -- --nocapture

LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_token_diff_vs_llama_cpp -- --nocapture
```

The token-diff parity check still emits
`[<think>, \n\n, </think>, \n\n, Hello]` with GT rank 0 every step.

Bench protocol was unchanged from E.3:

```bash
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=16 LARQL_QWEN35_BENCH_DECODE=4 \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_bench -- --nocapture
```

Result:

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.3 (+ DeltaNet/full-attn projections) | 0.33 | 0.33 | 21.16 GiB |
| **Phase E.4.1/E.4.2 first pass (+ CUDA Conv1D/recur)** | **0.33** | **0.33** | **21.16 GiB** |

The first pass preserves correctness but does **not** improve headline
throughput. The bottleneck has shifted from arithmetic in the scalar
recurrence to per-layer launch/synchronisation and CPU/GPU ping-pong:
Conv1D output, recurrence output, post-recurrence RMSNorm, z-gating,
and residual/FFN boundaries still cross back to the CPU every layer.
The remaining E.4 work should focus on fusing the DeltaNet block
around the recurrence output and moving per-head L2/RMSNorm/z-gate
operations onto the same device path; otherwise E.6-style
device-resident activations/CUDA graphs are required for the expected
multi-tok/s jump.

## 2026-05-12 update — Phase E.4.3 GPU per-head L2/RMSNorm

Added CUDA reductions for the remaining per-head DeltaNet norms:
Q/K L2 normalisation in the `[head_dim, n_k_heads]` dim-major layout
and post-recurrence RMSNorm in the head-major `[n_v_heads, head_dim]`
layout. Both are exposed as optional `ComputeBackend` hooks and keep
the CPU implementation as fallback. The CUDA module now validates
Conv1D, recurrence, L2, and RMSNorm against tiny CPU references.

Validation:

```bash
cargo check -p larql-inference --features cuda
cargo test -p larql-compute --features cuda cuda::deltanet -- --nocapture
cargo test -p larql-inference --lib qwen35 -- --nocapture

LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_token_diff_vs_llama_cpp -- --nocapture
```

The token-diff parity check still emits
`[<think>, \n\n, </think>, \n\n, Hello]` with GT rank 0 every step.

Bench protocol was unchanged from E.4.1/E.4.2:

```bash
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=16 LARQL_QWEN35_BENCH_DECODE=4 \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_bench -- --nocapture
```

Result:

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.4.1/E.4.2 first pass (+ CUDA Conv1D/recur) | 0.33 | 0.33 | 21.16 GiB |
| **Phase E.4.3 (+ CUDA per-head L2/RMSNorm)** | **0.32** | **0.33** | **21.16 GiB** |

Correctness is preserved, but the E.4.4 target (≥ 10 decode t/s) is
still unmet. The new reductions remove more CPU arithmetic but add
more tiny GPU launches and synchronising host returns. The next
meaningful speed step is a fused/device-resident DeltaNet block path
or the E.6 device-resident activation/weight pipeline; standalone
host-returning hooks are correctness plumbing, not enough throughput
plumbing.

## 2026-05-12 update — Phase 2d lazy-quant embed (105 → 20 GiB, −80.9 %)

Adds `QuantTensor::row_to_f32(token_id)` for the embed-lookup
pattern (a row read, not a matvec), then wires it into
`qwen35_forward_step` and the GGUF lazy loader. `embed_quant`
becomes a peer of `lm_head_quant` on `ModelWeights` and
`Qwen35Weights`.

| Config | Decode (t/s) | VmRSS |
|---|---:|---:|
| Phase 2c (lazy FFN/attn) | 0.23 | 24.07 GiB |
| **Phase 2d (+ embed lazy)** | **0.23** | **19.99 GiB** |
| Δ vs Phase 2c | same | **−4.08 GiB** |
| Δ vs baseline | −53 % | **−85.26 GiB (−81.0 %)** |

Speed unchanged — embed lookup is a single-row dequant per token,
amortised against the 256 matvecs/token already on the lazy path.

**llama.cpp's ~16 GiB target is now ~4 GiB away.** Remaining
chunks are smaller per-head SSM tensors (ssm_alpha / ssm_beta /
ssm_conv1d / ssm_norm) and the per-layer norm vectors. Closing
the last 4 GiB would require lazifying those too, but each one is
tiny (1-50 MB), so the engineering effort per GiB has crossed an
inflection point — Phase 3b (cache-tile batched Q4_K matvec) is
now a higher-priority lever.

**Parity preserved**: argmax bit-exact, GT rank 0 every step.

## 2026-05-12 update — Phase 2c lazy-quant full-attn q/k/v/o

Extends the lazy set to the four full-attention projections per
attn layer (16 attn layers × q/k/v/o = 64 additional matvecs/token).

| Config | Decode (t/s) | VmRSS |
|---|---:|---:|
| Phase 2b (lazy FFN + DeltaNet projs) | 0.20 | 29.62 GiB |
| **Phase 2c (+ full-attn q/k/v/o)** | **0.23** | **24.07 GiB** |
| Δ vs Phase 2b | **+15 %** | **−5.55 GiB** |
| Δ vs baseline | −53 % | **−81.18 GiB (−77.1 %)** |

Speed actually **improved slightly** (0.20 → 0.23 t/s): the
full-attn dense matvecs on x86 CPU run at f32 BLAS but as
single-vector sgemv (no batching across rows), whereas the
rayon-parallel Q4_K kernel splits each matvec's rows across cores.
On a 16-core box the parallelism wins even for these moderately-sized
matrices.

llama.cpp parity on RAM (~16 GiB) is now **~8 GiB away**. The
remaining big chunks are:
- `embed` `{vocab=248320, hidden=5120}` ≈ 5 GiB (Q4_K → ~1 GiB)
- Per-head SSM tensors (ssm_beta, ssm_alpha, ssm_conv1d, ssm_norm)
- Various per-layer norm vectors

Embed needs a different code path (row-lookup not matvec) — a
future `QuantTensor::row_to_f32(token_id)` would dequant one row
on demand. Per-head SSM tensors are small enough that the win is
marginal (<2 GiB total).

**Parity preserved**: `real_gguf_qwen35_token_diff_vs_llama_cpp`
still emits the same `[<think>, \n\n, </think>, \n\n, Hello]`
sequence with GT rank 0 every step.

## 2026-05-12 update — Phase 2b lazy-quant attn_qkv / attn_gate / ssm_out

Phase 2b extends the lazy-tensor set to the three big DeltaNet
projections per linear layer: `attn_qkv` `{conv_dim=10240, hidden=5120}`,
`attn_gate` `{value_dim=6144, hidden=5120}`, `ssm_out`
`{hidden=5120, value_dim=6144}`. 48 linear-attention layers × 3
tensors = 144 additional matvecs/token through the lazy path.

| Config | Decode (t/s) | VmRSS |
|---|---:|---:|
| Phase 3 (lazy FFN + AVX2 + rayon) | 0.20 | 46.65 GiB |
| **Phase 2b (+ attn_qkv / attn_gate / ssm_out)** | **0.20** | **29.62 GiB** |
| Δ vs Phase 3 | same | **−17.03 GiB** |
| Δ vs baseline | −59 % | **−75.63 GiB (−71.9 %)** |

Each linear-attention layer's three big projections sum to
~470 MB f32 → ~75 MB Q4_K (lossy 6.3×); 48 layers × 395 MB saved
= ~19 GiB. The observed 17 GiB drop matches that estimate.

Speed unchanged at 0.20 t/s — the extra 144 matvecs/token are
amortised by the same rayon row-parallelism that drove Phase 3.

Remaining ~30 GiB is mostly:
- `embed` `{vocab=248320, hidden=5120}` ≈ 5 GiB
- Full-attn projections (16 layers × q/k/v/o) ≈ 4-8 GiB
- Smaller per-layer SSM tensors and DeltaNet `ssm_norm` /
  per-head bias vectors
- Plus the dequantized layer attn_*_norm / ssm_norm vectors

**Parity preserved**: `real_gguf_qwen35_token_diff_vs_llama_cpp`
still emits the same `[<think>, \n\n, </think>, \n\n, Hello]`
sequence with logits `[28.18, 24.78, 25.47, 30.39, 21.66]` and
GT rank 0 every step.

llama.cpp parity on RAM (~16 GiB) is now ~13 GiB away.

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
