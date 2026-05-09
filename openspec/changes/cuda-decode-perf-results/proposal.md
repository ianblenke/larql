## Why

Consolidates the multi-session CUDA decode/prefill performance push
into one navigable record. Documents:

1. The 13 branches shipped (8 wins, 5 documented negative results),
   with empirical data and the proposal for each.
2. The bench numbers at every checkpoint vs llama-cpp-turboquant.
3. The remaining gap and its decomposition.
4. Concrete next-step paths with effort/risk/reward estimates.

This is a navigation aid, not a code change.

## What This Change Ships

A new top-level `openspec/changes/cuda-decode-perf-results/proposal.md`
(this file) that's the entry point for any future CUDA-decode work.
No code changes.

## Bench progression (RTX 4090 / sm_89, Gemma 3 4B Q4_K_M, 6-token prompt + 20 decode tokens)

| Checkpoint | Decode ms/tok | tok/s | Prefill ms | Notes |
|---|---:|---:|---:|---|
| Pre-session baseline | 9.62 | 103.9 | 18.0 | f32 throughout, no graph |
| `cuda-decode-cuda-graph` | 8.52 | 117.4 | 17.9 | CUDA Graph capture/replay |
| `cuda-attn-grid-split` | 8.31 | 120.3 | 17.9 | `d_split` grid parallelism |
| `cuda-prefill-tensor-cores` | 8.31 | 120.3 | **10.7** | f16 cuBLAS hgemm prefill (-40%) |
| `cuda-q4k-mmvq-warp-cooperative` | 8.23 | 121.5 | 10.7 | llama.cpp 4-warp/row |
| `cuda-fused-norm-add` | 8.19 | 122.1 | 10.7 | TensorRT-LLM-style residual fusion |
| `cuda-attn-wmma-f16kv` | **8.04** | **124.4** | 10.7 | f16 KV cache + variance collapse |
| **llama-cpp-turboquant** | **4.34** | **230.2** | **6.25** | Reference target |

Combined movement: decode -16% / +20% tok/s, prefill -40%, run-to-run
variance collapsed 24× (0.47 ms range → 0.02 ms range). Decode gap
with llama.cpp closed from 2.18× to 1.85×; prefill gap from 3.20×
to 1.71×.

## Branches shipped (chronological)

### Wins (8)

| Branch | Decode Δ | Mechanism |
|---|---:|---|
| `feat/cuda-decode-cuda-graph` | 9.62 → 8.52 | CUDA Graph capture eliminates per-call cuLaunchKernel overhead. f16 KV cache foundation; non-default stream + disabled event tracking for capture compatibility. |
| `feat/cuda-attn-grid-split` | 8.38 → 8.31 | `d_split = 4` adds head_dim parallelism to fused decode attention. Grid grows from `(num_q_heads,)` to `(num_q_heads, d_split)`. |
| `feat/cuda-prefill-tensor-cores` | (prefill 18.0 → 10.7) | f16 cuBLAS hgemm via Tensor Cores for prefill projection GEMM. f16 weight cache (`q4k_f16_device_cache`, `q6k_f16_device_cache`). |
| `feat/cuda-q4k-mmvq-warp-cooperative` | 8.50 → 8.23 | llama.cpp's 4-warps-per-row mmvq (vs our 1-warp-per-row, 4-rows-per-block). Shape-aware dispatch routes `kv` and `down` to coop, others to legacy. |
| `feat/cuda-fused-norm-add` | 8.23 → 8.19 | TensorRT-LLM `RMSNormPlugin`-style residual fusion: `dst += rms_norm(src) * scale` in one kernel. |
| `feat/cuda-attn-wmma-f16kv` | 8.22 → 8.04 | f16 KV cache (Phase 1 of WMMA Phase 2). Halves K/V slab footprint, halves K/V read bandwidth, **collapses run-to-run variance**. |

### Negative results (5, documented)

| Branch | Hypothesis | Outcome |
|---|---|---|
| `feat/cuda-q4k-mmvq-down-tile` | Tile size / arch tweaks help proj_down | rpb=4 / compute_61 already optimal across all shapes. proj_down asymmetry was profile-sync artifact. |
| `feat/cuda-tensor-cores-q4k` | cuBLAS hgemm-b1 beats dp4a for INT4 mmvq | hgemm-b1 is **2.0–5.0× SLOWER** than dp4a on every Gemma 3 4B shape. Tensor Cores' 16×16 output tile wastes 15/16 rows at batch=1. |
| `feat/cuda-attn-wmma-phase2` | NVRTC supports `<mma.h>` and WMMA fragments work | ✅ Toolchain works, kernel produces correct output. (Phase 2A only — kernel writing is Phase 2B follow-up.) |
| `feat/cuda-attn-wmma-kernel-v2` | Single-warp WMMA score-matmul beats SIMT on GQA | WMMA is **20–32% slower** at every n_ctx. GQA gives 12.5% fragment row utilization (2 of 16 rows real per kvh-group MMA). |
| `feat/cuda-attn-wmma-multi-warp` | Multi-warp d-tile reduction closes the WMMA-vs-SIMT gap | Multi-warp lifts WMMA by 8–11% over single-warp but is **still 13–27% slower** than SIMT on Gemma. The structural GQA fragment-row problem is orthogonal to warp-level parallelism. |

## Remaining gap with llama.cpp (3.7 ms decode, 4.4 ms prefill)

Bench profile (legacy non-graph + sync) puts the per-token decode
GPU time at:

| Bucket | ms | % | Optimization status |
|---|---:|---:|---|
| `attn_call` (fused decode attention) | 2.67 | 30% | SIMT optimal for Gemma's GQA per cuda-attn-wmma-multi-warp empirical result. Further gains require either non-GQA models, or 5-10 days of mma.sync PTX work. |
| `proj_down` (mmvq, hidden=10240, rows=2560) | 1.61 | 18% | Already on llama.cpp-style 4-warp coop. Further requires Marlin-style INT4-IMMA. |
| `proj_gate_up` (mmvq) | 1.41 | 16% | dp4a optimal at batch=1 per cuda-tensor-cores-q4k. Marlin INT4-IMMA is the only known path. |
| `norm_cpu` | 1.07 | 12% | Already fused (cuda-fused-norm-add). Marginal further gain. |
| `proj_qkv` (3 mmvq) | 0.85 | 10% | Could fuse to one concat-mmvq for ~0.07 ms. Plumbing cost > gain. |
| `residual_cpu` | 0.91 | 10% | Already fused. |
| `proj_wo` | 0.37 | 4% | Small budget. |

## Concrete next-step paths

### Path A: Marlin-style INT4-IMMA mmvq (5-10 days, est. -1.0 to -1.5 ms)

Custom kernel using `mma.sync.aligned.m16n8k32.s32.s8.s8.s32`
(sm_80+ INT8 Tensor Cores). On-the-fly dequant of Q4_K to INT8 in
shared memory, MMA with INT8 accumulator. Marlin's published
results show ~80% of HBM bandwidth utilization on RTX 4090 vs
our dp4a ~50%.

Risk: high. Custom PTX, weight repacking at layer-load, parity
bring-up.

### Path B: WMMA attention via mitigation #2 or #3 (3-10 days, est. -0.5 to -1.0 ms)

Per `cuda-attn-wmma-multi-warp`'s empirical result, mitigations
#1 (multi-warp) is empirically settled negative. Remaining:

- **#2** (drop kvh-grouped layout, replicate K per q_head): K
  bandwidth doubles, but for short context (our bench) bandwidth is
  small. **Untested empirically — could be the missing piece.**
- **#3** (raw `mma.sync.aligned` PTX): finer warp-level control
  than `nvcuda::wmma::*` allows.

Risk: high for #3, medium for #2. Either could still lose to SIMT.

### Path C: Speculative decoding (2-3 weeks, est. 1.5-2× throughput)

Lifts batch ≥ 8, unlocking Tensor Cores for both attention AND
mmvq. Requires draft-model integration, tree attention, verification
logic. Architectural change, biggest single potential win.

Risk: very high — multi-week scope, model-specific tuning.

### Path D: Q/K/V mmvq fusion (1-2 days, est. -0.07 to -0.15 ms)

Concatenate `[W_q | W_k | W_v]` per-layer, run one mmvq instead
of three. Saves 2 launches per layer (68 per token in graph mode)
and 2 redundant Q8_1 input reads.

Risk: low. Plumbing cost (CudaSlice → CudaView signature change
in `fused_decode_attention_device_kv_into`) is the main lift.
Modest gain; demonstrates the concat-weight pattern.

### Path E: norm + Q8_1 quantize fusion (1-2 days, est. -0.1 ms)

Fuse `rms_norm + quantize_q8_1` for the pre-attn and pre-FFN
inputs. Different launch geometries (1 block vs n_blocks of 32)
make this awkward to fuse cleanly.

Risk: low-medium. Marginal gain.

## Recommended next session

**Path D (Q/K/V mmvq fusion)** for a quick clean win, then **Path A
(Marlin INT4-IMMA mmvq)** as the headline multi-day effort. Path A
attacks the largest remaining bucket (49% of decode) with a known-
viable approach (Marlin's published results). Speculative decoding
(Path C) is a 2-3 week investment; defer until Marlin's 1.0-1.5 ms
savings are banked.

WMMA attention (Path B) is the most explored dead-end — three
proposals settled it negatively for Gemma's GQA layout. Don't
revisit unless targeting a non-GQA model.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the consolidated decode/prefill
  performance results contract.

## Impact

- **Affected files**: this proposal only. No code changes.
- **Affected systems**: documentation only.

## Risks and back-out

None — pure documentation.

## Acceptance bar

This proposal SHALL be the entry point for the next CUDA-decode
optimization session. The next contributor SHALL be able to
identify the highest-ROI tractable path within ~5 minutes of
reading.
