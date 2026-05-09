## Why

Path E from `cuda-decode-perf-results`: fuse the captured-decode
pipeline's pre-attn (`h → h_attn → h_attn_q8_1`) and pre-FFN
(`h → h_ffn → h_ffn_q8_1`) `rms_norm + quantize_q8_1` pairs into
a single kernel. Predicted savings: 2 launches per layer (68 per
token) + intermediate buffer write+read elimination.

This change implements Path E and **records the empirical
regression**: the fused kernel is **0.18 ms SLOWER** on average
than the unfused pair (8.22 ms vs 8.04 ms baseline), with the
familiar bimodal variance pattern (8.03 / 8.47).

## Empirical results

10-run avg, RTX 4090, Gemma 3 4B Q4_K, with
`LARQL_CUDA_PREFILL_TENSOR_CORES=1`:

| Variant | decode ms/tok | range |
|---|---:|---|
| Baseline (f16-kv tip) | **8.04** | 8.03–8.05 |
| Path E fused | 8.22 | 8.03–8.51 |
| Path E fused, `LARQL_CUDA_DECODE_GRAPH=0` | 9.37 | 9.19–9.54 |
| Baseline, `LARQL_CUDA_DECODE_GRAPH=0` | 9.05 | (5-run) |

The graph-off comparison confirms the fused kernel itself is
slower: 9.37 ms vs baseline 9.05 ms = **0.32 ms regression**
without any graph-capture variance to blame.

## Mechanism

The pair-of-kernels schedule:

| Phase | Block geometry | SMs busy |
|---|---|---:|
| `rms_norm_device_into` | 1 block × 1024 threads | 1 |
| `quantize_q8_1_device_into` | 80 blocks × 32 threads (1 warp/block, 1 Q8_1 block per warp) | up to 80 |

The fused kernel collapses both phases into one block:

| Phase | Block geometry | SMs busy |
|---|---|---:|
| Fused phase 1 (sum-of-squares reduction) | 1 block × 1024 threads | 1 |
| Fused phase 2 (write normalised values to smem) | 1 block × 1024 threads | 1 |
| Fused phase 3 (per-warp Q8_1 quantize, strided across n_blocks) | 1 block × 32 warps, 80/32 = 3 iters per warp | 1 |

The pair's quantize phase parallelises across **80 SMs** (one
SM per Q8_1 block); the fused phase 3 squeezes the same work
into **32 warps on 1 SM** with 3 strided iterations. Per-warp
work is the same, but wall-clock grows because the GPU's
parallelism is collapsed.

The captured-graph runtime sometimes overlaps the separate
`quantize_q8_1` launch with adjacent work (e.g., the K/V
projections in the next layer); the fused kernel's larger
single-block footprint can't.

## Why this is worth shipping (as a negative result)

This is the **third fusion-style negative result** of the session
(joining `cuda-q4k-qkv-fuse-v2` and `cuda-attn-wmma-multi-warp`).
The pattern is now empirically established:

> **Fusion that reduces launch count but collapses across-SM
> parallelism is a regression on Gemma 3 4B's GQA decode shape.**

Path E joins Path D (Q/K/V mmvq fusion) as a "would-be quick win"
that is invalidated by the existing shape-aware kernel
optimizations. Together they narrow the search space: any future
Path-D-or-E-like fusion idea has to first explain how it preserves
across-SM parallelism for the small/parallel kernels involved.

## What This Change Ships

The change is **kept on the branch** (compiles, parity-passes,
implements all infrastructure) but is **NOT merged** — production
stays on the unfused `rms_norm_device_into +
quantize_q8_1_device_into` pair.

- `RMS_NORM_QUANTIZE_Q8_1_SRC` PTX: 3-phase kernel (reduce →
  write normalised → per-warp Q8_1 quantize). Single block of
  1024 threads, smem ≥ max(bdim, n) × 4 bytes for the dual-purpose
  reduction-then-normalised buffer.
- `RMS_NORM_QUANTIZE_Q8_1_FUNC: OnceLock` cell + lazy load.
- `elem::rms_norm_quantize_q8_1_into(...)` Rust wrapper.
- Captured-decode pipeline calls the fused wrapper at the two
  pre-projection norm sites instead of the two separate calls.

The path is fully implemented and parity-verified. It just
loses 0.18-0.32 ms to the unfused baseline because of the
parallelism-collapse mechanism above.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the fused-norm-quantize path
  contract and documents its empirical regression on
  GQA-shaped Q4_K decode.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/elem.rs` — `RMS_NORM_QUANTIZE_Q8_1_SRC`,
    `RMS_NORM_QUANTIZE_Q8_1_FUNC`, `rms_norm_quantize_q8_1_into` wrapper.
  - `crates/larql-compute/src/cuda/decode.rs` — the two
    pre-projection norm sites in
    `run_decode_pipeline_into_scratch`.
- **Affected systems**: GPU only.

## Risks and back-out

- The branch is shippable in code but produces a regression; the
  recommendation is to NOT merge it.
- Future contributors who want to revisit the fusion need to
  redesign phase 3's parallelism (e.g., emit n_blocks worth of
  blocks in the launch, with shared memory carried across via
  cooperative-groups or a 2-launch in-place rewrite).

## Acceptance bar

- All 200+ existing tests pass with the fused path.
- Bench documents the empirical regression (10-run avg) vs the
  unfused baseline.
- Future contributors see this proposal first when considering
  norm + quantize fusion.
