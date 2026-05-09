## Why

Phase 1 (`cuda-attn-wmma-f16kv`) switched the CUDA decode K/V cache
storage from f32 to f16 — the prerequisite for WMMA-based attention
compute. This change is the **viability probe** for Phase 2 (the
actual WMMA attention kernel) plus a complete implementation sketch.

The full Phase 2 kernel rewrite is genuinely 5–10 hours of focused
CUDA debugging — multi-session work. This change ships only the
parts that can land cleanly today: a `<mma.h>`-based smoke test
that confirms the toolchain works, and a detailed proposal so the
next session can pick up without re-discovering the constraints.

## What This Change Ships

### Phase 2A: WMMA viability probe

- ADD `cuda_wmma_mma_sync_smoke_test` in `cuda::backend::tests`.
  Compiles a kernel that uses `nvcuda::wmma::*` (16×16×16 f16
  → f32) via NVRTC, runs it on a fixed `A @ B^T` test, and
  verifies max-element diff against an f32 host reference ≤ 1e-2.
  Confirms:
  - cudarc 0.19's NVRTC pipeline can compile `<mma.h>`-based
    kernels (with the right include path).
  - The toolchain's `wmma::load_matrix_sync` / `mma_sync` /
    `store_matrix_sync` produce arithmetically correct results
    on the dev box (sm_89, RTX 4090).
  - The `col_major` B-fragment loaded from row-major memory
    yields `A @ B^T` (the canonical attention `Q @ K^T`
    pattern when A=Q, B=K row-major).

### NVRTC include-path autodiscovery

The probe walks a candidate list (`/usr/local/cuda-12.5/...`,
`/opt/cuda/include`, etc.) at compile time. The full Phase 2
kernel will reuse this via a small helper in `cuda::nvrtc_paths`
or equivalent.

## Update: Phase 2B-as-sketched is dead — see `cuda-attn-wmma-kernel-v2`

The follow-up change `cuda-attn-wmma-kernel-v2` ran a head-to-head
microbench between SIMT and the Phase 2B-sketched WMMA score-matmul
on Gemma 3 4B's GQA shape. **WMMA loses 20–32% at every n_ctx
tested** with bit-exact parity. GQA gives only 12.5% fragment row
utilization × 25% warp utilization × 50% block-count utilization;
SIMT wins by structural parallelism even though Tensor Core
per-op throughput is higher.

The Phase 2B sketch below is preserved for reference but should NOT
be implemented as written. Future Phase 2 work has to use one of:

1. Multi-warp MMA per block (3-4 warps each issuing MMAs in parallel,
   not just warp 0)
2. Drop the kvh-grouped layout (replicate K per q_head in shared, or
   accept "wrong K" rows and mask)
3. Raw `mma.sync.aligned` PTX intrinsics for finer warp control

Estimated cost: 5–10 days with no guaranteed net win on Ada (sm_89).

## Why Phase 2B is NOT in this change

Phase 2B is the production WMMA attention kernel:

- New NVRTC kernel `fused_decode_attention_f32_wmma` that uses
  `wmma::fragment` for the score-loop matmul and the
  output-loop matmul.
- Reshape per-block work: pad `num_q_heads` (typically 8 for
  Gemma 3 4B) to 16 with zero rows, so a single 16×16×16 MMA
  call can cover all q_heads × 16 K positions in one shot.
- Q rotation has to land in shared-memory f16 (not the current
  f32 `q_rot`) so it can be loaded directly into a WMMA
  fragment.
- Softmax stays on f32 SIMT inside the block — only the score
  and output **matmuls** go to Tensor Cores.
- Output computation similarly: the per-d output loop becomes
  a `wmma::mma_sync(out_frag, scores_frag, V_frag)` for each
  16-wide head_dim tile.

Estimated cost: 5–10 hours of focused CUDA work. Multi-session.
The full proposal sketch with kernel pseudo-code is below.

### Phase 2B kernel sketch

```c
extern "C" __global__ void fused_decode_attention_f32_wmma(
    const float*  q,             // [num_q_heads, head_dim], f32
    const float*  k_new,
    const float*  v_new,
    unsigned short* k_cache,     // [max_seq, num_kv_heads, head_dim], f16
    unsigned short* v_cache,     // same, f16
    const float* q_norm, const float* k_norm,
    float*       out,            // [num_q_heads, head_dim]
    int num_q_heads,             // assumed ≤ 16; pad to 16 in shared
    int num_kv_heads,
    int head_dim,                // multiple of 16
    const int*   pos_dev,
    int max_seq, int rotary_dim,
    float rope_base, float eps, float qk_norm_offset,
    float attn_scale, float softcap,
    int  use_qk_norm
) {
    // Per-block: handle ALL q_heads (not 1 per block as before).
    // Grid: (1, 1, 1). Block: (32 × 4, 1, 1) = 128 threads = 4 warps.
    using namespace nvcuda::wmma;

    // Q is small (≤16 rows × head_dim cols). Load into shared as f16,
    // padded to 16 rows. Apply RoPE once.
    __shared__ __half q_smem[16 * MAX_HEAD_DIM];
    __shared__ __half k_smem[MAX_KV_TILE * MAX_HEAD_DIM];
    __shared__ __half v_smem[MAX_KV_TILE * MAX_HEAD_DIM];
    __shared__ float  scores_tile[16 * 16];

    // 1. Load Q into q_smem with RoPE applied (parallelised across
    //    block threads). Pad rows ≥ num_q_heads with zero.

    // 2. Per K tile of 16 positions: load K from cache into k_smem
    //    (with the new-token's K rotated and written to the cache
    //     first). Similarly V into v_smem.

    // 3. Score matmul (per-head_dim tile of 16):
    //    for (k_tile = 0; k_tile < head_dim; k_tile += 16) {
    //      fragment<matrix_a, 16,16,16, __half, row_major> q_frag;
    //      fragment<matrix_b, 16,16,16, __half, col_major> k_frag;
    //      fragment<accumulator, 16,16,16, float> s_frag;
    //      load_matrix_sync(q_frag, q_smem + k_tile, head_dim);
    //      load_matrix_sync(k_frag, k_smem + k_tile, head_dim);
    //      mma_sync(s_frag, q_frag, k_frag, s_frag);
    //    }
    //    store_matrix_sync(scores_tile, s_frag, 16, mem_row_major);

    // 4. SIMT softmax over scores_tile (16 q_heads × 16 K positions).

    // 5. Output matmul:
    //    fragment<matrix_a, 16,16,16, __half, row_major> p_frag;
    //    fragment<matrix_b, 16,16,16, __half, row_major> v_frag;
    //    fragment<accumulator, 16,16,16, float> out_frag;
    //    Convert scores_tile to f16, load into p_frag.
    //    For each d_tile of 16 in head_dim:
    //      load_matrix_sync(v_frag, v_smem + d_tile, head_dim);
    //      mma_sync(out_frag, p_frag, v_frag, out_frag);
    //    Store out_frag to global `out`.
}
```

### Predicted savings

The score-loop matmul is the dominant per-token attention cost
(~50% of the kernel's time per profile). With WMMA:

- Per kernel call's matmul work goes from ~5 K fmas (n_ctx ×
  head_dim) at 1 fma/cycle to ~5 × 32 fmas worth of MMA
  (16×16×16 = 4096 fmas in 4 cycles = 1024 fma-equiv/cycle).
  Net speedup on the matmul portion: 30-40×, but only ~50% of
  kernel time → real kernel speedup ~2×.
- Output-loop matmul: same shape, same speedup, same fraction.
- Non-matmul (softmax, RoPE, K/V append): unchanged.

Per-token attention budget (now 2.67 ms via profile, ~2 ms
real-time): expect ~1 ms reduction. Decode from 8.04 ms → ~7.0
ms / ~143 tok/s.

Combined with INT4-IMMA mmvq (separate change, similar effort):
projected decode 5.5–6.5 ms / 154–182 tok/s — within ~25% of
llama.cpp's 4.34 ms / 230.2 tok/s.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the WMMA viability contract.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/backend.rs::tests` — adds the
    smoke test.
- **Affected systems**: GPU only; only runs when
  `LARQL_CUDA_AVAILABLE=1`. Does not change any production
  code path.

## Risks and back-out

- Risk: NVRTC's include-path autodiscovery is host-specific. If
  the dev box's CUDA toolkit moves or a CI box has a different
  layout, the smoke test fails with a clear error.
  Mitigation: the candidate-path list is permissive; add new
  candidates as needed.
- No production back-out needed — this change adds zero
  production code paths.

## Acceptance bar

- `cuda_wmma_mma_sync_smoke_test` passes on the dev box (RTX
  4090, sm_89, CUDA 12.5).
- All 200+ existing tests still pass (no regression).
- Phase 2B implementation cost recorded as 5–10 hours, with
  the kernel sketch above as the starting point.
