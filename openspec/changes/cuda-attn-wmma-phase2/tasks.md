# cuda-attn-wmma-phase2 — tasks

## Phase 2A: viability probe (this change)

- [x] 1.1 Add `cuda_wmma_mma_sync_smoke_test` to
      `cuda::backend::tests`. Compiles a 16×16×16 f16 → f32
      WMMA kernel via NVRTC and verifies output ≤ 1e-2.
- [x] 1.2 NVRTC include-path autodiscovery (walk
      `/usr/local/cuda-{12.5,12.1}/...`, `/usr/local/cuda/...`,
      `/opt/cuda/...`).
- [x] 1.3 Confirm B-fragment col_major load over row-major
      memory yields `A @ B^T` (the canonical attention pattern).
- [x] 1.4 All 200+ unit + integration tests still pass.

## Phase 2B: production WMMA attention kernel (next session)

- [ ] 2.1 New NVRTC kernel `fused_decode_attention_f32_wmma`
      with grid `(1, 1, 1)`, block `(128, 1, 1) = 4 warps`.
      Pads `num_q_heads` to 16.
- [ ] 2.2 Q rotation lands in shared-memory f16 (not the
      current f32 `q_rot`), ready for `load_matrix_sync`.
- [ ] 2.3 Score-loop matmul: `wmma::mma_sync` over
      head_dim/16 tiles of (Q_padded[16, head_dim] @ K_tile[16,
      head_dim]^T). Writes 16×16 score tile to shared memory.
- [ ] 2.4 SIMT softmax + tile-stitching for variable-length
      n_ctx.
- [ ] 2.5 Output matmul: `wmma::mma_sync` over
      `(P_padded[16, 16] @ V_tile[16, head_dim])` per head_dim
      tile. Stores to global `out`.
- [ ] 2.6 Rust wrapper `fused_decode_attention_device_kv_into_wmma`.
- [ ] 2.7 Dispatch helper `attn_wmma_supported(num_q_heads,
      head_dim)` — returns true when num_q_heads ≤ 16 and
      head_dim is a multiple of 16. Default: dispatch through
      it; fall back to the legacy SIMT kernel otherwise.
      `LARQL_CUDA_ATTN_WMMA=0` forces legacy.
- [ ] 2.8 Parity tests pass at 1e-3 (relax to 5e-3 if the f16
      score-tile reduction loses too much precision).
- [ ] 2.9 Bench gate: decode 8.04 → ≤ 7.5 ms / ≥ 133 tok/s.

## Phase 2C: cleanup + archive (after 2B)

- [ ] 3.1 Document final bench numbers in proposal.md.
- [ ] 3.2 Archive both Phase 2A and 2B together.
