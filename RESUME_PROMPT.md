# larql resume prompt — feed this to a fresh session

## Where we are (state at end of session 2026-05-22)

**Five PRs landed this session** completing the long-context-prefill
arc on Qwen3.6-35B-A3B. The headline win is **flat 42 ms/tok GPU
prefill across a 32× context range (557 → 17658 tokens)** — the
shmem occupancy cliff is gone and the dynamic-shmem opt-in unlocks
prefill up to ~24K tokens per launch.

### PRs landed

| PR | What | Win |
|---|---|---|
| #246 | Phase 4e A/B switches + Step B routing + skip_ssm_out fix + shmem-by-n_ctx | 1.35-1.86× batched-over-per-token; closes "4× regression" myth |
| #247 | Same shmem fix on 3 spec-decode prefill variants | Spec-decode path gets the same unlock |
| #248 | 96 KB dynamic shmem opt-in on attention kernels | Prefill at >11K tokens stops falling back to CPU |
| #249 | 17K-token f16/iso3 head-to-head | Documents flat scaling; iso3 still pending 32K+ |
| #250 | Full 4K→17K scaling curve | 42 ms/tok flat across 32× range |

### Final on-hardware numbers (Qwen3.6-35B-A3B-vindex-v10, RTX 4090, max_seq=20000)

| N tok | wall_s | per-tok | VRAM peak |
|---|---|---|---|
| 557 | 24.3s | 43.6 ms | — |
| 1110 | 45.4s | 40.9 ms | — |
| 4419 | 183.7s | 41.6 ms | — |
| 8831 | 369.7s | 41.9 ms | — |
| 13245 | 557.2s | 42.1 ms | — |
| 17658 | 752.8s | 42.6 ms | 21284 MiB (f16) / 21412 MiB (iso3) |

vs pre-Phase-4 README baseline (4419 tok at 296.1s / 67 ms/tok at
max_seq=8192): **current binary is 1.61× faster with 2.4× more KV
cache headroom**.

### The "4× regression" story

Initial Phase 4e bench showed prefill at max_seq=20000 was ~4×
slower than the README baseline at max_seq=8192. Root cause was an
RTX 4090 occupancy bug in both decode-attn and prefill-attn CUDA
kernels: both sized shared memory by `opts.max_seq` (slab capacity)
instead of `opts.pos + 1` (actual cached context). At max_seq=20000
that's 80 KB shmem per block → 1 block/SM occupancy on Ada. PR #246
tracks `n_ctx` for shmem; PR #248 adds the dynamic-shmem opt-in so
n_ctx > 11K is supported. Step B's matmul routing itself delivered
~0% wall-time improvement — the matmul wasn't the GPU bottleneck.

### Bench env switches added this session

- `LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1` — forces
  `qwen35_forward_prefill` into the per-token loop without enabling
  any diagnostic dumping. The clean A/B switch.
- `LARQL_QWEN35_NO_BACKEND=1` — when built with `--features cuda`,
  skips the unconditional CudaBackend attach in the chat handler.
  Lets CPU batched-matmul gates fire from a cuda-built binary.

## What's actually next

Three genuine fresh-session-scope arcs, in order of priority:

### A. **Tiled-scores rework for 32K+ in a single launch** — substantial CUDA work

The shmem-by-n_ctx + 96 KB opt-in combination caps n_ctx at ~24K
per launch (`96·1024 / 4 − scratch − head_dim`). 32K+ prefill in a
single launch hits the Ada 100 KB max. Unblocking it needs an
online-softmax / FlashAttention-style tiling: move scores from
shmem to a streaming pattern.

Two-pass design:
1. Pass 1: stream over K cache in tiles, computing per-tile
   `max(scores)` and `sum(exp(scores - max))`. Combine across tiles
   into final softmax statistics.
2. Pass 2: stream over K cache again, computing the
   exp(score - max) / sum * V output.

Substantial — probably 1000+ LoC across CUDA source + host plumbing
+ parity tests. Unblocks the iso3 32K+ VRAM-savings bench (the
original Phase 3 thesis).

### B. **CPU rayon-parallel attention scan** — next CPU bottleneck

The Phase 4e CPU-only bench fit curve was
`T(N) ≈ 0.130·N + 5.70e-5·N²` (s). At 2212 tokens the O(N²)
attention term is already ~half the wall time. A CPU rayon
attention scan that matches `fused_prefill_attention_seq`'s
behaviour would unlock more for the CPU-only path.

This is more contained than (A) — pure CPU code, no kernel work.
Probably 300-500 LoC. Lower priority than (A) since CPU-only isn't
the production deployment, but cleaner contained PR.

### C. **GPU MoE scatter-gather measurement** — moderate effort

PR #241 added CPU scatter-gather for the MoE prefill path, gated on
`backend.is_none()`. With matmul_with_backend routing Q4_K to GPU,
we could try scatter-gather on GPU too. The existing `qwen35_moe_ffn_batch`
(PR #218) batches 8 experts per call; scatter-gather would dispatch
~128 experts per layer in a different pattern. Empirical: which is
faster on RTX 4090? Run a bench to find out.

### Smaller follow-ups noted but not urgent

- **Phase 4-final dump-env fallback**: `qwen35_forward_prefill` falls
  back to per-token loop when `LARQL_QWEN35_DUMP_*` is set. Teach the
  batched-prefill helpers to emit dump format too — not blocking.
- **GPU paired Q4_K matmul** — DeltaNet's per-row form uses
  `qwen35_paired_q4k_matvec`. A paired matmul would batch those for
  GPU prefill.

## Critical context the fresh session needs

- **Project memory**: read
  `~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/MEMORY.md`
  — especially `project_larql_driving_goal.md` and
  `project_batched_prefill_arc.md` (just updated with the 5-PR arc).

- **Standing rules** (same as 2026-05-21):
  - **Don't self-merge unattended.** User authorises each merge via
    `merge and continue`. Admin-merge OK when CI's only failures are
    pre-existing main-branch issues (3 known: openspec on main,
    macOS Metal stale trait impls, Ubuntu cuda runner missing nvcc).
  - **No `--no-verify` / hook skipping** without explicit permission.
  - **OpenSpec workflow** still applies — `make traceability` after
    test line shifts.

- **Hardware**: Threadripper PRO 5965WX (24 cores, 48 threads,
  AVX2 no AVX-512), 440 GB RAM, RTX 4090. Models at
  `/tank/ai/<org>/...`. CUDA is enabled in the dev build.

- **Branch state**: clean main as of PR #250 merge. Untracked
  diagnostic test files from prior sessions still present
  (`tests/test_gemma3_*.rs`, `tests/test_q6k_roundtrip.rs`,
  `tests/profile_4b_decode.rs`, `tests/test_v_proj_writer_roundtrip.rs`)
  — leave them, they're harmless.

## Quick start prompt for the next session

> Read `RESUME_PROMPT.md`. Five PRs landed last session (#246-#250)
> completing the long-context prefill arc on Qwen3.6-35B-A3B. The
> headline result is **flat 42 ms/tok GPU prefill across a 32×
> context range** (557 → 17658 tokens) — the shmem occupancy cliff
> is gone and dynamic-shmem opt-in unlocks prefill up to ~24K tokens
> per launch. Pre-Phase-4 baseline (4419 tok at 296s / 67 ms/tok) is
> beaten by 1.61× absolute with 2.4× more KV cache headroom.
>
> **Three next steps in order of priority:**
>
> 1. **Tiled-scores rework for 32K+ in a single launch.** Two-pass
>    online-softmax tiling — move scores from shmem to streaming.
>    Unblocks iso3 32K+ VRAM-savings bench (Phase 3 thesis). ~1000
>    LoC CUDA + host. Dedicated session warranted.
>
> 2. **CPU rayon-parallel attention scan.** O(N²) attention is
>    ~half of CPU batched wall time at 2K tokens. Pure CPU code,
>    ~300-500 LoC. Lower production priority than (1) but more
>    contained.
>
> 3. **GPU MoE scatter-gather measurement.** Empirical compare
>    against PR #218's `qwen35_moe_ffn_batch`. Bench-driven.
>
> Per-PR auth via `merge and continue`; admin-merge only when CI's
> only failures are pre-existing main-branch issues.
