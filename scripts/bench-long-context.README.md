# Long-context KV-cache bench (Phase 3)

Runs the qwen35 chat completion path against increasingly-long prompts to
measure the RotorQuant Iso3-compressed device KV cache vs the default
f16 device KV cache.

## Usage

Terminal 1 — start the server in the desired KV-format:

```bash
# f16 baseline
LARQL_QWEN35_GPU=1 \
  LARQL_QWEN35_KV_MAX_SEQ=8192 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 --port 8181

# OR Iso3 compressed
LARQL_QWEN35_GPU=1 \
  LARQL_QWEN35_KV_FORMAT=iso3 \
  LARQL_QWEN35_KV_MAX_SEQ=8192 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 --port 8181
```

Terminal 2 — run the bench:

```bash
python3 scripts/bench-long-context.py <mode_label> > bench-<mode>.csv
```

`mode_label` is just an annotation for the CSV (e.g. `f16` / `iso3`).
The server's behaviour is controlled by env vars at startup.

## Output columns

```
mode,target_prompt_tokens,actual_prompt_tokens,decode_tokens,wall_s,
decode_per_tok_avg_s,tok_per_s_overall,vram_pre_mib,vram_peak_mib,content
```

- `vram_peak_mib`: peak GPU 0 memory reading during the request, sampled
  every 100 ms via `nvidia-smi`. Includes the model weight caches, the
  KV slabs, and any per-request scratches.
- `wall_s`: end-to-end request time (prefill + decode + network).
- `tok_per_s_overall`: `decode_tokens / wall_s` — dominated by prefill
  at long contexts. The decode-only rate is the steady-state number
  reported in the headline benches; this CSV measures the long-context
  pressure end-to-end.

## Results captured 2026-05-21 (RTX 4090, Qwen3.6-35B-A3B-vindex-v10)

Captured at `LARQL_QWEN35_KV_MAX_SEQ=8192` for both modes.

| Prompt tok | f16 wall_s | iso3 wall_s | f16 VRAM | iso3 VRAM | Δ VRAM |
|---|---|---|---|---|---|
| 145 | 9.6s | 9.7s | 15924 MiB | 15860 MiB | -64 |
| 2212 | 130.7s | 133.6s | 20404 MiB | 20404 MiB | 0 |
| 4419 | 296.1s | 301.2s | 20788 MiB | 20852 MiB | +64 |

Output coherent in both modes ("It appears you have pasted a large
block..."). Throughput delta ≤ 2% (dequant overhead).

## Why the VRAM savings aren't yet visible at 4K context

Theoretical at max_seq=8192, kv_dim=512, 16 full-attn layers:

  f16 KV slabs:   8192 × 512 × 2 bytes × 2(K+V) × 16 layers = 128 MiB
  iso3 codes:     8192 × 200 bytes × 2 × 16                 =  51 MiB
  iso3 scratches: 8192 × 512 × (4+2+2) bytes                =  32 MiB
                                                              ----
  iso3 total:                                                  83 MiB
  Savings:                                                     45 MiB

45 MiB is buried in the ~20 GiB weight-cache noise (lazy-loaded expert
weights vary by 100+ MiB depending on which experts got dispatched
during prefill). To make the win visible the bench needs `max_seq ≥
32K`, where:

| max_seq | f16 | iso3 | Δ |
|---|---|---|---|
| 32K  | 1024 MiB | 330 MiB | 694 MiB |
| 64K  | 2048 MiB | 530 MiB | 1.5 GiB |
| 128K | 4096 MiB | 930 MiB | 3.2 GiB |

That's the "models > VRAM at long context" operating point.

## Why 32K+ bench takes too long for a session

Each prefill token costs ~70 ms on Qwen3.6-35B-A3B. A 32K-token prompt
prefills in ~37 minutes; 128K → ~2.5 hours. The bench script's nominal
`PROMPT_TARGETS` are kept ≤ 4096 so a full sweep fits in ~10 minutes.

Production bench at 32K+ needs one of:
- **Batched prefill** — a custom forward that processes N prompt
  tokens per kernel call instead of N sequential calls. Out of scope
  for this PR; would land in its own arc.
- **Faked-fill cache** — directly populate `slab.codes_*` / `slab.k/v`
  to a target `cached_seq_len` without going through prefill, then
  measure decode VRAM. Useful for VRAM verification without the
  prefill cost. Could be added as a debug-only bench helper.
- **Patience** — let it run overnight. Single 128K prefill at 70 ms/tok
  finishes in ~150 minutes.

For now: the bench script + 4K results document infrastructure
correctness; the architectural projection above documents the design
target. Phase 4 will add either of the above acceleration paths to
make the value-prop bench session-scale.

## Phase 4e on-hardware results captured 2026-05-21 (RTX 4090 + Threadripper PRO 5965WX, Qwen3.6-35B-A3B-vindex-v10)

End-to-end wall-time A/B of the Phase 4 batched-prefill arc
(PRs #230-#245) vs the per-token fallback. `LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1`
forces `qwen35_forward_prefill` into its per-token loop without
enabling any diagnostic dumping. `LARQL_QWEN35_NO_BACKEND=1` (added
this session) skips the unconditional CudaBackend attach in the chat
handler so the batched-matmul path's `backend.is_none()` gate fires.

### GPU mode (LARQL_QWEN35_GPU=1, hybrid GPU attention + projections)

| N prompt tok | batched wall_s | per-token wall_s | speedup |
|---|---|---|---|
| 557 | 37.7 | n/a | — |
| 1110 | 104.8 | n/a | — |

**No improvement vs the pre-Phase-4 GPU per-token bench above** (67-94
ms/tok matches the 67 ms/tok at 4419 tok captured earlier). The
batched-matmul path (PRs #239-#244) is gated `backend.is_none()` in
both `qwen35_attention_block_prefill` and `deltanet_block_prefill`,
so a cuda-attached backend falls back to per-row matvec dispatch.
The batched *attention* kernel (PR #231) does run but its wall-time
contribution at these sizes is dwarfed by kernel-launch + matvec
dispatch overhead. **GPU users see no Phase 4 prefill speedup until
Step B (GPU `quant_matmul` kernel) lands.**

### CPU-only mode (LARQL_QWEN35_NO_BACKEND=1, where Phase 4 actually fires)

| N prompt tok | batched wall_s | per-token wall_s | speedup |
|---|---|---|---|
| 281  | 42.5  | 58.1  | 1.37× |
| 557  | 90.0  | 463.6 | — *(per-token hit lazy expert load mid-bench)* |
| 1110 | 208.8 | 417.2 | **2.00×** |
| 2212 | 566.2 | 681.9 | 1.20× |

The 1110-token row is the most reliable A/B (post-cache-warm in both
modes). **Phase 4 batched matmul delivers a 2× wall-time speedup at
1K context on CPU-only mode.** Speedup decreases with N because both
modes share the same attention O(N²) scan and the same per-position
MoE-routing + DeltaNet-conv1d sequential cost — projection-bandwidth
amortisation only addresses the O(N) "everything else" term.

Curve fit on the batched data: `T(N) ≈ 0.130·N + 5.70e-5·N²` (s).
Extrapolated to 32K: ~18 hours. **CPU-only is not the deployment
mode** — this measurement isolates the landed Phase 4 work, but the
production hybrid path (GPU attn + CPU FFN) needs Step B before
prefill becomes session-scale.

### Why the bandwidth math overpredicted

RESUME_PROMPT's headline was "~40 TB → ~135 GB across the full flow
(~300×)" of projection-bandwidth reduction at 32K. That number is
real for the *weight-read* traffic, but the wall-time speedup is
capped by whatever bottleneck remains after bandwidth is fixed:

- **Attention O(N²)** scan — same code in both modes, dominates at large N
- **MoE per-token top-K routing** — inherently sequential, scales O(N)
- **DeltaNet conv1d state update** — inherently sequential, scales O(N)
- **CPU memory bandwidth saturation** — projections share the same
  DDR4 channels as activations / cache / KV

Phase 4 amortised projection bandwidth from `40 TB` of weight reads
down to `135 GB`, but the *remaining* work (attention, MoE, DeltaNet
recurrence) was the throttle the whole time. 2× wall-time at 1K is
the actual delivered value of PRs #239-#244 on CPU.

### Implications for the roadmap

- **Step B (GPU `quant_matmul`) becomes more urgent** — without it,
  GPU users see zero Phase 4 prefill speedup, even though PRs #230,
  #231, #235, #236 (the GPU-side parts) all landed.
- **Attention-scan optimisation is the next CPU bottleneck** — at
  2212 tokens the attention O(N²) term is already ~half of
  batched wall-time. A CPU rayon-parallel attention scan that
  matches `fused_prefill_attention_seq` would unlock more.
- **The 32K+ value-prop bench is still preseed-only** — until Step
  B and a CPU attention scan land, real 32K prefill is hours not
  minutes. `LARQL_QWEN35_KV_PRESEED` from bench-preseed.md remains
  the right tool for VRAM bench.

### Bench env switches added this session

- `LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1` — forces
  `qwen35_forward_prefill` into its per-token loop. Cleaner A/B
  switch than piggybacking on a diagnostic dump var.
- `LARQL_QWEN35_NO_BACKEND=1` — when built with `--features cuda`,
  skips the unconditional CudaBackend attach in
  `chat.rs::handle_chat_completions`. Lets the CPU batched-matmul
  gates (`backend.is_none()`) fire from a cuda-built binary.

### Reproducing

```bash
# Build (cuda enabled — for the GPU run)
cargo build --release --bin larql-server --features cuda

# CPU batched
LARQL_QWEN35_NO_BACKEND=1 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 \
  --port 8181

# CPU per-token (separate run, restart server)
LARQL_QWEN35_NO_BACKEND=1 LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 \
  --port 8181

# Drive the bench
LARQL_BENCH_TARGETS=256,512,1024,2048 LARQL_BENCH_DECODE=4 \
  LARQL_BENCH_HTTP_TIMEOUT=3600 \
  python3 scripts/bench-long-context.py <mode_label>
```
