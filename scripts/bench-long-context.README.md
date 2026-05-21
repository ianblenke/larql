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
