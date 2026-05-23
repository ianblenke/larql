# larql resume — session of 2026-05-14 (CI/test-infra + bench measurement arc)

> Sibling to the GPU-arc `RESUME_PROMPT.md`. The work captured here is
> orthogonal: it cleaned up CI debt, unblocked the bench measurement
> that #127 had been blocking, and surfaced one unverified correctness
> hypothesis (vocab/forward-pass) worth investigating before any
> further bench claims.

## What landed today

Eleven PRs (#122 → #131), all merged to `main` except #131 (which
was opened awaiting merge — check `gh pr view 131` before assuming).

| PR | Title | What it unblocks |
|---:|---|---|
| #122 | `feat(gguf-moe)`: unblock 35B-A3B vindex extraction | Qwen 3.6-35B-A3B GGUF → vindex (~2 min wall, was hours) |
| #123 | `fix(server)`: chat completions deadlock | `/v1/chat/completions` returns; was hanging forever |
| #124 | `docs(bench)`: close blockers proposal with measured numbers | Bench gap diagnosed: 153× algorithmic |
| #125 | `feat(convert)`: `--quant q4k` flag for `gguf-to-vindex` | One-step GGUF → fast-decode vindex (dense models) |
| #126 | `test(server)`: fix infra drift + bounded-time regression for #123 | Server test suite goes from broken → 287 tests green |
| #127 | `chore(lint)`: drop unused re-export + `is_none_or` fix | Removes the last build warnings on `vindex` / `models` |
| #128 | `test(compute,vindex)`: fix trait + struct-field drift | 145 previously-broken tests unblocked |
| #129 | `test(workspace)`: close last build-target drift | `cargo build --workspace --all-targets` clean |
| #130 | `ci`: tighten check/clippy to `--all-targets` | Catches the *next* drift class at PR time |
| #131 | `ci(cli)`: re-enable clippy on larql-cli | Last "intentionally skipped" gate closed |

Posture after today:
- `cargo build --release --workspace --all-targets` — clean.
- Every crate enforces `clippy -- -D warnings` on its full `--all-targets`.
- 145+ previously-uncompilable tests now run.
- `/v1/chat/completions` no longer deadlocks (regression test pins it).
- GGUF → vindex extraction works end-to-end for MoE.

## The bench number that came out of this

The chat-completion hang (filed as task #127) had been blocking the
end-to-end measurement of larql's production decode path. With #123
fixed, the number is now measurable:

| Config | Decode (t/s) | Notes |
|---|---:|---|
| llama.cpp `-ngl 999` (Gemma 3 4B Q4_K_M) | **238** | RTX 4090, 6.8 GB VRAM |
| llama.cpp `-ngl 0` (CPU only) | **16.2** | ~2.3 GB RSS |
| **larql `/v1/chat/completions`** (Gemma 3 4B Q4_K) | **0.106** | 32 tokens in 301 s; ~11 GB RSS |

The ~153× gap to llama.cpp CPU is **algorithmic, not micro-kernel**.
`crates/larql-inference/src/layer_graph/generate/cpu.rs::generate_via_cpu_q4k`
loops `predict_q4k` per token — each call re-runs the full forward
pass over the entire prompt-so-far (no KV cache). The kernel-level
AVX2 wins from the cpu-kquant arc (PRs #102–#119) are real and
measured; closing the algorithmic gap is what compounds them into a
user-visible speedup.

See `openspec/changes/bench-vs-llama-cpp-end-to-end-blockers/proposal.md`
for the full diagnosis and the "HUNG → measurable" transition.

## ⚠ Unverified correctness gap — vocab mismatch hypothesis

**The 0.106 tok/s number measures throughput, not correctness.** Output
samples from the run:

- Warmup ("hi", `max_tokens=1`) → `"ům"`
- 32-token bench → `" Wndې...DeutschesYaml RemLaravel铎 XNUMX∂</ிறு ดู DBHelper..."`
- `larql run` direct (no server, earlier in the session) → `"tragedy"`

This is **incoherent**, not just slow. A correct-but-slow forward pass
should produce coherent text at low tok/s, not multilingual gibberish.
The output pattern (English-looking tokens like "Wnd", "Tale",
"Linux" intermixed with Devanagari/Arabic/CJK characters) is
consistent with a **vocab mismatch** — the model emitting plausibly-
ranked token *ids* that decode to the wrong *strings*.

The hypothesis was identified in this session but **not investigated**.
The decode-speed work I documented above is independent of whether the
output is correct — but any further bench claim, perf chart, or
"close the gap" PR should validate correctness first.

### Diagnostic checklist

1. **Tokenizer audit.** Compare `output/gemma-3-4b-it-vindex/tokenizer.json`
   against the canonical Gemma 3 4B tokenizer:
   - vocab size matches (262208 expected)
   - special-token IDs match (`<bos>`, `<eos>`, `<start_of_turn>`,
     `<end_of_turn>`)
   - first ~100 vocab entries decode identically
2. **Token-ID parity vs llama.cpp.** Run the same prompt through
   `llama.cpp` and `larql` and compare the emitted token *IDs* (not
   the decoded strings) step-by-step:
   - IDs identical, strings differ → tokenizer mismatch (vocab issue)
   - IDs differ → forward-pass correctness bug (algorithmic, kernel,
     or weight load)
3. **Smoke test on a known-good vindex.** Try
   `output/gemma-3-1b-it-vindex` (smaller, same family) — does it
   also produce gibberish? If yes, the bug is family-wide; if no,
   it's specific to the 4b extraction.
4. **Compare against `extract` output.** The Gemma 3 4B vindex was
   extracted at some earlier date with some specific `larql extract`
   invocation. If we can re-extract from scratch with the current
   tooling and the output changes, the bug is in extraction; if not,
   it's in inference.

### Why this matters

The whole point of the bench-vs-llama.cpp arc is a head-to-head perf
claim. A number that's both 153× slow AND outputting gibberish is two
unrelated problems wearing one face. Fix correctness first, then the
perf-gap diagnosis (`O(N²)` no-KV-cache) stands on solid ground.

## Open arcs from here

In order of impact:

1. **Verify correctness on Gemma 3 4B** (the diagnostic checklist
   above). Cheap and gating — should be the next 30 min of work
   before any further bench effort.

2. **CPU Q4K KV cache** — the 153× perf win. Wire a KV cache
   through `crates/larql-inference/src/vindex/q4k_forward/hidden.rs`
   and its attention block. Metal already has it
   (`DecodeBackend::decode_token` + `populate_kv_layer` +
   `truncate_kv_cache`); CPU just doesn't use it. Probably 2-3
   focused PRs. Expected: ~10× speedup, closer to llama.cpp CPU.

3. **Hybrid SSM Q4_K writer** — extends PR #125. The current
   `--quant q4k` flag rejects hybrid SSM archs (Qwen 3.6 family)
   because the Q4_K attn writer at
   `crates/larql-vindex/src/format/weights/write_q4k/attn.rs` only
   iterates Q/K/V/O. DeltaNet layers need their own `ssm_*` tensor
   set. Probably 2-3 PRs. Unlocks `larql convert gguf-to-vindex
   Qwen3.6-35B-A3B-... --quant q4k --level all` and the MoE
   head-to-head bench.

4. **walk_path_audit resurrection** — gated behind `#[cfg(any())]`
   in PR #129. Needs `MaskedGateIndex`'s `impl GateIndex` split
   across `GateLookup`, `PatchOverrides`, `FfnRowAccess` (and its
   sub-supertraits). 1-2 hour focused session of method
   classification. Restores the per-path WalkFfn equivalence
   harness.

## Critical environment notes (read these)

- **No CUDA on this dev box** — decode defaults to CPU. The 0.106
  tok/s bench is CPU-only.
- **Linux x86_64, glibc.** The chat-completion `pick_template`
  deadlock fixed in #123 is glibc-specific (non-reentrant
  `pthread_rwlock_t` since 2.31). The fix avoids the pattern
  entirely — should be portable, but it's worth knowing the
  failure mode is glibc-only if you ever see it again on a
  different platform.
- **`larql serve` is a wrapper.** It spawns `larql-server` as a
  subprocess and the wrapper itself blocks in `wait4`. Don't gdb
  the wrapper PID — attach to the `larql-server` child.
  `--log-level` is passed by the wrapper and **overrides
  `RUST_LOG`**; use `--log-level trace` not env var when going
  through `larql serve`.
- **`ptrace_scope=1` on this host** — gdb attach to non-child
  processes blocked without sudo. Use instrumented `eprintln!` for
  live diagnosis. (This was how #123's deadlock was localized.)
- **`/health` returns 404** on `larql-server` — there's no health
  endpoint at that path. Use `/v1/models` for liveness.

## Standing rules

- **Don't self-merge PRs unattended.** The user authorizes each
  merge individually (`merge and continue`). Standing auth is
  per-PR, not session-wide.
  See `~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/feedback_unattended_merging.md`.
- **OpenSpec workflow.** Every code change references a capability
  under `openspec/specs/<name>/spec.md` (or for in-flight work,
  `openspec/changes/<id>/specs/...`). Scenarios link to tests via
  `<!-- test: <fqn> -->` annotations. Run `make ci` before pushing.
  See `CLAUDE.md`.
- **Q4K vindex format is the production fast-decode path.** f16
  vindexes work but go through the slow CPU dequant loop. The
  `interleaved_q4k.bin` / `attn_weights_q4k.bin` / `lm_head_q4.bin`
  triad is what production decode reads. Today's `--quant q4k`
  flag (#125) is the GGUF entry point.

## Quick start prompt for a fresh session

If the next session picks up where today ended, the highest-leverage
opening is the correctness check — diagnostic checklist above,
specifically step 2 (token-ID parity vs llama.cpp). A reasonable
opening prompt:

> Read RESUME_PROMPT_SESSION_2026-05-14.md. The 0.106 tok/s bench
> number locked in today (Gemma 3 4B Q4_K via `/v1/chat/completions`)
> measures throughput, but output samples were incoherent — could be
> a vocab mismatch or a forward-pass correctness bug. Run the
> diagnostic checklist in that file's "Unverified correctness gap"
> section: tokenizer audit → token-ID parity vs llama.cpp on the
> same prompt → smoke test the 1B / 270M variants. Report findings
> in under 200 words; don't fix anything yet — diagnose first.

Or for the CPU KV cache arc:

> Read RESUME_PROMPT_SESSION_2026-05-14.md. Start the CPU Q4K KV
> cache arc (option 2 in "Open arcs"). First PR: thread an optional
> `KvCache` parameter through `predict_q4k_hidden` and the
> attention block in
> `crates/larql-inference/src/vindex/q4k_forward/hidden.rs` without
> changing behavior — just plumbing. Verify it builds clean across
> the workspace and existing tests still pass. Open a PR when the
> plumbing compiles + tests pass.

Or for hybrid SSM Q4_K:

> Read RESUME_PROMPT_SESSION_2026-05-14.md option 3. Start the
> hybrid SSM Q4_K writer arc. First step: add an
> `arch.is_full_attention_layer(layer)` helper to
> `crates/larql-models/src/config.rs` (derived from
> `full_attention_interval`), then refactor
> `crates/larql-vindex/src/format/weights/write_q4k/attn.rs::write_attn_weights_q4k`
> to dispatch per layer. Test against
> `output/gguf-cache/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

## Files touched (for quick git blame / context)

| Path | PR | Why |
|---|---:|---|
| `crates/larql-vindex/src/extract/build.rs` | #122 | MoE down_meta early-out |
| `crates/larql-vindex/src/format/weights/write_f32.rs` | #122 | WeightSource quant_tensors fallback |
| `crates/larql-server/src/routes/openai/chat.rs` | #123 | pick_template above lock_weights_for_gen |
| `openspec/changes/bench-vs-llama-cpp-end-to-end-blockers/proposal.md` | #124 | Resolution section + bench numbers |
| `openspec/changes/bench-vs-llama-cpp-end-to-end-blockers/tasks.md` | #124 | Gap 1/Gap 2 closed |
| `crates/larql-cli/src/commands/extraction/convert_cmd.rs` | #125 | `--quant q4k` flag plumbing |
| `crates/larql-server/tests/common/mod.rs` | #126 | model_with_loaded_weights helper |
| `crates/larql-server/tests/test_http_embed.rs` | #126 | bounded-time regression tests |
| `crates/larql-vindex/src/format/weights/mod.rs` | #127 | drop unused re-export |
| `crates/larql-models/src/quant/lazy.rs` | #127 | is_none_or fix |
| `crates/larql-compute/tests/test_backend_matmul_quant.rs` | #128 | DecodeBackend trait stub realigned |
| `crates/larql-compute/examples/demo_architecture.rs` | #128 | full_pipeline_q4 arity fix |
| `crates/larql-vindex/tests/{test_vindex,compute_storage_regressions,persistence_regressions}.rs` | #128 | VindexModelConfig / ModelWeights field drift |
| `crates/larql-vindex/examples/demo_features.rs` | #128 | ModelWeights field drift |
| `crates/larql-inference/examples/{debug_layers,debug_gpu_step,debug_generate,walk_path_audit}.rs` | #129 | DecodeBackend arity fix + walk_path_audit stub |
| `crates/larql-server/benches/attention_service.rs` | #129 | LoadedModel/AppState field drift |
| `crates/larql-lql/src/executor/tests.rs` | #129 | ModelWeights + VindexModelConfig drift |
| `.github/workflows/larql-{server,cli,inference}.yml` | #130 | --all-targets tightening |
| `crates/larql-inference/Cargo.toml` | #130 | mech_interp_demo required-features |
| `.github/workflows/larql-cli.yml` | #131 | re-enable clippy gate |
