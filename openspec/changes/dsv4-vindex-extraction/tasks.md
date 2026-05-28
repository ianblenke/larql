## 1. V0 — Groundwork & decision gate

- [x] 1.1 GGUF tensor inventory (from the inference audit / gguf-dump): attn `q_a`[1024,4096]·`q_b`[1024,32768]·`kv_latent`[4096,512]·`output_a`[4096,8192]·`output_b`[8192,4096] (all Q4_K), HCA `attn_compress_kv/gate`[4096,512] Q4_K + `compress_ape`/`norm` F32, indexer `attn_q_b`[1024,8192] Q4_K + `compress_*` Q4_K + `proj`[4096,64] Q4_K, mHC `hc_*_fn`[16384,24] F32, MoE exps Q4_K + `ffn_down_shexp` Q6_K, norms/sinks/`gate_inp` F32, `gate_tid2eid` I32 (first 3 layers).
- [x] 1.2 FP8-KV decision: FP8 is a **runtime KV-cache** quant (`dsv4_fp8_kv`), not a stored weight — extraction has no FP8 tensor to store; it stores the Q4_K/Q6_K/F32 weights as-is and FP8-KV stays a runtime concern of the (existing) inference path. So no FP8 format support is needed in the vindex.
- [x] 1.3 Capabilities: added `ModelArchitecture::uses_dsv4_attention()` (default false; `true` for `DeepSeekV4Arch`) and a DSv4 branch in `capabilities.rs` that gates DSv4 out of the standard Q/K/V/O writers with a **distinct** feature message (DSv4 was previously not MLA-flagged, so it silently fell into the broken standard writer). 5 new tests; V2/V3 still MLA-rejected, llama still passes. The gate **flips to accept** when the DSv4 writer lands (V1+) — until then this is a clean reject, not the spec's end-state "accept".

## 2. V1 — Config + attention storage (the decisive unknown)

- [ ] 2.1 `DsV4VindexMeta` on `VindexModelConfig` (n_hc, indexer dims, fp8_kv, yarn) + `compress_ratio: u8` on `VindexLayerInfo`; serialize into `index.json`. Round-trip test.
- [ ] 2.2 `dsv4_attn.bin` + manifest: write low-rank Q (`attn_q_a/q_b`), latent KV (`attn_kv_latent`), grouped O (`attn_output_a/b`) as Q4_K/Q6_K passthrough + inline f32 norms/sinks.
- [ ] 2.3 Thin **test reader** for `dsv4_attn.bin` → reconstruct the attention half of `DsV4LayerWeightStorage`; assert weights equal the GGUF-loaded ones (byte/shape round-trip). **This is the V1 gate + the re-evaluation point.**

## 3. V2 — HCA compressor + indexer

- [ ] 3.1 `dsv4_hca.bin` + manifest: per-layer compressor (`attn_compress_kv/gate/ape/norm`) gated by `compress_ratio>0`; indexer (`indexer.compress_*`, `indexer.attn_q_b`, `indexer.proj`) gated by `compress_ratio==4`.
- [ ] 3.2 Test-reader round-trip for the HCA/indexer tensors (variant-aware: NoCompress/Compress/Indexer layers).

## 4. V3 — mHC + MoE + head

- [ ] 4.1 `dsv4_mhc.bin`: `hc_{attn,ffn,head}_{base,fn,scale}` bookends, all layers + head.
- [ ] 4.2 Routed MoE experts + shared expert + router (`ffn_gate_inp`, hash `ffn_gate_tid2eid` for the first 3 layers, `exp_probs_b`) via the existing generic MoE extraction path; verify the hash table + bias survive.
- [ ] 4.3 lm_head + token-embed + final norm via existing generic writers.

## 5. V4 — End-to-end extraction + round-trip

- [ ] 5.1 Wire all writers into `build_vindex` behind the `deepseek_v4` branch; produce a full DSv4-Flash vindex from the real GGUF (`#[ignore]`, ~172 GB in / ~161 GB out).
- [ ] 5.2 Full round-trip: load every layer from the produced vindex via the test reader → reconstruct `DsV4LayerWeightStorage[]` equal (byte/shape) to the GGUF resident load.
- [ ] 5.3 `make ci` green; traceability; openspec validate. Hand-off note for the follow-up **dsv4-vindex serving reader** change (loads this vindex into the resident forward).
