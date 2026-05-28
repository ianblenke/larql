## 1. V0 — Groundwork & decision gate

- [ ] 1.1 Confirm DSv4-Flash GGUF tensor inventory vs `DsV4TensorKind` (already audited for inference) — the extraction source set, with shapes + quant types per kind.
- [ ] 1.2 Decide FP8-KV handling (store FP8 bytes vs dequant to f16) — record in design D3/D5.
- [ ] 1.3 Capabilities: add a `deepseek_v4` branch to `capabilities.rs` (allow extraction; distinct from V2/V3 MLA reject) + a unit test that V4 is accepted and V2/V3 still rejected.

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
