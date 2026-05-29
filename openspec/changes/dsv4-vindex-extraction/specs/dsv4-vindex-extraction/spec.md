## ADDED Requirements

### Requirement: DSv4 architecture is extractable to vindex

The vindex extraction pipeline SHALL recognize the `deepseek_v4`
architecture as extractable (distinct from the classic-MLA `deepseek_v2`/
`deepseek_v3`, which remain rejected) and SHALL produce a vindex from a
DeepSeek-V4-Flash GGUF.

#### Scenario: V4 accepted, V2/V3 still rejected

- **WHEN** the extraction capabilities gate evaluates an architecture
- **THEN** `deepseek_v4` SHALL be accepted for extraction, while
  `deepseek_v2`/`deepseek_v3` SHALL still be rejected as unsupported MLA

### Requirement: DSv4 attention + structural weights are represented

The vindex format SHALL store DSv4's non-standard weights that the
generic Q/K/V/O writer cannot represent: the low-rank Q (`attn_q_a/q_b`),
the latent KV (`attn_kv_latent`), the grouped low-rank output projection
(`attn_output_a/b`), the per-head attention sinks and post-projection
RMSNorms, and the per-layer attention variant (`compress_ratio`). The
stored quantized bytes SHALL match the GGUF (no dequantize/recompress).

#### Scenario: Attention weights round-trip from the vindex

- **WHEN** DSv4 attention weights are extracted to a vindex and then read
  back
- **THEN** the reconstructed low-rank Q, latent KV, and grouped O weights
  SHALL equal the GGUF-loaded weights (same shapes + quantized bytes),
  and each layer's `compress_ratio` SHALL be recorded

<!-- test: larql_inference::attention::dsv4_vindex_attn::tests::attn_weights_round_trip_losslessly -->
<!-- test: larql_inference::attention::dsv4_vindex_attn::tests::no_sinks_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_attn::tests::round_tripped_raw_builds_quant_tensor -->
<!-- test: larql_inference::attention::dsv4_vindex_attn::tests::malformed_blobs_are_typed_errors -->
<!-- test: larql_inference::attention::dsv4_vindex_attn::tests::real_gguf_attn_round_trips_to_storage -->

### Requirement: HCA, indexer, and mHC weights are extracted

The vindex SHALL store, per layer where present, the HCA compressor
(`attn_compress_*`), the lightning indexer (`indexer.*`, on
`compress_ratio == 4` layers), and the mHC residual bookends
(`hc_{attn,ffn,head}_*`). Indexer top-k masks SHALL NOT be precomputed —
only the indexer weights are stored (selection stays runtime-dynamic).

#### Scenario: Variant-gated structural weights round-trip

- **WHEN** a Compress layer and an Indexer layer are extracted
- **THEN** the compressor weights SHALL be present for both, the indexer
  weights SHALL be present only for the Indexer layer, and the mHC
  bookends SHALL be present for every layer — all round-tripping to the
  GGUF-loaded values

<!-- test: larql_inference::attention::dsv4_vindex_hca::tests::compressor_only_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_hca::tests::indexer_layer_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_hca::tests::nocompress_layer_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_hca::tests::malformed_blobs_are_typed_errors -->
<!-- test: larql_inference::attention::dsv4_vindex_hca::tests::real_gguf_hca_round_trips_to_storage -->
<!-- test: larql_inference::attention::dsv4_vindex_mhc::tests::layer_mhc_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_mhc::tests::head_mhc_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_mhc::tests::empty_mhc_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_mhc::tests::malformed_blobs_are_typed_errors -->
<!-- test: larql_inference::attention::dsv4_vindex_mhc::tests::real_gguf_mhc_round_trips_to_storage -->

### Requirement: Routed and shared MoE reuse generic extraction

The extraction pipeline SHALL extract DSv4's routed experts, shared
expert, router gate, the first-3-layer hash routing table, and the
routing bias through the existing generic MoE path, surviving the
round-trip intact.

#### Scenario: MoE weights and hash routing round-trip

- **WHEN** a hash-routed MoE layer (first 3) and a gated MoE layer are
  extracted
- **THEN** the expert + shared-expert weights, the gate, the hash
  token→expert table, and the routing bias SHALL all reload equal to the
  GGUF-loaded values

<!-- test: larql_inference::attention::dsv4_vindex_moe::tests::gated_layer_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_moe::tests::hash_layer_round_trips -->
<!-- test: larql_inference::attention::dsv4_vindex_moe::tests::malformed_blobs_are_typed_errors -->
<!-- test: larql_inference::attention::dsv4_vindex_moe::tests::real_gguf_moe_round_trips_to_storage -->

### Requirement: Full-model extraction round-trips

A complete DSv4-Flash vindex produced from the GGUF SHALL reconstruct,
layer by layer, per-layer storage equal (shapes + quantized bytes) to a
direct GGUF resident load of the same model.

#### Scenario: Whole-model round-trip

- **WHEN** every layer is loaded from a produced DSv4-Flash vindex
- **THEN** the reconstructed per-layer storage SHALL equal the GGUF
  resident load for all layers and the global head/embeddings
