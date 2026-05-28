## ADDED Requirements

### Requirement: Compressed-cache serialization

DSv4 SHALL provide a versioned binary wire format that serializes a
layer's **compressed** CSA/HCA KV cache (and, on Indexer layers, the
indexer's compressed KV), together with the `compress_ratio` and the
compressor `overlap_state`, and deserializes it back losslessly. The
uncompressed sliding-window (`raw`) cache and the `pending_cur` tail
SHALL NOT be serialized (Zero-SWA).

#### Scenario: Compressed cache round-trips losslessly

- **WHEN** a layer's `DsV4LayerHcaCache` is serialized and then
  deserialized
- **THEN** the restored compressed entries SHALL equal the originals
  bit-exactly, with the same `compress_ratio`, dims, and indexer-present
  flag, and the `raw` cache SHALL be empty
<!-- test: larql_inference::attention::dsv4_kv_persist::hca_compressed_round_trips_losslessly -->
<!-- test: larql_inference::attention::dsv4_kv_persist::hca_with_indexer_and_overlap_round_trips -->

#### Scenario: No-compress layer serializes as empty compressed cache

- **WHEN** a NoCompress (pure-SWA) layer's cache is serialized
- **THEN** the blob SHALL encode an empty compressed cache and
  deserialize to one, without error
<!-- test: larql_inference::attention::dsv4_kv_persist::no_compress_layer_round_trips_as_empty -->

#### Scenario: Unknown format version is a typed error

- **WHEN** a blob with a bad magic or unknown version is deserialized
- **THEN** deserialization SHALL return a typed error, not panic
<!-- test: larql_inference::attention::dsv4_kv_persist::unsupported_version_is_typed_error -->
<!-- test: larql_inference::attention::dsv4_kv_persist::bad_magic_is_typed_error -->
<!-- test: larql_inference::attention::dsv4_kv_persist::truncated_blob_is_typed_error -->

### Requirement: Prefix-keyed on-disk store

DSv4 SHALL provide a content-addressed on-disk store mapping a hash of
the prompt token-id prefix — taken at `lcm(m, m')`-token block
boundaries and salted by a model identifier — to the per-layer
serialized compressed-KV blobs. Lookups SHALL return the longest cached
block-prefix of a given token sequence. Writes SHALL be atomic and the
store SHALL enforce a size cap.

#### Scenario: Put then longest-prefix get returns the same caches

- **WHEN** the per-layer compressed caches for a block-aligned prefix
  are written, then a superset token sequence is looked up
- **THEN** the store SHALL return that prefix's hit length and the same
  compressed caches it stored

#### Scenario: Miss returns no hit

- **WHEN** a token sequence shares no cached block-prefix
- **THEN** the lookup SHALL return no hit, and prefill SHALL proceed cold

#### Scenario: Size cap evicts least-recently-used entries

- **WHEN** writes would exceed the configured size cap
- **THEN** the store SHALL evict least-recently-used prefixes to stay
  under the cap, and surviving entries SHALL still load correctly

### Requirement: Zero-SWA prefill reuse is transparent

On a prefix hit, DSv4 SHALL reconstruct each layer's cache by loading the
compressed entries and recomputing only the last `n_win·L` tokens of the
prefix to restore the sliding-window tail and compressor state, then
prefill the uncached suffix. The reconstructed state SHALL produce the
same model output as a cold full prefill of the identical prompt, within
the documented numerical tolerance. The feature SHALL be opt-in; with no
cache supplied the cold prefill path SHALL be unchanged.

#### Scenario: Cache hit is transparent to output

- **WHEN** the same prompt is prefilled cold and via a prefix-cache hit
- **THEN** the per-position logits SHALL agree within the documented
  tolerance and the greedy next token SHALL be identical, including for
  a prompt whose suffix is not block-aligned

#### Scenario: Reuse recomputes only the SWA tail

- **WHEN** a prefix hit of length `H` (a multiple of the block size) is
  reused
- **THEN** the number of recomputed tokens SHALL be `min(H, n_win·L)` —
  bounded independent of `H` — not the full `H`

#### Scenario: Disabled cache leaves cold prefill unchanged

- **WHEN** no prefix cache is supplied
- **THEN** the prefill path SHALL be byte-for-byte identical to the
  pre-feature cold prefill
