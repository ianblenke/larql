## 1. P0 — Groundwork

- [ ] 1.1 Confirm the exact reconstructable state of `DsV4LayerHcaCache` post-prefill (what `raw`/`pending_cur`/`overlap_state` hold at a 128-block boundary) by instrumenting the existing cached prefill on a real-GGUF prompt — establishes the recompute target for D5.
- [ ] 1.2 Confirm the `lcm(m, m') = 128` block boundary lands both the cr=4 and cr=128 compressed streams on a clean chunk (no `pending_cur` carry) — so a block-aligned prefix is self-contained.
- [ ] 1.3 Decide the `model_id` salt (GGUF file content hash vs path+mtime) — record in design D3.

## 2. P1 — Serialization wire format (no store, no reuse)

- [x] 2.1 New `dsv4_kv_persist.rs`: `serialize_layer_cache(&DsV4LayerCache) -> Vec<u8>` + `deserialize_layer_cache(&[u8]) -> Result<DsV4LayerCache, KvPersistError>`. Operates at the per-layer `DsV4LayerCache` enum (NoCompress | Hca) so it covers both variants uniformly. Serializes the **compressed** cache (+ indexer compressed) + `compress_ratio` + both overlap states; `raw`/`pending_cur` omitted (Zero-SWA). Versioned LE header (magic `D4KV`, version, tag); hand-rolled, no serde.
- [x] 2.2 Round-trip tests (synthetic, no GGUF): `hca_compressed_round_trips_losslessly` + `hca_with_indexer_and_overlap_round_trips` — compressed/indexer-compressed rows bit-exact, compress_ratio/dims/overlap preserved, `raw`+`pending_cur` come back empty.
- [x] 2.3 NoCompress-layer (pure SWA) handling: `no_compress_layer_round_trips_as_empty` (shape-only shell → empty cache) + `empty_compressed_round_trips` (HCA layer with no chunk yet).
- [x] 2.4 Version/format-mismatch handling: typed `KvPersistError` (BadMagic / UnsupportedVersion / UnknownTag / Truncated), never a panic — `bad_magic_is_typed_error`, `unsupported_version_is_typed_error`, `unknown_tag_is_typed_error`, `truncated_blob_is_typed_error` (every truncation length).

## 3. P2 — Prefix-keyed on-disk store

- [x] 3.1 `DsV4PrefixCache::open(root, model_id, max_bytes)` — content-addressed dir tree `<root>/<model_id>/<prefix_hash>/{tokens.bin, layer_{i}.kvz}`; in-memory index rebuilt by scanning on open (`reopen_rebuilds_index`). Sweeps leftover `.tmp.*` dirs.
- [x] 3.2 `put(token_ids, &[DsV4LayerCache])` (the per-layer enum, matching P1's serializer) at a 128-block boundary — atomic write (populate `.tmp.<key>.<nonce>` dir, then `rename`), one `layer_{i}.kvz` per layer + `tokens.bin`. `get_longest_prefix(token_ids) -> Option<(hit_len, Vec<DsV4LayerCache>)>`. Non-aligned `put` → `NotBlockAligned` (`put_rejects_unaligned`).
- [x] 3.3 Prefix hashing at 128-token block boundaries via stable FNV-1a salted by `model_id` (`block_prefix_hashes`, one O(n) pass); longest-prefix match, **verified against `tokens.bin`** so a hash collision can never return the wrong cache (`longest_prefix_wins`, `model_id_isolates`).
- [x] 3.4 Size-capped LRU eviction by `last_used` (`size_cap_evicts_lru`: A touched → B evicted, C survives, survivors still load). Atomicity via tmp-dir rename.
- [x] 3.5 Store round-trip (`put_then_get_longest_prefix_round_trips`) + miss/short-prefix (`no_shared_prefix_misses`). All 7 tests on tempdirs.

## 4. P3 — Zero-SWA prefill reuse (the payoff)

- [ ] 4.1 `dsv4_prefill_with_prefix_cache(gguf/resident layers, hp, head, token_ids, &mut DsV4PrefixCache)`: longest-prefix `get`, seed each layer's `compressed`, recompute the last `min(hit_len, n_win·L)` tokens to restore `raw`/`pending`/`overlap`, then prefill the suffix; `put` new block boundaries through.
- [ ] 4.2 **Transparency test (load-bearing, real-GGUF, ignored):** logits after a cache-hit prefill equal a cold full prefill within the documented tolerance, at several prefix lengths incl. a non-block-aligned suffix. Greedy next-token identical. Backs "Cache hit is transparent to output".
- [ ] 4.3 Reuse-cost assertion: the recompute window is `O(n_win·L)`, independent of hit length (count forward-token invocations on a short vs long hit). Backs "Reuse recomputes only the SWA tail".
- [ ] 4.4 Opt-in gate: feature is off by default (`DsV4PrefixCache` constructed explicitly); the existing cold prefill path is byte-for-byte unchanged when no cache is passed. Test the disabled path is untouched.

## 5. P4 — Wire-up & docs

- [ ] 5.1 Expose the prefix-cache entry point from `dsv4_generate` behind an explicit `Option<&mut DsV4PrefixCache>` arg (default `None`).
- [ ] 5.2 Bench: cold full-prefill vs warm prefix-hit prefill wall time at a long prefix (real-GGUF, ignored) — report the speedup.
- [ ] 5.3 `make ci` green; traceability regenerated; openspec validate.
