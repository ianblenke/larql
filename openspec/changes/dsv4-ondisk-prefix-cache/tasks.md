## 1. P0 — Groundwork

- [ ] 1.1 Confirm the exact reconstructable state of `DsV4LayerHcaCache` post-prefill (what `raw`/`pending_cur`/`overlap_state` hold at a 128-block boundary) by instrumenting the existing cached prefill on a real-GGUF prompt — establishes the recompute target for D5.
- [ ] 1.2 Confirm the `lcm(m, m') = 128` block boundary lands both the cr=4 and cr=128 compressed streams on a clean chunk (no `pending_cur` carry) — so a block-aligned prefix is self-contained.
- [ ] 1.3 Decide the `model_id` salt (GGUF file content hash vs path+mtime) — record in design D3.

## 2. P1 — Serialization wire format (no store, no reuse)

- [ ] 2.1 New `dsv4_kv_persist.rs`: `serialize_hca_cache(&DsV4LayerHcaCache) -> Vec<u8>` + `deserialize_hca_cache(&[u8]) -> DsV4LayerHcaCache` for the **compressed** cache (+ indexer compressed) + `compress_ratio` + `overlap_state`; `raw`/`pending_cur` omitted (Zero-SWA). Versioned header.
- [ ] 2.2 Round-trip test (synthetic, no GGUF): `deserialize(serialize(c))` reproduces the compressed entries bit-exactly + the same dims/compress_ratio. Backs "Compressed cache round-trips losslessly".
- [ ] 2.3 NoCompress-layer (pure SWA) handling: such layers have no compressed entries → serialize an empty/sentinel blob; deserialize yields an empty compressed cache. Test it.
- [ ] 2.4 Version/format-mismatch handling: a bad magic or unknown version → typed error, not panic. Test it.

## 3. P2 — Prefix-keyed on-disk store

- [ ] 3.1 `DsV4PrefixCache::open(root, model_id)` — content-addressed dir tree `<root>/<model_id>/<prefix_hash>/layer_{i}.kvz`; in-memory index rebuilt by scanning on open.
- [ ] 3.2 `put(token_ids_prefix, &[DsV4LayerHcaCache])` at a 128-block boundary — atomic write (`.tmp` + rename), one blob per layer. `get_longest_prefix(token_ids) -> Option<(hit_len, Vec<DsV4LayerHcaCache>)>`.
- [ ] 3.3 Prefix hashing at 128-token block boundaries, salted by `model_id`; longest-prefix match over present blocks.
- [ ] 3.4 Size-capped LRU eviction (by mtime); atomicity + eviction tests on a tempdir.
- [ ] 3.5 Store round-trip test: put then get-longest-prefix returns the same compressed caches; partial-prefix and miss cases.

## 4. P3 — Zero-SWA prefill reuse (the payoff)

- [ ] 4.1 `dsv4_prefill_with_prefix_cache(gguf/resident layers, hp, head, token_ids, &mut DsV4PrefixCache)`: longest-prefix `get`, seed each layer's `compressed`, recompute the last `min(hit_len, n_win·L)` tokens to restore `raw`/`pending`/`overlap`, then prefill the suffix; `put` new block boundaries through.
- [ ] 4.2 **Transparency test (load-bearing, real-GGUF, ignored):** logits after a cache-hit prefill equal a cold full prefill within the documented tolerance, at several prefix lengths incl. a non-block-aligned suffix. Greedy next-token identical. Backs "Cache hit is transparent to output".
- [ ] 4.3 Reuse-cost assertion: the recompute window is `O(n_win·L)`, independent of hit length (count forward-token invocations on a short vs long hit). Backs "Reuse recomputes only the SWA tail".
- [ ] 4.4 Opt-in gate: feature is off by default (`DsV4PrefixCache` constructed explicitly); the existing cold prefill path is byte-for-byte unchanged when no cache is passed. Test the disabled path is untouched.

## 5. P4 — Wire-up & docs

- [ ] 5.1 Expose the prefix-cache entry point from `dsv4_generate` behind an explicit `Option<&mut DsV4PrefixCache>` arg (default `None`).
- [ ] 5.2 Bench: cold full-prefill vs warm prefix-hit prefill wall time at a long prefix (real-GGUF, ignored) — report the speedup.
- [ ] 5.3 `make ci` green; traceability regenerated; openspec validate.
