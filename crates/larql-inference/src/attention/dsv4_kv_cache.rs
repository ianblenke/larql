//! Per-layer KV cache storage for DSv4 MLA attention.
//!
//! DSv4 uses Multi-head Latent Attention (MLA) where K and V share a
//! single low-rank `kv_a` tensor of shape `(n_pos, head_dim)` — not
//! per-head as in GQA. This collapses the per-token cache footprint
//! to one `head_dim`-sized row, which for DSv4-Flash (head_dim=512) is
//! 2 KB at f32. A full 128K-context cache for one layer is ~256 MB;
//! all 43 layers ≈ 11 GB. (A 1M-context cache is ~86 GB — feasible on
//! servers, not laptops.)
//!
//! The compressor and indexer variants store the SAME `kv_a` cache —
//! the compressor reads recent `kv_a` rows and produces compressed
//! positions on-the-fly during attention rather than caching them
//! separately. So one struct works for all 3 attention variants.
//!
//! ## API shape
//! ```ignore
//! let mut cache = DsV4LayerKvCache::with_capacity(max_seq_len, head_dim);
//! cache.append(new_kv_rows.view());     // grow by N tokens (prefill or decode)
//! let cached = cache.view_current();    // (current_len, head_dim) borrowed view
//! let pos    = cache.current_len();     // absolute position for the next token
//! cache.clear();                         // reset for new sequence
//! ```
//!
//! No attention integration here — this is just the storage. The
//! follow-up PR wires `view_current()` into the attention block so
//! the decode forward can reuse cached K/V instead of re-running
//! prefill from scratch each step.

use ndarray::{s, Array1, Array2, ArrayView2};

use super::dsv4_compressor_prefill::CompressorOverlapState;

/// Per-layer DSv4 MLA KV cache.
///
/// Holds a `(max_seq_len, head_dim)` buffer pre-allocated to capacity,
/// with a separate `current_len` tracking how many rows have been
/// filled. The unused tail is zeroed at construction.
///
/// Append-only and clear-only (no random delete) — typical streaming
/// LLM decode loop usage. If you need to retract tokens (e.g. for
/// rollback), use `clear` and re-prefill from the desired prefix.
#[derive(Clone)]
pub struct DsV4LayerKvCache {
    /// Storage: `(max_seq_len, head_dim)`. Rows `[0..current_len)` are
    /// valid; rows `[current_len..max_seq_len)` are zero-initialized.
    buf: Array2<f32>,
    /// Number of valid rows in `buf`.
    current_len: usize,
}

impl DsV4LayerKvCache {
    /// Build an empty cache with capacity for `max_seq_len` tokens of
    /// `head_dim`-sized KV rows. Both arguments must be positive.
    pub fn with_capacity(max_seq_len: usize, head_dim: usize) -> Self {
        assert!(max_seq_len > 0, "max_seq_len must be > 0");
        assert!(head_dim > 0, "head_dim must be > 0");
        Self {
            buf: Array2::<f32>::zeros((max_seq_len, head_dim)),
            current_len: 0,
        }
    }

    /// Total capacity (max tokens this cache can hold).
    pub fn max_seq_len(&self) -> usize {
        self.buf.shape()[0]
    }

    /// Per-token row width.
    pub fn head_dim(&self) -> usize {
        self.buf.shape()[1]
    }

    /// Number of valid cached rows.
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Borrowed view of the valid prefix: `(current_len, head_dim)`.
    /// Empty (zero-row) view if `current_len == 0`.
    pub fn view_current(&self) -> ArrayView2<'_, f32> {
        self.buf.slice(s![..self.current_len, ..])
    }

    /// Append `new_rows` to the cache, increasing `current_len` by
    /// `new_rows.shape()[0]`. Panics if:
    /// - `new_rows` head_dim mismatches `self.head_dim()`
    /// - appending would overflow `max_seq_len`
    pub fn append(&mut self, new_rows: ArrayView2<f32>) {
        let n_new = new_rows.shape()[0];
        let head_dim = new_rows.shape()[1];
        assert_eq!(
            head_dim,
            self.head_dim(),
            "head_dim mismatch: append {} vs cache {}",
            head_dim,
            self.head_dim()
        );
        assert!(
            self.current_len + n_new <= self.max_seq_len(),
            "append would overflow: {} + {} > capacity {}",
            self.current_len,
            n_new,
            self.max_seq_len()
        );
        for r in 0..n_new {
            for d in 0..head_dim {
                self.buf[[self.current_len + r, d]] = new_rows[[r, d]];
            }
        }
        self.current_len += n_new;
    }

    /// Reset `current_len` to 0. Does not zero the underlying buffer
    /// (those rows are no longer reachable via `view_current` and
    /// will be overwritten on the next `append`).
    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Newly-built cache is empty + correctly sized.
    #[test]
    fn new_cache_is_empty_with_correct_capacity() {
        let cache = DsV4LayerKvCache::with_capacity(128, 512);
        assert_eq!(cache.max_seq_len(), 128);
        assert_eq!(cache.head_dim(), 512);
        assert_eq!(cache.current_len(), 0);
        assert_eq!(cache.view_current().shape(), &[0, 512]);
    }

    /// Single-row append: grows current_len by 1; view returns the row.
    #[test]
    fn append_single_row_grows_view() {
        let mut cache = DsV4LayerKvCache::with_capacity(8, 4);
        let row = Array2::<f32>::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        cache.append(row.view());
        assert_eq!(cache.current_len(), 1);
        let view = cache.view_current();
        assert_eq!(view.shape(), &[1, 4]);
        assert_eq!(view[[0, 0]], 1.0);
        assert_eq!(view[[0, 3]], 4.0);
    }

    /// Multi-row prefill append, then per-token decode appends.
    #[test]
    fn append_prefill_then_decode_grows_cumulatively() {
        let mut cache = DsV4LayerKvCache::with_capacity(16, 3);
        // Prefill: 5 tokens.
        let prefill = Array2::<f32>::from_shape_fn((5, 3), |(t, d)| (t * 10 + d) as f32);
        cache.append(prefill.view());
        assert_eq!(cache.current_len(), 5);
        // Decode step 1: 1 new token.
        let decode_1 = Array2::<f32>::from_shape_vec((1, 3), vec![100.0, 101.0, 102.0]).unwrap();
        cache.append(decode_1.view());
        assert_eq!(cache.current_len(), 6);
        // Decode step 2.
        let decode_2 = Array2::<f32>::from_shape_vec((1, 3), vec![200.0, 201.0, 202.0]).unwrap();
        cache.append(decode_2.view());
        assert_eq!(cache.current_len(), 7);
        // View: 7 rows.
        let view = cache.view_current();
        assert_eq!(view.shape(), &[7, 3]);
        // Prefill rows unchanged.
        assert_eq!(view[[0, 0]], 0.0); // (0*10 + 0)
        assert_eq!(view[[4, 2]], 42.0); // (4*10 + 2)
                                        // Decode rows appended at the right positions.
        assert_eq!(view[[5, 0]], 100.0);
        assert_eq!(view[[5, 2]], 102.0);
        assert_eq!(view[[6, 0]], 200.0);
        assert_eq!(view[[6, 2]], 202.0);
    }

    /// View skips the unused tail.
    #[test]
    fn view_excludes_uninitialized_rows() {
        let mut cache = DsV4LayerKvCache::with_capacity(10, 2);
        let row = Array2::<f32>::from_shape_vec((1, 2), vec![5.0, 7.0]).unwrap();
        cache.append(row.view());
        let view = cache.view_current();
        // Only 1 row visible despite 10-row capacity.
        assert_eq!(view.shape(), &[1, 2]);
    }

    /// Clear resets length; next append starts from 0.
    #[test]
    fn clear_resets_for_new_sequence() {
        let mut cache = DsV4LayerKvCache::with_capacity(4, 2);
        let row =
            Array2::<f32>::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        cache.append(row.view());
        assert_eq!(cache.current_len(), 3);
        cache.clear();
        assert_eq!(cache.current_len(), 0);
        assert_eq!(cache.view_current().shape(), &[0, 2]);
        // After clear, appending starts at row 0 again.
        let new_row = Array2::<f32>::from_shape_vec((1, 2), vec![99.0, 88.0]).unwrap();
        cache.append(new_row.view());
        assert_eq!(cache.current_len(), 1);
        assert_eq!(cache.view_current()[[0, 0]], 99.0);
    }

    /// head_dim mismatch on append panics with diagnostic.
    #[test]
    #[should_panic(expected = "head_dim mismatch")]
    fn append_head_dim_mismatch_panics() {
        let mut cache = DsV4LayerKvCache::with_capacity(4, 8);
        let wrong_width = Array2::<f32>::zeros((1, 4));
        cache.append(wrong_width.view());
    }

    /// Overflowing capacity panics.
    #[test]
    #[should_panic(expected = "append would overflow")]
    fn append_overflow_panics() {
        let mut cache = DsV4LayerKvCache::with_capacity(4, 2);
        let too_many = Array2::<f32>::zeros((5, 2));
        cache.append(too_many.view());
    }

    /// Capacity-exact append fills exactly to max_seq_len.
    #[test]
    fn append_exactly_to_capacity() {
        let mut cache = DsV4LayerKvCache::with_capacity(3, 2);
        let exact = Array2::<f32>::from_shape_fn((3, 2), |(t, d)| (t + d) as f32);
        cache.append(exact.view());
        assert_eq!(cache.current_len(), 3);
        assert_eq!(cache.current_len(), cache.max_seq_len());
        // One more row would overflow.
    }

    /// Zero-arg constructors panic.
    #[test]
    #[should_panic(expected = "max_seq_len must be > 0")]
    fn zero_max_seq_len_panics() {
        let _ = DsV4LayerKvCache::with_capacity(0, 4);
    }

    #[test]
    #[should_panic(expected = "head_dim must be > 0")]
    fn zero_head_dim_panics() {
        let _ = DsV4LayerKvCache::with_capacity(4, 0);
    }
}

/// Per-layer KV cache for the HCA attention variants (Compress and
/// Indexer). Holds **two** sub-caches:
///
/// - `raw`: the per-token kv_a rows, shape `(n_raw_pos, head_dim)` —
///   same as [`DsV4LayerKvCache`] in the NoCompress path. Capacity =
///   `max_seq_len`.
/// - `compressed`: the per-chunk compressed kv rows, shape
///   `(n_comp, head_dim)`. Capacity = `max_seq_len.div_ceil(compress_ratio)`.
///
/// The two caches are decoupled — callers append to each one
/// independently as the compressor schedule dictates (one raw row per
/// new token, one compressed row every `compress_ratio` tokens via
/// the compressor pipeline). This module does not enforce the
/// compressor's chunk-completion semantics — that's the cached HCA
/// attention's responsibility in the follow-up PR.
///
/// `compress_ratio` is stored so the cached attention can derive the
/// expected n_comp from n_raw without re-passing it.
#[derive(Clone)]
pub struct DsV4LayerHcaCache {
    /// Raw per-token kv_a cache.
    pub raw: DsV4LayerKvCache,
    /// Compressed-per-chunk kv cache.
    pub compressed: DsV4LayerKvCache,
    /// Compress ratio for this layer (typically 4 or 128 in DSv4-Flash).
    pub compress_ratio: usize,
    /// Partial-chunk buffer of cur (post-attn-norm hidden) rows that
    /// haven't yet formed a complete chunk for the compressor. Each
    /// entry is one row of `n_embd` f32s. At most `compress_ratio - 1`
    /// entries held at any time — the cached HCA Compress attention
    /// drains these in `compress_ratio`-sized batches as new tokens
    /// arrive, calling the matching `dsv4_compressor_step_coff{1,2}`
    /// and appending the result to `compressed`.
    pub pending_cur: Vec<Array1<f32>>,
    /// Overlap state for the cr=4 compressor step. Empty / unused for
    /// other compress_ratios (the coff=1 cached step doesn't read it).
    pub overlap_state: CompressorOverlapState,
}

impl DsV4LayerHcaCache {
    /// Build an empty HCA cache sized for up to `max_seq_len` tokens
    /// of `head_dim`-sized rows, with chunks of `compress_ratio`
    /// tokens producing one compressed row each.
    ///
    /// The compressed sub-cache is sized to `max_seq_len.div_ceil(compress_ratio)`
    /// so it covers the case where the final partial chunk also
    /// produces a compressed row.
    pub fn with_capacity(max_seq_len: usize, head_dim: usize, compress_ratio: usize) -> Self {
        assert!(compress_ratio > 0, "compress_ratio must be > 0");
        let max_n_comp = max_seq_len.div_ceil(compress_ratio);
        Self {
            raw: DsV4LayerKvCache::with_capacity(max_seq_len, head_dim),
            // Pad max_n_comp to at least 1 so the constructor doesn't
            // panic when max_seq_len < compress_ratio (a legal but
            // edge-case configuration where no compressed positions
            // will ever be produced).
            compressed: DsV4LayerKvCache::with_capacity(max_n_comp.max(1), head_dim),
            compress_ratio,
            pending_cur: Vec::new(),
            overlap_state: CompressorOverlapState::empty(),
        }
    }

    /// Current number of raw kv_a rows (equivalently, current sequence
    /// position).
    pub fn current_raw_len(&self) -> usize {
        self.raw.current_len()
    }

    /// Current number of compressed kv rows.
    pub fn current_compressed_len(&self) -> usize {
        self.compressed.current_len()
    }

    /// Reset all sub-caches + pending buffer + overlap state for a new
    /// sequence. After this, the cache behaves as if freshly
    /// constructed (modulo retained capacity).
    pub fn clear(&mut self) {
        self.raw.clear();
        self.compressed.clear();
        self.pending_cur.clear();
        self.overlap_state.clear();
    }
}

#[cfg(test)]
mod hca_tests {
    use super::*;
    use ndarray::Array2;

    /// Constructor sizes both sub-caches correctly.
    #[test]
    fn hca_with_capacity_sizes_both_subcaches() {
        let cache = DsV4LayerHcaCache::with_capacity(128, 512, 4);
        assert_eq!(cache.raw.max_seq_len(), 128);
        assert_eq!(cache.raw.head_dim(), 512);
        // max_n_comp = ceil(128 / 4) = 32.
        assert_eq!(cache.compressed.max_seq_len(), 32);
        assert_eq!(cache.compressed.head_dim(), 512);
        assert_eq!(cache.compress_ratio, 4);
        assert_eq!(cache.current_raw_len(), 0);
        assert_eq!(cache.current_compressed_len(), 0);
    }

    /// compress_ratio=128 with max_seq_len=128: exactly 1 compressed
    /// position fits.
    #[test]
    fn hca_compress_ratio_128_one_compressed_slot() {
        let cache = DsV4LayerHcaCache::with_capacity(128, 512, 128);
        // ceil(128 / 128) = 1.
        assert_eq!(cache.compressed.max_seq_len(), 1);
    }

    /// Partial chunks count: max_seq_len=10, compress_ratio=4 → 3 chunks
    /// (8 covers chunks 0,1; partial chunk 2 covers positions 8,9 → still
    /// 1 more compressed slot if the partial gets compressed).
    #[test]
    fn hca_partial_chunk_gets_a_compressed_slot() {
        let cache = DsV4LayerHcaCache::with_capacity(10, 4, 4);
        // ceil(10 / 4) = 3 (two full chunks + one partial).
        assert_eq!(cache.compressed.max_seq_len(), 3);
    }

    /// max_seq_len < compress_ratio: max_n_comp = 0 would panic on the
    /// inner constructor, so we pad to 1.
    #[test]
    fn hca_max_seq_len_below_compress_ratio_still_constructible() {
        let cache = DsV4LayerHcaCache::with_capacity(2, 4, 4);
        // ceil(2 / 4) = 1 → pad to 1, not 0.
        assert_eq!(cache.compressed.max_seq_len(), 1);
        // Raw is still sized as requested.
        assert_eq!(cache.raw.max_seq_len(), 2);
    }

    /// Independent appends work on both sub-caches.
    #[test]
    fn hca_independent_appends() {
        let mut cache = DsV4LayerHcaCache::with_capacity(16, 4, 4);
        // Append 4 raw rows.
        let raw_rows = Array2::<f32>::from_shape_fn((4, 4), |(t, d)| (t * 10 + d) as f32);
        cache.raw.append(raw_rows.view());
        // Append 1 compressed row (representing the chunk that just
        // closed in the compressor pipeline).
        let comp_row =
            Array2::<f32>::from_shape_vec((1, 4), vec![100.0, 101.0, 102.0, 103.0]).unwrap();
        cache.compressed.append(comp_row.view());

        assert_eq!(cache.current_raw_len(), 4);
        assert_eq!(cache.current_compressed_len(), 1);
        assert_eq!(cache.raw.view_current().shape(), &[4, 4]);
        assert_eq!(cache.compressed.view_current().shape(), &[1, 4]);
    }

    /// Clear resets both sub-caches.
    #[test]
    fn hca_clear_resets_both() {
        let mut cache = DsV4LayerHcaCache::with_capacity(8, 4, 4);
        let row = Array2::<f32>::zeros((1, 4));
        cache.raw.append(row.view());
        cache.compressed.append(row.view());
        assert_eq!(cache.current_raw_len(), 1);
        assert_eq!(cache.current_compressed_len(), 1);
        cache.clear();
        assert_eq!(cache.current_raw_len(), 0);
        assert_eq!(cache.current_compressed_len(), 0);
    }

    /// compress_ratio=0 panics.
    #[test]
    #[should_panic(expected = "compress_ratio must be > 0")]
    fn hca_zero_compress_ratio_panics() {
        let _ = DsV4LayerHcaCache::with_capacity(8, 4, 0);
    }

    // ── Pending buffer + overlap state extension tests ──

    /// Fresh cache: pending_cur empty + overlap state empty.
    #[test]
    fn hca_extension_initializes_empty() {
        let cache = DsV4LayerHcaCache::with_capacity(8, 4, 4);
        assert!(cache.pending_cur.is_empty());
        assert!(cache.overlap_state.kv_prev_last.is_none());
        assert!(cache.overlap_state.score_prev_last.is_none());
    }

    /// Pending buffer push + len.
    #[test]
    fn hca_pending_cur_grows_with_push() {
        let mut cache = DsV4LayerHcaCache::with_capacity(8, 4, 4);
        let row_a = Array1::<f32>::from_vec(vec![1.0, 2.0, 3.0]);
        let row_b = Array1::<f32>::from_vec(vec![4.0, 5.0, 6.0]);
        cache.pending_cur.push(row_a.clone());
        cache.pending_cur.push(row_b.clone());
        assert_eq!(cache.pending_cur.len(), 2);
        assert_eq!(cache.pending_cur[0], row_a);
        assert_eq!(cache.pending_cur[1], row_b);
    }

    /// Overlap state can be populated and read.
    #[test]
    fn hca_overlap_state_can_be_set_and_read() {
        let mut cache = DsV4LayerHcaCache::with_capacity(8, 4, 4);
        let kv_prev = Array2::<f32>::ones((4, 4));
        let score_prev = Array2::<f32>::zeros((4, 4));
        cache.overlap_state.kv_prev_last = Some(kv_prev.clone());
        cache.overlap_state.score_prev_last = Some(score_prev.clone());
        assert_eq!(
            cache.overlap_state.kv_prev_last.as_ref().unwrap()[[0, 0]],
            1.0
        );
        assert_eq!(
            cache.overlap_state.score_prev_last.as_ref().unwrap()[[3, 3]],
            0.0
        );
    }

    /// clear() resets everything: raw + compressed + pending + overlap.
    #[test]
    fn hca_clear_resets_all_four_pieces() {
        let mut cache = DsV4LayerHcaCache::with_capacity(8, 4, 4);
        // Populate every piece.
        let row2d = Array2::<f32>::ones((1, 4));
        cache.raw.append(row2d.view());
        cache.compressed.append(row2d.view());
        cache.pending_cur.push(Array1::<f32>::ones(3));
        cache.overlap_state.kv_prev_last = Some(Array2::<f32>::ones((4, 4)));
        cache.overlap_state.score_prev_last = Some(Array2::<f32>::zeros((4, 4)));

        cache.clear();
        assert_eq!(cache.current_raw_len(), 0);
        assert_eq!(cache.current_compressed_len(), 0);
        assert!(cache.pending_cur.is_empty());
        assert!(cache.overlap_state.kv_prev_last.is_none());
        assert!(cache.overlap_state.score_prev_last.is_none());
    }
}
