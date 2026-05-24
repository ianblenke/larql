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

use ndarray::{s, Array2, ArrayView2};

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
