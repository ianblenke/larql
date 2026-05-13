//! Lazy-quantized tensor — holds raw GGUF bytes + tensor type and
//! dispatches matvec without materialising the full f32 matrix.
//!
//! Motivation: dequantising a 27 B Qwen3.6 model to f32 at load time
//! consumes ~100 GiB of RAM (PR #85 bench-baseline). Keeping each
//! tensor in its native Q4_K / Q5_K / Q6_K form drops that by 5×.
//! The `matvec` path here is intentionally per-row so it can dispatch
//! to the existing `q4k_row_dot` / `q6k_row_dot` kernels for the
//! quantized types and fall back to a per-row dequant + dot for
//! anything without a fused row kernel.

use std::sync::Arc;

use ndarray::Array1;

use crate::quant::ggml::{
    dequantize, q4k_q8k_row_dot, q4k_row_dot, q5k_q8k_row_dot, q6k_q8k_row_dot, q6k_row_dot,
    q8_0_q8k_row_dot, q8_0_row_dot,
    quantize_to_q8_k, tensor_data_size, type_name, Q8_K_BLOCK_BYTES, TYPE_F32, TYPE_Q4_K,
    TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0,
};
use crate::ModelError;

/// A `[rows, cols]` matrix held in its native GGML quantised
/// representation. Cheap to clone (`Arc<[u8]>` under the hood).
///
/// Supports view semantics: `byte_offset` + `byte_len` carve out a
/// contiguous sub-range of the shared `Arc<[u8]>`. The original
/// `from_raw` constructor produces a view spanning the whole buffer;
/// `expert_slice` produces a per-expert sub-view of a 3D-packed MoE
/// tensor without copying. All internal `data[i..j]` accesses go
/// through `bytes()`, which respects the current view's offset.
/// Backing storage for a `QuantTensor`. Either an owned heap buffer
/// (the existing `from_raw` / `from_f32_rows` path that copies bytes)
/// or an `Arc<Mmap>` window with `byte_offset`/`byte_len` carving out
/// a per-tensor sub-range (Phase G.1 ktransformers-style zero-copy).
///
/// The Mmap variant means weights stay resident in the page cache,
/// not in RSS — RSS only reflects actually-touched pages. The
/// `prefetch_willneed` method on `QuantTensor` calls
/// `Mmap::advise_range(WillNeed, ...)` so the kernel starts paging
/// in the next active expert's bytes during the current expert's
/// compute.
#[derive(Clone)]
pub enum QuantBacking {
    Heap(Arc<[u8]>),
    Mmap(Arc<memmap2::Mmap>),
}

impl QuantBacking {
    #[inline]
    fn as_full_slice(&self) -> &[u8] {
        match self {
            Self::Heap(b) => b,
            Self::Mmap(m) => &m[..],
        }
    }
}

#[derive(Clone)]
pub struct QuantTensor {
    /// Raw GGUF tensor bytes (shared parent buffer). For block-
    /// quantised types, contiguous super-blocks fill the `cols`
    /// axis first, then rows.
    data: QuantBacking,
    /// Byte offset into `data` where this view starts.
    byte_offset: usize,
    /// Byte length of this view (= `rows * row_bytes`).
    byte_len: usize,
    /// GGML tensor type id (e.g. `TYPE_Q6_K`).
    tensor_type: u32,
    rows: usize,
    cols: usize,
    /// Bytes per row (= `tensor_data_size(type, cols)`). Cached so
    /// matvec doesn't recompute it per row.
    row_bytes: usize,
}

impl QuantTensor {
    /// Build from raw GGUF bytes. The buffer length MUST be
    /// `tensor_data_size(tensor_type, rows * cols)`.
    pub fn from_raw(
        data: Vec<u8>,
        tensor_type: u32,
        rows: usize,
        cols: usize,
    ) -> Result<Self, ModelError> {
        let expected = tensor_data_size(tensor_type, rows * cols)?;
        if data.len() != expected {
            return Err(ModelError::Parse(format!(
                "QuantTensor::from_raw: data len {} != expected {} for {} {}×{}",
                data.len(),
                expected,
                type_name(tensor_type),
                rows,
                cols,
            )));
        }
        let row_bytes = tensor_data_size(tensor_type, cols)?;
        let byte_len = data.len();
        Ok(Self {
            data: QuantBacking::Heap(data.into()),
            byte_offset: 0,
            byte_len,
            tensor_type,
            rows,
            cols,
            row_bytes,
        })
    }

    /// Phase G.1 — mmap-backed constructor. Shares the parent mmap
    /// `Arc<Mmap>` and carves out `[offset .. offset + len]`. No
    /// bytes are copied. The mmap stays alive as long as any tensor
    /// view holds the `Arc`, so partial drops are safe.
    pub fn from_mmap_region(
        mmap: Arc<memmap2::Mmap>,
        byte_offset: usize,
        byte_len: usize,
        tensor_type: u32,
        rows: usize,
        cols: usize,
    ) -> Result<Self, ModelError> {
        let expected = tensor_data_size(tensor_type, rows * cols)?;
        if byte_len != expected {
            return Err(ModelError::Parse(format!(
                "QuantTensor::from_mmap_region: byte_len {byte_len} != expected {expected} for {} {rows}×{cols}",
                type_name(tensor_type),
            )));
        }
        if byte_offset.checked_add(byte_len).map_or(true, |end| end > mmap.len()) {
            return Err(ModelError::Parse(format!(
                "QuantTensor::from_mmap_region: range {byte_offset}+{byte_len} out of mmap (len {})",
                mmap.len()
            )));
        }
        let row_bytes = tensor_data_size(tensor_type, cols)?;
        Ok(Self {
            data: QuantBacking::Mmap(mmap),
            byte_offset,
            byte_len,
            tensor_type,
            rows,
            cols,
            row_bytes,
        })
    }

    /// Build from an f32 row-major buffer. Stored as `TYPE_F32`
    /// internally — useful for synthetic test fallbacks where no
    /// real quantisation is involved.
    pub fn from_f32_rows(rows: usize, cols: usize, values: &[f32]) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut bytes = Vec::with_capacity(rows * cols * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let row_bytes = cols * 4;
        let byte_len = bytes.len();
        Self {
            data: QuantBacking::Heap(bytes.into()),
            byte_offset: 0,
            byte_len,
            tensor_type: TYPE_F32,
            rows,
            cols,
            row_bytes,
        }
    }

    /// Bytes of this view (respects `byte_offset` and `byte_len`).
    #[inline]
    fn bytes(&self) -> &[u8] {
        let all = self.data.as_full_slice();
        &all[self.byte_offset..self.byte_offset + self.byte_len]
    }

    /// Phase G.1 — issue a `MADV_WILLNEED` hint for this tensor's
    /// byte range. No-op for `Heap`-backed tensors (already resident).
    /// For `Mmap`-backed views, signals the kernel to start paging
    /// in this range from the page cache / underlying file. Called
    /// before per-expert dispatch in `swiglu_moe_lazy` so the next
    /// expert's pages arrive while the current expert is computing.
    pub fn prefetch_willneed(&self) {
        if let QuantBacking::Mmap(mmap) = &self.data {
            // `advise_range` clips to page boundaries internally on
            // memmap2 0.9+. Ignore errors — prefetch is a hint.
            let _ = mmap.advise_range(
                memmap2::Advice::WillNeed,
                self.byte_offset,
                self.byte_len,
            );
        }
    }

    /// Per-expert 2D slice of a 3D-packed MoE weight tensor.
    ///
    /// Qwen3.6-35B-A3B (and similar MoE GGUFs) pack the
    /// `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` tensors
    /// as 3D with the expert axis outermost: GGUF dims
    /// `[hidden, ffn_dim, num_experts]` (fastest-first) which the
    /// loader presents to us as a flat 2D `[num_experts * out_dim,
    /// in_dim]` (so `rows` is divisible by `num_experts`).
    ///
    /// `expert_slice(E, num_experts)` returns a new `QuantTensor`
    /// view of shape `[rows / num_experts, cols]` (the per-expert
    /// 2D weight matrix in the standard matvec layout `y =
    /// expert_w @ x`). The slice shares the parent's `Arc<[u8]>`;
    /// no copy or re-decoding. Byte alignment is guaranteed because
    /// every super-block (256 elements for Q4_K / Q5_K / Q6_K) fits
    /// inside one row's `row_bytes`, and each expert occupies a
    /// whole `rows_per_expert * row_bytes` contiguous chunk.
    pub fn expert_slice(&self, expert_id: usize, num_experts: usize) -> Result<Self, ModelError> {
        if num_experts == 0 || self.rows % num_experts != 0 {
            return Err(ModelError::Parse(format!(
                "QuantTensor::expert_slice: rows {} not divisible by num_experts {}",
                self.rows, num_experts,
            )));
        }
        if expert_id >= num_experts {
            return Err(ModelError::Parse(format!(
                "QuantTensor::expert_slice: expert_id {} out of range [0..{})",
                expert_id, num_experts,
            )));
        }
        let rows_per_expert = self.rows / num_experts;
        let bytes_per_expert = rows_per_expert * self.row_bytes;
        let expert_byte_offset = self.byte_offset + expert_id * bytes_per_expert;
        // Defensive: the parent's `byte_len` must contain every
        // expert's slice. With well-formed GGUF this is implied by
        // the divisibility check above, but verify explicitly.
        if expert_id * bytes_per_expert + bytes_per_expert > self.byte_len {
            return Err(ModelError::Parse(format!(
                "QuantTensor::expert_slice: expert {} slice extends past parent view (byte_offset + len = {} > parent byte_len {})",
                expert_id,
                expert_id * bytes_per_expert + bytes_per_expert,
                self.byte_len,
            )));
        }
        Ok(Self {
            data: self.data.clone(),
            byte_offset: expert_byte_offset,
            byte_len: bytes_per_expert,
            tensor_type: self.tensor_type,
            rows: rows_per_expert,
            cols: self.cols,
            row_bytes: self.row_bytes,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn tensor_type(&self) -> u32 {
        self.tensor_type
    }

    /// `M @ x` where `M` is this tensor (`[rows, cols]`) and `x` is
    /// a column vector (`[cols]`). Returns `[rows]`.
    pub fn matvec(&self, x: &Array1<f32>) -> Result<Array1<f32>, ModelError> {
        if x.len() != self.cols {
            return Err(ModelError::Parse(format!(
                "QuantTensor::matvec: x len {} != cols {}",
                x.len(),
                self.cols,
            )));
        }
        let x_slice = x
            .as_slice()
            .ok_or_else(|| ModelError::Parse("QuantTensor::matvec: x must be contiguous".into()))?;
        let mut out = Array1::<f32>::zeros(self.rows);
        let out_slice = out
            .as_slice_mut()
            .expect("Array1 is contiguous by construction");
        let view_bytes = self.bytes();
        match self.tensor_type {
            TYPE_Q4_K => {
                use rayon::prelude::*;
                let rb = self.row_bytes;
                let data = view_bytes;
                let use_q8k = x_slice.len().is_multiple_of(256)
                    && std::env::var("LARQL_Q4K_USE_F32_DOT").ok().as_deref() != Some("1");
                if use_q8k {
                    // E.8 step 2c: thread-local cache of the Q8_K
                    // conversion. In MoE the same `x` is fed into 16+
                    // matvec calls (gate × 8 experts + up × 8 experts);
                    // the cache makes those reuse one quantise instead
                    // of doing it per call.
                    with_q8k_for(x_slice, |q8k_bytes| {
                        out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                            let row = &data[r * rb..(r + 1) * rb];
                            *out_r = q4k_q8k_row_dot(row, q8k_bytes).expect("q4k_q8k_row_dot");
                        });
                    });
                } else {
                    out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                        let row = &data[r * rb..(r + 1) * rb];
                        *out_r = q4k_row_dot(row, x_slice).expect("q4k_row_dot");
                    });
                }
                let _ = Q8_K_BLOCK_BYTES;
            }
            TYPE_Q6_K => {
                use rayon::prelude::*;
                let rb = self.row_bytes;
                let data = view_bytes;
                // E.8 step 2e: AVX2 Q6_K × Q8_K fast path (LM_HEAD is
                // 7 % of all-CPU MoE decode time). Mirrors the Q4_K /
                // Q5_K patterns. Falls back to the legacy `q6k_row_dot`
                // scalar (Q4_K_S has an AVX2 dequant pass) when
                // `x.len() % 256 != 0` or via `LARQL_Q6K_USE_F32_DOT=1`.
                let use_q8k = x_slice.len().is_multiple_of(256)
                    && std::env::var("LARQL_Q6K_USE_F32_DOT").ok().as_deref() != Some("1");
                if use_q8k {
                    with_q8k_for(x_slice, |q8k_bytes| {
                        out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                            let row = &data[r * rb..(r + 1) * rb];
                            *out_r = q6k_q8k_row_dot(row, q8k_bytes).expect("q6k_q8k_row_dot");
                        });
                    });
                } else {
                    out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                        let row = &data[r * rb..(r + 1) * rb];
                        *out_r = q6k_row_dot(row, x_slice).expect("q6k_row_dot");
                    });
                }
            }
            TYPE_Q5_K => {
                // E.8 step 2d: AVX2 Q5_K × Q8_K fast path. FFN_DOWN is
                // 50.5 % of all-CPU MoE decode time per the
                // LARQL_QWEN35_FINE_PROFILE breakdown — this is the
                // single biggest hot spot on the CPU value-prop tier.
                // Path mirrors the Q4_K case: pre-quantise `x` to Q8_K
                // once, then per-row int-arithmetic dot via
                // `q5k_q8k_row_dot`. The legacy dequant-per-row path
                // stays for shapes that aren't a multiple of 256, and
                // can be forced via `LARQL_Q5K_USE_DEQUANT_DOT=1`.
                let use_q8k = x_slice.len().is_multiple_of(256)
                    && std::env::var("LARQL_Q5K_USE_DEQUANT_DOT")
                        .ok()
                        .as_deref()
                        != Some("1");
                if use_q8k {
                    use rayon::prelude::*;
                    let rb = self.row_bytes;
                    let data = view_bytes;
                    with_q8k_for(x_slice, |q8k_bytes| {
                        out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                            let row = &data[r * rb..(r + 1) * rb];
                            *out_r = q5k_q8k_row_dot(row, q8k_bytes).expect("q5k_q8k_row_dot");
                        });
                    });
                } else {
                    for r in 0..self.rows {
                        let row = &view_bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                        let deq = dequantize(row, self.tensor_type, self.cols)?;
                        out_slice[r] = deq.iter().zip(x_slice).map(|(a, b)| a * b).sum();
                    }
                }
            }
            TYPE_Q8_0 => {
                // Legacy Q8_0 — 32-element blocks of int8 + f16 scale.
                // Mixed-quant GGUFs like Qwen3.6-35B-A3B-UD-Q4_K_M use
                // Q8_0 for attn projections + shared-expert FFN.
                //
                // E.8 step 2f: AVX2 Q8_0 × Q8_K fast path. When
                // `x.len() % 256 == 0` (a multiple of one Q8_K
                // super-block), pre-quantise `x` once and dispatch
                // the int8 × int8 → int32 dot via `q8_0_q8k_row_dot`.
                // Otherwise fall back to the f32-input `q8_0_row_dot`
                // (AVX2 on x86_64, scalar elsewhere). Force the f32
                // path for A/B with `LARQL_Q8_0_USE_F32_DOT=1`.
                use rayon::prelude::*;
                let rb = self.row_bytes;
                let data = view_bytes;
                let use_q8k = x_slice.len().is_multiple_of(256)
                    && std::env::var("LARQL_Q8_0_USE_F32_DOT").ok().as_deref() != Some("1");
                if use_q8k {
                    with_q8k_for(x_slice, |q8k_bytes| {
                        out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                            let row = &data[r * rb..(r + 1) * rb];
                            *out_r = q8_0_q8k_row_dot(row, q8k_bytes).expect("q8_0_q8k_row_dot");
                        });
                    });
                } else {
                    out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                        let row = &data[r * rb..(r + 1) * rb];
                        *out_r = q8_0_row_dot(row, x_slice).expect("q8_0_row_dot");
                    });
                }
            }
            TYPE_F32 => {
                for r in 0..self.rows {
                    let row_bytes = &view_bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                    let mut acc = 0.0_f32;
                    for (i, b) in row_bytes.chunks_exact(4).enumerate() {
                        let v = f32::from_le_bytes(b.try_into().unwrap());
                        acc += v * x_slice[i];
                    }
                    out_slice[r] = acc;
                }
            }
            other => {
                return Err(ModelError::Parse(format!(
                    "QuantTensor::matvec: unsupported tensor type id {other} ({})",
                    type_name(other),
                )));
            }
        }
        Ok(out)
    }

    /// Bytes resident for this tensor view. Useful for bench
    /// assertions. Equal to `rows * row_bytes`.
    pub fn bytes_resident(&self) -> usize {
        self.byte_len
    }

    /// Borrow the raw GGML bytes for this view. Used by GPU dispatch
    /// — the matvec kernels read the same bit layout the CPU
    /// `q4k_row_dot` / `q6k_row_dot` use, so no conversion is
    /// required. Respects `expert_slice` views (returns only the
    /// expert's bytes, not the parent's).
    pub fn raw_bytes(&self) -> &[u8] {
        self.bytes()
    }

    /// Dequantise a single row to `Array1<f32>`. Hot path for embed
    /// lookups where the caller wants one row by index without
    /// materialising the entire matrix.
    pub fn row_to_f32(&self, row: usize) -> Result<Array1<f32>, ModelError> {
        if row >= self.rows {
            return Err(ModelError::Parse(format!(
                "QuantTensor::row_to_f32: row {row} out of range [0..{})",
                self.rows
            )));
        }
        let view_bytes = self.bytes();
        let row_bytes = &view_bytes[row * self.row_bytes..(row + 1) * self.row_bytes];
        match self.tensor_type {
            TYPE_F32 => {
                let vec: Vec<f32> = row_bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                Ok(Array1::from(vec))
            }
            // For block-quantised types just delegate to the bulk
            // dequantizer over a single row's worth of bytes. The
            // result vec is `self.cols` elements.
            _ => {
                let vec = dequantize(row_bytes, self.tensor_type, self.cols)?;
                Ok(Array1::from(vec))
            }
        }
    }
}

/// Thread-local Q8_K conversion cache used by `QuantTensor::matvec`'s
/// Q4_K and Q5_K arms (E.8 step 2c amortisation).
///
/// MoE feeds the same `x` (post-norm hidden state) into 16+ matvec
/// calls per layer (gate × 8 experts + up × 8 experts). Without
/// caching, each call re-runs `quantize_to_q8_k(x)` — a 256-element
/// scan + Q8_K block-pack. The cache keys on `(x.as_ptr(), x.len())`:
/// consecutive calls with the same `x` slice reuse the cached bytes
/// inside one thread.
///
/// Caveat: `x.as_ptr()` can ABA-collide if a `Vec<f32>` is dropped
/// and a new one lands at the same address. In the larql forward path
/// `x` is kept alive across all sibling matvec calls in one MoE FFN,
/// so this doesn't fire in practice.
#[inline]
fn with_q8k_for<R>(x: &[f32], f: impl FnOnce(&[u8]) -> R) -> R {
    thread_local! {
        static CACHE: std::cell::RefCell<Option<(usize, usize, Vec<u8>)>> =
            const { std::cell::RefCell::new(None) };
    }
    CACHE.with(|cell| {
        let mut cell = cell.borrow_mut();
        let ptr = x.as_ptr() as usize;
        let len = x.len();
        let needs_refresh =
            !matches!(*cell, Some((p, l, _)) if p == ptr && l == len);
        if needs_refresh {
            *cell = Some((ptr, len, quantize_to_q8_k(x)));
        }
        let bytes = cell.as_ref().expect("just initialised").2.as_slice();
        f(bytes)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrip_matvec_matches_scalar_dot() {
        let rows = 7;
        let cols = 5;
        let values: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let qt = QuantTensor::from_f32_rows(rows, cols, &values);
        let x = Array1::from(vec![0.5_f32, -0.25, 0.125, 1.0, -0.7]);
        let out_qt = qt.matvec(&x).unwrap();
        // Reference: scalar loop. Avoiding `Array2::dot` keeps this
        // crate's tests off the BLAS link path (`larql-models` does
        // not depend on `blas-src`).
        for r in 0..rows {
            let mut expected = 0.0f32;
            for c in 0..cols {
                expected += values[r * cols + c] * x[c];
            }
            assert!(
                (out_qt[r] - expected).abs() < 1e-5,
                "row {r}: qt={} ref={}",
                out_qt[r],
                expected,
            );
        }
    }

    #[test]
    fn matvec_wrong_x_len_errors() {
        let qt = QuantTensor::from_f32_rows(2, 3, &[1.0; 6]);
        let x = Array1::from(vec![1.0_f32; 4]);
        assert!(qt.matvec(&x).is_err());
    }

    #[test]
    fn row_to_f32_returns_each_row_in_order() {
        let rows = 4;
        let cols = 3;
        let values: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.5 - 1.0).collect();
        let qt = QuantTensor::from_f32_rows(rows, cols, &values);
        for r in 0..rows {
            let out = qt.row_to_f32(r).unwrap();
            assert_eq!(out.len(), cols);
            for c in 0..cols {
                assert!((out[c] - values[r * cols + c]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn row_to_f32_oob_errors() {
        let qt = QuantTensor::from_f32_rows(2, 3, &[0.0; 6]);
        assert!(qt.row_to_f32(2).is_err());
    }

    /// `expert_slice` carves out a 3D-packed MoE tensor's per-expert
    /// 2D submatrix without copying. The parent in this test is
    /// `[num_experts=4, rows_per_expert=2, cols=3]` flattened as a
    /// 2D `[rows=8, cols=3]` f32 tensor. Each expert's slice must
    /// match the corresponding rows of the original buffer.
    #[test]
    fn expert_slice_returns_per_expert_subview_f32() {
        let num_experts = 4;
        let rows_per_expert = 2;
        let cols = 3;
        let rows = num_experts * rows_per_expert; // = 8
                                                  // values[expert * 6 + row * 3 + col] = unique tag.
        let values: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
        let parent = QuantTensor::from_f32_rows(rows, cols, &values);
        assert_eq!(parent.bytes_resident(), rows * cols * 4);

        for e in 0..num_experts {
            let slice = parent.expert_slice(e, num_experts).unwrap();
            assert_eq!(slice.shape(), [rows_per_expert, cols]);
            assert_eq!(slice.tensor_type(), parent.tensor_type());
            // The slice's bytes must equal the parent rows
            // `e * rows_per_expert .. (e + 1) * rows_per_expert`.
            for r in 0..rows_per_expert {
                let row = slice.row_to_f32(r).unwrap();
                for c in 0..cols {
                    let parent_row = e * rows_per_expert + r;
                    let want = values[parent_row * cols + c];
                    assert!(
                        (row[c] - want).abs() < 1e-6,
                        "expert {e} row {r} col {c}: got {} want {want}",
                        row[c],
                    );
                }
            }
        }
    }

    /// `expert_slice` shares the parent's backing Arc — verified via
    /// `Arc::strong_count` on the heap-backed variant. Cheap clone,
    /// no per-expert copy.
    #[test]
    fn expert_slice_shares_arc_with_parent() {
        let parent = QuantTensor::from_f32_rows(4, 2, &[1.0_f32; 8]);
        let arc = match &parent.data {
            QuantBacking::Heap(arc) => Arc::clone(arc),
            QuantBacking::Mmap(_) => panic!("from_f32_rows builds heap-backed tensors"),
        };
        // After the explicit clone above the parent's Arc has +1
        // strong refcount. expert_slice clones the enum (which clones
        // the inner Arc) so each slice further bumps by +1.
        let arc_before = Arc::strong_count(&arc);
        let _s0 = parent.expert_slice(0, 2).unwrap();
        let _s1 = parent.expert_slice(1, 2).unwrap();
        let arc_after = Arc::strong_count(&arc);
        assert_eq!(arc_after, arc_before + 2);
    }

    #[test]
    fn expert_slice_rejects_mismatched_num_experts() {
        let qt = QuantTensor::from_f32_rows(7, 3, &[0.0; 21]);
        // 7 rows not divisible by 4 experts.
        assert!(qt.expert_slice(0, 4).is_err());
    }

    #[test]
    fn expert_slice_rejects_oob_expert_id() {
        let qt = QuantTensor::from_f32_rows(8, 3, &[0.0; 24]);
        assert!(qt.expert_slice(4, 4).is_err());
    }

    /// Matvec on a sliced expert produces the same result as matvec
    /// on a freshly-built QuantTensor of the same rows. Confirms the
    /// view semantics work end-to-end through the existing `matvec`
    /// path (which uses `self.bytes()` to address its data).
    #[test]
    fn expert_slice_matvec_matches_standalone() {
        let num_experts = 3;
        let rows_per_expert = 4;
        let cols = 5;
        let rows = num_experts * rows_per_expert;
        let values: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.07 - 0.3).collect();
        let parent = QuantTensor::from_f32_rows(rows, cols, &values);
        let x = Array1::from(vec![0.4_f32, -0.2, 0.1, 0.05, -0.15]);

        for e in 0..num_experts {
            let slice = parent.expert_slice(e, num_experts).unwrap();
            let slice_out = slice.matvec(&x).unwrap();

            // Build an equivalent standalone tensor from the same
            // rows and matvec it directly.
            let start = e * rows_per_expert * cols;
            let end = start + rows_per_expert * cols;
            let standalone = QuantTensor::from_f32_rows(rows_per_expert, cols, &values[start..end]);
            let standalone_out = standalone.matvec(&x).unwrap();

            assert_eq!(slice_out.len(), standalone_out.len());
            for r in 0..rows_per_expert {
                assert!(
                    (slice_out[r] - standalone_out[r]).abs() < 1e-6,
                    "expert {e} row {r}: slice={} standalone={}",
                    slice_out[r],
                    standalone_out[r],
                );
            }
        }
    }
}
