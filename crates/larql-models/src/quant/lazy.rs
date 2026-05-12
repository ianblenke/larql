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
    dequantize, q4k_row_dot, q6k_row_dot, tensor_data_size, type_name, TYPE_F32, TYPE_Q4_K,
    TYPE_Q5_K, TYPE_Q6_K,
};
use crate::ModelError;

/// A `[rows, cols]` matrix held in its native GGML quantised
/// representation. Cheap to clone (`Arc<[u8]>` under the hood).
#[derive(Clone)]
pub struct QuantTensor {
    /// Raw GGUF tensor bytes (one row's worth of data, repeated).
    /// For block-quantised types, contiguous super-blocks fill the
    /// `cols` axis first, then rows.
    data: Arc<[u8]>,
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
        Ok(Self {
            data: data.into(),
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
        Self {
            data: bytes.into(),
            tensor_type: TYPE_F32,
            rows,
            cols,
            row_bytes,
        }
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
        match self.tensor_type {
            TYPE_Q4_K => {
                use rayon::prelude::*;
                let rb = self.row_bytes;
                let data = &self.data;
                // Parallelise per-row; each thread holds one
                // accumulator. The kernel itself is already SIMD
                // (AVX2 / NEON), so this stacks data-parallel and
                // thread-parallel speedups.
                out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                    let row = &data[r * rb..(r + 1) * rb];
                    *out_r = q4k_row_dot(row, x_slice).expect("q4k_row_dot");
                });
            }
            TYPE_Q6_K => {
                use rayon::prelude::*;
                let rb = self.row_bytes;
                let data = &self.data;
                out_slice.par_iter_mut().enumerate().for_each(|(r, out_r)| {
                    let row = &data[r * rb..(r + 1) * rb];
                    *out_r = q6k_row_dot(row, x_slice).expect("q6k_row_dot");
                });
            }
            TYPE_Q5_K => {
                // No fused row-dot kernel for Q5_K yet; dequant the
                // row (256-element blocks) and dot. Allocates per row,
                // which is fine for an lm_head matvec (one call per
                // decode step).
                for r in 0..self.rows {
                    let row = &self.data[r * self.row_bytes..(r + 1) * self.row_bytes];
                    let deq = dequantize(row, self.tensor_type, self.cols)?;
                    out_slice[r] = deq.iter().zip(x_slice).map(|(a, b)| a * b).sum();
                }
            }
            TYPE_F32 => {
                for r in 0..self.rows {
                    let row_bytes = &self.data[r * self.row_bytes..(r + 1) * self.row_bytes];
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

    /// Bytes resident for this tensor. Useful for bench assertions.
    pub fn bytes_resident(&self) -> usize {
        self.data.len()
    }

    /// Borrow the raw GGML bytes. Used by GPU dispatch — the matvec
    /// kernels read the same bit layout the CPU `q4k_row_dot` /
    /// `q6k_row_dot` use, so no conversion is required.
    pub fn raw_bytes(&self) -> &[u8] {
        &self.data
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
        let row_bytes = &self.data[row * self.row_bytes..(row + 1) * self.row_bytes];
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
}
