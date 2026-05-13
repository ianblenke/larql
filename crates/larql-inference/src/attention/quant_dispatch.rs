//! Backend-aware quantised matvec dispatch — bridges `QuantTensor`
//! (in `larql-models`) and `larql_compute::backend::QuantMatVec`
//! (the GPU dispatcher).
//!
//! Why this lives here: `larql-models` deliberately doesn't depend on
//! `larql-compute` (the model definition crate stays free of CUDA /
//! Metal feature gates). So the bridge function — "if a GPU backend
//! is attached, route `QuantTensor::matvec` through it" — lives in
//! `larql-inference`, where both deps are present.
//!
//! Phase E.1 of `qwen35-gpu-forward`. The fallback path is the
//! existing rayon CPU dispatch already living on `QuantTensor::matvec`.

use larql_compute::ComputeBackend;
use larql_compute::QuantFormat;
use larql_models::quant::ggml::{TYPE_F32, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K};
use larql_models::quant::lazy::QuantTensor;
use ndarray::Array1;

/// Map a GGML tensor type id to the corresponding
/// `larql_compute::QuantFormat`. Returns `None` for unsupported
/// formats (e.g. Q5_K — no GPU kernel today).
pub fn ggml_type_to_quant_format(t: u32) -> Option<QuantFormat> {
    match t {
        TYPE_Q4_K => Some(QuantFormat::Q4_K),
        TYPE_Q6_K => Some(QuantFormat::Q6_K),
        TYPE_F32 => Some(QuantFormat::F32),
        TYPE_Q5_K => None,
        _ => None,
    }
}

/// Dispatch one Qwen3.6 forward matvec to the GPU backend when
/// possible, falling back to the rayon CPU path otherwise.
///
/// `out[N] = W[N, K] · x[K]` where `qt` is `[rows=N, cols=K]`.
pub fn matvec_with_backend(
    qt: &QuantTensor,
    x: &Array1<f32>,
    backend: Option<&dyn ComputeBackend>,
) -> Array1<f32> {
    let shape = qt.shape();
    let rows = shape[0];
    let cols = shape[1];

    // GPU fast path: format must be GPU-supported AND the backend
    // must actually accept it (returns Some).
    if let Some(b) = backend {
        if let Some(format) = ggml_type_to_quant_format(qt.tensor_type()) {
            let bytes = qt.raw_bytes();
            let x_slice = x.as_slice().expect("Array1 contiguous");
            if let Some(out) = b.quant_matvec(format, bytes, x_slice, rows, cols) {
                return Array1::from(out);
            }
        }
    }

    // CPU fallback (rayon + AVX2/NEON inside `QuantTensor::matvec`).
    // Diagnostic env var prints the first 20 fallback dispatches so we
    // can see at a glance which tensors aren't taking the GPU path:
    //   LARQL_QWEN35_DISPATCH_TRACE=1
    // type=13 is GGML's Q5_K — the format that, on Qwen3.6-27B-Q4_K_S,
    // covers `attn_qkv` (DeltaNet), `ssm_out`, `ffn_down`, and the
    // full-attn `attn_k`/`attn_v`. Those four are the dominant
    // per-token cost (E.6.A.9 fine-profile finding).
    if std::env::var("LARQL_QWEN35_DISPATCH_TRACE").is_ok() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        let n = CALLS.fetch_add(1, Ordering::Relaxed);
        if n < 20 {
            eprintln!(
                "[dispatch] CPU fallback: type={} rows={} cols={}",
                qt.tensor_type(),
                rows,
                cols
            );
        }
    }
    qt.matvec(x).expect("QuantTensor::matvec CPU fallback")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without a backend, dispatch reduces to `QuantTensor::matvec`
    /// — already covered by `quant::lazy` tests but verified here so
    /// the bridge's plumbing (shape extraction, slice access) holds.
    #[test]
    fn no_backend_falls_back_to_cpu_matvec() {
        let rows = 5;
        let cols = 4;
        let values: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let qt = QuantTensor::from_f32_rows(rows, cols, &values);
        let x = Array1::from(vec![0.5_f32, -0.25, 0.125, 1.0]);
        let cpu = qt.matvec(&x).unwrap();
        let dispatched = matvec_with_backend(&qt, &x, None);
        for r in 0..rows {
            assert!(
                (cpu[r] - dispatched[r]).abs() < 1e-6,
                "row {r}: cpu={} dispatched={}",
                cpu[r],
                dispatched[r],
            );
        }
    }
}
