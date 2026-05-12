//! Compute backend interface.
//!
//! `ComputeBackend` is the umbrella trait every caller takes as
//! `&dyn ComputeBackend`. It supertraits four narrower traits, each in
//! its own module so it's easy to read what a backend has to provide:
//!
//! | Sub-trait                     | What's there                                  |
//! |-------------------------------|-----------------------------------------------|
//! | [`MatMul`]                    | f32 / f16 matmul, gemv, batch matmul          |
//! | [`QuantMatVec`]               | unified `quant_matvec` + per-format helpers   |
//! | [`DecodeBackend`]             | KV-cached decode + prefill + MoE hook         |
//! | (umbrella) `ComputeBackend`   | `name`, `device_info`, [`Capability`] probe   |
//!
//! Most callers stay typed against `&dyn ComputeBackend`; the
//! sub-trait split is mainly an implementation-side organising
//! principle. Callers that want to branch on a specific accelerator
//! (e.g. "use f32_gemv if the backend has it, otherwise fall back to
//! matmul_transb") should use [`Capability`] + [`ComputeBackend::supports`]
//! instead of probing for `None` returns.

pub mod capability;
pub mod decode;
pub mod helpers;
pub mod matmul;
pub mod quant_matvec;

pub use capability::Capability;
pub use decode::DecodeBackend;
pub use helpers::{dot_proj_gpu, matmul_gpu};
pub use matmul::{MatMul, MatMulOp};
pub use quant_matvec::QuantMatVec;

/// Hardware compute backend — the umbrella trait every caller binds.
///
/// Combines [`MatMul`] + [`QuantMatVec`] + [`DecodeBackend`] plus
/// metadata (`name`, `device_info`) and an explicit
/// [`Capability::supports`](Self::supports) probe. Most callers
/// shouldn't care which sub-trait a method comes from.
pub trait ComputeBackend: MatMul + QuantMatVec + DecodeBackend + Send + Sync {
    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Device info string (for logging/diagnostics).
    fn device_info(&self) -> String {
        self.name().to_string()
    }

    /// Whether this backend accelerates `cap`. Callers can branch on
    /// this *before* calling, instead of pattern-matching on `None`
    /// returns from probe methods.
    ///
    /// Default returns `false` for everything; backends override to
    /// enable. See [`Capability`] for the menu.
    fn supports(&self, _cap: Capability) -> bool {
        false
    }

    /// Optional Qwen3.6 Gated DeltaNet causal Conv1D step.
    ///
    /// Implementations mutate `state` in place and return the
    /// convolution output. Shape convention is row-major:
    /// `weight[d_conv, conv_dim]`, `state[d_conv - 1, conv_dim]`,
    /// `new[conv_dim]`.
    fn qwen35_causal_conv1d_step(
        &self,
        _weight: &[f32],
        _state: &mut [f32],
        _new: &[f32],
        _d_conv: usize,
        _conv_dim: usize,
        _sequence_pos: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Optional Qwen3.6 Gated DeltaNet recurrence step.
    ///
    /// Shape convention is row-major for the ndarray layouts used by
    /// `larql-inference`: `q[s, h_k]`, `k[s, h_k]`, `v[s, h_v]`, and
    /// `state[s, s, h_v]` with `h_v` as the fastest-moving axis.
    /// Implementations mutate `state` in place and return
    /// `out[s, h_v]`.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_deltanet_step(
        &self,
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _log_g: &[f32],
        _beta: &[f32],
        _state: &mut [f32],
        _s: usize,
        _h_k: usize,
        _h_v: usize,
        _sequence_pos: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Optional Qwen3.6 per-head L2 normalisation.
    ///
    /// Shape convention is row-major for a logical
    /// `[head_dim, n_heads]` array, i.e. `x[d * n_heads + h]`.
    /// Implementations return the same layout.
    fn qwen35_l2_normalize_per_head(
        &self,
        _x: &[f32],
        _head_dim: usize,
        _n_heads: usize,
        _eps: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Optional Qwen3.6 per-head RMSNorm.
    ///
    /// Shape convention is head-major flat layout,
    /// `x[h * head_dim + d]`; `weight[d]` is shared across heads.
    fn qwen35_rms_norm_heads(
        &self,
        _x: &[f32],
        _weight: &[f32],
        _num_heads: usize,
        _head_dim: usize,
        _eps: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Optional fused Qwen3.6 DeltaNet post-projection chain (Phase
    /// E.6.A). Composes conv1d → silu → split + reshape → L2 Q/K →
    /// recurrence → reshape → rms_norm_heads → silu(z) * o on the
    /// device with a single sync at the end. Caller has already
    /// computed the projection outputs (`qkv_mixed`, `z`, `log_g`,
    /// `beta`) and supplies them as host slices.
    ///
    /// Returns the post-rms-norm, post-silu-z output `[value_dim]` in
    /// head-major layout, ready for the `ssm_out` projection. `None`
    /// means the backend declined to handle this call (CPU fallback).
    #[allow(clippy::too_many_arguments)]
    fn qwen35_deltanet_postproj_step(
        &self,
        _qkv_mixed: &[f32],
        _ssm_conv1d_weight: &[f32],
        _log_g: &[f32],
        _beta: &[f32],
        _z: &[f32],
        _ssm_norm_weight: &[f32],
        _conv_state: &mut [f32],
        _recurrent_state: &mut [f32],
        _head_v_dim: usize,
        _n_v_heads: usize,
        _n_k_heads: usize,
        _d_conv: usize,
        _eps: f32,
        _sequence_pos: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Expose the concrete type for safe downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}
