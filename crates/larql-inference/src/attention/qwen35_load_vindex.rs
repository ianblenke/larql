//! Vindex → `Qwen35Weights` adapter — Phase 2 of `vindex-qwen35moe-reader`.
//!
//! Mirrors [`super::qwen35_load::load_qwen35_weights`] (which loads
//! from a GGUF mmap via `larql-models::load_gguf`) but reads from a
//! vindex directory produced by `larql convert gguf-to-vindex
//! --quant q4k`: the writer side shipped in PR #147 + the router
//! fixes in PRs #152 / #155.
//!
//! This file lands the **module skeleton** + canonical error type +
//! public signature. The per-layer assembly bodies (DeltaNet /
//! full-attn / MoE 256-expert packing) are scoped out as tasks
//! 2b.2..2b.4 in
//! `openspec/changes/vindex-qwen35moe-reader/step-2b-design.md` and
//! left as a `todo!()` that names the design-doc breadcrumb.
//!
//! Why ship the stub now: getting the module wiring + imports +
//! signature right in isolation lets the assembly PRs be pure
//! per-layer logic, not file-skeleton churn.

use std::path::Path;

use larql_models::quant::ggml::{dequantize, TYPE_Q4_K, TYPE_Q6_K};
use larql_models::quant::lazy::QuantTensor;
use larql_models::{ModelArchitecture, ModelWeights};
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::qwen35_forward::Qwen35Weights;

/// Errors surfaced by [`load_qwen35_weights_from_vindex`].
#[derive(Debug, thiserror::Error)]
pub enum VindexLoadError {
    #[error("vindex error: {0}")]
    Vindex(#[from] larql_vindex::VindexError),

    #[error("model error: {0}")]
    Model(#[from] larql_models::ModelError),

    #[error(
        "vindex {0:?} reports arch `{1}`, but only `qwen35` and `qwen35moe` reach this loader"
    )]
    UnexpectedArch(std::path::PathBuf, String),

    #[error(
        "tensor `{key}` not found in vindex {dir:?} — was the vindex built with PR #147+ and \
         the router-write fixes from PRs #152/#155?"
    )]
    MissingTensor {
        dir: std::path::PathBuf,
        key: String,
    },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Reconstruct a `Qwen35Weights` from a vindex directory.
///
/// **Status: stub** — task 2b.5 of `vindex-qwen35moe-reader`. The
/// public signature is final; the body is a `todo!()` until tasks
/// 2b.2..2b.4 (per-layer assembly) land. Callers SHOULD treat any
/// successful compile as a forward-compat check only — calling this
/// function at runtime panics with a breadcrumb to the design doc.
pub fn load_qwen35_weights_from_vindex(
    vindex_dir: &Path,
) -> Result<Qwen35Weights, VindexLoadError> {
    // Sanity-check the arch so the next session's runtime tests at
    // least see a clean refusal on non-qwen35 vindexes.
    let config = larql_vindex::load_vindex_config(vindex_dir)?;
    let arch_family = config
        .model_config
        .as_ref()
        .map(|c| c.model_type.clone())
        .unwrap_or_default();
    if arch_family != "qwen35" && arch_family != "qwen35moe" {
        return Err(VindexLoadError::UnexpectedArch(
            vindex_dir.to_path_buf(),
            arch_family,
        ));
    }

    todo!(
        "2b.3 (full-attn) + 2b.4 (MoE) bridges, then delegate to \
         `qwen35_load::load_qwen35_weights(&weights, &*arch)`. \
         The 2b.2 DeltaNet bridge below is ready; the standard \
         attn / Q-K-V-O bridge and MoE-256-expert packed bridge are \
         the remaining concrete blockers. See \
         openspec/changes/vindex-qwen35moe-reader/step-2b-design.md."
    )
}

/// **Task 2b.2** — bridge DeltaNet Q4_K bytes into a `ModelWeights`.
///
/// Inserts the three lazy-quant matmul tensors (`attn_qkv`,
/// `attn_gate`, `ssm_out`) into `weights.quant_tensors` so the
/// existing GGUF-side `load_qwen35_weights` (which is data-source
/// agnostic) finds them via `weights.quant_tensors.get(&key)`. The
/// two smaller matmuls without a `*_quant` slot in
/// `DeltaNetLayerWeights` (`ssm_alpha`, `ssm_beta`) get dequantised
/// to f32 and inserted into `weights.tensors`.
///
/// No-op when the vindex carries no DeltaNet bytes
/// (`idx.has_deltanet_q4k() == false`) — every non-hybrid arch
/// falls through cleanly.
///
/// Looks up arch keys via `attn_qkv_key` / `attn_gate_key` /
/// `ssm_alpha_key` / `ssm_beta_key` / `ssm_out_key`. The Qwen35 arch
/// handler returns the **HF-normalised** form
/// (`layers.{L}.self_attn.qkv_proj.weight` etc) which matches the
/// writer-side keys recorded by PR #147's manifest. The historical
/// router-key mismatch (PR #155 / task 2b.0b) does NOT affect
/// DeltaNet matmuls — the GGUF→HF remap table covers `attn_qkv.` →
/// `self_attn.qkv_proj.`, so there's no naming-skew gotcha here.
pub fn populate_deltanet_quant_tensors(
    idx: &VectorIndex,
    arch: &dyn ModelArchitecture,
    weights: &mut ModelWeights,
) -> Result<(), VindexLoadError> {
    if !idx.has_deltanet_q4k() {
        return Ok(());
    }
    let n_layers = weights.num_layers;
    for layer in 0..n_layers {
        if !arch.is_linear_attention_layer(layer) {
            continue;
        }
        let Some(slots) = idx.deltanet_q4k_layer_data(layer) else {
            continue;
        };

        // Fixed-order keys mirror the writer's tensor enumeration
        // in `write_q4k::deltanet::write_deltanet_weights_q4k`.
        let keys: [Option<String>; 5] = [
            arch.attn_qkv_key(layer),
            arch.attn_gate_key(layer),
            arch.ssm_alpha_key(layer),
            arch.ssm_beta_key(layer),
            arch.ssm_out_key(layer),
        ];

        for (i, (bytes, fmt, shape)) in slots.iter().enumerate() {
            let Some(key) = keys[i].as_ref() else {
                continue;
            };
            if shape.len() != 2 {
                continue;
            }
            let (rows, cols) = (shape[0], shape[1]);
            let tensor_type = match *fmt {
                "Q4_K" => TYPE_Q4_K,
                "Q6_K" => TYPE_Q6_K,
                _ => continue,
            };

            // Three tensors have a `*_quant` slot in
            // `DeltaNetLayerWeights` (attn_qkv / attn_gate / ssm_out
            // — indices 0, 1, 4). The other two (ssm_alpha, ssm_beta
            // — indices 2, 3) are dense-only on the consumer side,
            // so dequantise to f32 and feed `weights.tensors`. The
            // existing `load_qwen35_weights` looks them up via
            // `get_tensor` which only reads `weights.tensors`.
            let is_dense_only = matches!(i, 2 | 3);
            if is_dense_only {
                let floats = dequantize(bytes, tensor_type, rows * cols)?;
                let arr = Array2::from_shape_vec((rows, cols), floats).map_err(|e| {
                    VindexLoadError::Vindex(larql_vindex::VindexError::Parse(e.to_string()))
                })?;
                weights.tensors.insert(key.clone(), arr.into_shared());
            } else {
                let qt = QuantTensor::from_raw(bytes.to_vec(), tensor_type, rows, cols)?;
                weights.quant_tensors.insert(key.clone(), qt);
            }
        }
    }
    Ok(())
}

/// **Task 2b.3b** — bridge full-attn Q/K/V/O vindex bytes into a
/// `ModelWeights`. All 4 projections have `*_quant` slots in
/// `Qwen35AttentionLayerWeights`, so every tensor lands in
/// `weights.quant_tensors` — no dequant fallback needed (unlike
/// the DeltaNet bridge's `ssm_alpha`/`ssm_beta` dense-only case).
///
/// Walks every layer L where `arch.is_linear_attention_layer(L) ==
/// false` and pulls 4 (bytes, fmt, shape) tuples via the sparse-
/// aware accessor from PR #163 (`attn_q4k_sparse_layer_data`). The
/// existing dense accessor's `layer * 4` arithmetic would read
/// wrong bytes for hybrid arches whose manifest only carries
/// entries for the 10 full-attention layers.
///
/// V is typically Q6_K (writer's higher-precision choice for the
/// value projection); Q/K/O are Q4_K. The fmt tag string per
/// tensor determines the `tensor_type` constant passed to
/// `QuantTensor::from_raw`.
pub fn populate_attn_quant_tensors(
    idx: &VectorIndex,
    arch: &dyn ModelArchitecture,
    weights: &mut ModelWeights,
) -> Result<(), VindexLoadError> {
    let n_layers = weights.num_layers;
    for layer in 0..n_layers {
        if arch.is_linear_attention_layer(layer) {
            continue;
        }
        let Some(slots) = idx.attn_q4k_sparse_layer_data(layer) else {
            continue;
        };

        // Fixed-order keys mirror the writer's tensor enumeration
        // in `write_q4k::attn::write_attn_weights_q4k`.
        let keys: [String; 4] = [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ];

        for (i, (bytes, fmt, shape)) in slots.iter().enumerate() {
            if shape.len() != 2 {
                continue;
            }
            let (rows, cols) = (shape[0], shape[1]);
            let tensor_type = match *fmt {
                "Q4_K" => TYPE_Q4_K,
                "Q6_K" => TYPE_Q6_K,
                _ => continue,
            };
            let qt = QuantTensor::from_raw(bytes.to_vec(), tensor_type, rows, cols)?;
            weights.quant_tensors.insert(keys[i].clone(), qt);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Non-qwen35 vindex must be refused with `UnexpectedArch` —
    /// guards against the loader being mis-called from
    /// `larql-server`'s dispatch (Step 2c).
    ///
    /// Disabled until a tiny mock vindex fixture lands; the body
    /// references the right error variant so the import + signature
    /// stay verified at compile time.
    #[test]
    #[ignore = "needs a synthetic non-qwen35 vindex fixture; revisit when 2b.3 lands"]
    fn rejects_non_qwen35_arch() {
        let _ = load_qwen35_weights_from_vindex(Path::new("/dev/null"));
    }

    /// Compile-time check that the error type carries the expected
    /// variants. Catches drift if a future PR drops or renames any
    /// of the breadcrumb error cases the design doc references.
    #[test]
    fn error_variants_compile() {
        let dir = std::path::PathBuf::from("/tmp/no-such-vindex");
        let _e1 = VindexLoadError::UnexpectedArch(dir.clone(), "llama".into());
        let _e2 = VindexLoadError::MissingTensor {
            dir,
            key: "layers.0.ssm_norm.weight".into(),
        };
    }
}
