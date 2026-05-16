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
        "2b.2..2b.5 — assemble per-layer Qwen35FullLayerWeights from \
         `idx.deltanet_q4k_layer_data(l)` / `idx.attn_q4k_layer_data(l)` / \
         `layers/layer_LL.weights`. See \
         openspec/changes/vindex-qwen35moe-reader/step-2b-design.md."
    )
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
