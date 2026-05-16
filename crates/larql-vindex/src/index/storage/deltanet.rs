//! DeltaNet weight loader + per-layer accessor.
//!
//! Loads the per-linear-attention-layer DeltaNet matmul weights
//! (`attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out`) from
//! `deltanet_weights_q4k.bin` plus the matching manifest. Mirrors
//! [`super::attn`]'s `load_attn_q4k` shape; lives in its own file so
//! the qwen35 hybrid-attention surface is clearly separated from the
//! standard Q/K/V/O loader.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::index::core::VectorIndex;
use crate::index::storage::vindex_storage::{
    DeltanetManifestEntry, VindexStorage, DELTANET_TENSORS_PER_LAYER,
};
use crate::mmap_util::mmap_optimized;

impl VectorIndex {
    /// Load `deltanet_weights_q4k.bin` + its manifest into the storage
    /// façade. No-op (returns `Ok(())`) when the file is absent — every
    /// non-hybrid vindex falls through here.
    pub fn load_deltanet_q4k(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(DELTANET_WEIGHTS_Q4K_BIN);
        if !path.exists() {
            return Ok(());
        }
        let file = std::fs::File::open(&path)?;
        let mmap = Arc::new(unsafe { mmap_optimized(&file)? });

        let manifest_path = dir.join(DELTANET_WEIGHTS_Q4K_MANIFEST_JSON);
        let manifest = if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<DeltanetManifestEntry> = json
                .iter()
                .map(|e| {
                    let key = e["key"]
                        .as_str()
                        .ok_or_else(|| {
                            VindexError::Parse(
                                "deltanet_weights_q4k_manifest entry missing `key` field".into(),
                            )
                        })?
                        .to_string();
                    let offset = e["offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["length"].as_u64().unwrap_or(0) as usize;
                    let format = e["format"]
                        .as_str()
                        .ok_or_else(|| {
                            VindexError::Parse(
                                "deltanet_weights_q4k_manifest entry missing `format` field".into(),
                            )
                        })?
                        .to_string();
                    Ok::<_, VindexError>(DeltanetManifestEntry {
                        key,
                        offset,
                        length,
                        format,
                    })
                })
                .collect::<Result<Vec<_>, VindexError>>()?;
            Some(entries)
        } else {
            None
        };

        Arc::make_mut(&mut self.storage).set_deltanet_q4k(mmap, manifest);
        Ok(())
    }

    /// Whether a `deltanet_weights_q4k.bin` mmap has been loaded into
    /// the storage façade. Useful for the qwen35 dispatch check.
    pub fn has_deltanet_q4k(&self) -> bool {
        self.storage.has_deltanet_q4k()
    }

    /// Get the 5 DeltaNet matmul tensors for one linear-attention layer:
    /// `[(attn_qkv, fmt), (attn_gate, fmt), (ssm_alpha, fmt),
    ///   (ssm_beta, fmt), (ssm_out, fmt)]`.
    ///
    /// `layer` is the **global** layer index. Returns `None` when no
    /// `deltanet_weights_q4k.bin` was loaded, when the layer is a
    /// full-attention layer (its weights live in `attn_weights_q4k.bin`
    /// instead), or when any of the 5 expected entries is missing.
    pub fn deltanet_q4k_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); DELTANET_TENSORS_PER_LAYER]> {
        let arr = self.storage.deltanet_q4k_layer_data(layer)?;
        let mut out: [(&[u8], &str); DELTANET_TENSORS_PER_LAYER] =
            [(&[], ""); DELTANET_TENSORS_PER_LAYER];
        for i in 0..DELTANET_TENSORS_PER_LAYER {
            let (view, fmt) = arr[i];
            out[i] = (view.as_slice(), fmt);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vindex_with_deltanet(payload: &[u8], manifest: serde_json::Value) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(DELTANET_WEIGHTS_Q4K_BIN), payload).unwrap();
        std::fs::write(
            tmp.path().join(DELTANET_WEIGHTS_Q4K_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        tmp
    }

    fn empty_vindex() -> VectorIndex {
        VectorIndex::empty(1, 2560)
    }

    /// Missing `deltanet_weights_q4k.bin` SHALL be a clean no-op: every
    /// vindex predating PR #147 (and every non-hybrid arch) lacks the
    /// file, and the loader must succeed without writing storage.
    #[test]
    fn load_deltanet_q4k_is_noop_when_file_absent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut idx = empty_vindex();
        idx.load_deltanet_q4k(tmp.path())
            .expect("missing file must be a clean no-op");
        assert!(!idx.has_deltanet_q4k());
        assert!(idx.deltanet_q4k_layer_data(0).is_none());
    }

    /// A linear layer with all 5 expected tensors must surface as
    /// `Some([...; 5])` in fixed-order. A neighbouring full-attention
    /// layer (with no entries) must return `None`. A layer with a
    /// missing tensor must also return `None` — partial layers would
    /// mislead the forward.
    #[test]
    fn deltanet_q4k_layer_data_returns_5_entries_for_complete_layer() {
        let one_tensor = 144;
        let total = one_tensor * 5;
        let payload = vec![0u8; total];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.qkv_proj.weight",
                "shape": [8192, 2048],
                "format": "Q4_K",
                "offset": 0,
                "length": one_tensor,
            },
            {
                "key": "layers.0.attn_gate.weight",
                "shape": [4096, 2048],
                "format": "Q4_K",
                "offset": one_tensor,
                "length": one_tensor,
            },
            {
                "key": "layers.0.ssm_alpha.weight",
                "shape": [32, 2048],
                "format": "Q4_K",
                "offset": one_tensor * 2,
                "length": one_tensor,
            },
            {
                "key": "layers.0.ssm_beta.weight",
                "shape": [32, 2048],
                "format": "Q4_K",
                "offset": one_tensor * 3,
                "length": one_tensor,
            },
            {
                "key": "layers.0.ssm_out.weight",
                "shape": [2048, 4096],
                "format": "Q4_K",
                "offset": one_tensor * 4,
                "length": one_tensor,
            },
            // Layer 1 — only the qkv tensor. Partial → accessor returns None.
            {
                "key": "layers.1.self_attn.qkv_proj.weight",
                "shape": [8192, 2048],
                "format": "Q4_K",
                "offset": 0,
                "length": one_tensor,
            },
        ]);
        let tmp = make_vindex_with_deltanet(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_deltanet_q4k(tmp.path())
            .expect("loader must succeed");

        assert!(idx.has_deltanet_q4k());
        let layer0 = idx.deltanet_q4k_layer_data(0).expect("layer 0 complete");
        assert_eq!(layer0.len(), DELTANET_TENSORS_PER_LAYER);
        assert!(layer0.iter().all(|(_, fmt)| *fmt == "Q4_K"));

        assert!(
            idx.deltanet_q4k_layer_data(1).is_none(),
            "partial layer must not be returned — would mislead the forward"
        );
        assert!(
            idx.deltanet_q4k_layer_data(99).is_none(),
            "out-of-range layer must be None"
        );
    }

    /// Missing `format` field in a manifest entry must surface as an
    /// error at load time, not silent default to Q4_K. Mirrors the
    /// attn loader's ROADMAP-P0 contract.
    #[test]
    fn load_deltanet_q4k_rejects_manifest_without_format() {
        let payload = vec![0u8; 144];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.qkv_proj.weight",
                "shape": [8192, 2048],
                "offset": 0,
                "length": 144,
            }
        ]);
        let tmp = make_vindex_with_deltanet(&payload, manifest);
        let mut idx = empty_vindex();
        let err = idx
            .load_deltanet_q4k(tmp.path())
            .expect_err("missing `format` must surface");
        let msg = format!("{err:?}");
        assert!(msg.contains("format"), "{msg}");
    }
}
