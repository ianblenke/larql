//! Full DSv4 GGUF → typed-storage loader.
//!
//! Composes:
//! - [`super::dsv4_hyperparams_load::DsV4Hyperparams::from_gguf`] (8h-4b-1)
//! - [`super::dsv4_layer_variants::detect_layer_variant`] (8h-4b-2)
//! - [`super::dsv4_gguf_reader::read_dsv4_layer_tensors_from_gguf`] (8h-2d)
//! - [`super::dsv4_storage_build::build_layer_storage`] (8h-2c)
//!
//! Each layer's variant (compress_ratio + has_indexer) is detected
//! from the GGUF tensors themselves; the loader then reads, dequantizes,
//! and shapes the per-layer tensors into the typed storage struct.

use larql_models::detect::ModelError;
use larql_models::loading::gguf::GgufFile;

use super::dsv4_gguf_reader::read_dsv4_layer_tensors_from_gguf;
use super::dsv4_hyperparams_load::DsV4MetadataError;
use super::dsv4_layer_variants::{detect_layer_variant, DsV4LayerVariant};
use super::dsv4_storage::DsV4LayerWeightStorage;
use super::dsv4_storage_build::{build_layer_storage, DsV4BuildError, DsV4Hyperparams};

/// Errors that can arise during full DSv4 loading.
#[derive(Debug)]
pub enum DsV4LoadError {
    /// GGUF metadata extraction failed (8h-4b-1).
    Metadata(DsV4MetadataError),
    /// Raw tensor I/O failed (8h-2d).
    TensorIo(ModelError),
    /// Per-layer storage construction failed (8h-2c).
    Build {
        layer_index: usize,
        cause: DsV4BuildError,
    },
}

impl std::fmt::Display for DsV4LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4LoadError::Metadata(e) => write!(f, "DSv4 metadata: {e}"),
            DsV4LoadError::TensorIo(e) => write!(f, "DSv4 tensor I/O: {e}"),
            DsV4LoadError::Build { layer_index, cause } => {
                write!(f, "DSv4 build layer {layer_index}: {cause}")
            }
        }
    }
}

impl std::error::Error for DsV4LoadError {}

impl From<DsV4MetadataError> for DsV4LoadError {
    fn from(e: DsV4MetadataError) -> Self {
        DsV4LoadError::Metadata(e)
    }
}

impl From<ModelError> for DsV4LoadError {
    fn from(e: ModelError) -> Self {
        DsV4LoadError::TensorIo(e)
    }
}

/// Load a single DSv4 layer into typed storage.
///
/// Detects the variant from the layer's tensors, reads all per-kind
/// f32 buffers, and constructs the [`DsV4LayerWeightStorage`].
pub fn load_dsv4_layer(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    layer_index: usize,
) -> Result<(DsV4LayerWeightStorage, DsV4LayerVariant), DsV4LoadError> {
    let variant = detect_layer_variant(gguf, layer_index, hp.head_dim);
    let raw = read_dsv4_layer_tensors_from_gguf(gguf, layer_index)?;
    let compress_ratio = variant.compress_ratio.unwrap_or(0);
    let storage = build_layer_storage(raw, hp, compress_ratio)
        .map_err(|cause| DsV4LoadError::Build { layer_index, cause })?;
    Ok((storage, variant))
}

/// Load every layer of a DSv4 model.
///
/// `n_layer` is the model's `block_count`. Returns one
/// `(storage, variant)` pair per layer in `0..n_layer`.
pub fn load_dsv4_layers(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    n_layer: usize,
) -> Result<Vec<(DsV4LayerWeightStorage, DsV4LayerVariant)>, DsV4LoadError> {
    let mut out = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        out.push(load_dsv4_layer(gguf, hp, l)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_layer_variants::AttnVariantTag;
    use super::*;

    /// End-to-end real-GGUF load of layer 0 (the cheapest layer —
    /// hash-routed, no compressor, no indexer). Verifies the full
    /// pipeline from GGUF bytes to typed storage works on the real
    /// 172 GB DSv4-Flash artifact.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn load_layer_0_from_real_gguf() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let (storage, variant) = load_dsv4_layer(&gguf, &hp, 0).expect("load layer 0");

        // Layer 0: hash routing, no compressor, no indexer.
        assert!(variant.uses_hash_routing);
        assert_eq!(variant.compress_ratio, None);
        assert!(!variant.has_indexer);
        assert_eq!(variant.attention_variant_tag(), AttnVariantTag::NoCompress);

        // Storage: main attention buffers all populated; HCA optional
        // pieces absent.
        assert_eq!(storage.attn_norm.len(), hp.n_embd);
        assert_eq!(storage.q_a_norm.len(), hp.q_lora_rank);
        assert_eq!(storage.kv_a_norm.len(), hp.head_dim);
        assert_eq!(storage.wq_a.shape(), &[hp.q_lora_rank, hp.n_embd]);
        assert_eq!(
            storage.wq_b.shape(),
            &[hp.n_head * hp.head_dim, hp.q_lora_rank]
        );
        assert_eq!(storage.wkv.shape(), &[hp.head_dim, hp.n_embd]);
        assert!(storage.attn_sinks.is_some());
        assert_eq!(storage.attn_sinks.as_ref().unwrap().len(), hp.n_head);
        // wo_a is (n_groups, o_lora_rank, group_dim).
        let group_dim = hp.head_dim * (hp.n_head / hp.n_groups);
        assert_eq!(
            storage.wo_a.shape(),
            &[hp.n_groups, hp.o_lora_rank, group_dim]
        );
        // wo_b is (n_embd, n_groups * o_lora_rank).
        assert_eq!(
            storage.wo_b.shape(),
            &[hp.n_embd, hp.n_groups * hp.o_lora_rank]
        );

        // HCA pieces all None.
        assert!(storage.compressor.is_none());
        assert!(storage.indexer.is_none());
        assert!(storage.compressor_params.is_none());
        assert!(storage.indexer_compressor_params.is_none());
        assert!(storage.indexer_params.is_none());
        assert!(storage.top_k.is_none());

        // Spot-check finiteness on a small subset.
        for &v in storage.attn_norm.iter().take(16) {
            assert!(v.is_finite(), "non-finite attn_norm");
        }
        for &v in storage.q_a_norm.iter().take(16) {
            assert!(v.is_finite(), "non-finite q_a_norm");
        }
    }

    /// Same, for layer 4 (the cheapest indexer + compressor layer):
    /// regular routing + compress_ratio=4 + indexer. This exercises
    /// the heaviest variant path.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn load_layer_4_indexer_path_from_real_gguf() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let (storage, variant) = load_dsv4_layer(&gguf, &hp, 4).expect("load layer 4");

        assert!(!variant.uses_hash_routing);
        assert_eq!(variant.compress_ratio, Some(4));
        assert!(variant.has_indexer);
        assert_eq!(variant.attention_variant_tag(), AttnVariantTag::Indexer);

        // Compressor + indexer storage populated.
        let comp = storage.compressor.as_ref().expect("compressor present");
        let coff = 2; // compress_ratio = 4
        assert_eq!(comp.wkv.shape(), &[coff * hp.head_dim, hp.n_embd]);
        assert_eq!(comp.norm.len(), hp.head_dim);

        let idx = storage.indexer.as_ref().expect("indexer present");
        let ihead = hp.indexer_head_size.unwrap();
        assert_eq!(idx.compressor.norm.len(), ihead);
        assert_eq!(
            idx.wq_b.shape(),
            &[hp.n_index_head.unwrap() * ihead, hp.q_lora_rank]
        );
        assert_eq!(idx.wproj.shape(), &[hp.n_index_head.unwrap(), hp.n_embd]);

        assert_eq!(storage.top_k, hp.top_k);
    }
}
