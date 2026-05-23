//! DSv4 hyperparameter struct + storage constructor from pre-dequantized
//! tensor buffers.
//!
//! This is the shape-conversion layer between "I've got a `Vec<f32>`
//! for each tensor name" and "I want a [`DsV4LayerWeightStorage`]".
//! GGUF byte-reading (i.e. opening a file, finding tensor by name, and
//! running dequantize) is the next stage; that produces the
//! `HashMap<DsV4TensorKind, Vec<f32>>` this module's constructor
//! consumes.
//!
//! Separating "convert bytes to f32" (Stage 8h-2d) from "shape into
//! typed storage" (this stage) keeps both pieces independently testable
//! and makes synthetic-tensor unit tests trivial (no GGUF file needed).

use std::collections::HashMap;

use ndarray::{Array1, Array2, Array3};

use larql_models::architectures::deepseek_v4_tensors::DsV4TensorKind;

use super::dsv4_attn_block::DsV4AttnBlockParams;
use super::dsv4_compressor_prefill::CompressorParams;
use super::dsv4_indexer::IndexerParams;
use super::dsv4_rope_tail::DsV4RopeMode;
use super::dsv4_storage::{CompressorStorage, DsV4LayerWeightStorage, IndexerStorage};

/// DSv4 model-wide hyperparameters. Set once for the whole model (no
/// per-layer variation except `compress_ratio`, which is handed to the
/// loader separately).
#[derive(Clone, Copy, Debug)]
pub struct DsV4Hyperparams {
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub n_groups: usize,
    pub o_lora_rank: usize,
    pub n_rot: usize,
    pub rope_base: f64,
    pub rope_mode: DsV4RopeMode,
    pub window_size: usize,
    pub norm_eps: f32,
    /// `Some(_)` iff the model has an indexer (at least one layer with
    /// `compress_ratio == 4`).
    pub indexer_head_size: Option<usize>,
    pub n_index_head: Option<usize>,
    pub top_k: Option<usize>,
}

impl DsV4Hyperparams {
    fn attn_params(&self) -> DsV4AttnBlockParams {
        DsV4AttnBlockParams {
            n_embd: self.n_embd,
            n_head: self.n_head,
            head_dim: self.head_dim,
            q_lora_rank: self.q_lora_rank,
            n_groups: self.n_groups,
            o_lora_rank: self.o_lora_rank,
            n_rot: self.n_rot,
            rope_base: self.rope_base,
            rope_mode: self.rope_mode,
            window_size: self.window_size,
            norm_eps: self.norm_eps,
            yarn: None,
        }
    }
}

/// Error returned by [`build_layer_storage`].
#[derive(Debug)]
pub enum DsV4BuildError {
    /// A required tensor for the variant wasn't in the input map.
    MissingTensor(DsV4TensorKind),
    /// The tensor's element count doesn't match the expected shape.
    ShapeMismatch {
        kind: DsV4TensorKind,
        expected: Vec<usize>,
        got_elems: usize,
    },
    /// Indexer variant requested but indexer hyperparameters weren't set.
    MissingIndexerHyperparams,
    /// `compress_ratio == 4` requires indexer support.
    IndexerRequiredForCompressRatio4,
}

impl std::fmt::Display for DsV4BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4BuildError::MissingTensor(k) => write!(f, "missing tensor {k}"),
            DsV4BuildError::ShapeMismatch {
                kind,
                expected,
                got_elems,
            } => write!(
                f,
                "shape mismatch for {kind}: expected {expected:?} (= {} elements), got {got_elems}",
                expected.iter().product::<usize>()
            ),
            DsV4BuildError::MissingIndexerHyperparams => {
                write!(f, "indexer hyperparams (head_size, n_head, top_k) required")
            }
            DsV4BuildError::IndexerRequiredForCompressRatio4 => {
                write!(f, "compress_ratio == 4 requires indexer support")
            }
        }
    }
}

impl std::error::Error for DsV4BuildError {}

/// Shape a pre-dequantized tensor map into a [`DsV4LayerWeightStorage`].
///
/// - `tensors`: per-tensor-kind dequantized `Vec<f32>` (row-major).
/// - `hp`: model-wide hyperparameters.
/// - `compress_ratio`: the per-layer compress_ratio (0 = no compress,
///   4 = indexer, other positive = no-indexer compress).
///
/// Validates that every tensor required for the variant is present
/// and has the expected element count. Missing tensors → error;
/// off-by-one shapes → error.
pub fn build_layer_storage(
    tensors: HashMap<DsV4TensorKind, Vec<f32>>,
    hp: &DsV4Hyperparams,
    compress_ratio: usize,
) -> Result<DsV4LayerWeightStorage, DsV4BuildError> {
    // ── Main attention (always required) ──
    let attn_norm = take_vec(&tensors, DsV4TensorKind::AttnNorm, &[hp.n_embd])?;
    let wq_a = take_2d(&tensors, DsV4TensorKind::AttnQA, hp.q_lora_rank, hp.n_embd)?;
    let q_a_norm = take_vec(&tensors, DsV4TensorKind::AttnQANorm, &[hp.q_lora_rank])?;
    let wq_b = take_2d(
        &tensors,
        DsV4TensorKind::AttnQB,
        hp.n_head * hp.head_dim,
        hp.q_lora_rank,
    )?;
    let wkv = take_2d(&tensors, DsV4TensorKind::AttnKv, hp.head_dim, hp.n_embd)?;
    let kv_a_norm = take_vec(&tensors, DsV4TensorKind::AttnKvANorm, &[hp.head_dim])?;
    let attn_sinks = tensors
        .get(&DsV4TensorKind::AttnSinks)
        .map(|v| Array1::from(v.clone()));

    let group_heads = hp.n_head / hp.n_groups;
    let group_dim = hp.head_dim * group_heads;
    let low_dim = hp.o_lora_rank * hp.n_groups;
    let wo_a = take_3d(
        &tensors,
        DsV4TensorKind::AttnOutA,
        hp.n_groups,
        hp.o_lora_rank,
        group_dim,
    )?;
    let wo_b = take_2d(&tensors, DsV4TensorKind::AttnOutB, hp.n_embd, low_dim)?;

    let mut storage = DsV4LayerWeightStorage {
        attn_norm,
        wq_a,
        q_a_norm,
        wq_b,
        wkv,
        kv_a_norm,
        attn_sinks,
        wo_a,
        wo_b,
        attn_params: hp.attn_params(),
        compressor: None,
        compressor_params: None,
        indexer: None,
        indexer_compressor_params: None,
        indexer_params: None,
        top_k: None,
    };

    if compress_ratio == 0 {
        return Ok(storage);
    }

    // ── Compressor ──
    let coff = if compress_ratio == 4 { 2 } else { 1 };
    let n_kv = coff * hp.head_dim;
    storage.compressor = Some(CompressorStorage {
        wkv: take_2d(&tensors, DsV4TensorKind::AttnCompressorKv, n_kv, hp.n_embd)?,
        wgate: take_2d(
            &tensors,
            DsV4TensorKind::AttnCompressorGate,
            n_kv,
            hp.n_embd,
        )?,
        ape: take_2d(
            &tensors,
            DsV4TensorKind::AttnCompressorApe,
            compress_ratio,
            n_kv,
        )?,
        norm: take_vec(&tensors, DsV4TensorKind::AttnCompressorNorm, &[hp.head_dim])?,
    });
    storage.compressor_params = Some(CompressorParams {
        head_dim: hp.head_dim,
        n_embd: hp.n_embd,
        compress_ratio,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode: hp.rope_mode,
        norm_eps: hp.norm_eps,
    });

    if compress_ratio != 4 {
        return Ok(storage);
    }

    // ── Indexer (compress_ratio == 4 only) ──
    let ihead = hp
        .indexer_head_size
        .ok_or(DsV4BuildError::MissingIndexerHyperparams)?;
    let inh = hp
        .n_index_head
        .ok_or(DsV4BuildError::MissingIndexerHyperparams)?;
    let topk = hp.top_k.ok_or(DsV4BuildError::MissingIndexerHyperparams)?;

    let idx_n_kv = 2 * ihead;
    let idx_comp = CompressorStorage {
        wkv: take_2d(
            &tensors,
            DsV4TensorKind::IndexerCompressorKv,
            idx_n_kv,
            hp.n_embd,
        )?,
        wgate: take_2d(
            &tensors,
            DsV4TensorKind::IndexerCompressorGate,
            idx_n_kv,
            hp.n_embd,
        )?,
        ape: take_2d(
            &tensors,
            DsV4TensorKind::IndexerCompressorApe,
            compress_ratio,
            idx_n_kv,
        )?,
        norm: take_vec(&tensors, DsV4TensorKind::IndexerCompressorNorm, &[ihead])?,
    };
    let idx_wq_b = take_2d(
        &tensors,
        DsV4TensorKind::IndexerAttnQB,
        inh * ihead,
        hp.q_lora_rank,
    )?;
    let idx_wproj = take_2d(&tensors, DsV4TensorKind::IndexerProj, inh, hp.n_embd)?;
    storage.indexer = Some(IndexerStorage {
        compressor: idx_comp,
        wq_b: idx_wq_b,
        wproj: idx_wproj,
    });
    storage.indexer_compressor_params = Some(CompressorParams {
        head_dim: ihead,
        n_embd: hp.n_embd,
        compress_ratio,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode: hp.rope_mode,
        norm_eps: hp.norm_eps,
    });
    storage.indexer_params = Some(IndexerParams {
        n_embd: hp.n_embd,
        q_lora_rank: hp.q_lora_rank,
        n_index_head: inh,
        n_index_head_size: ihead,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode: hp.rope_mode,
    });
    storage.top_k = Some(topk);

    Ok(storage)
}

// ── Helpers ───────────────────────────────────────────────────────────

fn take_vec(
    tensors: &HashMap<DsV4TensorKind, Vec<f32>>,
    kind: DsV4TensorKind,
    expected: &[usize],
) -> Result<Vec<f32>, DsV4BuildError> {
    let v = tensors
        .get(&kind)
        .cloned()
        .ok_or(DsV4BuildError::MissingTensor(kind))?;
    let want: usize = expected.iter().product();
    if v.len() != want {
        return Err(DsV4BuildError::ShapeMismatch {
            kind,
            expected: expected.to_vec(),
            got_elems: v.len(),
        });
    }
    Ok(v)
}

fn take_2d(
    tensors: &HashMap<DsV4TensorKind, Vec<f32>>,
    kind: DsV4TensorKind,
    dim0: usize,
    dim1: usize,
) -> Result<Array2<f32>, DsV4BuildError> {
    let v = take_vec(tensors, kind, &[dim0, dim1])?;
    Ok(Array2::from_shape_vec((dim0, dim1), v).expect("shape verified by take_vec"))
}

fn take_3d(
    tensors: &HashMap<DsV4TensorKind, Vec<f32>>,
    kind: DsV4TensorKind,
    dim0: usize,
    dim1: usize,
    dim2: usize,
) -> Result<Array3<f32>, DsV4BuildError> {
    let v = take_vec(tensors, kind, &[dim0, dim1, dim2])?;
    Ok(Array3::from_shape_vec((dim0, dim1, dim2), v).expect("shape verified by take_vec"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_hp() -> DsV4Hyperparams {
        DsV4Hyperparams {
            n_embd: 64,
            n_head: 4,
            head_dim: 64,
            q_lora_rank: 16,
            n_groups: 2,
            o_lora_rank: 8,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            indexer_head_size: None,
            n_index_head: None,
            top_k: None,
        }
    }

    /// Build a minimal valid tensor map for the no-compress variant.
    fn no_compress_tensors(hp: &DsV4Hyperparams) -> HashMap<DsV4TensorKind, Vec<f32>> {
        let group_heads = hp.n_head / hp.n_groups;
        let group_dim = hp.head_dim * group_heads;
        let low_dim = hp.o_lora_rank * hp.n_groups;
        let mut m = HashMap::new();
        m.insert(DsV4TensorKind::AttnNorm, vec![1.0; hp.n_embd]);
        m.insert(
            DsV4TensorKind::AttnQA,
            vec![0.01; hp.q_lora_rank * hp.n_embd],
        );
        m.insert(DsV4TensorKind::AttnQANorm, vec![1.0; hp.q_lora_rank]);
        m.insert(
            DsV4TensorKind::AttnQB,
            vec![0.01; hp.n_head * hp.head_dim * hp.q_lora_rank],
        );
        m.insert(DsV4TensorKind::AttnKv, vec![0.01; hp.head_dim * hp.n_embd]);
        m.insert(DsV4TensorKind::AttnKvANorm, vec![1.0; hp.head_dim]);
        m.insert(DsV4TensorKind::AttnSinks, vec![-1.0; hp.n_head]);
        m.insert(
            DsV4TensorKind::AttnOutA,
            vec![0.01; hp.n_groups * hp.o_lora_rank * group_dim],
        );
        m.insert(DsV4TensorKind::AttnOutB, vec![0.01; hp.n_embd * low_dim]);
        m
    }

    #[test]
    fn build_no_compress_storage() {
        let hp = base_hp();
        let tensors = no_compress_tensors(&hp);
        let storage = build_layer_storage(tensors, &hp, 0).expect("no-compress build");
        assert!(storage.compressor.is_none());
        assert!(storage.indexer.is_none());
        assert_eq!(storage.attn_norm.len(), hp.n_embd);
        assert_eq!(storage.wq_a.shape(), &[hp.q_lora_rank, hp.n_embd]);
    }

    #[test]
    fn build_compress_storage() {
        let hp = base_hp();
        let mut tensors = no_compress_tensors(&hp);
        // Compressor with compress_ratio=2 → coff=1.
        let compress_ratio = 2usize;
        let n_kv = hp.head_dim;
        tensors.insert(
            DsV4TensorKind::AttnCompressorKv,
            vec![0.01; n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorGate,
            vec![0.01; n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorApe,
            vec![0.01; compress_ratio * n_kv],
        );
        tensors.insert(DsV4TensorKind::AttnCompressorNorm, vec![1.0; hp.head_dim]);
        let storage = build_layer_storage(tensors, &hp, compress_ratio).expect("compress build");
        assert!(storage.compressor.is_some());
        assert!(storage.indexer.is_none());
        assert_eq!(
            storage.compressor_params.unwrap().compress_ratio,
            compress_ratio
        );
    }

    #[test]
    fn build_indexer_storage() {
        let mut hp = base_hp();
        hp.indexer_head_size = Some(16);
        hp.n_index_head = Some(2);
        hp.top_k = Some(2);
        let ihead = 16usize;
        let inh = 2usize;
        let compress_ratio = 4usize;

        let mut tensors = no_compress_tensors(&hp);
        // Main compressor (coff=2).
        let main_n_kv = 2 * hp.head_dim;
        tensors.insert(
            DsV4TensorKind::AttnCompressorKv,
            vec![0.01; main_n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorGate,
            vec![0.01; main_n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorApe,
            vec![0.01; compress_ratio * main_n_kv],
        );
        tensors.insert(DsV4TensorKind::AttnCompressorNorm, vec![1.0; hp.head_dim]);
        // Indexer compressor (coff=2).
        let idx_n_kv = 2 * ihead;
        tensors.insert(
            DsV4TensorKind::IndexerCompressorKv,
            vec![0.01; idx_n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::IndexerCompressorGate,
            vec![0.01; idx_n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::IndexerCompressorApe,
            vec![0.01; compress_ratio * idx_n_kv],
        );
        tensors.insert(DsV4TensorKind::IndexerCompressorNorm, vec![1.0; ihead]);
        // Indexer score weights.
        tensors.insert(
            DsV4TensorKind::IndexerAttnQB,
            vec![0.01; inh * ihead * hp.q_lora_rank],
        );
        tensors.insert(DsV4TensorKind::IndexerProj, vec![0.01; inh * hp.n_embd]);

        let storage = build_layer_storage(tensors, &hp, compress_ratio).expect("indexer build");
        assert!(storage.compressor.is_some());
        assert!(storage.indexer.is_some());
        assert_eq!(storage.top_k, Some(2));
    }

    #[test]
    fn missing_tensor_errors() {
        let hp = base_hp();
        let mut tensors = no_compress_tensors(&hp);
        tensors.remove(&DsV4TensorKind::AttnQB);
        let err = match build_layer_storage(tensors, &hp, 0) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            DsV4BuildError::MissingTensor(k) => assert_eq!(k, DsV4TensorKind::AttnQB),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn shape_mismatch_errors() {
        let hp = base_hp();
        let mut tensors = no_compress_tensors(&hp);
        // Wrong size for AttnNorm (too short).
        tensors.insert(DsV4TensorKind::AttnNorm, vec![1.0; 7]);
        let err = match build_layer_storage(tensors, &hp, 0) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            DsV4BuildError::ShapeMismatch {
                kind, got_elems, ..
            } => {
                assert_eq!(kind, DsV4TensorKind::AttnNorm);
                assert_eq!(got_elems, 7);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn compress_ratio_4_requires_indexer_hyperparams() {
        let hp = base_hp(); // no indexer hp set
        let mut tensors = no_compress_tensors(&hp);
        // Add main compressor only.
        let n_kv = 2 * hp.head_dim;
        let compress_ratio = 4;
        tensors.insert(
            DsV4TensorKind::AttnCompressorKv,
            vec![0.01; n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorGate,
            vec![0.01; n_kv * hp.n_embd],
        );
        tensors.insert(
            DsV4TensorKind::AttnCompressorApe,
            vec![0.01; compress_ratio * n_kv],
        );
        tensors.insert(DsV4TensorKind::AttnCompressorNorm, vec![1.0; hp.head_dim]);
        let err = match build_layer_storage(tensors, &hp, compress_ratio) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            DsV4BuildError::MissingIndexerHyperparams => {}
            other => panic!("wrong error: {other:?}"),
        }
    }
}
