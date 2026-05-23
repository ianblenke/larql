//! Owned storage for per-layer DSv4 weights + borrowed-view methods.
//!
//! The attention blocks built in Stages 8a, 8f, 8g consume borrowed
//! weight structs (`DsV4AttnBlockWeights<'a>` etc.) whose lifetimes
//! tie back to f32 buffers somewhere. This module owns those buffers
//! per layer and exposes `as_*_weights()` methods that hand out the
//! matching borrowed view.
//!
//! Sub-structs:
//! - [`CompressorStorage`] — owned weights for the compressor (used
//!   by the HCA paths).
//! - [`IndexerStorage`] — owned weights for the indexer's secondary
//!   compressor + Q-up/score-proj.
//! - [`DsV4LayerWeightStorage`] — full per-layer container.
//!
//! GGUF loading (Stage 8h-2c) populates these from a real DSv4 GGUF.
//! For now the only public constructor is [`DsV4LayerWeightStorage::
//! synthetic`], used by tests to verify the storage→view contract.

use ndarray::{Array1, Array2, Array3};

use super::dsv4_attn_block::{DsV4AttnBlockParams, DsV4AttnBlockWeights};
use super::dsv4_attn_block_compress::{DsV4AttnBlockCompressParams, DsV4AttnBlockCompressWeights};
use super::dsv4_attn_block_indexer::{DsV4AttnBlockIndexerParams, DsV4AttnBlockIndexerWeights};
use super::dsv4_attn_dispatch::DsV4AttnLayer;
use super::dsv4_compressor_prefill::{CompressorParams, CompressorWeights};
use super::dsv4_indexer::{IndexerParams, IndexerWeights};

/// Owned weights for the DSv4 compressor (a per-layer HCA producer).
#[derive(Clone)]
pub struct CompressorStorage {
    pub wkv: Array2<f32>,
    pub wgate: Array2<f32>,
    pub ape: Array2<f32>,
    pub norm: Vec<f32>,
}

impl CompressorStorage {
    pub fn as_weights(&self) -> CompressorWeights<'_> {
        CompressorWeights {
            wkv: self.wkv.view(),
            wgate: self.wgate.view(),
            ape: self.ape.view(),
            norm: &self.norm,
        }
    }
}

/// Owned weights for the DSv4 indexer (a per-layer top-k selector for
/// the compressed positions). Includes its own smaller compressor plus
/// the Q-up + score-projection matrices.
#[derive(Clone)]
pub struct IndexerStorage {
    pub compressor: CompressorStorage,
    pub wq_b: Array2<f32>,
    pub wproj: Array2<f32>,
}

impl IndexerStorage {
    pub fn as_compressor_weights(&self) -> CompressorWeights<'_> {
        self.compressor.as_weights()
    }
    pub fn as_indexer_weights(&self) -> IndexerWeights<'_> {
        IndexerWeights {
            wq_b: self.wq_b.view(),
            wproj: self.wproj.view(),
        }
    }
}

/// All per-layer DSv4 attention-side weights (FFN side will live in a
/// sibling struct in Stage 8h-3). Owns the f32 buffers; emits borrowed
/// views via `as_*_weights()`.
///
/// Variant: the presence of `compressor` and `indexer` determines
/// which dispatcher arm the caller should build:
/// - `compressor.is_none()` → NoCompress (Stage 8a)
/// - `compressor.is_some() && indexer.is_none()` → Compress (Stage 8f)
/// - both → Indexer (Stage 8g)
#[derive(Clone)]
pub struct DsV4LayerWeightStorage {
    // ── Main attention ──
    pub attn_norm: Vec<f32>,
    pub wq_a: Array2<f32>,
    pub q_a_norm: Vec<f32>,
    pub wq_b: Array2<f32>,
    pub wkv: Array2<f32>,
    pub kv_a_norm: Vec<f32>,
    pub attn_sinks: Option<Array1<f32>>,
    pub wo_a: Array3<f32>,
    pub wo_b: Array2<f32>,
    pub attn_params: DsV4AttnBlockParams,
    // ── Optional HCA pieces ──
    pub compressor: Option<CompressorStorage>,
    pub compressor_params: Option<CompressorParams>,
    pub indexer: Option<IndexerStorage>,
    pub indexer_compressor_params: Option<CompressorParams>,
    pub indexer_params: Option<IndexerParams>,
    pub top_k: Option<usize>,
}

impl DsV4LayerWeightStorage {
    /// Borrow as the no-compress (compress_ratio=0) weight struct.
    /// Always valid — the main attention weights are present in every
    /// variant.
    pub fn as_attn_weights(&self) -> DsV4AttnBlockWeights<'_> {
        DsV4AttnBlockWeights {
            attn_norm: &self.attn_norm,
            wq_a: self.wq_a.view(),
            q_a_norm: &self.q_a_norm,
            wq_b: self.wq_b.view(),
            wkv: self.wkv.view(),
            kv_a_norm: &self.kv_a_norm,
            wo_a: self.wo_a.view(),
            wo_b: self.wo_b.view(),
            attn_sinks: self.attn_sinks.as_ref().map(|a| a.view()),
        }
    }

    /// Borrow as the HCA-no-indexer weight struct. Returns `None` if
    /// `compressor` is absent.
    pub fn as_compress_weights(&self) -> Option<DsV4AttnBlockCompressWeights<'_>> {
        let comp = self.compressor.as_ref()?;
        Some(DsV4AttnBlockCompressWeights {
            attn: self.as_attn_weights(),
            compressor: comp.as_weights(),
        })
    }

    /// Borrow as the HCA-with-indexer weight struct. Returns `None`
    /// if `indexer` is absent (or `compressor` is absent).
    pub fn as_indexer_weights(&self) -> Option<DsV4AttnBlockIndexerWeights<'_>> {
        let comp = self.compressor.as_ref()?;
        let idx = self.indexer.as_ref()?;
        Some(DsV4AttnBlockIndexerWeights {
            attn: self.as_attn_weights(),
            compressor: comp.as_weights(),
            indexer_compressor: idx.as_compressor_weights(),
            indexer: idx.as_indexer_weights(),
        })
    }

    /// Build the dispatcher variant for this layer based on which
    /// optional pieces are populated. Borrowing carries the storage's
    /// lifetime so the result is valid as long as `self` is.
    pub fn dispatcher_layer<'a>(
        &'a self,
        compress_params: &'a Option<DsV4AttnBlockCompressParams>,
        indexer_params: &'a Option<DsV4AttnBlockIndexerParams>,
    ) -> DsV4AttnLayer<'a, 'a> {
        if let (Some(idx_w), Some(p)) = (self.as_indexer_weights(), indexer_params) {
            DsV4AttnLayer::Indexer {
                weights: idx_w,
                params: p,
            }
        } else if let (Some(c_w), Some(p)) = (self.as_compress_weights(), compress_params) {
            DsV4AttnLayer::Compress {
                weights: c_w,
                params: p,
            }
        } else {
            DsV4AttnLayer::NoCompress {
                weights: self.as_attn_weights(),
                params: &self.attn_params,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_attn_dispatch::dsv4_attn_layer;
    use super::super::dsv4_rope_tail::DsV4RopeMode;
    use super::*;

    /// Common helper: build a no-compress storage with synthetic but
    /// shape-correct weights for the spec-shaped block in the unit tests.
    fn synthetic_no_compress(n_embd: usize) -> DsV4LayerWeightStorage {
        let n_head = 4;
        let head_dim = 64;
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;
        let attn_params = DsV4AttnBlockParams {
            n_embd,
            n_head,
            head_dim,
            q_lora_rank,
            n_groups,
            o_lora_rank,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            yarn: None,
        };
        DsV4LayerWeightStorage {
            attn_norm: vec![1.0; n_embd],
            wq_a: Array2::<f32>::from_shape_fn((q_lora_rank, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.01).sin()
            }),
            q_a_norm: vec![1.0; q_lora_rank],
            wq_b: Array2::<f32>::from_shape_fn((n_head * head_dim, q_lora_rank), |(i, j)| {
                ((i + j) as f32 * 0.013).cos() * 0.1
            }),
            wkv: Array2::<f32>::from_shape_fn((head_dim, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.007).sin() * 0.05
            }),
            kv_a_norm: vec![1.0; head_dim],
            attn_sinks: Some(Array1::<f32>::from_elem(n_head, -1.0)),
            wo_a: Array3::<f32>::from_shape_fn((n_groups, o_lora_rank, group_dim), |(g, r, j)| {
                ((g + r + j) as f32 * 0.005).cos() * 0.05
            }),
            wo_b: Array2::<f32>::from_shape_fn((n_embd, low_dim), |(i, j)| {
                ((i + j) as f32 * 0.011).sin() * 0.1
            }),
            attn_params,
            compressor: None,
            compressor_params: None,
            indexer: None,
            indexer_compressor_params: None,
            indexer_params: None,
            top_k: None,
        }
    }

    fn add_compressor(storage: &mut DsV4LayerWeightStorage, compress_ratio: usize, coff: usize) {
        let head_dim = storage.attn_params.head_dim;
        let n_embd = storage.attn_params.n_embd;
        let n_kv = coff * head_dim;
        storage.compressor = Some(CompressorStorage {
            wkv: Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.013).cos() * 0.05
            }),
            wgate: Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.017).sin() * 0.05
            }),
            ape: Array2::<f32>::from_shape_fn((compress_ratio, n_kv), |(r, k)| {
                ((r + k) as f32 * 0.03).sin() * 0.05
            }),
            norm: vec![1.0; head_dim],
        });
        storage.compressor_params = Some(CompressorParams {
            head_dim,
            n_embd,
            compress_ratio,
            n_rot: storage.attn_params.n_rot,
            rope_base: storage.attn_params.rope_base,
            rope_mode: storage.attn_params.rope_mode,
            norm_eps: storage.attn_params.norm_eps,
        });
    }

    fn add_indexer(
        storage: &mut DsV4LayerWeightStorage,
        compress_ratio: usize,
        indexer_head_size: usize,
        n_index_head: usize,
        top_k: usize,
    ) {
        let n_embd = storage.attn_params.n_embd;
        let q_lora_rank = storage.attn_params.q_lora_rank;
        let idx_n_kv = 2 * indexer_head_size; // coff=2 for compress_ratio=4
        let idx_comp = CompressorStorage {
            wkv: Array2::<f32>::from_shape_fn((idx_n_kv, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.014).cos() * 0.05
            }),
            wgate: Array2::<f32>::from_shape_fn((idx_n_kv, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.016).sin() * 0.05
            }),
            ape: Array2::<f32>::from_shape_fn((compress_ratio, idx_n_kv), |(r, k)| {
                ((r + k) as f32 * 0.025).sin() * 0.05
            }),
            norm: vec![1.0; indexer_head_size],
        };
        storage.indexer = Some(IndexerStorage {
            compressor: idx_comp,
            wq_b: Array2::<f32>::from_shape_fn(
                (n_index_head * indexer_head_size, q_lora_rank),
                |(i, j)| ((i + j) as f32 * 0.011).sin() * 0.1,
            ),
            wproj: Array2::<f32>::from_shape_fn((n_index_head, n_embd), |(i, j)| {
                ((i + j) as f32 * 0.012).cos() * 0.1
            }),
        });
        storage.indexer_compressor_params = Some(CompressorParams {
            head_dim: indexer_head_size,
            n_embd,
            compress_ratio,
            n_rot: storage.attn_params.n_rot,
            rope_base: storage.attn_params.rope_base,
            rope_mode: storage.attn_params.rope_mode,
            norm_eps: storage.attn_params.norm_eps,
        });
        storage.indexer_params = Some(IndexerParams {
            n_embd,
            q_lora_rank,
            n_index_head,
            n_index_head_size: indexer_head_size,
            n_rot: storage.attn_params.n_rot,
            rope_base: storage.attn_params.rope_base,
            rope_mode: storage.attn_params.rope_mode,
        });
        storage.top_k = Some(top_k);
    }

    /// Storage's view → dispatcher path produces finite output for
    /// the no-compress variant.
    #[test]
    fn storage_no_compress_dispatches_correctly() {
        let n_embd = 64;
        let storage = synthetic_no_compress(n_embd);
        let layer = storage.dispatcher_layer(&None, &None);
        assert_eq!(layer.variant_name(), "no_compress");

        let x =
            Array2::<f32>::from_shape_fn((6, n_embd), |(t, d)| ((t * 7 + d) as f32 * 0.013).sin());
        let out = dsv4_attn_layer(x.view(), &layer, 0);
        assert_eq!(out.shape(), &[6, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Storage with compressor populated → Compress variant.
    #[test]
    fn storage_with_compressor_dispatches_compress() {
        let n_embd = 64;
        let mut storage = synthetic_no_compress(n_embd);
        add_compressor(&mut storage, 2, 1);

        let compress_params = Some(DsV4AttnBlockCompressParams {
            attn: storage.attn_params,
            compressor: storage.compressor_params.unwrap(),
        });
        let layer = storage.dispatcher_layer(&compress_params, &None);
        assert_eq!(layer.variant_name(), "compress");
        assert_eq!(layer.compress_ratio(), 2);

        let x =
            Array2::<f32>::from_shape_fn((8, n_embd), |(t, d)| ((t * 7 + d) as f32 * 0.013).sin());
        let out = dsv4_attn_layer(x.view(), &layer, 0);
        assert_eq!(out.shape(), &[8, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Storage with both compressor + indexer → Indexer variant.
    #[test]
    fn storage_with_indexer_dispatches_indexer() {
        let n_embd = 64;
        let mut storage = synthetic_no_compress(n_embd);
        add_compressor(&mut storage, 4, 2);
        add_indexer(&mut storage, 4, 16, 2, 2);

        let compress_params = Some(DsV4AttnBlockCompressParams {
            attn: storage.attn_params,
            compressor: storage.compressor_params.unwrap(),
        });
        let indexer_params = Some(DsV4AttnBlockIndexerParams {
            attn: storage.attn_params,
            compressor: storage.compressor_params.unwrap(),
            indexer_compressor: storage.indexer_compressor_params.unwrap(),
            indexer: storage.indexer_params.unwrap(),
            top_k: storage.top_k.unwrap(),
        });
        let layer = storage.dispatcher_layer(&compress_params, &indexer_params);
        assert_eq!(layer.variant_name(), "indexer");
        assert_eq!(layer.compress_ratio(), 4);

        let x =
            Array2::<f32>::from_shape_fn((16, n_embd), |(t, d)| ((t * 7 + d) as f32 * 0.013).sin());
        let out = dsv4_attn_layer(x.view(), &layer, 0);
        assert_eq!(out.shape(), &[16, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Variant selection priority: indexer beats compress beats
    /// no-compress when params are present.
    #[test]
    fn dispatcher_priority_indexer_over_compress_over_nocompress() {
        let n_embd = 64;
        let storage = synthetic_no_compress(n_embd);

        // Without compressor or indexer params → NoCompress.
        let l = storage.dispatcher_layer(&None, &None);
        assert_eq!(l.variant_name(), "no_compress");

        // Even if we pass compress_params, no compressor in storage →
        // still NoCompress (since as_compress_weights returns None).
        let comp_p = DsV4AttnBlockCompressParams {
            attn: storage.attn_params,
            compressor: CompressorParams {
                head_dim: storage.attn_params.head_dim,
                n_embd,
                compress_ratio: 2,
                n_rot: 0,
                rope_base: 10000.0,
                rope_mode: DsV4RopeMode::Neox,
                norm_eps: 1e-5,
            },
        };
        let some_comp_p = Some(comp_p);
        let l2 = storage.dispatcher_layer(&some_comp_p, &None);
        assert_eq!(l2.variant_name(), "no_compress");
    }
}
