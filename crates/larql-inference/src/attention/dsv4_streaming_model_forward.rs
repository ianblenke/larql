//! Memory-bounded DSv4 model forward via per-layer streaming.
//!
//! The full 43-layer DSv4-Flash model holds ~1.1 TB of FFN weights in
//! aggregate (26 GB f32 per layer × 43), far beyond typical hardware
//! RAM. This module runs the model by loading one layer at a time,
//! running [`dsv4_per_layer_forward`] on the current residual, then
//! dropping the layer's storage before loading the next.
//!
//! Peak RAM stays at ~26 GB (one layer + the residual + the global
//! head) while the effective N-layer forward proceeds end-to-end.
//!
//! Use this as the production entry point for full-model forward on
//! real DSv4-Flash artifacts. The non-streaming
//! [`super::dsv4_model_forward::dsv4_model_forward`] is still useful
//! for small-N tests and synthetic-weight microbenchmarks where
//! holding all layers in memory at once is cheap.
//!
//! ## Per-layer dispatch params
//!
//! The attention dispatch arm (NoCompress / Compress / Indexer) is
//! chosen per-layer from the variant detected in the GGUF. This module
//! constructs the matching [`DsV4AttnBlockCompressParams`] /
//! [`DsV4AttnBlockIndexerParams`] from the model hyperparameters; the
//! caller doesn't have to pre-build per-variant params themselves.

use ndarray::{Array2, Array3};

use larql_models::loading::gguf::GgufFile;

use super::dsv4_attn_block::DsV4AttnBlockParams;
use super::dsv4_attn_block_compress::DsV4AttnBlockCompressParams;
use super::dsv4_attn_block_indexer::DsV4AttnBlockIndexerParams;
use super::dsv4_compressor_prefill::CompressorParams;
use super::dsv4_ffn_block::Dsv4FfnParams;
use super::dsv4_full_loader::{load_dsv4_layer, DsV4LoadError};
use super::dsv4_head_storage::DsV4HeadStorage;
use super::dsv4_indexer::IndexerParams;
use super::dsv4_layer_variants::DsV4LayerVariant;
use super::dsv4_mhc_bookend::MhcParams;
use super::dsv4_model_forward::{broadcast_to_streams, dsv4_hc_head, embed_tokens};
use super::dsv4_per_layer::{dsv4_per_layer_forward, DsV4LayerParams, DsV4LayerWeights};
use super::dsv4_rope_tail::DsV4RopeMode;
use super::dsv4_storage_build::DsV4Hyperparams;

/// Build the per-layer [`DsV4LayerParams`] (MoE + mHC scalar config)
/// from model-wide hyperparameters. The same params apply to every
/// layer in DSv4-Flash; per-layer variation is in the attention
/// dispatch arm only.
fn build_layer_params(hp: &DsV4Hyperparams) -> DsV4LayerParams {
    DsV4LayerParams {
        mhc: MhcParams {
            n_embd: hp.n_embd,
            n_hc: hp.n_hc,
            sinkhorn_iters: 2,
            hc_eps: 1e-5,
            norm_eps: hp.norm_eps,
        },
        ffn: Dsv4FfnParams {
            n_expert: hp.n_expert,
            n_expert_used: hp.n_expert_used,
            norm_eps: hp.norm_eps,
            routed_swiglu_limit: 7.0,
            shared_swiglu_limit: 0.0,
            expert_weights_norm: hp.expert_weights_norm,
            expert_weights_scale: hp.expert_weights_scale,
        },
    }
}

/// Build [`DsV4AttnBlockCompressParams`] for a given compress_ratio.
fn build_compress_params(
    hp: &DsV4Hyperparams,
    compress_ratio: usize,
) -> DsV4AttnBlockCompressParams {
    let attn = DsV4AttnBlockParams {
        n_embd: hp.n_embd,
        n_head: hp.n_head,
        head_dim: hp.head_dim,
        q_lora_rank: hp.q_lora_rank,
        n_groups: hp.n_groups,
        o_lora_rank: hp.o_lora_rank,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode: DsV4RopeMode::Neox,
        window_size: hp.window_size,
        norm_eps: hp.norm_eps,
        yarn: hp.yarn,
    };
    DsV4AttnBlockCompressParams {
        attn,
        compressor: CompressorParams {
            head_dim: hp.head_dim,
            n_embd: hp.n_embd,
            compress_ratio,
            n_rot: hp.n_rot,
            rope_base: hp.rope_base,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: hp.norm_eps,
        },
    }
}

/// Build [`DsV4AttnBlockIndexerParams`] for a given compress_ratio.
/// Requires `hp.indexer_head_size`, `n_index_head`, `top_k` to be set.
fn build_indexer_params(hp: &DsV4Hyperparams, compress_ratio: usize) -> DsV4AttnBlockIndexerParams {
    let cp = build_compress_params(hp, compress_ratio);
    DsV4AttnBlockIndexerParams {
        attn: cp.attn,
        compressor: cp.compressor,
        indexer_compressor: CompressorParams {
            head_dim: hp.indexer_head_size.expect("indexer_head_size set"),
            n_embd: hp.n_embd,
            compress_ratio,
            n_rot: hp.n_rot,
            rope_base: hp.rope_base,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: hp.norm_eps,
        },
        indexer: IndexerParams {
            n_embd: hp.n_embd,
            q_lora_rank: hp.q_lora_rank,
            n_index_head: hp.n_index_head.expect("n_index_head set"),
            n_index_head_size: hp.indexer_head_size.expect("indexer_head_size set"),
            n_rot: hp.n_rot,
            rope_base: hp.rope_base,
            rope_mode: DsV4RopeMode::Neox,
        },
        top_k: hp.top_k.expect("top_k set"),
    }
}

/// Pre-built per-variant dispatch params, owned by the caller for the
/// duration of the streaming forward.
///
/// The streaming loop borrows from these per-layer based on the
/// detected [`DsV4LayerVariant`], so the structs need to outlive every
/// per-layer iteration.
struct DispatchParamsPool {
    no_cp: Option<DsV4AttnBlockCompressParams>,
    no_ip: Option<DsV4AttnBlockIndexerParams>,
    cp_4: Option<DsV4AttnBlockCompressParams>,
    ip_4: Option<DsV4AttnBlockIndexerParams>,
    cp_128: Option<DsV4AttnBlockCompressParams>,
}

impl DispatchParamsPool {
    fn new(hp: &DsV4Hyperparams) -> Self {
        let cp_4 = Some(build_compress_params(hp, 4));
        let ip_4 = if hp.indexer_head_size.is_some() {
            Some(build_indexer_params(hp, 4))
        } else {
            None
        };
        let cp_128 = Some(build_compress_params(hp, 128));
        Self {
            no_cp: None,
            no_ip: None,
            cp_4,
            ip_4,
            cp_128,
        }
    }

    fn pick(
        &self,
        variant: &DsV4LayerVariant,
    ) -> (
        &Option<DsV4AttnBlockCompressParams>,
        &Option<DsV4AttnBlockIndexerParams>,
    ) {
        match (variant.compress_ratio, variant.has_indexer) {
            (None, _) => (&self.no_cp, &self.no_ip),
            (Some(4), true) => (&self.cp_4, &self.ip_4),
            (Some(4), false) => (&self.cp_4, &self.no_ip),
            (Some(128), _) => (&self.cp_128, &self.no_ip),
            (Some(other), _) => panic!(
                "unsupported compress_ratio {other} (only 4 and 128 are known for DSv4-Flash)"
            ),
        }
    }
}

/// Run the full DSv4 model forward by streaming layers one at a time.
///
/// Loads each requested layer from the GGUF, runs the per-layer
/// forward against the running residual, then drops the layer storage
/// before loading the next. The global head (token embedding +
/// output_norm + lm_head + mHC head) is pre-loaded and reused.
///
/// `layer_indices` is the sequence of layers to run (typically `0..43`
/// for a full DSv4-Flash forward, or a prefix for partial forwards).
/// `position_offset` is 0 for prefill from scratch.
///
/// Returns logits of shape `(token_ids.len(), head.n_vocab)`.
pub fn dsv4_streaming_model_forward(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    head: &DsV4HeadStorage,
    token_ids: &[u32],
    layer_indices: &[usize],
    position_offset: usize,
) -> Result<Array2<f32>, DsV4LoadError> {
    let layer_p = build_layer_params(hp);
    let pool = DispatchParamsPool::new(hp);

    let head_w = head.as_weights();
    let head_p = head.params(hp);

    // 1. Embed + broadcast to 4-stream initial residual.
    let x = embed_tokens(token_ids, head.token_embd_view());
    let mut residual: Array3<f32> = broadcast_to_streams(x.view(), hp.n_hc);

    // 2. Stream each layer.
    for &layer_idx in layer_indices {
        let (storage, variant) = load_dsv4_layer(gguf, hp, layer_idx)?;
        let (cp_ref, ip_ref) = pool.pick(&variant);
        let layer_w = DsV4LayerWeights {
            mhc_attn: storage
                .mhc_attn
                .as_ref()
                .expect("mhc_attn loaded by full loader")
                .as_weights(),
            attn: storage.dispatcher_layer(cp_ref, ip_ref),
            mhc_ffn: storage
                .mhc_ffn
                .as_ref()
                .expect("mhc_ffn loaded by full loader")
                .as_weights(),
            ffn: storage
                .ffn
                .as_ref()
                .expect("ffn loaded by full loader")
                .as_weights(),
        };
        residual = dsv4_per_layer_forward(
            residual.view(),
            &layer_w,
            &layer_p,
            Some(token_ids),
            position_offset,
        );
        // storage drops here, freeing ~26 GB before the next iteration.
    }

    // 3. mHC head → 1-stream pooled.
    let mut pooled = dsv4_hc_head(residual.view(), &head_w, &head_p);

    // 4. Final RMSNorm.
    let n_tokens = token_ids.len();
    let n_embd = hp.n_embd;
    for t in 0..n_tokens {
        let mut sumsq = 0.0_f32;
        for d in 0..n_embd {
            let v = pooled[[t, d]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n_embd as f32 + head_p.norm_eps).sqrt();
        for d in 0..n_embd {
            pooled[[t, d]] *= inv * head_w.output_norm[d];
        }
    }

    // 5. lm_head projection.
    let n_vocab = head.n_vocab;
    let mut logits = Array2::<f32>::zeros((n_tokens, n_vocab));
    for t in 0..n_tokens {
        for v in 0..n_vocab {
            let mut acc = 0.0_f32;
            for d in 0..n_embd {
                acc += pooled[[t, d]] * head_w.output_lm_head[[v, d]];
            }
            logits[[t, v]] = acc;
        }
    }
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::dsv4_head_storage::load_dsv4_head;

    /// Build the dispatch pool, then exercise the pick logic against
    /// each known DSv4-Flash variant tag. No GGUF I/O — uses synthetic
    /// hyperparameters with the indexer fields populated.
    #[test]
    fn dispatch_pool_picks_correct_variant_arm() {
        let hp = DsV4Hyperparams {
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
            indexer_head_size: Some(16),
            n_index_head: Some(2),
            top_k: Some(2),
            n_hc: 4,
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 8,
            n_expert_shared: 1,
            expert_weights_norm: true,
            expert_weights_scale: 1.0,
            yarn: None,
            rope_base_swa: None,
        };
        let pool = DispatchParamsPool::new(&hp);

        // NoCompress: both None.
        let v = DsV4LayerVariant {
            uses_hash_routing: true,
            compress_ratio: None,
            has_indexer: false,
        };
        let (cp, ip) = pool.pick(&v);
        assert!(cp.is_none());
        assert!(ip.is_none());

        // Indexer cr=4: both Some.
        let v = DsV4LayerVariant {
            uses_hash_routing: false,
            compress_ratio: Some(4),
            has_indexer: true,
        };
        let (cp, ip) = pool.pick(&v);
        assert!(cp.is_some());
        assert!(ip.is_some());
        assert_eq!(cp.as_ref().unwrap().compressor.compress_ratio, 4);

        // Compress-only cr=128: cp Some, ip None.
        let v = DsV4LayerVariant {
            uses_hash_routing: false,
            compress_ratio: Some(128),
            has_indexer: false,
        };
        let (cp, ip) = pool.pick(&v);
        assert!(cp.is_some());
        assert!(ip.is_none());
        assert_eq!(cp.as_ref().unwrap().compressor.compress_ratio, 128);
    }

    /// End-to-end real-GGUF: streaming forward over layers 0..3.
    /// Same coverage scope as the per_layer / 2-layer model_forward
    /// smokes, but through the new reusable streaming helper.
    /// Expected wall: ~110 s release (3 × ~30 s loads + forward).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn streaming_helper_3_layer_smoke() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        // 16 tokens (compress_ratio=4 at layer 2 needs n_tokens >= 4).
        let n_tokens = 16;
        let token_ids: Vec<u32> = (0..n_tokens)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();

        let logits = dsv4_streaming_model_forward(&gguf, &hp, &head, &token_ids, &[0, 1, 2], 0)
            .expect("streaming forward");

        assert_eq!(logits.shape(), &[n_tokens, head.n_vocab]);
        let n_nonfinite = logits.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(n_nonfinite, 0, "{n_nonfinite} non-finite logits");
        let total: f32 = logits.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "logits all zero");
    }
}
