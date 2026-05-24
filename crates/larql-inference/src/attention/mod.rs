//! Attention computation — RoPE, GQA, causal masking, GPU dispatch.
//!
//! Submodules:
//! - `rope`: Rotary Position Embeddings (full and partial rotation)
//! - `gqa`: Grouped-Query Attention with BLAS-fused dot products
//! - `block`: CPU attention block (norm → proj → RoPE → GQA → O → residual)
//! - `gpu`: GPU-accelerated attention, KV-capture, Q4 projection

pub mod block;
pub mod decode;
pub mod deltanet_block;
pub mod deltanet_recurrence;
pub mod deltanet_state;
pub mod dsv4_attn_block;
pub mod dsv4_attn_block_compress;
pub mod dsv4_attn_block_indexer;
pub mod dsv4_attn_dispatch;
pub mod dsv4_compressor;
pub mod dsv4_compressor_prefill;
pub mod dsv4_decode_loop;
pub mod dsv4_ffn_block;
pub mod dsv4_fp8_kv;
pub mod dsv4_full_loader;
pub mod dsv4_gguf_reader;
pub mod dsv4_grouped_o_proj;
pub mod dsv4_head_storage;
pub mod dsv4_hyperparams_load;
pub mod dsv4_indexer;
pub mod dsv4_layer_smoke;
pub mod dsv4_layer_variants;
pub mod dsv4_masked_attn;
pub mod dsv4_mhc;
pub mod dsv4_mhc_bookend;
pub mod dsv4_model_forward;
pub mod dsv4_moe_dispatch;
pub mod dsv4_moe_ops;
pub mod dsv4_moe_routing;
pub mod dsv4_per_layer;
pub mod dsv4_rope_tail;
pub mod dsv4_rope_tail_yarn;
pub mod dsv4_sampling;
pub mod dsv4_storage;
pub mod dsv4_storage_build;
pub mod dsv4_streaming_model_forward;
pub mod dsv4_swa;
pub mod dsv4_topk_logits;
pub mod dsv4_yarn_config;
pub mod fine_profile;
pub mod gpu;
pub mod gpu_tier;
pub mod gqa;
pub mod q4k_direct;
pub mod q4k_prefill;
pub mod quant_dispatch;
pub mod qwen35_attn;
pub mod qwen35_block;
pub mod qwen35_forward;
pub mod qwen35_load;
pub mod qwen35_load_vindex;
pub mod rope;

use ndarray::Array2;

/// Per-head attention weights for the last token position.
pub struct AttentionWeights {
    /// Per-head attention distribution for the last sequence position.
    /// `heads[h][j]` = attention weight from last token to position j.
    pub heads: Vec<Vec<f32>>,
}

/// Per-head attention weights for every query position.
///
/// `heads[h][i][j]` = attention weight from query position `i` to source
/// position `j`. Rows are padded to the full sequence length; causal-future
/// entries are zero.
pub struct AttentionAllWeights {
    pub heads: Vec<Vec<Vec<f32>>>,
}

/// Shared KV pair: post-RoPE K and post-V-norm V from a source layer.
pub type SharedKV = (Array2<f32>, Array2<f32>);

// ── Re-exports: preserve `crate::attention::*` paths ──

pub use block::{
    run_attention_block, run_attention_block_replace_head_residual_delta,
    run_attention_block_replace_pre_o_head, run_attention_block_shared,
    run_attention_block_shared_with_pre_o, run_attention_block_subtract_pre_o_heads,
    run_attention_block_with_kv_out, run_attention_block_with_kv_out_with_cache,
    run_attention_block_with_pre_o, run_attention_block_with_pre_o_and_all_attention_weights,
    run_attention_block_with_pre_o_and_reduced_qk_attention_weights,
    run_attention_block_zero_pre_o_heads,
};
pub use decode::{
    gqa_attention_decode_step, run_attention_block_decode_step,
    run_attention_block_decode_step_backend, KvCache,
};
pub use gpu::{
    q4_attention_proj, run_attention_block_gpu, run_attention_with_kv,
    run_attention_with_kv_backend,
};
pub use gqa::{gqa_attention, gqa_attention_with_all_weights, gqa_attention_with_weights};
pub use rope::{apply_rope, apply_rope_partial, apply_rope_partial_at};
