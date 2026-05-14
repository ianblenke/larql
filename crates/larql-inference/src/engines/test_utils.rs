//! Synthetic test fixtures for engine and layer-graph unit tests.
//!
//! Three helpers:
//! - `make_test_weights()` — fully functional 2-layer ModelWeights (no disk I/O)
//! - `make_test_vindex(weights)` — in-memory VectorIndex with random gate vectors
//! - `make_test_tokenizer(vocab_size)` — WordLevel tokenizer mapping token N to "[N]"
//!
//! Dimensions: vocab=32, hidden=16, intermediate=32, 2 q-heads, 1 kv-head,
//! head_dim=8, 2 layers. Forward pass ≈ 10 ms on CPU.

use larql_models::{detect_from_json, ModelWeights, WeightArray};
use ndarray::Array2;
use std::collections::HashMap;

/// Build a synthetic `ModelWeights` with all tensors populated.
/// Uses `TinyModelArch` key conventions (e.g. `"0.attn.q_proj.weight"`).
pub fn make_test_weights() -> ModelWeights {
    make_test_weights_with(SyntheticDims::tiny())
}

/// Caller-supplied dimensions for [`make_test_weights_with`].
/// Useful when you need the same synthetic-weights factory at a
/// larger shape — e.g. for benchmarking the attention-service
/// runner under realistic dims, without paying for a real
/// model checkpoint.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticDims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
}

impl SyntheticDims {
    /// Tiny: vocab=32, hidden=16, 2 layers. Default for unit tests.
    /// Forward pass < 1 ms on CPU.
    pub fn tiny() -> Self {
        Self {
            vocab: 32,
            hidden: 16,
            intermediate: 32,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            num_layers: 2,
        }
    }

    /// Gemma-3-4B-shaped: vocab=262144, hidden=2560, 33 layers,
    /// num_q=8, num_kv=4, head_dim=320, intermediate=10240.
    /// Useful for realistic-shape benches; weight build is
    /// noticeable (~1 GB f32) so prefer caching with `OnceLock`.
    pub fn gemma_3_4b_like() -> Self {
        Self {
            vocab: 32,
            hidden: 2560,
            intermediate: 10240,
            num_q_heads: 8,
            num_kv_heads: 4,
            head_dim: 320,
            num_layers: 33,
        }
    }

    /// Smaller Gemma-shaped fixture for benches that don't need the
    /// full 33 layers — same hidden/heads but only 4 layers.
    /// Build cost ≈ 100 MB f32; useful for parametric sweeps.
    pub fn gemma_3_4b_4layer() -> Self {
        Self {
            vocab: 32,
            hidden: 2560,
            intermediate: 10240,
            num_q_heads: 8,
            num_kv_heads: 4,
            head_dim: 320,
            num_layers: 4,
        }
    }
}

/// Build synthetic weights at caller-supplied dimensions. The body
/// is identical to [`make_test_weights`] but parameterised. Used by
/// the `larql-server` benchmark harness to reach realistic Gemma
/// shapes without a real checkpoint.
pub fn make_test_weights_with(dims: SyntheticDims) -> ModelWeights {
    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": dims.hidden,
        "num_hidden_layers": dims.num_layers,
        "intermediate_size": dims.intermediate,
        "head_dim": dims.head_dim,
        "num_attention_heads": dims.num_q_heads,
        "num_key_value_heads": dims.num_kv_heads,
        "vocab_size": dims.vocab,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut rng_state = 0xdeadbeef_u64;

    // LCG giving values in [-scale, +scale]
    let mut rand_mat = |rows: usize, cols: usize, scale: f32| -> WeightArray {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng_state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data)
            .unwrap()
            .into_shared()
    };

    let hidden = dims.hidden;
    // Embed + lm_head
    let embed = rand_mat(dims.vocab, hidden, 0.1);
    let lm_head = rand_mat(dims.vocab, hidden, 0.1);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm (ones → valid unweighted RMSNorm fallback)
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

    let q_dim = dims.num_q_heads * dims.head_dim;
    let kv_dim = dims.num_kv_heads * dims.head_dim;
    let inter = dims.intermediate;

    for layer in 0..dims.num_layers {
        // Attention projections
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, hidden, 0.1));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, hidden, 0.1));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, hidden, 0.1));
        tensors.insert(arch.attn_o_key(layer), rand_mat(hidden, q_dim, 0.1));
        // FFN — missing tensors cause panic, so always provide them
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(inter, hidden, 0.1));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(inter, hidden, 0.1));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(hidden, inter, 0.1));
        // Layer norms
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; hidden]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        embed_quant: None,
        lm_head,
        lm_head_quant: None,
        position_embed: None,
        quant_tensors: HashMap::new(),
        arch,
        num_layers: dims.num_layers,
        hidden_size: hidden,
        intermediate_size: inter,
        vocab_size: dims.vocab,
        head_dim: dims.head_dim,
        num_q_heads: dims.num_q_heads,
        num_kv_heads: dims.num_kv_heads,
        rope_base: 10_000.0,
    }
}

/// Build an in-memory `VectorIndex` with random gate vectors per layer.
/// The VectorIndex has no Q4K or interleaved data — `predict_honest` falls
/// through to the CPU path, and `WalkFfn` routes through the sparse fallback
/// that uses `weights.tensors`.
pub fn make_test_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let n_features = weights.intermediate_size;
    let hidden = weights.hidden_size;

    // Each layer gets an independent LCG seed so gate matrices are distinct.
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..weights.num_layers)
        .map(|l| {
            let mut state = 0xabcdef_u64.wrapping_add(l as u64 * 0x9e3779b97f4a7c15);
            let data: Vec<f32> = (0..n_features * hidden)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
                })
                .collect();
            Some(Array2::from_shape_vec((n_features, hidden), data).unwrap())
        })
        .collect();

    let down_meta = vec![None; weights.num_layers];
    larql_vindex::VectorIndex::new(gate_vectors, down_meta, weights.num_layers, hidden)
}

/// Build a `tokenizers::Tokenizer` with a vocabulary of `vocab_size` tokens.
/// Token N decodes to `"[N]"`, so token IDs from `make_test_weights()` all
/// decode to valid (if meaningless) strings.
pub fn make_test_tokenizer(vocab_size: usize) -> tokenizers::Tokenizer {
    // WordLevel::builder().vocab() requires an AHashMap.
    // Build a simple BPE-less tokenizer via JSON serialization instead.
    let mut vocab_json = serde_json::Map::new();
    for i in 0..vocab_size as u64 {
        vocab_json.insert(format!("[{i}]"), serde_json::Value::Number(i.into()));
    }
    // Add UNK token at the end
    vocab_json.insert("[UNK]".into(), serde_json::Value::Number(vocab_size.into()));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_json,
            "unk_token": "[UNK]"
        }
    });

    let bytes = serde_json::to_vec(&tokenizer_json).expect("JSON serialization failed");
    tokenizers::Tokenizer::from_bytes(&bytes).expect("synthetic tokenizer construction failed")
}

/// All three synthetic fixtures bundled together. Build once per test module
/// via `OnceLock`; each field is cheaply borrowed.
pub struct TestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}

impl TestFixtures {
    pub fn build() -> Self {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        Self {
            weights,
            tokenizer,
            index,
        }
    }
}

// ── Alternate-arch fixtures ─────────────────────────────────────────────
//
// `make_test_weights` uses the `tinymodel` arch which leaves many optional
// branches dormant (no bias keys, no QK norm, no post norms, gated FFN
// only). The fixtures below pin those branches by routing through a
// real arch impl that enables them. Each fixture provides exactly the
// tensors + vectors the matching forward path needs to reach finite
// output without panicking.

fn rand_mat_seeded(rows: usize, cols: usize, scale: f32, seed: u64) -> WeightArray {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .unwrap()
        .into_shared()
}

/// Build a synthetic `ModelWeights` configured as a Gemma 3-style arch.
///
/// Enables the dormant branches in `attention/{block, gpu}.rs` and
/// `forward/layer.rs` that tinymodel never reaches:
/// - **QK norm** — `attn_q_norm_key` / `attn_k_norm_key` return Some,
///   so `rms_norm_heads` runs in `run_attention_block_core`
/// - **post norms** — `has_post_norms()` is true; pre/post FFN norm keys
///   are populated, the FFN dispatch routes through the post-norm arm
/// - **GeluTanh activation** — `activation()` is `GeluTanh`, exercising
///   the gelu-tanh gate-up branches in `ffn/weight.rs` and `attention`
/// - **`embed_scale = sqrt(hidden)`** — non-1.0 embed scaling
/// - **`norm_weight_offset = 1.0`** — non-zero offset added to every
///   norm weight at runtime (the saved weights are deltas)
/// - **sliding-window/global pattern** — layer 0 is a sliding-window
///   layer, layer 5 (and multiples of 6) is global; both are exercised
///   in the 2-layer fixture only at layer 0/1 (both sliding), but
///   `is_sliding_window_layer()` still returns sensible values.
pub fn make_gemma3_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;

    let arch_json = serde_json::json!({
        "model_type": "gemma3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        // Non-default scaling: exercises the `res_mult != 1.0` branch in
        // `forward/layer.rs::run_ffn` and `attention/gpu.rs::run_attention_block_gpu`.
        "residual_multiplier": 0.5,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    // Embed + lm_head — small, non-zero so post-norm RMS doesn't divide by 0.
    let embed = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x9e3779b9);
    let lm_head = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0xa1b2c3d4);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm — Gemma3 uses norm_weight_offset=1.0, so the saved
    // weight is the *delta* off identity. Zeros → unit-scale norm at
    // runtime (offset=1 + weight=0 → 1.0).
    vectors.insert(arch.final_norm_key().to_string(), vec![0.0; HIDDEN]);

    let mut seed_counter: u64 = 0xdeadbeef;
    let mut next_seed = || {
        seed_counter = seed_counter.wrapping_add(0x9e3779b97f4a7c15);
        seed_counter
    };

    for layer in 0..NUM_LAYERS {
        // Attention projections
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(HIDDEN, q_dim, 0.1, next_seed()),
        );

        // FFN — gated (SwiGLU/GeluTanh)
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(HIDDEN, INTER, 0.1, next_seed()),
        );

        // Layer norms — input + post-attention. norm_weight_offset=1.0
        // means saved weights are deltas; zeros = identity.
        vectors.insert(arch.input_layernorm_key(layer), vec![0.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.0; HIDDEN]);
        // Gemma3-specific: pre/post FFN norms (post-norms branch).
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.0; HIDDEN]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.0; HIDDEN]);
        }

        // QK norm — per-head dim weights.
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.0; HEAD_DIM]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.0; HEAD_DIM]);
        }
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        embed_quant: None,
        lm_head,
        lm_head_quant: None,
        position_embed: None,
        quant_tensors: HashMap::new(),
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

/// Build a synthetic `ModelWeights` configured as a Starcoder2-style arch.
///
/// Enables the dormant branches:
/// - **LayerNorm (not RMSNorm)** — `norm_type()` is `LayerNorm`
/// - **Non-gated FFN** — `ffn_type()` is `Standard` (non-gated),
///   exercising the `else` arm in `ffn/weight.rs::dense_ffn_forward_backend`
/// - **FFN bias** — `ffn_up_bias_key` / `ffn_down_bias_key` return Some,
///   so the `add_bias` calls fire
/// - **Attention bias** — `attn_q_bias_key` / `attn_k_bias_key` /
///   `attn_v_bias_key` / `attn_o_bias_key` return Some
/// - **c_fc/c_proj naming** — `ffn_up_key` / `ffn_down_key` use
///   `mlp.c_fc.weight` / `mlp.c_proj.weight` rather than gate/up/down
/// - **GeluTanh activation** — `activation()` is `GeluTanh`
pub fn make_starcoder2_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;

    let arch_json = serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
        // Non-default scaling: exercises the `res_mult != 1.0` branch in
        // the no-post-norms arm of `forward/layer.rs::run_ffn` and the
        // `attention_multiplier()` branch in `attention/gpu.rs`.
        "residual_multiplier": 0.5,
        "attention_multiplier": 2.0,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    let embed = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x12345678);
    let lm_head = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x87654321);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let mut seed_counter: u64 = 0xfeedbabe;
    let mut next_seed = || {
        seed_counter = seed_counter.wrapping_add(0x9e3779b97f4a7c15);
        seed_counter
    };

    for layer in 0..NUM_LAYERS {
        // Attention projections
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(HIDDEN, q_dim, 0.1, next_seed()),
        );

        // Attention biases — Starcoder2 has them.
        if let Some(k) = arch.attn_q_bias_key(layer) {
            vectors.insert(k, vec![0.01; q_dim]);
        }
        if let Some(k) = arch.attn_k_bias_key(layer) {
            vectors.insert(k, vec![0.01; kv_dim]);
        }
        if let Some(k) = arch.attn_v_bias_key(layer) {
            vectors.insert(k, vec![0.01; kv_dim]);
        }
        if let Some(k) = arch.attn_o_bias_key(layer) {
            vectors.insert(k, vec![0.01; HIDDEN]);
        }

        // FFN — non-gated (c_fc + c_proj). The keys map through
        // arch.ffn_up_key/ffn_down_key which return c_fc/c_proj naming.
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(HIDDEN, INTER, 0.1, next_seed()),
        );
        // Add gate too — gated-only code paths may probe it regardless
        // of arch's ffn_type; safer to populate it so a stray lookup
        // doesn't panic the test.
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );

        // FFN biases — Starcoder2 has them.
        if let Some(k) = arch.ffn_up_bias_key(layer) {
            vectors.insert(k, vec![0.01; INTER]);
        }
        if let Some(k) = arch.ffn_down_bias_key(layer) {
            vectors.insert(k, vec![0.01; HIDDEN]);
        }

        // Layer norms — Starcoder2 uses LayerNorm, norm_weight_offset=0,
        // so weights are the actual scale.
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        embed_quant: None,
        lm_head,
        lm_head_quant: None,
        position_embed: None,
        quant_tensors: HashMap::new(),
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma3_fixture_has_qk_norm_and_post_norm_keys() {
        let w = make_gemma3_test_weights();
        // QK norm keys present
        for layer in 0..w.num_layers {
            let qk = w.arch.attn_q_norm_key(layer).expect("Gemma3 has q_norm");
            let kk = w.arch.attn_k_norm_key(layer).expect("Gemma3 has k_norm");
            assert!(w.vectors.contains_key(&qk), "missing {qk}");
            assert!(w.vectors.contains_key(&kk), "missing {kk}");
        }
        // post-norms branch
        assert!(
            w.arch.has_post_norms(),
            "Gemma3 fixture must have post_norms=true"
        );
        // embed scale is sqrt(hidden)
        let expected = (w.hidden_size as f32).sqrt();
        assert!((w.arch.embed_scale() - expected).abs() < 1e-6);
        // norm offset is 1.0
        assert_eq!(w.arch.norm_weight_offset(), 1.0);
        // family
        assert_eq!(w.arch.family(), "gemma3");
    }

    #[test]
    fn starcoder2_fixture_has_layernorm_and_biases() {
        use larql_models::{FfnType, NormType};
        let w = make_starcoder2_test_weights();
        assert_eq!(w.arch.family(), "starcoder2");
        // LayerNorm (not RMSNorm)
        assert!(matches!(w.arch.norm_type(), NormType::LayerNorm));
        // Standard (non-gated) FFN
        assert!(matches!(w.arch.ffn_type(), FfnType::Standard));
        // FFN keys use c_fc/c_proj naming
        let up_key = w.arch.ffn_up_key(0);
        assert!(
            up_key.contains("c_fc"),
            "expected c_fc in {up_key}, got {up_key}"
        );
        let down_key = w.arch.ffn_down_key(0);
        assert!(down_key.contains("c_proj"), "expected c_proj in {down_key}");
        // Attention biases present
        for layer in 0..w.num_layers {
            let qb = w
                .arch
                .attn_q_bias_key(layer)
                .expect("Starcoder2 has q bias");
            assert!(w.vectors.contains_key(&qb), "missing {qb}");
        }
        // FFN biases present
        for layer in 0..w.num_layers {
            let upb = w
                .arch
                .ffn_up_bias_key(layer)
                .expect("Starcoder2 has up bias");
            assert!(w.vectors.contains_key(&upb), "missing {upb}");
        }
    }

    #[test]
    fn gemma3_fixture_shape_matches_dims() {
        let w = make_gemma3_test_weights();
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.embed.shape(), &[w.vocab_size, w.hidden_size]);
        assert_eq!(w.lm_head.shape(), &[w.vocab_size, w.hidden_size]);
    }

    #[test]
    fn starcoder2_fixture_shape_matches_dims() {
        let w = make_starcoder2_test_weights();
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.embed.shape(), &[w.vocab_size, w.hidden_size]);
        assert_eq!(w.lm_head.shape(), &[w.vocab_size, w.hidden_size]);
    }
}
