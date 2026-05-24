//! High-level DSv4 text generation entry point.
//!
//! [`dsv4_generate`] is the one-shot "give me N tokens from DSv4"
//! API that wires together:
//! - per-variant cache allocation via
//!   [`super::dsv4_kv_cache::DsV4LayerCache::for_variant`]
//! - cached prefill via
//!   [`super::dsv4_streaming_model_forward::dsv4_streaming_model_forward_cached`]
//! - cached single-token decode loop
//! - sampling via
//!   [`super::dsv4_sampling::dsv4_sample_next_token`]
//!
//! Use this as the production entry point for DSv4-Flash text
//! generation. The lower-level building blocks remain available for
//! callers that need finer control (per-step token inspection, custom
//! cache management, etc.).

use rand::Rng;

use larql_models::loading::gguf::GgufFile;

use super::dsv4_decode_loop::DecodeConfig;
use super::dsv4_full_loader::DsV4LoadError;
use super::dsv4_head_storage::DsV4HeadStorage;
use super::dsv4_kv_cache::DsV4LayerCache;
use super::dsv4_layer_variants::detect_layer_variant;
use super::dsv4_sampling::dsv4_sample_next_token;
use super::dsv4_storage_build::DsV4Hyperparams;
use super::dsv4_streaming_model_forward::dsv4_streaming_model_forward_cached;

/// Generate up to `decode_config.max_new_tokens` new tokens given a
/// prompt, running DSv4 in cached-decode mode for each step after the
/// prefill.
///
/// Returns the full token sequence: `prompt + generated`. The
/// generated portion has at most `max_new_tokens` entries (fewer if
/// the EOS token from `decode_config` was sampled).
///
/// `layer_indices` is the sequence of layers to run (typically
/// `0..n_layer` for a full DSv4-Flash forward; a prefix runs a
/// partial model — useful for testing).
///
/// Cache pre-allocation: each layer's cache is sized to
/// `prompt.len() + max_new_tokens`. The variant is detected from the
/// GGUF metadata, so callers don't need to know per-layer variant
/// shapes themselves.
pub fn dsv4_generate(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    head: &DsV4HeadStorage,
    layer_indices: &[usize],
    prompt: &[u32],
    decode_config: DecodeConfig,
    rng: &mut impl Rng,
) -> Result<Vec<u32>, DsV4LoadError> {
    assert!(!prompt.is_empty(), "prompt must be non-empty");

    let max_seq_len = prompt.len() + decode_config.max_new_tokens;
    let mut layer_caches: Vec<DsV4LayerCache> = layer_indices
        .iter()
        .map(|&i| {
            let variant = detect_layer_variant(gguf, i, hp.head_dim);
            DsV4LayerCache::for_variant(hp, &variant, max_seq_len)
        })
        .collect();

    // 1. Prefill — runs the full prompt through the cached forward.
    let mut logits = dsv4_streaming_model_forward_cached(
        gguf,
        hp,
        head,
        prompt,
        layer_indices,
        0,
        Some(&mut layer_caches),
    )?;
    let mut tokens = prompt.to_vec();

    // 2. Decode loop. Each iteration: sample from the last logit row,
    //    append the new token, then run one cached forward step on
    //    just that new token to produce logits for the *next* token.
    for i in 0..decode_config.max_new_tokens {
        let last_row_idx = logits.shape()[0] - 1;
        let last_row = logits.row(last_row_idx);
        let next = dsv4_sample_next_token(last_row, decode_config.sampling, rng);
        tokens.push(next);

        if decode_config.eos_token == Some(next) {
            break;
        }
        // Skip the trailing forward if this was the final iteration —
        // its logits would be discarded.
        if i + 1 >= decode_config.max_new_tokens {
            break;
        }

        let pos = tokens.len() - 1; // absolute position of the new token
        logits = dsv4_streaming_model_forward_cached(
            gguf,
            hp,
            head,
            &[next],
            layer_indices,
            pos,
            Some(&mut layer_caches),
        )?;
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_head_storage::load_dsv4_head;
    use super::super::dsv4_sampling::SamplingConfig;
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Real-GGUF smoke: 16-token prompt + 2 greedy decode steps on
    /// layers 0..3 of DSv4-Flash. Verifies the entire one-shot
    /// generation API works end-to-end.
    ///
    /// Wall: ~500 s release (prefill ~110 s + 2 decode steps ~110 s
    /// each + head load + lm_head matmuls).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn generate_real_gguf_3_layer_greedy_smoke() {
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

        let n_prompt = 16;
        let prompt: Vec<u32> = (0..n_prompt)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();
        let decode_config = DecodeConfig {
            max_new_tokens: 2,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);

        let tokens = dsv4_generate(
            &gguf,
            &hp,
            &head,
            &[0, 1, 2],
            &prompt,
            decode_config,
            &mut rng,
        )
        .expect("generate");

        // prompt (16) + 2 generated = 18 tokens.
        assert_eq!(tokens.len(), n_prompt + 2);
        // Prompt prefix preserved.
        assert_eq!(&tokens[..n_prompt], &prompt[..]);
        // Generated tokens are valid IDs (< n_vocab).
        for &t in &tokens[n_prompt..] {
            assert!(
                (t as usize) < head.n_vocab,
                "generated token {t} out of vocab"
            );
        }
    }
}
