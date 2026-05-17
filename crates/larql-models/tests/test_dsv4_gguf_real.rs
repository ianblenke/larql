//! Real DeepSeek V4 Flash GGUF integration smoke — Scope 1.
//!
//! Opens the `antirez/deepseek-v4-gguf` Q4KExperts imatrix file
//! (164.6 GB at `/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/…`) via
//! `GgufFile::open` (header-only, no tensor data is read) and
//! confirms that:
//!
//! 1. The GGUF→HF metadata bridge maps `general.architecture =
//!    "deepseek4"` through to `config.model_type = "deepseek4"`.
//! 2. `detect_from_json` routes that model_type to `DeepSeekV4Arch`
//!    (not the V3 `DeepSeekArch` fallback).
//! 3. The expected MoE / MLA scalars are populated from the GGUF
//!    `deepseek4.*` keys: 43 layers, hidden=4096, 256 experts,
//!    6 used, 1 shared, q_lora_rank=1024, kv_lora_rank=512.
//!
//! Ignored by default — the file is local-only and the test would
//! fail on CI machines that don't have `/tank`. Run with:
//!
//! ```
//! cargo test -p larql-models --test test_dsv4_gguf_real -- --ignored
//! ```

use std::path::Path;

const DSV4_GGUF: &str = "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/\
    DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf";

#[test]
#[ignore = "requires /tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/…imatrix.gguf (164 GB local-only)"]
fn dsv4_gguf_header_parses_and_routes_to_v4_arch() {
    let path = Path::new(DSV4_GGUF);
    assert!(
        path.exists(),
        "DSv4 GGUF not present at {DSV4_GGUF} — skip with --skip or download first"
    );

    let gguf =
        larql_models::loading::gguf::GgufFile::open(path).expect("GgufFile::open on DSv4 GGUF");

    // GGUF→HF config synthesis.
    let cfg = gguf.to_config_json();
    assert_eq!(cfg["model_type"], "deepseek4", "model_type bridge");
    assert_eq!(cfg["num_hidden_layers"], 43);
    assert_eq!(cfg["hidden_size"], 4096);
    assert_eq!(cfg["num_experts"], 256);
    assert_eq!(cfg["num_experts_per_tok"], 6);
    assert_eq!(cfg["n_shared_experts"], 1);
    assert_eq!(cfg["q_lora_rank"], 1024);
    // V4 fuses KV down-projection; the gguf bridge falls back to
    // `attention.key_length` (=512) when explicit kv_lora_rank is absent.
    assert_eq!(cfg["kv_lora_rank"], 512);

    // Arch routing.
    let arch = larql_models::detect_from_json(&cfg);
    assert_eq!(arch.family(), "deepseek4");
    assert!(arch.is_moe());
    assert!(arch.uses_mla());
    assert_eq!(arch.num_experts(), 256);
    assert_eq!(arch.num_experts_per_token(), 6);
    assert_eq!(arch.num_shared_experts(), 1);
    assert_eq!(arch.q_lora_rank(), 1024);
    assert_eq!(arch.kv_lora_rank(), 512);

    // Tensor-key conventions match the GGUF post-normalisation form
    // (`blk.{L}.` → `layers.{L}.`).
    assert_eq!(
        arch.mla_q_a_key(0),
        Some("layers.0.attn_q_a.weight".to_string())
    );
    assert_eq!(
        arch.mla_kv_a_key(0),
        Some("layers.0.attn_kv.weight".to_string())
    );
    assert_eq!(arch.mla_kv_b_key(0), None);
}

/// Browse-tier vindex extraction smoke — Scope 2.
///
/// Validates that a vindex produced by
/// `larql convert gguf-to-vindex --level browse` on the real DSv4
/// GGUF has the expected structural shape: 43 layers, 256 experts
/// per layer, hidden=4096, vocab=129280, embedding f32 file sized
/// `vocab × hidden × 4`. The conversion itself is invoked manually
/// (it takes ~40 minutes and writes ~350 GB) — this test only opens
/// the result. Skipped automatically when the vindex directory
/// doesn't exist.
#[test]
#[ignore = "requires browse-tier vindex at /tank/ai/deepseek-ai/DeepSeek-V4-Flash-vindex-v1/"]
fn dsv4_browse_vindex_structure_matches_gguf_metadata() {
    let dir = Path::new("/tank/ai/deepseek-ai/DeepSeek-V4-Flash-vindex-v1");
    if !dir.exists() {
        panic!(
            "browse-tier vindex not present at {dir:?} — re-run the conversion:\n  \
             larql convert gguf-to-vindex --level browse --output {dir:?} \
             /tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/*.gguf"
        );
    }

    let cfg_path = dir.join("index.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("read index.json");
    let cfg: serde_json::Value =
        serde_json::from_slice(&cfg_bytes).expect("parse index.json as JSON");

    assert_eq!(cfg["family"], "deepseek4");
    assert_eq!(cfg["num_layers"], 43);
    assert_eq!(cfg["hidden_size"], 4096);
    assert_eq!(cfg["vocab_size"], 129280);
    assert_eq!(cfg["extract_level"], "browse");

    // 43 layers × 256 experts + 1 shared = 257 expert groups; the
    // per-layer record holds num_experts=256 (routed) with
    // num_features_per_expert=2048 (V4 expert intermediate width).
    let layer0 = &cfg["layers"][0];
    assert_eq!(layer0["num_experts"], 256);
    assert_eq!(layer0["num_features_per_expert"], 2048);
    // 256 routed × 2048 + 1 shared × 2048 = 526336 total features.
    assert_eq!(layer0["num_features"], 526336);

    // embeddings.bin shape: vocab × hidden × sizeof(f32).
    let embed_path = dir.join("embeddings.bin");
    let embed_meta = std::fs::metadata(&embed_path).expect("stat embeddings.bin");
    let expected_embed_size: u64 = 129280 * 4096 * 4;
    assert_eq!(embed_meta.len(), expected_embed_size);
}
