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
