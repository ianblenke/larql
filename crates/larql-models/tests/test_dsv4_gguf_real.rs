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

/// Q4_K vindex extraction smoke — DSv4 scope 3.
///
/// Validates the new MLA-aware Q4_K writer's outputs:
/// `mla_weights_q4k.bin` + manifest (5 tensors × 43 layers),
/// `dsv4_aux.bin` + manifest (V4-specific HC / compressor / indexer
/// / sinks / norms), and per-layer MoE expert files.
///
/// Manually invoked via:
/// ```
/// larql convert gguf-to-vindex --level inference --quant q4k \
///   --output /tank/ai/deepseek-ai/DeepSeek-V4-Flash-vindex-v2 \
///   /tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/*.gguf
/// ```
/// Ignored by default — opens the extracted vindex only.
#[test]
#[ignore = "requires Q4_K vindex at /tank/ai/deepseek-ai/DeepSeek-V4-Flash-vindex-v2/"]
fn dsv4_q4k_vindex_manifests_describe_mla_and_aux() {
    let dir = Path::new("/tank/ai/deepseek-ai/DeepSeek-V4-Flash-vindex-v2");
    if !dir.exists() {
        panic!("Q4_K vindex not present at {dir:?} — re-run conversion (see doc above)");
    }

    // MLA manifest: 5 tensors × 43 layers = 215 entries, all Q4_K.
    let mla_bytes = std::fs::read(dir.join("mla_weights_q4k_manifest.json")).unwrap();
    let mla: serde_json::Value = serde_json::from_slice(&mla_bytes).unwrap();
    let mla_entries = mla.as_array().expect("mla manifest is a JSON array");
    assert_eq!(mla_entries.len(), 215, "5 MLA tensors × 43 layers");
    for e in mla_entries {
        assert_eq!(e["format"], "Q4_K");
    }
    // Spot-check shapes of layer 0's MLA tensors.
    let layer0: Vec<&serde_json::Value> = mla_entries
        .iter()
        .filter(|e| e["key"].as_str().unwrap().starts_with("layers.0."))
        .collect();
    assert_eq!(layer0.len(), 5);
    let by_key: std::collections::HashMap<&str, &serde_json::Value> = layer0
        .iter()
        .map(|e| (e["key"].as_str().unwrap(), *e))
        .collect();
    assert_eq!(
        by_key["layers.0.attn_q_a.weight"]["shape"],
        serde_json::json!([1024, 4096])
    );
    assert_eq!(
        by_key["layers.0.attn_q_b.weight"]["shape"],
        serde_json::json!([32768, 1024])
    );
    assert_eq!(
        by_key["layers.0.attn_kv.weight"]["shape"],
        serde_json::json!([512, 4096])
    );
    assert_eq!(
        by_key["layers.0.attn_output_a.weight"]["shape"],
        serde_json::json!([8192, 4096])
    );
    assert_eq!(
        by_key["layers.0.attn_output_b.weight"]["shape"],
        serde_json::json!([4096, 8192])
    );

    // V4 aux manifest: covers HC + compressor + indexer + sinks etc.
    let aux_bytes = std::fs::read(dir.join("dsv4_aux_manifest.json")).unwrap();
    let aux: serde_json::Value = serde_json::from_slice(&aux_bytes).unwrap();
    let aux_entries = aux.as_array().expect("aux manifest is a JSON array");
    let aux_keys: std::collections::HashSet<String> = aux_entries
        .iter()
        .map(|e| e["key"].as_str().unwrap().to_string())
        .collect();
    // Per-layer presence checks. Universal-on-every-layer:
    // norms (attn_sinks, attn_q_a_norm, attn_kv_a_norm) and hyper-
    // connection (hc_{attn,ffn}_{base,fn,scale}).
    for layer in 0..43usize {
        for required in &[
            "attn_sinks.weight",
            "attn_q_a_norm.weight",
            "attn_kv_a_norm.weight",
            "hc_attn_base.weight",
            "hc_attn_fn.weight",
            "hc_attn_scale.weight",
            "hc_ffn_base.weight",
            "hc_ffn_fn.weight",
            "hc_ffn_scale.weight",
        ] {
            let key = format!("layers.{layer}.{required}");
            assert!(aux_keys.contains(&key), "missing aux entry: {key}");
        }
    }
    // Variable-presence tensors. The DSv4 GGUF reports
    // `deepseek4.hash_layer_count = 3` and a 44-entry
    // `attention.compress_ratios` array whose zeros (layers 0,1) and
    // non-zeros gate which auxiliary blocks each layer carries.
    let layers_with = |suffix: &str| -> Vec<usize> {
        let mut v: Vec<usize> = (0..43)
            .filter(|l| aux_keys.contains(&format!("layers.{l}.{suffix}")))
            .collect();
        v.sort();
        v
    };
    // exp_probs_b.bias: only on the 40 layers that use the learned
    // router (layers 3..42); first 3 use the hash-routing tables.
    assert_eq!(layers_with("exp_probs_b.bias"), (3..43).collect::<Vec<_>>());
    // attn_compressor: present on every layer with non-zero
    // compress_ratio — layers 2..42.
    assert_eq!(
        layers_with("attn_compressor_norm.weight"),
        (2..43).collect::<Vec<_>>()
    );
    // Indexer: present on the 21 even layers starting at 2.
    let expected_indexer: Vec<usize> = (2..43).step_by(2).collect();
    assert_eq!(layers_with("indexer.proj.weight"), expected_indexer);
    // Model-level entries.
    assert!(aux_keys.contains("output_hc_base.weight"));
    assert!(aux_keys.contains("output_hc_scale.weight"));
    assert!(aux_keys.contains("output_hc_fn.weight"));

    // Per-layer MoE expert files.
    let layers_dir = dir.join("layers");
    assert!(layers_dir.is_dir(), "layers/ directory present");
    let n_layer_files = std::fs::read_dir(&layers_dir).unwrap().count();
    assert_eq!(n_layer_files, 43, "one MoE-expert file per layer");

    // Top-level index reports the Q4_K extraction completed.
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    assert_eq!(cfg["quant"], "q4k");
    assert_eq!(cfg["has_model_weights"], true);
}
