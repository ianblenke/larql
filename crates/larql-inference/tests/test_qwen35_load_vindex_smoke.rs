//! Live smoke against the Qwen 3.6 35B-A3B vindex (task 2b.6 of
//! `vindex-qwen35moe-reader`). Validates that PRs #161 / #163 / #164 /
//! #165 / #166's bridges + orchestrator produce a structurally
//! complete `Qwen35Weights` from a real vindex on disk.
//!
//! Run with:
//!
//! ```sh
//! LARQL_QWEN36_VINDEX=/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v3 \
//!   cargo test -p larql-inference --release \
//!   --test test_qwen35_load_vindex_smoke -- --nocapture
//! ```
//!
//! Skips silently when `LARQL_QWEN36_VINDEX` is unset — keeps CI green
//! without the 60 GB vindex on disk.

use std::path::PathBuf;

use larql_inference::attention::qwen35_block::Qwen35LayerWeights;
use larql_inference::attention::qwen35_load_vindex::load_qwen35_weights_from_vindex;

#[test]
fn qwen35_35b_a3b_vindex_loads_into_structurally_complete_qwen35weights() {
    let Ok(dir) = std::env::var("LARQL_QWEN36_VINDEX") else {
        eprintln!(
            "skipping: set LARQL_QWEN36_VINDEX=/path/to/Qwen3.6-35B-A3B-vindex to run \
             this smoke against the real 60 GB vindex"
        );
        return;
    };
    let path = PathBuf::from(&dir);

    let weights = load_qwen35_weights_from_vindex(&path).unwrap_or_else(|e| {
        panic!("load_qwen35_weights_from_vindex({dir}) failed: {e}");
    });

    // Structural shape: 40 layers, every linear layer has DeltaNet,
    // every full-attn layer has Qwen35Attention.
    assert_eq!(
        weights.layers.len(),
        40,
        "Qwen 3.6 35B-A3B has 40 layers; got {}",
        weights.layers.len()
    );

    let mut linear = 0usize;
    let mut full = 0usize;
    let mut moe_present = 0usize;
    for (i, layer) in weights.layers.iter().enumerate() {
        match &layer.block {
            Qwen35LayerWeights::Linear(dn) => {
                linear += 1;
                assert!(
                    dn.attn_qkv_quant.is_some(),
                    "layer {i} (linear): attn_qkv_quant must be populated from deltanet bytes"
                );
                assert!(
                    dn.attn_gate_quant.is_some(),
                    "layer {i} (linear): attn_gate_quant must be populated"
                );
                assert!(
                    dn.ssm_out_quant.is_some(),
                    "layer {i} (linear): ssm_out_quant must be populated"
                );
                assert!(
                    !dn.ssm_norm.is_empty(),
                    "layer {i} (linear): ssm_norm vector empty"
                );
                assert!(
                    !dn.ssm_dt.is_empty(),
                    "layer {i} (linear): ssm_dt vector empty"
                );
                assert!(
                    !dn.ssm_a.is_empty(),
                    "layer {i} (linear): ssm_a vector empty"
                );
                // ssm_alpha + ssm_beta are dense f32 [n_v_heads, hidden].
                assert!(
                    dn.ssm_alpha.ncols() > 0,
                    "layer {i} (linear): ssm_alpha empty"
                );
                assert!(
                    dn.ssm_beta.ncols() > 0,
                    "layer {i} (linear): ssm_beta empty"
                );
            }
            Qwen35LayerWeights::Attention(at) => {
                full += 1;
                assert!(
                    at.attn_q_quant.is_some(),
                    "layer {i} (full-attn): attn_q_quant must be populated"
                );
                assert!(
                    at.attn_k_quant.is_some(),
                    "layer {i} (full-attn): attn_k_quant must be populated"
                );
                assert!(
                    at.attn_v_quant.is_some(),
                    "layer {i} (full-attn): attn_v_quant must be populated"
                );
                assert!(
                    at.attn_output_quant.is_some(),
                    "layer {i} (full-attn): attn_output_quant must be populated"
                );
            }
        }
        if layer.moe.is_some() {
            moe_present += 1;
        }
    }

    // full_attention_interval=4 → layers 3, 7, 11, …, 39 are full-attn (10 layers);
    // the other 30 are DeltaNet.
    assert_eq!(linear, 30, "expected 30 linear-attention layers");
    assert_eq!(full, 10, "expected 10 full-attention layers");
    assert_eq!(moe_present, 40, "every layer must carry MoE (qwen35moe)");

    assert!(
        !weights.final_norm.is_empty(),
        "final_norm vector must be populated from norms.bin"
    );
    assert!(
        weights.lm_head_quant.is_some(),
        "lm_head_quant must be populated from lm_head_q4.bin"
    );

    eprintln!(
        "smoke OK — 40 layers ({linear} linear + {full} full-attn), \
         {moe_present} MoE blocks, final_norm + lm_head_quant present"
    );
}
