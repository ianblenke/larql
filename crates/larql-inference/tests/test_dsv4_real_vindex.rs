//! Smoke test for the DSv4 vindex loader.

#[test]
#[ignore = "requires Q4K vindex at /tank/ai/deepseek-ai/DeepSeek-V4-Flash-vindex-v3/"]
fn dsv4_loads_real_q4k_vindex_and_runs_one_forward_step() {
    use larql_inference::attention::dsv4_forward::{dsv4_forward_step, Dsv4KvCache};
    use larql_inference::attention::dsv4_load_vindex::load_dsv4_weights_from_vindex;

    let dir = std::path::Path::new("/tank/ai/deepseek-ai/DeepSeek-V4-Flash-vindex-v3");
    if !dir.exists() {
        panic!("vindex not present at {dir:?}");
    }
    let weights = load_dsv4_weights_from_vindex(dir).expect("load DSv4 vindex");
    assert_eq!(weights.layers.len(), 43, "43 layers");
    // Derive dims again to match the loader's view.
    let cfg = larql_vindex::load_vindex_config(dir).expect("load config");
    let dims = larql_inference::attention::dsv4_forward::Dsv4Dims {
        hidden: cfg.hidden_size,
        q_lora_rank: 1024,
        kv_lora_rank: 512,
        n_heads: 64,
        head_dim: 512,
        output_intermediate: 8192,
        ffn_dim: 2048,
        num_experts: 256,
        experts_per_token: 6,
        num_shared_experts: 1,
        vocab: cfg.vocab_size,
        norm_eps: 1e-6,
        expert_weights_scale: 1.5,
    };
    let mut cache = Dsv4KvCache::new(43, dims.kv_lora_rank);
    // Run a single forward step for token id 0 (BOS).
    let start = std::time::Instant::now();
    let logits = dsv4_forward_step(0, &weights, &dims, &mut cache);
    let elapsed = start.elapsed();
    eprintln!("dsv4 forward took {elapsed:?}");
    assert_eq!(logits.len(), dims.vocab);
    assert!(logits.iter().all(|v| v.is_finite()), "logits all finite");
}
