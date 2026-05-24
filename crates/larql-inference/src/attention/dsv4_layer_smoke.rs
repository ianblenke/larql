//! End-to-end smoke test: real-GGUF layer 0 attention forward.
//!
//! Loads a single layer's worth of real DSv4-Flash weights via the
//! full GGUF→storage pipeline (Stages 8h-4b-1..4a), gets the
//! borrowed dispatcher arm via Stage 8h-2b, and runs Stage 8h-1's
//! attention dispatcher on a synthetic 1-token input.
//!
//! Purpose: this is the first real test of the load→compute chain on
//! actual model weights. Synthetic-tensor tests verify each piece in
//! isolation; this is the first one that fires the full chain on real
//! data. Anything that goes wrong here (shape mismatches, NaN propagation,
//! wrong RoPE base, missing weights) surfaces here rather than mid-way
//! through a 22-minute full-model load.
//!
//! Layer 0 is the cheapest: hash-routed FFN (which we currently skip
//! since the int routing table isn't loaded yet), no compressor, no
//! indexer. Just the standard attention path (Stage 8a).

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use larql_models::loading::gguf::GgufFile;

    use crate::attention::dsv4_attn_dispatch::dsv4_attn_layer;
    use crate::attention::dsv4_full_loader::load_dsv4_layer;
    use crate::attention::dsv4_hyperparams_load::DsV4MetadataError;
    use crate::attention::dsv4_storage_build::DsV4Hyperparams;

    /// End-to-end: load layer 0 + run a 1-token forward through the
    /// attention dispatcher. Expect finite output of shape (1, n_embd).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn smoke_layer_0_attention_forward() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp: Result<DsV4Hyperparams, DsV4MetadataError> = DsV4Hyperparams::from_gguf(&gguf);
        let hp = hp.expect("hyperparams");
        let (storage, variant) = load_dsv4_layer(&gguf, &hp, 0).expect("load layer 0");

        // Layer 0 is NoCompress in the real DSv4-Flash model.
        assert_eq!(variant.compress_ratio, None);
        assert!(!variant.has_indexer);

        // Build a single-token input — synthetic but with the right
        // n_embd (4096). Use a small magnitude so SwiGLU/softmax don't
        // saturate immediately.
        let n_tokens = 1;
        let n_embd = hp.n_embd;
        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(_, d)| {
            ((d as f32 * 0.0013).sin()) * 0.1
        });

        // Build the no-compress dispatcher arm directly (no compress or
        // indexer params needed since variant is NoCompress).
        let layer = storage.dispatcher_layer(&None, &None);
        assert_eq!(layer.variant_name(), "no_compress");

        let out = dsv4_attn_layer(x.view(), &layer, 0);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);
        // Finite-ness is the headline correctness check — if anything in
        // RoPE/SWA/grouped-o-proj goes off the rails we get NaN/Inf.
        let n_nonfinite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            n_nonfinite, 0,
            "{n_nonfinite} non-finite values in layer-0 attention output"
        );
        // Non-trivial: at least one nonzero output.
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "layer-0 attention output is all zeros");
    }
}
