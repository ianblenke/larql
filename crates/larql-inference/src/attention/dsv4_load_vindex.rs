//! DSv4 vindex reader — scope 4.
//!
//! Loads `Dsv4Weights` from a vindex produced by `larql convert
//! gguf-to-vindex --level inference --quant q4k` on a `deepseek4`
//! GGUF. The vindex layout was established in scope 2 (browse-tier)
//! + scope 3 (Q4_K writer for MLA attention + V4 aux blocks +
//! existing MoE per-layer files).
//!
//! What this loader reads:
//!
//! | File / manifest                          | Source                | Used for                         |
//! |------------------------------------------|-----------------------|----------------------------------|
//! | `embeddings.bin`                         | scope 2 browse writer | `Dsv4Weights.embed`              |
//! | `mla_weights_q4k.bin` + manifest         | scope 3 attn_mla      | per-layer Q-down/up + KV-down + output_a/b |
//! | `dsv4_aux.bin` + manifest                | scope 3 dsv4_aux      | per-layer norms (attn_norm, ffn_norm, q_a_norm, kv_a_norm) + sinks + router bias |
//! | `layers/layer_{LL}.weights`              | existing MoE writer   | 256 routed gate/up/down per layer |
//! | `interleaved_q4k.bin` + manifest         | existing FFN writer   | shared-expert gate/up/down + per-layer norms |
//! | `norms.bin`                              | existing norms writer | `ffn_gate_inp` (router) per layer |
//! | `lm_head_q4.bin`                         | existing lm_head writer | `Dsv4Weights.lm_head` |
//!
//! What's intentionally missing (deferred to follow-ups):
//! - HC / compressor / indexer aux tensors — written but unused.
//! - I32 hash-routing tables — skipped by the GGUF integer-tensor
//!   loader (scope 2) and so never enter the writer either.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use larql_models::quant::ggml::TYPE_F32;
use larql_models::quant::lazy::QuantTensor;

use super::dsv4_forward::{
    dsv4_forward_step, Dsv4AttnWeights, Dsv4Dims, Dsv4KvCache, Dsv4LayerWeights, Dsv4MoeWeights,
    Dsv4Weights,
};
use crate::layer_graph::generate::{
    Detokenizer, EosConfig, GenerateResult, Sampler, SamplingConfig,
};

/// Errors surfaced by [`load_dsv4_weights_from_vindex`].
#[derive(Debug, thiserror::Error)]
pub enum Dsv4LoadError {
    #[error("vindex error: {0}")]
    Vindex(#[from] larql_vindex::VindexError),

    #[error("model error: {0}")]
    Model(#[from] larql_models::ModelError),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse: {0}")]
    Parse(String),

    #[error("missing manifest entry: {0} (in {1})")]
    MissingManifestEntry(String, String),
}

/// Top-level loader. Reads the vindex directory and produces a
/// `Dsv4Weights` ready for [`super::dsv4_forward::dsv4_forward_step`].
pub fn load_dsv4_weights_from_vindex(dir: &Path) -> Result<Dsv4Weights, Dsv4LoadError> {
    let cfg = larql_vindex::load_vindex_config(dir)?;
    let arch_family = cfg
        .model_config
        .as_ref()
        .map(|c| c.model_type.clone())
        .unwrap_or_default();
    if arch_family != "deepseek4" {
        return Err(Dsv4LoadError::Parse(format!(
            "expected deepseek4 vindex at {dir:?}, got {arch_family:?}"
        )));
    }

    let num_layers = cfg.num_layers;
    let hidden = cfg.hidden_size;

    let dims = derive_dims(&cfg);

    // ── embed ──
    let (embed_arr, _scale) = larql_vindex::load_vindex_embeddings(dir)?;
    let embed = embed_arr.into_shared();
    if embed.shape() != [cfg.vocab_size, hidden] {
        return Err(Dsv4LoadError::Parse(format!(
            "embed shape {:?} mismatches vocab {} × hidden {}",
            embed.shape(),
            cfg.vocab_size,
            hidden
        )));
    }

    // ── mla_weights_q4k.bin ──
    let mla_mmap = mmap_file(&dir.join("mla_weights_q4k.bin"))?;
    let mla_entries: Vec<ManifestEntry> =
        parse_manifest(&dir.join("mla_weights_q4k_manifest.json"))?;

    // ── dsv4_aux.bin (norms, sinks, router bias) ──
    let aux_mmap = mmap_file(&dir.join("dsv4_aux.bin"))?;
    let aux_entries: Vec<AuxEntry> = parse_aux_manifest(&dir.join("dsv4_aux_manifest.json"))?;

    // ── per-layer MoE expert files (layers/layer_LL.weights) ──
    let layer_files: Vec<_> = (0..num_layers)
        .map(|l| dir.join("layers").join(format!("layer_{l:02}.weights")))
        .collect();

    // ── shared experts (interleaved_q4k.bin) ──
    let interleaved_mmap = mmap_file(&dir.join("interleaved_q4k.bin"))?;
    let interleaved_entries: Vec<ManifestEntry> =
        parse_manifest(&dir.join("interleaved_q4k_manifest.json"))?;

    // ── attn_norm / ffn_norm / router (norms.bin + weight_manifest.json) ──
    let norms_mmap = mmap_file(&dir.join("norms.bin"))?;
    let weight_entries: Vec<WeightEntry> =
        parse_weight_manifest(&dir.join("weight_manifest.json"))?;

    // ── lm_head ──
    let lm_head_mmap = mmap_file(&dir.join("lm_head_q4.bin"))?;
    let (lm_head_offset, lm_head_length, lm_head_type, lm_head_rows, lm_head_cols) =
        find_lm_head_descriptor(&weight_entries, cfg.vocab_size, hidden)?;
    let lm_head = QuantTensor::from_mmap_region(
        Arc::clone(&lm_head_mmap),
        lm_head_offset,
        lm_head_length,
        lm_head_type,
        lm_head_rows,
        lm_head_cols,
    )?;

    // Build per-layer weights.
    let mut layers = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let attn = build_attn_weights(
            layer,
            &dims,
            &mla_mmap,
            &mla_entries,
            &aux_mmap,
            &aux_entries,
            &norms_mmap,
            &weight_entries,
        )?;
        let moe = build_moe_weights(
            layer,
            &dims,
            &layer_files[layer],
            &interleaved_mmap,
            &interleaved_entries,
            &aux_mmap,
            &aux_entries,
            &norms_mmap,
            &weight_entries,
        )?;
        layers.push(Dsv4LayerWeights { attn, moe });
    }

    let final_norm = find_norm_vector(&norms_mmap, &weight_entries, "norm.weight", hidden)?;

    Ok(Dsv4Weights {
        embed,
        layers,
        final_norm,
        lm_head,
    })
}

fn derive_dims(cfg: &larql_vindex::VindexConfig) -> Dsv4Dims {
    let mc = cfg.model_config.as_ref();
    let hidden = cfg.hidden_size;
    let n_heads = mc.map(|c| c.num_q_heads).unwrap_or(64);
    let head_dim = mc.map(|c| c.head_dim).unwrap_or(512);
    let moe = mc.and_then(|c| c.moe.as_ref());
    let num_experts = moe.map(|m| m.num_experts).unwrap_or(256);
    let top_k = moe.map(|m| m.top_k).unwrap_or(6);
    let shared = moe.map(|m| m.shared_expert as usize).unwrap_or(1);
    let ffn_dim = moe.and_then(|m| m.moe_intermediate_size).unwrap_or(2048);
    // q_lora_rank / kv_lora_rank / rope_dim / rope_base aren't
    // surfaced through `VindexModelConfig` yet — DSv4 publishes them
    // in GGUF metadata but the writer doesn't propagate. Default to
    // the published model's values; a follow-up should add explicit
    // fields.
    Dsv4Dims {
        hidden,
        q_lora_rank: 1024,
        kv_lora_rank: 512,
        n_heads,
        head_dim,
        output_intermediate: 8192,
        ffn_dim,
        num_experts,
        experts_per_token: top_k,
        num_shared_experts: shared,
        vocab: cfg.vocab_size,
        norm_eps: 1e-6,
        expert_weights_scale: 1.5,
        rope_dim: 64,
        rope_base: mc.map(|c| c.rope_base).unwrap_or(10_000.0),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_attn_weights(
    layer: usize,
    dims: &Dsv4Dims,
    mla_mmap: &Arc<memmap2::Mmap>,
    mla: &[ManifestEntry],
    aux_mmap: &Arc<memmap2::Mmap>,
    aux: &[AuxEntry],
    norms_mmap: &Arc<memmap2::Mmap>,
    weights: &[WeightEntry],
) -> Result<Dsv4AttnWeights, Dsv4LoadError> {
    let q_a = manifest_quant(mla, mla_mmap, layer, "attn_q_a.weight")?;
    let q_b = manifest_quant(mla, mla_mmap, layer, "attn_q_b.weight")?;
    let kv = manifest_quant(mla, mla_mmap, layer, "attn_kv.weight")?;
    let output_a = manifest_quant(mla, mla_mmap, layer, "attn_output_a.weight")?;
    let output_b = manifest_quant(mla, mla_mmap, layer, "attn_output_b.weight")?;
    let attn_norm = find_norm_vector(
        norms_mmap,
        weights,
        &format!("layers.{layer}.input_layernorm.weight"),
        dims.hidden,
    )?;
    let q_a_norm = aux_vector(
        aux,
        aux_mmap,
        layer,
        "attn_q_a_norm.weight",
        dims.q_lora_rank,
    )?;
    let kv_a_norm = aux_vector(
        aux,
        aux_mmap,
        layer,
        "attn_kv_a_norm.weight",
        dims.kv_lora_rank,
    )?;
    let sinks = aux_vector(aux, aux_mmap, layer, "attn_sinks.weight", dims.n_heads)?;
    Ok(Dsv4AttnWeights {
        q_a,
        q_b,
        kv,
        output_a,
        output_b,
        attn_norm,
        q_a_norm,
        kv_a_norm,
        sinks,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_moe_weights(
    layer: usize,
    dims: &Dsv4Dims,
    layer_file: &Path,
    interleaved_mmap: &Arc<memmap2::Mmap>,
    interleaved: &[ManifestEntry],
    aux_mmap: &Arc<memmap2::Mmap>,
    aux: &[AuxEntry],
    norms_mmap: &Arc<memmap2::Mmap>,
    weights: &[WeightEntry],
) -> Result<Dsv4MoeWeights, Dsv4LoadError> {
    // Router: stored in norms.bin as a flat F32 buffer of
    // `num_experts * hidden` elements; the weight_manifest records
    // it as a vector (1-D). Read the raw bytes and build a
    // QuantTensor with the proper 2-D shape.
    let router_key = format!("layers.{layer}.ffn_gate_inp.weight");
    let (offset, length, _dtype, _rows, _cols) = find_weight_descriptor(weights, &router_key)?;
    let router_bytes = &norms_mmap[offset..offset + length];
    if router_bytes.len() != dims.num_experts * dims.hidden * 4 {
        return Err(Dsv4LoadError::Parse(format!(
            "router {router_key}: expected {} bytes, got {}",
            dims.num_experts * dims.hidden * 4,
            router_bytes.len()
        )));
    }
    let mut router_f32 = Vec::with_capacity(dims.num_experts * dims.hidden);
    for chunk in router_bytes.chunks_exact(4) {
        router_f32.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let router = QuantTensor::from_f32_rows(dims.num_experts, dims.hidden, &router_f32);
    let router_bias = aux_vector_opt(aux, aux_mmap, layer, "exp_probs_b.bias", dims.num_experts);

    // Routed experts: each entry in the per-layer file is a fused
    // gate+up block + a separate down block, interleaved per expert.
    let (expert_gate_ups, expert_downs) = load_per_layer_moe(layer_file, dims)?;

    // Shared experts: live in interleaved_q4k.bin. The writer stores
    // them as separate `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp`
    // tensors; we fuse gate+up into a single QuantTensor by reading
    // the byte ranges contiguously (they're adjacent in the file by
    // the writer's convention).
    let shared_gate = manifest_quant(
        interleaved,
        interleaved_mmap,
        layer,
        "ffn_gate_shexp.weight",
    )?;
    let shared_up = manifest_quant(interleaved, interleaved_mmap, layer, "ffn_up_shexp.weight")?;
    let shared_down = manifest_quant(
        interleaved,
        interleaved_mmap,
        layer,
        "ffn_down_shexp.weight",
    )?;
    // Shared-expert gate/up are separate tensors (not fused) per
    // the writer's `ffn_gate_shexp` / `ffn_up_shexp` layout — the
    // forward does two matvecs for them.

    let ffn_norm = find_norm_vector(
        norms_mmap,
        weights,
        &format!("layers.{layer}.post_attention_layernorm.weight"),
        dims.hidden,
    )?;

    Ok(Dsv4MoeWeights {
        router,
        router_bias,
        expert_gate_ups,
        expert_downs,
        shared_gate,
        shared_up,
        shared_down,
        ffn_norm,
    })
}

/// Read the per-layer MoE file. The existing PerExpert writer (in
/// `larql_vindex::format::weights::write_layers`) stores each layer
/// as a header + offset table + `num_experts` × (gate_up bytes,
/// down bytes) interleaved entries. Each gate_up block is the
/// fused `[2 * ffn_dim, hidden]` matrix; each down block is
/// `[hidden, ffn_dim]`. The loader mmaps the file once and
/// constructs `num_experts` zero-copy `QuantTensor` views per
/// expert (one for gate_up, one for down).
fn load_per_layer_moe(
    path: &Path,
    dims: &Dsv4Dims,
) -> Result<(Vec<QuantTensor>, Vec<QuantTensor>), Dsv4LoadError> {
    let mmap = mmap_file(path)?;
    let parsed = larql_vindex::format::weights::write_layers::parse_layer_weights_header(&mmap[..])
        .ok_or_else(|| Dsv4LoadError::Parse(format!("layer file {path:?}: header parse failed")))?;
    let (fmt, num_entries, _inter, _hidden, offsets) = parsed;

    let dtype = match fmt {
        larql_vindex::format::weights::write_layers::LayerWeightFormat::Q4_K => {
            larql_models::quant::ggml::TYPE_Q4_K
        }
        larql_vindex::format::weights::write_layers::LayerWeightFormat::Q6_K => {
            larql_models::quant::ggml::TYPE_Q6_K
        }
        larql_vindex::format::weights::write_layers::LayerWeightFormat::Q8_0 => {
            larql_models::quant::ggml::TYPE_Q8_0
        }
        larql_vindex::format::weights::write_layers::LayerWeightFormat::F32 => TYPE_F32,
        larql_vindex::format::weights::write_layers::LayerWeightFormat::F16 => {
            larql_models::quant::ggml::TYPE_F16
        }
        other => {
            return Err(Dsv4LoadError::Parse(format!(
                "layer file {path:?}: unsupported quant format {other:?}"
            )))
        }
    };

    if num_entries != dims.num_experts {
        return Err(Dsv4LoadError::Parse(format!(
            "layer file {path:?}: expected {} experts, got {}",
            dims.num_experts, num_entries
        )));
    }

    let mut gate_ups = Vec::with_capacity(num_entries);
    let mut downs = Vec::with_capacity(num_entries);
    for (gu_off, gu_bytes, dn_off, dn_bytes) in offsets {
        let gu = QuantTensor::from_mmap_region(
            Arc::clone(&mmap),
            gu_off,
            gu_bytes,
            dtype,
            2 * dims.ffn_dim,
            dims.hidden,
        )?;
        let dn = QuantTensor::from_mmap_region(
            Arc::clone(&mmap),
            dn_off,
            dn_bytes,
            dtype,
            dims.hidden,
            dims.ffn_dim,
        )?;
        gate_ups.push(gu);
        downs.push(dn);
    }
    Ok((gate_ups, downs))
}

// ─── Manifest helpers ───────────────────────────────────────────────

#[derive(Clone, Debug, serde::Deserialize)]
struct ManifestEntry {
    key: String,
    shape: Vec<usize>,
    format: String,
    offset: u64,
    length: u64,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct AuxEntry {
    key: String,
    shape: Vec<usize>,
    dtype: String,
    offset: u64,
    length: u64,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct WeightEntry {
    key: String,
    kind: String,
    shape: Vec<usize>,
    offset: u64,
    length: u64,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    format: Option<String>,
}

fn parse_manifest(path: &Path) -> Result<Vec<ManifestEntry>, Dsv4LoadError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| Dsv4LoadError::Parse(e.to_string()))
}

fn parse_aux_manifest(path: &Path) -> Result<Vec<AuxEntry>, Dsv4LoadError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| Dsv4LoadError::Parse(e.to_string()))
}

fn parse_weight_manifest(path: &Path) -> Result<Vec<WeightEntry>, Dsv4LoadError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| Dsv4LoadError::Parse(e.to_string()))
}

fn manifest_quant(
    entries: &[ManifestEntry],
    mmap: &Arc<memmap2::Mmap>,
    layer: usize,
    suffix: &str,
) -> Result<QuantTensor, Dsv4LoadError> {
    let key = format!("layers.{layer}.{suffix}");
    let e = entries.iter().find(|e| e.key == key).ok_or_else(|| {
        Dsv4LoadError::MissingManifestEntry(key.clone(), "mla/interleaved".into())
    })?;
    let dtype = match e.format.as_str() {
        "Q4_K" => larql_models::quant::ggml::TYPE_Q4_K,
        "Q6_K" => larql_models::quant::ggml::TYPE_Q6_K,
        "Q8_0" => larql_models::quant::ggml::TYPE_Q8_0,
        "F16" => larql_models::quant::ggml::TYPE_F16,
        "F32" => larql_models::quant::ggml::TYPE_F32,
        other => return Err(Dsv4LoadError::Parse(format!("unknown format tag {other}"))),
    };
    let (rows, cols) = (e.shape[0], e.shape[1]);
    Ok(QuantTensor::from_mmap_region(
        Arc::clone(mmap),
        e.offset as usize,
        e.length as usize,
        dtype,
        rows,
        cols,
    )?)
}

fn aux_vector(
    entries: &[AuxEntry],
    mmap: &Arc<memmap2::Mmap>,
    layer: usize,
    suffix: &str,
    expected_len: usize,
) -> Result<Arc<[f32]>, Dsv4LoadError> {
    aux_vector_opt(entries, mmap, layer, suffix, expected_len).ok_or_else(|| {
        Dsv4LoadError::MissingManifestEntry(format!("layers.{layer}.{suffix}"), "aux".into())
    })
}

fn aux_vector_opt(
    entries: &[AuxEntry],
    mmap: &Arc<memmap2::Mmap>,
    layer: usize,
    suffix: &str,
    expected_len: usize,
) -> Option<Arc<[f32]>> {
    let key = format!("layers.{layer}.{suffix}");
    let e = entries.iter().find(|e| e.key == key)?;
    if e.dtype != "f32" {
        return None;
    }
    let n = expected_len.max(e.shape.iter().product::<usize>());
    let start = e.offset as usize;
    let end = start + (n * 4);
    if end > mmap.len() {
        return None;
    }
    let bytes = &mmap[start..end];
    let mut floats = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(floats.into())
}

fn find_norm_vector(
    mmap: &Arc<memmap2::Mmap>,
    entries: &[WeightEntry],
    key: &str,
    _expected_len: usize,
) -> Result<Arc<[f32]>, Dsv4LoadError> {
    let e = entries
        .iter()
        .find(|e| e.key == key)
        .ok_or_else(|| Dsv4LoadError::MissingManifestEntry(key.into(), "weight_manifest".into()))?;
    let start = e.offset as usize;
    let end = start + e.length as usize;
    if end > mmap.len() {
        return Err(Dsv4LoadError::Parse(format!(
            "norm {key}: range {start}+{} out of bounds (mmap {})",
            e.length,
            mmap.len(),
        )));
    }
    let bytes = &mmap[start..end];
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out.into())
}

fn find_weight_descriptor(
    entries: &[WeightEntry],
    key: &str,
) -> Result<(usize, usize, u32, usize, usize), Dsv4LoadError> {
    let e = entries
        .iter()
        .find(|e| e.key == key)
        .ok_or_else(|| Dsv4LoadError::MissingManifestEntry(key.into(), "weight_manifest".into()))?;
    let dtype = match e.format.as_deref() {
        Some("Q4_K") => larql_models::quant::ggml::TYPE_Q4_K,
        Some("Q6_K") => larql_models::quant::ggml::TYPE_Q6_K,
        Some("Q8_0") => larql_models::quant::ggml::TYPE_Q8_0,
        Some("F16") => larql_models::quant::ggml::TYPE_F16,
        Some("F32") | None => TYPE_F32,
        Some(other) => {
            return Err(Dsv4LoadError::Parse(format!(
                "weight_manifest {key}: unknown format {other}"
            )))
        }
    };
    let (rows, cols) = match e.shape.as_slice() {
        [r, c] => (*r, *c),
        [r] => (1, *r),
        _ => {
            return Err(Dsv4LoadError::Parse(format!(
                "weight_manifest {key}: unexpected shape {:?}",
                e.shape
            )))
        }
    };
    Ok((e.offset as usize, e.length as usize, dtype, rows, cols))
}

fn find_lm_head_descriptor(
    entries: &[WeightEntry],
    vocab: usize,
    hidden: usize,
) -> Result<(usize, usize, u32, usize, usize), Dsv4LoadError> {
    // The lm_head writer emits `lm_head_q4.bin` containing the
    // tensor in Q4_K. The weight_manifest entry records the LOGICAL
    // shape (vocab × hidden) but omits the format; infer it from
    // `file == "lm_head_q4.bin"`.
    let entry = entries
        .iter()
        .find(|e| e.key == "lm_head.weight" || e.key == "output.weight");
    if let Some(e) = entry {
        let dtype = if e.file.as_deref() == Some("lm_head_q4.bin") {
            larql_models::quant::ggml::TYPE_Q4_K
        } else {
            match e.format.as_deref() {
                Some("Q4_K") => larql_models::quant::ggml::TYPE_Q4_K,
                Some("Q8_0") => larql_models::quant::ggml::TYPE_Q8_0,
                Some("F16") => larql_models::quant::ggml::TYPE_F16,
                _ => TYPE_F32,
            }
        };
        let (rows, cols) = match e.shape.as_slice() {
            [r, c] => (*r, *c),
            _ => (vocab, hidden),
        };
        return Ok((e.offset as usize, e.length as usize, dtype, rows, cols));
    }
    // Fallback: entire `lm_head_q4.bin` as a single Q4_K tensor.
    let total_q4k_bytes = (vocab * hidden / 256) * 144;
    Ok((
        0,
        total_q4k_bytes,
        larql_models::quant::ggml::TYPE_Q4_K,
        vocab,
        hidden,
    ))
}

fn mmap_file(path: &Path) -> Result<Arc<memmap2::Mmap>, Dsv4LoadError> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(Arc::new(mmap))
}

/// Derive the runtime `Dsv4Dims` from a `Dsv4Weights`'s embedded
/// shapes. Mirrors `derive_dims` but doesn't need the vindex config
/// — useful for callers (like the server dispatch) that have the
/// cached weights but not the config.
pub fn dsv4_dims_from_weights(weights: &Dsv4Weights) -> Dsv4Dims {
    let hidden = weights.embed.shape()[1];
    let vocab = weights.embed.shape()[0];
    let [_, kv_lora_rank] = weights.layers[0].attn.kv.shape();
    let [_, q_lora_rank] = weights.layers[0].attn.q_a.shape();
    let [q_b_rows, _] = weights.layers[0].attn.q_b.shape();
    let [output_intermediate, _] = weights.layers[0].attn.output_a.shape();
    let [_, ffn_dim] = weights.layers[0].moe.shared_down.shape();
    let num_experts = weights.layers[0].moe.expert_gate_ups.len();
    // V4 published values for fields we can't reconstruct from the
    // weights alone (n_heads, top-k, scales).
    Dsv4Dims {
        hidden,
        q_lora_rank,
        kv_lora_rank,
        n_heads: 64,
        head_dim: q_b_rows / 64,
        output_intermediate,
        ffn_dim,
        num_experts,
        experts_per_token: 6,
        num_shared_experts: 1,
        vocab,
        norm_eps: 1e-6,
        expert_weights_scale: 1.5,
        rope_dim: 64,
        rope_base: 10_000.0,
    }
}

/// Full generate driver — mirrors `qwen35_generate_with_sampling`
/// for the DSv4 path. Prefill on the prompt tokens, then decode up
/// to `max_tokens` via the sampler. Returns the same
/// `GenerateResult` shape the server's chat handler consumes.
pub fn dsv4_generate_with_sampling(
    weights: &Dsv4Weights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult {
    let dims = dsv4_dims_from_weights(weights);

    if prompt_ids.is_empty() {
        return GenerateResult::empty_error(crate::layer_graph::generate::GenerateError::Other {
            reason: "prompt_ids is empty".into(),
        });
    }

    let num_layers = weights.layers.len();
    let mut cache = Dsv4KvCache::new(num_layers, dims.kv_lora_rank);

    // Prefill: only the last token's logits matter for the first
    // decode step. Earlier tokens advance the KV-latent cache.
    let prefill_start = Instant::now();
    let mut last_logits = None;
    for &tok in prompt_ids {
        last_logits = Some(dsv4_forward_step(tok, weights, &dims, &mut cache));
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut sampler = Sampler::new(sampling);
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(prompt_ids);

    let mut out: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms: Vec<f64> = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    for _ in 0..max_tokens {
        let step_start = Instant::now();
        let Some(logits) = last_logits.take() else {
            break;
        };
        let Some(next_id) =
            sampler.sample_with_history(logits.as_slice().unwrap_or(&[]), &generated)
        else {
            break;
        };
        let text = detok.push(next_id);
        if eos.is_eos(next_id, &text) {
            break;
        }
        out.push((text, 1.0));
        generated.push(next_id);
        last_logits = Some(dsv4_forward_step(next_id, weights, &dims, &mut cache));
        decode_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);
    }

    GenerateResult {
        tokens: out,
        prefill_ms,
        decode_ms,
        stage_timings: crate::layer_graph::generate::StageTimings::default(),
        error: None,
    }
}
