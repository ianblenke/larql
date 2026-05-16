//! Stage 3 — per-layer FFN weights for MoE models (§5.12).
//!
//! Two MoE expert layouts are supported, both writing the same
//! `layers/layer_{L:02}.weights` files with `num_entries=num_experts`:
//!
//! 1. **`ExpertFormat::PackedBF16`** (Gemma 4 26B A4B hybrid MoE):
//!    Source ships one BF16 tensor per projection per layer carrying
//!    every expert stacked (`[num_experts, 2*inter, hidden]`). Decoded
//!    via [`quantize_moe_entries`].
//!
//! 2. **`ExpertFormat::PerExpert`** (Qwen 3.6 35B-A3B / Mixtral-style):
//!    Source ships one f32 tensor per expert per projection
//!    (`mlp.experts.{e}.{gate,up,down}_proj.weight`). 256 experts ×
//!    3 projections × 40 layers for qwen35moe.  GGUFs that pack these
//!    into a 3-D `ffn_*_exps.weight` tensor are surfaced through the
//!    same per-expert HF aliases by `load_gguf_lazy_tensors`, so
//!    [`WeightSource::get_tensor`] returns the dequantised per-expert
//!    `[inter, hidden]` matrix at the right key.
//!
//! For dense models (`is_moe() == false`): no-op —
//! `interleaved_q4k.bin` (stage 2) remains the primary FFN store.

use std::path::Path;

use crate::error::VindexError;

use super::super::write_f32::WeightSource;
use super::super::write_layers::{
    quantize_dense_entry, quantize_moe_entries, write_layer_weights, LayerEntry, LayerWeightFormat,
};

pub(super) fn write_per_layer_moe_q4k(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
) -> Result<(), VindexError> {
    let arch = source.arch();
    if !arch.is_moe() {
        return Ok(());
    }

    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = arch.config().hidden_size;
    let fmt = LayerWeightFormat::Q4_K;

    match arch.expert_format() {
        larql_models::ExpertFormat::PackedBF16 => {
            for layer in 0..num_layers {
                let gu_key = arch.packed_experts_gate_up_key(layer);
                let dn_key = arch.packed_experts_down_key(layer);
                let gu_bytes = gu_key.as_ref().and_then(|k| source.get_packed_bf16(k));
                let dn_bytes = dn_key.as_ref().and_then(|k| source.get_packed_bf16(k));

                if let (Some(gu), Some(dn)) = (gu_bytes, dn_bytes) {
                    let entries =
                        quantize_moe_entries(&gu, &dn, num_experts, moe_inter, hidden, fmt)?;
                    write_layer_weights(dir, layer, fmt, &entries, moe_inter, hidden)?;
                }
            }
        }
        larql_models::ExpertFormat::PerExpert => {
            for layer in 0..num_layers {
                let entries =
                    build_per_expert_entries(source, layer, num_experts, moe_inter, hidden, fmt)?;
                // If no expert tensors were resolvable at this layer
                // (e.g. an upstream loader returned None for every key),
                // skip emitting an empty `layer_LL.weights` file rather
                // than writing a degenerate header — a present-but-empty
                // file would silently mislead readers downstream.
                if entries.is_empty() {
                    continue;
                }
                write_layer_weights(dir, layer, fmt, &entries, moe_inter, hidden)?;
            }
        }
        // Other expert formats (e.g. PackedMxfp4 for GPT-OSS) are not
        // yet wired through this writer. The original guard pre-2026-05
        // returned Ok(()) for anything that wasn't PackedBF16; preserve
        // that behaviour here so non-target arches keep building.
        larql_models::ExpertFormat::PackedMxfp4 => return Ok(()),
    }
    Ok(())
}

/// Resolve and quantise one layer's per-expert tensors into
/// `LayerEntry` instances. Returns an empty Vec when no expert at this
/// layer surfaced gate/up/down — i.e. the caller should skip the file.
fn build_per_expert_entries(
    source: &dyn WeightSource,
    layer: usize,
    num_experts: usize,
    moe_inter: usize,
    hidden: usize,
    fmt: LayerWeightFormat,
) -> Result<Vec<LayerEntry>, VindexError> {
    let arch = source.arch();
    let mut entries: Vec<LayerEntry> = Vec::with_capacity(num_experts);

    for expert_id in 0..num_experts {
        let gate_key = arch.expert_ffn_gate_key(layer, expert_id);
        let up_key = arch.expert_ffn_up_key(layer, expert_id);
        let down_key = arch.expert_ffn_down_key(layer, expert_id);

        let gate = gate_key.as_ref().and_then(|k| source.get_tensor(k));
        let up = up_key.as_ref().and_then(|k| source.get_tensor(k));
        let down = down_key.as_ref().and_then(|k| source.get_tensor(k));

        let (Some((gate_f32, _, _)), Some((up_f32, _, _)), Some((down_f32, _, _))) =
            (gate, up, down)
        else {
            // Any expert missing → skip the layer entirely. The dense
            // `quantize_dense_entry` requires all three tensors, and a
            // partial layer would mislead the manifest. Producing a
            // half-populated layer file is worse than producing none —
            // a downstream loader would assume `num_entries=num_experts`
            // and read garbage offsets.
            return Ok(Vec::new());
        };

        let entry = quantize_dense_entry(&gate_f32, &up_f32, &down_f32, moe_inter, hidden, fmt)?;
        entries.push(entry);
    }

    Ok(entries)
}
