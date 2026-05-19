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
    pad_cols_to_256, quantize_f32, quantize_moe_entries, write_layer_weights, LayerEntry,
    LayerWeightFormat,
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

    // Maps a `LayerWeightFormat` (per-file) to the GGML tensor_type
    // we'd accept for raw-byte passthrough at the gate_up slot.
    // Returns `None` for formats we don't have a passthrough kernel
    // for at the MoE expert tier yet.
    fn passthrough_tensor_type(fmt: LayerWeightFormat) -> Option<u32> {
        match fmt {
            LayerWeightFormat::Q4_K => Some(larql_models::quant::ggml::TYPE_Q4_K),
            LayerWeightFormat::Q6_K => Some(larql_models::quant::ggml::TYPE_Q6_K),
            _ => None,
        }
    }

    for expert_id in 0..num_experts {
        let gate_key = arch.expert_ffn_gate_key(layer, expert_id);
        let up_key = arch.expert_ffn_up_key(layer, expert_id);
        let down_key = arch.expert_ffn_down_key(layer, expert_id);

        // Try raw byte-concat passthrough for the gate+up half of the
        // entry: in K-quant formats each [inter, hidden] row is an
        // independent run of super-blocks (no cross-row state), so
        // bytewise concatenation of two raw-quant tensors of the same
        // format equals the bytes of quantizing their f32 concat. This
        // preserves imatrix-aware quantization for the MoE gate/up
        // path — PR #195/#196/#197 established the same pattern for
        // attn/deltanet writers. For Coder-Next, gate/up are both
        // Q4_K and hidden is 2048 (multiple of 256), so passthrough
        // fires; the f32-round-trip below stays as the safe fallback.
        //
        // `down` is treated separately below — when source format
        // matches target, byte-passthrough; otherwise the existing
        // f32 dequant + pad + requant runs (no behavioural change).
        let target_ttype = passthrough_tensor_type(fmt);
        let gate_raw = gate_key
            .as_ref()
            .and_then(|k| source.get_quant_raw(k))
            .filter(|(_, t, _, _)| Some(*t) == target_ttype);
        let up_raw = up_key
            .as_ref()
            .and_then(|k| source.get_quant_raw(k))
            .filter(|(_, t, _, _)| Some(*t) == target_ttype);
        let down_raw = down_key
            .as_ref()
            .and_then(|k| source.get_quant_raw(k))
            .filter(|(_, t, _, _)| Some(*t) == target_ttype);

        let gate_up_passthrough = match (&gate_raw, &up_raw) {
            (Some((gb, _, gr, gc)), Some((ub, _, ur, uc)))
                if gr == ur && gc == uc && *gc == hidden && *gr == moe_inter =>
            {
                let mut bytes = Vec::with_capacity(gb.len() + ub.len());
                bytes.extend_from_slice(gb);
                bytes.extend_from_slice(ub);
                Some(bytes)
            }
            _ => None,
        };

        let down_passthrough = match &down_raw {
            Some((db, _, dr, dc))
                if *dr == hidden
                    && *dc == moe_inter
                    && moe_inter.is_multiple_of(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS) =>
            {
                Some(db.clone())
            }
            _ => None,
        };

        // Resolve gate_up bytes: try raw concat first, then fall back
        // to f32 dequant + interleave + quantize.
        let gate_up = if let Some(bytes) = gate_up_passthrough {
            bytes
        } else {
            let gate = gate_key.as_ref().and_then(|k| source.get_tensor(k));
            let up = up_key.as_ref().and_then(|k| source.get_tensor(k));
            let (Some((gate_f32, _, _)), Some((up_f32, _, _))) = (gate, up) else {
                // Missing — skip the layer entirely (a half-populated
                // layer file would mislead downstream readers that
                // assume `num_entries=num_experts`).
                return Ok(Vec::new());
            };
            let mut gate_up_f32 = Vec::with_capacity(gate_f32.len() + up_f32.len());
            gate_up_f32.extend_from_slice(&gate_f32);
            gate_up_f32.extend_from_slice(&up_f32);
            quantize_f32(&gate_up_f32, fmt)?
        };

        // Resolve down bytes: same pattern — raw passthrough when
        // source matches target, else f32 dequant + pad + quantize.
        let down = if let Some(bytes) = down_passthrough {
            bytes
        } else {
            let Some((down_f32, _, _)) = down_key.as_ref().and_then(|k| source.get_tensor(k))
            else {
                return Ok(Vec::new());
            };
            let (down_padded, _) = pad_cols_to_256(&down_f32, hidden, moe_inter);
            quantize_f32(&down_padded, fmt)?
        };

        entries.push(LayerEntry { gate_up, down });
    }

    Ok(entries)
}
