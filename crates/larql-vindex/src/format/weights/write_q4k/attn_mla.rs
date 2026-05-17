//! MLA attention Q4_K writer — DeepSeek V4 Flash (DSv4 scope 3).
//!
//! Emits `mla_weights_q4k.bin` + sidecar manifest, mirroring the
//! standard-attention [`super::attn`] writer pattern but for the 5
//! MLA tensors per layer (DSv4 has no Q/K/V/O):
//!
//! | Slot | Key (post-norm)                  | Shape          | Format |
//! |------|----------------------------------|----------------|--------|
//! | 0    | `layers.{L}.attn_q_a.weight`     | `[4096, 1024]` | Q4_K   |
//! | 1    | `layers.{L}.attn_q_b.weight`     | `[1024, 32768]`| Q4_K   |
//! | 2    | `layers.{L}.attn_kv.weight`      | `[4096, 512]`  | Q4_K   |
//! | 3    | `layers.{L}.attn_output_a.weight`| `[4096, 8192]` | Q4_K   |
//! | 4    | `layers.{L}.attn_output_b.weight`| `[8192, 4096]` | Q4_K   |
//!
//! Source rows are 256-aligned in the GGUF (DSv4 dims are all
//! powers of 2 ≥ 512), so the row-padding helper used by
//! `super::attn` is a no-op here — kept in the call for shape
//! consistency.

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::super::manifest::Q4kManifestEntry;
use super::super::write_f32::WeightSource;
use super::{pad_rows_to_block, QuantBlockFormat};

/// Write the 5 MLA matmul tensors per layer to
/// `mla_weights_q4k.bin` + manifest.
///
/// All 5 slots emit Q4_K (no Q6_K override for the MLA writer — the
/// attention low-rank A/B / Q_b path is dense matmul and the Q4_K /
/// Q6_K quality delta is negligible relative to the per-expert Q4_K
/// already dominating the model). The expert paths in
/// [`super::moe_layers`] retain their own dtype mix.
pub(super) fn write_mla_weights_q4k(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    let arch = source.arch();
    let path = dir.join(MLA_WEIGHTS_Q4K_BIN);
    let mut file = BufWriter::new(std::fs::File::create(&path)?);
    let mut offset: u64 = 0;
    let mut manifest: Vec<Q4kManifestEntry> = Vec::with_capacity(num_layers * 5);

    for layer in 0..num_layers {
        callbacks.on_layer_start(COMP_ATTN_Q4K, layer, num_layers);

        // Build (key, optional_tensor) tuples for the 5 slots. Missing
        // tensors are skipped rather than failing — the reader's
        // manifest will simply omit them and the forward can decide
        // what to do (e.g. on an arch variant that drops the
        // low-rank output split, output_a/b would be absent).
        let slots: [(Option<String>, &str); 5] = [
            (arch.mla_q_a_key(layer), "attn_q_a"),
            (arch.mla_q_b_key(layer), "attn_q_b"),
            (arch.mla_kv_a_key(layer), "attn_kv"),
            (
                Some(format!("layers.{layer}.attn_output_a.weight")),
                "attn_output_a",
            ),
            (
                Some(format!("layers.{layer}.attn_output_b.weight")),
                "attn_output_b",
            ),
        ];

        for (key_opt, _label) in &slots {
            let Some(key) = key_opt.as_ref() else {
                continue;
            };
            let Some((data, rows, cols)) = source.get_tensor(key) else {
                continue;
            };
            let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
            let _ = rows;
            let bytes = quantize_q4_k(&padded);
            file.write_all(&bytes)?;
            let length = bytes.len() as u64;
            manifest.push(Q4kManifestEntry {
                key: key.clone(),
                shape: vec![rows, padded_cols],
                format: QuantBlockFormat::Q4K,
                offset,
                length,
            });
            offset += length;
        }

        callbacks.on_layer_done(COMP_ATTN_Q4K, layer, 0.0);
    }

    file.flush()?;
    drop(file);

    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(MLA_WEIGHTS_Q4K_MANIFEST_JSON), manifest_json)?;
    Ok(())
}
