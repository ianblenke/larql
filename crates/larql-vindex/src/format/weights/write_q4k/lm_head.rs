//! Stage 6 — `lm_head_q4.bin`.
//!
//! Q4_K of the output projection matrix. Falls back to embed_tokens
//! when the architecture ties the embed and lm_head weights (Gemma,
//! Qwen, etc.); the source layer surfaces that via `source.lm_head()`.
//! Manifest entry is appended to the running norms manifest so
//! `weight_manifest.json` references everything in one list.

use std::path::Path;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::super::write_f32::{kind, WeightEntry, WeightSource};
use super::pad_rows_to_block;

pub(super) fn write_lm_head_q4k(
    source: &dyn WeightSource,
    dir: &Path,
    norm_entries: &mut Vec<WeightEntry>,
) -> Result<(), VindexError> {
    if let Some((data, rows, cols)) = source.lm_head() {
        // Truncate to logical vocab so the on-disk row count matches
        // `config.vocab_size` (and therefore matches what the loader
        // expects). Some GGUFs ship `token_embd` / lm_head with extra
        // SIMD-alignment rows beyond the logical vocab — see
        // `build::write_embeddings` for the matching truncation on the
        // embed side.
        let logical_vocab = source.arch().config().vocab_size.unwrap_or(rows);
        let (truncated_data, truncated_rows) = if rows > logical_vocab {
            let truncated: Vec<f32> = data[..logical_vocab * cols].to_vec();
            (truncated, logical_vocab)
        } else {
            (data, rows)
        };
        let (padded, padded_cols) = pad_rows_to_block(&truncated_data, truncated_rows, cols);
        let q_bytes = quantize_q4_k(&padded);
        std::fs::write(dir.join(LM_HEAD_Q4_BIN), &q_bytes)?;
        // Record in norms manifest so a single weight_manifest.json references
        // everything non-quantised-via-layout. Shape records the stored
        // `padded_cols` — callers route through the matvec dispatch which
        // uses shape[1] as `K`, so the padding stays invisible provided the
        // input activation buffer is zero-padded to match.
        norm_entries.push(WeightEntry {
            key: "lm_head.weight".into(),
            kind: kind::TENSOR_Q4K.into(),
            shape: vec![truncated_rows, padded_cols],
            offset: 0,
            length: q_bytes.len() as u64,
            file: LM_HEAD_Q4_BIN.into(),
        });
    }
    Ok(())
}
