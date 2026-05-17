//! DeepSeek V4 Flash auxiliary-tensor writer — DSv4 scope 3.
//!
//! Emits `dsv4_aux.bin` + sidecar manifest holding the V4-specific
//! tensors that have no analogue in other architectures:
//!
//! - **Per-layer norms** beyond the standard `attn_norm` / `ffn_norm`:
//!   `attn_q_a_norm` (1024), `attn_kv_a_norm` (512).
//! - **Per-layer attention sinks**: `attn_sinks` (64).
//! - **Per-layer router bias**: `exp_probs_b.bias` (256).
//! - **Hyper-connection** (attn + ffn): `hc_{attn,ffn}_{base,scale}`
//!   (1-D, F32) and `hc_{attn,ffn}_fn` (2-D, 16384×24, F16).
//! - **Output hyper-connection** (model-level):
//!   `output_hc_{base,scale}` + `output_hc_fn`.
//! - **Per-layer attention compressor** (variable shape — only
//!   present on layers where `compress_ratios[layer+1] > 0`):
//!   `attn_compressor_{ape,gate,kv,norm}`.
//! - **Per-layer indexer** (variable, present on some layers):
//!   `indexer.attn_q_b`, `indexer.proj`,
//!   `indexer_compressor_{ape,gate,kv,norm}`.
//!
//! Each entry is written byte-for-byte as the dense GGUF loader
//! produced it (F32 form post-`dequantize` — F16 source is widened
//! to F32 by the load path; storage cost is small because these
//! tensors are tiny relative to the matmul body). The manifest
//! records `(key, dtype="f32", shape, offset, length)` so the
//! scope-4 reader can mmap a sized slice without re-scanning the
//! GGUF.
//!
//! **Not yet covered**: I32 `ffn_gate_tid2eid` hash-routing tables
//! (first 3 layers). The dense GGUF loader skips integer tensors,
//! so they're not in the `WeightSource`. Scope 4 will either
//! re-read them from the GGUF directly or add a raw-bytes side
//! channel.

use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;
use crate::format::filenames::*;

use super::super::write_f32::WeightSource;

/// One DSv4-aux tensor in the sidecar manifest.
///
/// `shape` is the *post-dim-swap* shape — i.e. matches what
/// `WeightSource::get_tensor` returns (`(data, rows, cols)`). 1-D
/// entries have `shape.len() == 1`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dsv4AuxEntry {
    pub key: String,
    pub shape: Vec<usize>,
    pub dtype: &'static str,
    pub offset: u64,
    pub length: u64,
}

/// Per-layer DSv4-aux key suffixes we probe for. The writer probes
/// `layers.{L}.{suffix}` for each combination and emits an entry
/// when the source has the tensor.
///
/// 1-D entries (norms, sinks, bias, hc base/scale).
const PER_LAYER_AUX_VEC_SUFFIXES: &[&str] = &[
    "attn_q_a_norm.weight",
    "attn_kv_a_norm.weight",
    "attn_sinks.weight",
    "exp_probs_b.bias",
    "hc_attn_base.weight",
    "hc_attn_scale.weight",
    "hc_ffn_base.weight",
    "hc_ffn_scale.weight",
    "attn_compressor_norm.weight",
    "indexer_compressor_norm.weight",
];

/// 2-D per-layer aux entries (HC fn matrices + compressor + indexer
/// matmul tensors).
const PER_LAYER_AUX_MAT_SUFFIXES: &[&str] = &[
    "hc_attn_fn.weight",
    "hc_ffn_fn.weight",
    "attn_compressor_ape.weight",
    "attn_compressor_gate.weight",
    "attn_compressor_kv.weight",
    "indexer.attn_q_b.weight",
    "indexer.proj.weight",
    "indexer_compressor_ape.weight",
    "indexer_compressor_gate.weight",
    "indexer_compressor_kv.weight",
];

/// Model-level aux entries (not per-layer): `output_norm` is already
/// written by the norms writer; `output_hc_*` is V4-specific.
const MODEL_AUX_VEC_KEYS: &[&str] = &["output_hc_base.weight", "output_hc_scale.weight"];

const MODEL_AUX_MAT_KEYS: &[&str] = &["output_hc_fn.weight"];

/// Write `dsv4_aux.bin` + manifest under `dir`. Probes every
/// per-layer + global aux suffix in the source; absent tensors are
/// silently skipped (the manifest only records what was present).
pub(super) fn write_dsv4_aux(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
) -> Result<(), VindexError> {
    let path = dir.join(DSV4_AUX_BIN);
    let mut file = BufWriter::new(std::fs::File::create(&path)?);
    let mut offset: u64 = 0;
    let mut manifest: Vec<Dsv4AuxEntry> = Vec::new();

    let mut write_vec = |file: &mut BufWriter<std::fs::File>,
                         offset: &mut u64,
                         manifest: &mut Vec<Dsv4AuxEntry>,
                         key: String,
                         data: &[f32]|
     -> Result<(), VindexError> {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;
        let length = bytes.len() as u64;
        manifest.push(Dsv4AuxEntry {
            key,
            shape: vec![data.len()],
            dtype: "f32",
            offset: *offset,
            length,
        });
        *offset += length;
        Ok(())
    };

    let mut write_mat = |file: &mut BufWriter<std::fs::File>,
                         offset: &mut u64,
                         manifest: &mut Vec<Dsv4AuxEntry>,
                         key: String,
                         data: &[f32],
                         rows: usize,
                         cols: usize|
     -> Result<(), VindexError> {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;
        let length = bytes.len() as u64;
        manifest.push(Dsv4AuxEntry {
            key,
            shape: vec![rows, cols],
            dtype: "f32",
            offset: *offset,
            length,
        });
        *offset += length;
        Ok(())
    };

    for layer in 0..num_layers {
        let prefix = format!("layers.{layer}.");
        for suffix in PER_LAYER_AUX_VEC_SUFFIXES {
            let key = format!("{prefix}{suffix}");
            if let Some(v) = source.get_vector(&key) {
                write_vec(&mut file, &mut offset, &mut manifest, key, &v)?;
            }
        }
        for suffix in PER_LAYER_AUX_MAT_SUFFIXES {
            let key = format!("{prefix}{suffix}");
            if let Some((data, rows, cols)) = source.get_tensor(&key) {
                write_mat(
                    &mut file,
                    &mut offset,
                    &mut manifest,
                    key,
                    &data,
                    rows,
                    cols,
                )?;
            }
        }
    }
    for key in MODEL_AUX_VEC_KEYS {
        if let Some(v) = source.get_vector(key) {
            write_vec(
                &mut file,
                &mut offset,
                &mut manifest,
                (*key).to_string(),
                &v,
            )?;
        }
    }
    for key in MODEL_AUX_MAT_KEYS {
        if let Some((data, rows, cols)) = source.get_tensor(key) {
            write_mat(
                &mut file,
                &mut offset,
                &mut manifest,
                (*key).to_string(),
                &data,
                rows,
                cols,
            )?;
        }
    }

    file.flush()?;
    drop(file);

    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(DSV4_AUX_MANIFEST_JSON), manifest_json)?;
    Ok(())
}
