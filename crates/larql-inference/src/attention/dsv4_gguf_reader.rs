//! Read DSv4 per-layer tensors from a GGUF file.
//!
//! Layered on:
//! - [`larql_models::loading::gguf::GgufFile`] — open + parse headers
//! - [`larql_models::quant::ggml::dequantize`] — convert raw bytes → f32
//! - [`larql_models::architectures::deepseek_v4_tensors`] — name schema
//!
//! Produces the `HashMap<DsV4TensorKind, Vec<f32>>` that
//! [`super::dsv4_storage_build::build_layer_storage`] consumes.
//!
//! Two entry points:
//! - [`rekey_dsv4_layer_tensors`] — pure: re-key a name→vec map by
//!   `DsV4TensorKind`. Synthetic testable.
//! - [`read_dsv4_layer_tensors_from_gguf`] — open-shard, byte-read,
//!   dequantize, then re-key.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use larql_models::architectures::deepseek_v4_tensors::{
    all_kinds, tensor_name_of, try_parse_name, DsV4TensorKind,
};
use larql_models::detect::ModelError;
use larql_models::loading::gguf::GgufFile;
use larql_models::quant::ggml::{dequantize, is_integer_type, tensor_data_size};

/// Re-key a generic `(raw_gguf_name → Vec<f32>)` map by
/// `DsV4TensorKind` for a given `layer_index`.
///
/// Pure-logic helper: takes the output of any byte-reading step and
/// drops everything that doesn't belong to the requested layer (or
/// can't be parsed by the DSv4 schema).
pub fn rekey_dsv4_layer_tensors(
    raw_map: HashMap<String, Vec<f32>>,
    layer_index: usize,
) -> HashMap<DsV4TensorKind, Vec<f32>> {
    let mut out = HashMap::new();
    for (name, vec) in raw_map {
        let Some((kind, layer)) = try_parse_name(&name) else {
            continue;
        };
        match (kind.is_per_layer(), layer) {
            (true, Some(l)) if l == layer_index => {
                out.insert(kind, vec);
            }
            (false, None) => {
                // Global tensor — DSv4 forward loop pulls these
                // separately, but caller may want to receive them. Skip
                // here to keep the per-layer map clean.
            }
            _ => {}
        }
    }
    out
}

/// Read per-layer DSv4 tensors from a `GgufFile`, dequantize each to
/// `f32`, and re-key by `DsV4TensorKind`.
///
/// Returns a `HashMap` with one entry per DSv4 tensor kind that's
/// actually present in the GGUF for layer `layer_index`. Missing
/// tensors are simply absent from the map — the downstream
/// [`super::dsv4_storage_build::build_layer_storage`] will reject any
/// that the variant requires.
pub fn read_dsv4_layer_tensors_from_gguf(
    gguf: &GgufFile,
    layer_index: usize,
) -> Result<HashMap<DsV4TensorKind, Vec<f32>>, ModelError> {
    // Build the per-layer name → kind index up front.
    let mut want: HashMap<String, DsV4TensorKind> = HashMap::new();
    for &kind in all_kinds() {
        if !kind.is_per_layer() {
            continue;
        }
        want.insert(tensor_name_of(kind, layer_index), kind);
    }

    // For each matching tensor in the GGUF, read its bytes + dequantize.
    // Integer-typed tensors (e.g. `ffn_gate_tid2eid` is I32) have no
    // meaningful f32 dequantization — they're routing tables consumed
    // elsewhere. Skip them here; future code can read them separately.
    let mut out: HashMap<DsV4TensorKind, Vec<f32>> = HashMap::new();
    let mut shard_files: HashMap<usize, File> = HashMap::new();
    for info in &gguf.tensor_infos {
        let Some(&kind) = want.get(info.name()) else {
            continue;
        };
        if is_integer_type(info.tensor_type()) {
            continue;
        }
        let shard_idx = info.shard_idx();
        let abs_offset = gguf
            .shard_data_offset(info)
            .checked_add(info.offset())
            .ok_or_else(|| {
                ModelError::Parse(format!(
                    "DSv4 layer {layer_index}: {} offset overflow",
                    info.name()
                ))
            })?;
        let n_elements: u64 = info.dims().iter().product();
        let data_size = tensor_data_size(info.tensor_type(), n_elements as usize)?;

        let path: PathBuf = gguf.shard_path(info).to_path_buf();
        let f = shard_files.entry(shard_idx).or_insert_with_key(|_| {
            File::open(&path)
                .unwrap_or_else(|e| panic!("DSv4 layer {layer_index}: open shard {path:?}: {e}"))
        });
        f.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; data_size];
        f.read_exact(&mut buf)?;

        let vec = dequantize(&buf, info.tensor_type(), n_elements as usize)?;
        out.insert(kind, vec);
    }
    Ok(out)
}

/// Read per-layer DSv4 **integer** tensors (e.g. `ffn_gate_tid2eid`),
/// keyed by [`DsV4TensorKind`]. Returns `Vec<i32>` per kind so the
/// downstream loader can shape it into the right routing table.
///
/// Skips non-integer types — callers should pair this with
/// [`read_dsv4_layer_tensors_from_gguf`] for the f32 weights.
pub fn read_dsv4_layer_int_tensors_from_gguf(
    gguf: &GgufFile,
    layer_index: usize,
) -> Result<HashMap<DsV4TensorKind, Vec<i32>>, ModelError> {
    let mut want: HashMap<String, DsV4TensorKind> = HashMap::new();
    for &kind in all_kinds() {
        if !kind.is_per_layer() {
            continue;
        }
        want.insert(tensor_name_of(kind, layer_index), kind);
    }

    let mut out: HashMap<DsV4TensorKind, Vec<i32>> = HashMap::new();
    let mut shard_files: HashMap<usize, File> = HashMap::new();
    for info in &gguf.tensor_infos {
        let Some(&kind) = want.get(info.name()) else {
            continue;
        };
        if !is_integer_type(info.tensor_type()) {
            continue;
        }
        let shard_idx = info.shard_idx();
        let abs_offset = gguf
            .shard_data_offset(info)
            .checked_add(info.offset())
            .ok_or_else(|| {
                ModelError::Parse(format!(
                    "DSv4 layer {layer_index}: {} offset overflow",
                    info.name()
                ))
            })?;
        let n_elements: u64 = info.dims().iter().product();
        let data_size = tensor_data_size(info.tensor_type(), n_elements as usize)?;
        // DSv4 integer tensors observed so far are all I32 (type 26).
        // tensor_data_size returns the right byte count for any int width.
        let path: PathBuf = gguf.shard_path(info).to_path_buf();
        let f = shard_files.entry(shard_idx).or_insert_with_key(|_| {
            File::open(&path)
                .unwrap_or_else(|e| panic!("DSv4 layer {layer_index}: open shard {path:?}: {e}"))
        });
        f.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; data_size];
        f.read_exact(&mut buf)?;
        // Decode bytes as i32 little-endian. Only I32 (type 26) handled
        // here — other integer widths are rare in DSv4 and can extend
        // this later if needed.
        if info.tensor_type() != larql_models::quant::ggml::TYPE_I32 {
            continue;
        }
        let elems = n_elements as usize;
        if buf.len() < elems * 4 {
            return Err(ModelError::Parse(format!(
                "DSv4 layer {layer_index}: {} I32 buffer too short",
                info.name()
            )));
        }
        let mut vec = Vec::with_capacity(elems);
        for chunk in buf[..elems * 4].chunks_exact(4) {
            vec.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out.insert(kind, vec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rekey_picks_only_requested_layer() {
        let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
        raw.insert("blk.3.attn_norm.weight".to_string(), vec![1.0; 4]);
        raw.insert("blk.3.attn_q_a.weight".to_string(), vec![2.0; 8]);
        // Different layer — should be dropped.
        raw.insert("blk.4.attn_norm.weight".to_string(), vec![9.0; 4]);
        // Global tensor — dropped by the per-layer view.
        raw.insert("token_embd.weight".to_string(), vec![0.0; 16]);
        // Non-DSv4 — dropped.
        raw.insert("blk.3.foo_bar.weight".to_string(), vec![5.0; 4]);

        let out = rekey_dsv4_layer_tensors(raw, 3);
        assert_eq!(out.len(), 2);
        assert_eq!(out.get(&DsV4TensorKind::AttnNorm).unwrap()[0], 1.0);
        assert_eq!(out.get(&DsV4TensorKind::AttnQA).unwrap()[0], 2.0);
    }

    #[test]
    fn rekey_handles_indexer_dotted_names() {
        let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
        raw.insert("blk.5.indexer.attn_q_b.weight".to_string(), vec![1.0; 8]);
        raw.insert("blk.5.indexer.proj.weight".to_string(), vec![2.0; 4]);
        let out = rekey_dsv4_layer_tensors(raw, 5);
        assert_eq!(out.len(), 2);
        assert!(out.contains_key(&DsV4TensorKind::IndexerAttnQB));
        assert!(out.contains_key(&DsV4TensorKind::IndexerProj));
    }

    /// `exp_probs_b` carries no suffix in the real DSv4 GGUF (the
    /// schema was speculative in initial draft — calibrated 2026-05-23).
    #[test]
    fn rekey_handles_no_suffix_exp_probs_b() {
        let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
        raw.insert("blk.2.exp_probs_b".to_string(), vec![1.0, 2.0, 3.0]);
        let out = rekey_dsv4_layer_tensors(raw, 2);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out.get(&DsV4TensorKind::FfnExpProbsB).unwrap(),
            &vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn rekey_layer_mismatch_drops_everything() {
        let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
        raw.insert("blk.0.attn_norm.weight".to_string(), vec![1.0; 4]);
        raw.insert("blk.1.attn_norm.weight".to_string(), vec![2.0; 4]);
        let out = rekey_dsv4_layer_tensors(raw, 7);
        assert!(out.is_empty());
    }

    #[test]
    fn rekey_empty_input_yields_empty_output() {
        let out = rekey_dsv4_layer_tensors(HashMap::new(), 0);
        assert!(out.is_empty());
    }

    /// Real-GGUF: layer 0's `ffn_gate_tid2eid` is I32 with shape
    /// `[6, 129280]` = (n_expert_used, n_vocab). Verify the int reader
    /// loads it correctly.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_loads_ffn_gate_tid2eid() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let map = read_dsv4_layer_int_tensors_from_gguf(&gguf, 0).expect("load int tensors");
        let table = map
            .get(&DsV4TensorKind::FfnGateTid2Eid)
            .expect("ffn_gate_tid2eid present on layer 0");
        // n_expert_used=6, n_vocab=129280 → 775680 entries.
        assert_eq!(table.len(), 6 * 129280);
        // Indices should be in [0, n_expert) = [0, 256).
        let max_idx = *table.iter().max().unwrap();
        let min_idx = *table.iter().min().unwrap();
        assert!(
            min_idx >= 0 && max_idx < 256,
            "tid2eid out of expected expert range: min={min_idx} max={max_idx}"
        );

        // Layers 3+ should NOT have tid2eid (verified by the empty map).
        let layer_3_map =
            read_dsv4_layer_int_tensors_from_gguf(&gguf, 3).expect("load int tensors for layer 3");
        assert!(
            !layer_3_map.contains_key(&DsV4TensorKind::FfnGateTid2Eid),
            "layer 3 should not have ffn_gate_tid2eid (hash routing is first 3 layers)"
        );
    }
}
