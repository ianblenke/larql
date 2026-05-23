//! DeepSeek V4 GGUF tensor name schema.
//!
//! Maps the per-layer and global tensors expected by the DSv4 attention
//! and FFN blocks (Stages 8a, 8c, 8d, 8f, 8g) to GGUF tensor name
//! strings. Used by the per-layer GGUF loader to populate the typed
//! weight structs.
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:88-160`
//! (`llama_model_deepseek4::llama_model_deepseek4` — the create_tensor
//! calls per layer).
//!
//! Naming convention follows llama.cpp's `LLM_TENSOR_FOO_BAR →
//! "blk.%d.foo_bar"` (or `"foo_bar"` for non-layered tensors). The
//! DSv4-specific tensor name strings aren't in the local llama.cpp
//! checkout's `LLM_TENSOR_NAMES` table yet — the PR may extend it
//! before merge — so a few of the compressor/HC names are inferred
//! from the enum naming. If a real DSv4 GGUF uses a different string
//! for a given tensor, it's a per-entry override in [`tensor_name_of`]
//! rather than a structural change to the schema.
//!
//! ## Tensors per layer
//! - Attention:
//!   - `attn_norm`              — input RMSNorm gain
//!   - `attn_q_a`               — Q low-rank "down"
//!   - `attn_q_a_norm`          — RMSNorm gain on q_a output
//!   - `attn_q_b`               — Q low-rank "up"
//!   - `attn_kv`                — single-head KV projection
//!   - `attn_kv_a_norm`         — RMSNorm gain on kv output
//!   - `attn_sinks`             — per-head attention sink logits
//!   - `attn_out_a` / `attn_out_b` — grouped low-rank output projection
//! - mHC (per-layer residual bookends):
//!   - `hc_attn_base`, `hc_attn_fn`, `hc_attn_scale`
//!   - `hc_ffn_base`,  `hc_ffn_fn`,  `hc_ffn_scale`
//! - Compressor (HCA, when compress_ratio > 0):
//!   - `attn_compressor_kv`, `attn_compressor_gate`,
//!     `attn_compressor_ape`, `attn_compressor_norm`
//! - Indexer (compress_ratio == 4):
//!   - `indexer_compressor_kv` / `_gate` / `_ape` / `_norm`
//!   - `indexer_attn_q_b`, `indexer_proj`
//! - MoE FFN:
//!   - `ffn_norm`, `ffn_gate_inp`, `ffn_exp_probs_b` (bias),
//!     `ffn_gate_tid2eid` (hash routing, first 3 layers only)
//!   - `ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`
//!   - `ffn_gate_shexp`, `ffn_down_shexp`, `ffn_up_shexp` (shared experts)
//!
//! ## Global tensors
//! - `token_embd`, `output_norm`, `output`
//! - `output_hc_base`, `output_hc_fn`, `output_hc_scale` (mHC bookend
//!   at the head)

use std::fmt;

/// Kind of DSv4 tensor — covers every weight the reference forward
/// path consumes. Variants are split into per-layer (require a layer
/// index) and global (don't).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DsV4TensorKind {
    // ── Global ──
    TokenEmbd,
    OutputNorm,
    Output,
    OutputHcBase,
    OutputHcFn,
    OutputHcScale,
    // ── Per-layer: attention ──
    AttnNorm,
    AttnQA,
    AttnQANorm,
    AttnQB,
    AttnKv,
    AttnKvANorm,
    AttnSinks,
    AttnOutA,
    AttnOutB,
    // ── Per-layer: mHC ──
    HcAttnBase,
    HcAttnFn,
    HcAttnScale,
    HcFfnBase,
    HcFfnFn,
    HcFfnScale,
    // ── Per-layer: compressor ──
    AttnCompressorKv,
    AttnCompressorGate,
    AttnCompressorApe,
    AttnCompressorNorm,
    // ── Per-layer: indexer ──
    IndexerCompressorKv,
    IndexerCompressorGate,
    IndexerCompressorApe,
    IndexerCompressorNorm,
    IndexerAttnQB,
    IndexerProj,
    // ── Per-layer: FFN / MoE ──
    FfnNorm,
    FfnGateInp,
    FfnExpProbsB,
    FfnGateTid2Eid,
    FfnGateExps,
    FfnDownExps,
    FfnUpExps,
    FfnGateShexp,
    FfnDownShexp,
    FfnUpShexp,
}

impl DsV4TensorKind {
    /// True iff this kind requires a layer index to form a name.
    pub fn is_per_layer(self) -> bool {
        !matches!(
            self,
            DsV4TensorKind::TokenEmbd
                | DsV4TensorKind::OutputNorm
                | DsV4TensorKind::Output
                | DsV4TensorKind::OutputHcBase
                | DsV4TensorKind::OutputHcFn
                | DsV4TensorKind::OutputHcScale
        )
    }

    /// Short stem matching the llama.cpp `LLM_TENSOR_FOO_BAR` → "foo_bar"
    /// convention (lower snake case with no "blk.N." or ".weight"
    /// decorators).
    pub fn stem(self) -> &'static str {
        match self {
            DsV4TensorKind::TokenEmbd => "token_embd",
            DsV4TensorKind::OutputNorm => "output_norm",
            DsV4TensorKind::Output => "output",
            DsV4TensorKind::OutputHcBase => "output_hc_base",
            DsV4TensorKind::OutputHcFn => "output_hc_fn",
            DsV4TensorKind::OutputHcScale => "output_hc_scale",
            DsV4TensorKind::AttnNorm => "attn_norm",
            DsV4TensorKind::AttnQA => "attn_q_a",
            DsV4TensorKind::AttnQANorm => "attn_q_a_norm",
            DsV4TensorKind::AttnQB => "attn_q_b",
            DsV4TensorKind::AttnKv => "attn_kv",
            DsV4TensorKind::AttnKvANorm => "attn_kv_a_norm",
            DsV4TensorKind::AttnSinks => "attn_sinks",
            DsV4TensorKind::AttnOutA => "attn_out_a",
            DsV4TensorKind::AttnOutB => "attn_out_b",
            DsV4TensorKind::HcAttnBase => "hc_attn_base",
            DsV4TensorKind::HcAttnFn => "hc_attn_fn",
            DsV4TensorKind::HcAttnScale => "hc_attn_scale",
            DsV4TensorKind::HcFfnBase => "hc_ffn_base",
            DsV4TensorKind::HcFfnFn => "hc_ffn_fn",
            DsV4TensorKind::HcFfnScale => "hc_ffn_scale",
            DsV4TensorKind::AttnCompressorKv => "attn_compressor_kv",
            DsV4TensorKind::AttnCompressorGate => "attn_compressor_gate",
            DsV4TensorKind::AttnCompressorApe => "attn_compressor_ape",
            DsV4TensorKind::AttnCompressorNorm => "attn_compressor_norm",
            DsV4TensorKind::IndexerCompressorKv => "indexer_compressor_kv",
            DsV4TensorKind::IndexerCompressorGate => "indexer_compressor_gate",
            DsV4TensorKind::IndexerCompressorApe => "indexer_compressor_ape",
            DsV4TensorKind::IndexerCompressorNorm => "indexer_compressor_norm",
            // The two existing indexer tensors that ARE in the local
            // llama-arch.cpp use the "indexer.attn_q_b" / "indexer.proj"
            // dotted form — pin that here to match the actual GGUFs.
            DsV4TensorKind::IndexerAttnQB => "indexer.attn_q_b",
            DsV4TensorKind::IndexerProj => "indexer.proj",
            DsV4TensorKind::FfnNorm => "ffn_norm",
            DsV4TensorKind::FfnGateInp => "ffn_gate_inp",
            DsV4TensorKind::FfnExpProbsB => "ffn_exp_probs_b",
            DsV4TensorKind::FfnGateTid2Eid => "ffn_gate_tid2eid",
            DsV4TensorKind::FfnGateExps => "ffn_gate_exps",
            DsV4TensorKind::FfnDownExps => "ffn_down_exps",
            DsV4TensorKind::FfnUpExps => "ffn_up_exps",
            DsV4TensorKind::FfnGateShexp => "ffn_gate_shexp",
            DsV4TensorKind::FfnDownShexp => "ffn_down_shexp",
            DsV4TensorKind::FfnUpShexp => "ffn_up_shexp",
        }
    }

    /// True iff this is a `.bias` tensor (`ffn_exp_probs_b`) rather than
    /// a `.weight`. The DSv4 schema is almost all `.weight`; only the
    /// expert-prob bias is the exception.
    pub fn is_bias(self) -> bool {
        matches!(self, DsV4TensorKind::FfnExpProbsB)
    }

    /// `.weight` / `.bias` suffix the loader should look for.
    pub fn suffix(self) -> &'static str {
        if self.is_bias() {
            "bias"
        } else {
            "weight"
        }
    }
}

impl fmt::Display for DsV4TensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stem())
    }
}

/// Build the GGUF tensor name for a `(kind, layer_index)` pair.
///
/// For global tensors `layer_index` is ignored. For per-layer tensors,
/// the form is `"blk.{layer_index}.{stem}.{suffix}"`. For global, it's
/// `"{stem}.{suffix}"`.
pub fn tensor_name_of(kind: DsV4TensorKind, layer_index: usize) -> String {
    let stem = kind.stem();
    let suffix = kind.suffix();
    if kind.is_per_layer() {
        format!("blk.{layer_index}.{stem}.{suffix}")
    } else {
        format!("{stem}.{suffix}")
    }
}

/// Reverse lookup: given a GGUF tensor name, recover the
/// `(kind, layer_index)` pair if it matches the DSv4 schema.
///
/// Returns `None` for names that don't fit the DSv4 pattern (e.g.
/// tensors from a different architecture).
pub fn try_parse_name(name: &str) -> Option<(DsV4TensorKind, Option<usize>)> {
    // Strip the suffix.
    let (head, suffix) = name.rsplit_once('.')?;
    if suffix != "weight" && suffix != "bias" {
        return None;
    }

    // Per-layer? "blk.N.<stem>"
    if let Some(rest) = head.strip_prefix("blk.") {
        let (idx_str, stem) = rest.split_once('.')?;
        let layer: usize = idx_str.parse().ok()?;
        let kind = kind_from_stem(stem)?;
        if !kind.is_per_layer() {
            return None;
        }
        return Some((kind, Some(layer)));
    }

    // Global tensor.
    let kind = kind_from_stem(head)?;
    if kind.is_per_layer() {
        return None;
    }
    Some((kind, None))
}

fn kind_from_stem(stem: &str) -> Option<DsV4TensorKind> {
    use DsV4TensorKind::*;
    let k = match stem {
        "token_embd" => TokenEmbd,
        "output_norm" => OutputNorm,
        "output" => Output,
        "output_hc_base" => OutputHcBase,
        "output_hc_fn" => OutputHcFn,
        "output_hc_scale" => OutputHcScale,
        "attn_norm" => AttnNorm,
        "attn_q_a" => AttnQA,
        "attn_q_a_norm" => AttnQANorm,
        "attn_q_b" => AttnQB,
        "attn_kv" => AttnKv,
        "attn_kv_a_norm" => AttnKvANorm,
        "attn_sinks" => AttnSinks,
        "attn_out_a" => AttnOutA,
        "attn_out_b" => AttnOutB,
        "hc_attn_base" => HcAttnBase,
        "hc_attn_fn" => HcAttnFn,
        "hc_attn_scale" => HcAttnScale,
        "hc_ffn_base" => HcFfnBase,
        "hc_ffn_fn" => HcFfnFn,
        "hc_ffn_scale" => HcFfnScale,
        "attn_compressor_kv" => AttnCompressorKv,
        "attn_compressor_gate" => AttnCompressorGate,
        "attn_compressor_ape" => AttnCompressorApe,
        "attn_compressor_norm" => AttnCompressorNorm,
        "indexer_compressor_kv" => IndexerCompressorKv,
        "indexer_compressor_gate" => IndexerCompressorGate,
        "indexer_compressor_ape" => IndexerCompressorApe,
        "indexer_compressor_norm" => IndexerCompressorNorm,
        "indexer.attn_q_b" => IndexerAttnQB,
        "indexer.proj" => IndexerProj,
        "ffn_norm" => FfnNorm,
        "ffn_gate_inp" => FfnGateInp,
        "ffn_exp_probs_b" => FfnExpProbsB,
        "ffn_gate_tid2eid" => FfnGateTid2Eid,
        "ffn_gate_exps" => FfnGateExps,
        "ffn_down_exps" => FfnDownExps,
        "ffn_up_exps" => FfnUpExps,
        "ffn_gate_shexp" => FfnGateShexp,
        "ffn_down_shexp" => FfnDownShexp,
        "ffn_up_shexp" => FfnUpShexp,
        _ => return None,
    };
    Some(k)
}

/// Enumerate every DSv4 tensor kind — useful for cataloguing a model
/// (e.g. "for each layer, expect these N tensors").
pub fn all_kinds() -> &'static [DsV4TensorKind] {
    use DsV4TensorKind::*;
    &[
        TokenEmbd,
        OutputNorm,
        Output,
        OutputHcBase,
        OutputHcFn,
        OutputHcScale,
        AttnNorm,
        AttnQA,
        AttnQANorm,
        AttnQB,
        AttnKv,
        AttnKvANorm,
        AttnSinks,
        AttnOutA,
        AttnOutB,
        HcAttnBase,
        HcAttnFn,
        HcAttnScale,
        HcFfnBase,
        HcFfnFn,
        HcFfnScale,
        AttnCompressorKv,
        AttnCompressorGate,
        AttnCompressorApe,
        AttnCompressorNorm,
        IndexerCompressorKv,
        IndexerCompressorGate,
        IndexerCompressorApe,
        IndexerCompressorNorm,
        IndexerAttnQB,
        IndexerProj,
        FfnNorm,
        FfnGateInp,
        FfnExpProbsB,
        FfnGateTid2Eid,
        FfnGateExps,
        FfnDownExps,
        FfnUpExps,
        FfnGateShexp,
        FfnDownShexp,
        FfnUpShexp,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_layer_name_format() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::AttnNorm, 7),
            "blk.7.attn_norm.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::AttnQA, 0),
            "blk.0.attn_q_a.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::FfnGateInp, 12),
            "blk.12.ffn_gate_inp.weight"
        );
    }

    #[test]
    fn global_name_format() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::TokenEmbd, 0),
            "token_embd.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::OutputHcFn, 999),
            "output_hc_fn.weight"
        );
    }

    #[test]
    fn ffn_exp_probs_b_uses_bias_suffix() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::FfnExpProbsB, 3),
            "blk.3.ffn_exp_probs_b.bias"
        );
    }

    #[test]
    fn indexer_uses_dotted_stem() {
        assert_eq!(
            tensor_name_of(DsV4TensorKind::IndexerAttnQB, 5),
            "blk.5.indexer.attn_q_b.weight"
        );
        assert_eq!(
            tensor_name_of(DsV4TensorKind::IndexerProj, 5),
            "blk.5.indexer.proj.weight"
        );
    }

    #[test]
    fn round_trip_per_layer() {
        // Every per-layer kind should round-trip through name → parse.
        for &kind in all_kinds() {
            if !kind.is_per_layer() {
                continue;
            }
            let name = tensor_name_of(kind, 42);
            let parsed = try_parse_name(&name)
                .unwrap_or_else(|| panic!("failed to parse {name} for {kind:?}"));
            assert_eq!(parsed, (kind, Some(42)), "round-trip mismatch on {name}");
        }
    }

    #[test]
    fn round_trip_global() {
        for &kind in all_kinds() {
            if kind.is_per_layer() {
                continue;
            }
            let name = tensor_name_of(kind, 0);
            let parsed = try_parse_name(&name)
                .unwrap_or_else(|| panic!("failed to parse {name} for {kind:?}"));
            assert_eq!(parsed, (kind, None), "round-trip mismatch on {name}");
        }
    }

    #[test]
    fn parse_rejects_unrelated_names() {
        assert_eq!(try_parse_name("blk.0.attn_q.weight"), None);
        assert_eq!(try_parse_name("ssm.in.weight"), None);
        assert_eq!(try_parse_name("garbage"), None);
        // No `.weight` / `.bias` suffix.
        assert_eq!(try_parse_name("blk.0.attn_norm"), None);
        // Per-layer kind used without a layer index.
        assert_eq!(try_parse_name("attn_norm.weight"), None);
        // Global kind used with a layer index.
        assert_eq!(try_parse_name("blk.0.token_embd.weight"), None);
    }

    #[test]
    fn is_per_layer_classification() {
        assert!(!DsV4TensorKind::TokenEmbd.is_per_layer());
        assert!(!DsV4TensorKind::OutputHcFn.is_per_layer());
        assert!(DsV4TensorKind::AttnNorm.is_per_layer());
        assert!(DsV4TensorKind::IndexerProj.is_per_layer());
        assert!(DsV4TensorKind::HcAttnFn.is_per_layer());
    }

    #[test]
    fn all_kinds_are_unique() {
        let kinds = all_kinds();
        let mut seen: std::collections::HashSet<DsV4TensorKind> = std::collections::HashSet::new();
        for &k in kinds {
            assert!(seen.insert(k), "duplicate kind {k:?}");
        }
    }

    #[test]
    fn all_stems_are_unique() {
        let kinds = all_kinds();
        let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        for &k in kinds {
            assert!(seen.insert(k.stem()), "duplicate stem {}", k.stem());
        }
    }
}
