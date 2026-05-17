//! DeepSeek V4 Flash — MoE + MLA + hyper-connection + indexer + hash-routing.
//!
//! Distinct from DeepSeek V3 (`deepseek.rs`) — V3 was the standard
//! MoE-MLA transformer with `attn_q_a` / `attn_kv_a` / `attn_kv_b`
//! tensors. V4 fuses kv down-projection (`attn_kv` produces both
//! `kv_a` and the positional `kv_a_pe` slice), splits output into
//! low-rank `attn_output_{a,b}`, replaces routed-layer routing with
//! hash tables for the first 3 layers (`ffn_gate_tid2eid`), and
//! introduces per-layer hyper-connection + indexer + compressor
//! blocks that have no V3 analogue.
//!
//! The tensor inventory (from antirez's `DeepSeek-V4-Flash-Q4KExperts`
//! GGUF):
//!
//! | Per-layer tensor (GGUF `blk.{L}.<name>.weight`) | Shape | Dtype | Notes |
//! |---|---|---|---|
//! | `attn_norm` | `[4096]` | F32 | input layernorm |
//! | `attn_q_a` | `[4096, 1024]` | Q8_0 | Q down-projection |
//! | `attn_q_a_norm` | `[1024]` | F32 | post-down-Q layernorm |
//! | `attn_q_b` | `[1024, 32768]` | Q8_0 | Q up-projection (64 heads × 512 dim) |
//! | `attn_kv` | `[4096, 512]` | Q8_0 | joint KV down-projection |
//! | `attn_kv_a_norm` | `[512]` | F32 | post-down-KV layernorm |
//! | `attn_output_a` | `[4096, 8192]` | Q8_0 | output low-rank A |
//! | `attn_output_b` | `[8192, 4096]` | Q8_0 | output low-rank B |
//! | `attn_sinks` | `[64]` | F32 | per-head attention sink |
//! | `ffn_norm` | `[4096]` | F32 | post-attn layernorm |
//! | `ffn_gate_inp` | `[4096, 256]` | F16 | learned router |
//! | `exp_probs_b.bias` | `[256]` | F32 | router bias |
//! | `ffn_gate_exps` | `[4096, 2048, 256]` | Q4_K | 256 routed experts (gate) |
//! | `ffn_up_exps` | `[4096, 2048, 256]` | Q4_K | 256 routed experts (up) |
//! | `ffn_down_exps` | `[2048, 4096, 256]` | Q4_K | 256 routed experts (down) |
//! | `ffn_gate_shexp` | `[4096, 2048]` | Q8_0 | 1 shared expert (gate) |
//! | `ffn_up_shexp` | `[4096, 2048]` | Q8_0 | 1 shared expert (up) |
//! | `ffn_down_shexp` | `[2048, 4096]` | Q8_0 | 1 shared expert (down) |
//! | `ffn_gate_tid2eid` (L<3 only) | `[6, 129280]` | I32 | hash-routing table |
//! | `hc_{attn,ffn}_{base,fn,scale}` | varies | F16/F32 | hyper-connection (4-way) |
//! | `attn_compressor_{ape,gate,kv,norm}` | varies | F16/F32 | per-layer attn compressor |
//! | `indexer.{attn_q_b,proj}`, `indexer_compressor_{ape,gate,kv,norm}` | varies | F16/F32 | sparse attention indexer |
//!
//! Plus model-level: `token_embd` (F16, `[4096, 129280]`), `output`
//! (Q8_0, `[4096, 129280]`), `output_norm` (F32), and
//! `output_hc_{base,fn,scale}` (HC at the head too).
//!
//! Status: **Scope 1** — arch + tensor-key conventions + detection only.
//! Inference, the V4 forward, and the vindex writer/reader land in
//! later scopes. The auxiliary blocks (HC / indexer / compressor /
//! hash-routing / `attn_sinks` / `exp_probs_b`) don't yet have arch
//! accessors — when those land they'll surface as new `Option<String>`
//! key methods on this struct so the writer can name them.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct DeepSeekV4Arch {
    config: ModelConfig,
}

impl DeepSeekV4Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for DeepSeekV4Arch {
    fn family(&self) -> &str {
        "deepseek4"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── Tensor key conventions ──
    //
    // GGUF→HF normalisation in `loading/gguf.rs::normalize_gguf_key`
    // rewrites `blk.N.` → `layers.N.` and leaves the V4-specific
    // suffixes (`attn_q_a`, `attn_kv`, `ffn_gate_inp`,
    // `ffn_*_exps`, `ffn_*_shexp`, `hc_*`, etc.) intact. So every
    // per-layer key below is `layers.{layer}.<gguf-suffix>.weight`.

    fn key_prefixes_to_strip(&self) -> &[&str] {
        // V4 has no `model.` or `transformer.` wrapper in the GGUF —
        // post-normalisation keys start at `layers.` (or `token_embd.`
        // → `embed_tokens.` for the input embedding).
        &[]
    }

    fn embed_key(&self) -> &str {
        // GGUF→HF map rewrites `token_embd.` → `embed_tokens.`.
        "embed_tokens.weight"
    }

    fn final_norm_key(&self) -> &str {
        // GGUF→HF map rewrites `output_norm.` → `norm.`.
        "norm.weight"
    }

    // Attention: V4 uses MLA with the q_a/q_b/kv layout. There's no
    // standalone Q / K / V projection — Q comes from `attn_q_a` →
    // `attn_q_a_norm` → `attn_q_b`, K and V are sliced out of
    // `attn_kv` → `attn_kv_a_norm`. The legacy `attn_q_key` /
    // `attn_k_key` / `attn_v_key` slots return the V4-specific
    // names to let the loader at least find *something*; correct
    // V4 inference must go through the MLA accessors below.
    fn attn_q_key(&self, layer: usize) -> String {
        format!("{}attn_q_a.weight", self.layer_prefix(layer))
    }
    fn attn_k_key(&self, layer: usize) -> String {
        // V4 has no standalone K projection — K is sliced from
        // `attn_kv` after `attn_kv_a_norm`. Return the fused tensor
        // key so name-lookup tools can find it; the inference path
        // uses `mla_kv_a_key` instead.
        format!("{}attn_kv.weight", self.layer_prefix(layer))
    }
    fn attn_v_key(&self, layer: usize) -> String {
        format!("{}attn_kv.weight", self.layer_prefix(layer))
    }
    fn attn_o_key(&self, layer: usize) -> String {
        // V4 splits the output projection into low-rank A+B; this slot
        // returns the A half. The B half is exposed only via the
        // explicit `attn_output_b_key` helper below (to be added
        // when the writer needs it in Scope 3).
        format!("{}attn_output_a.weight", self.layer_prefix(layer))
    }

    fn input_layernorm_key(&self, layer: usize) -> String {
        format!("{}attn_norm.weight", self.layer_prefix(layer))
    }
    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        format!("{}ffn_norm.weight", self.layer_prefix(layer))
    }
    fn pre_feedforward_layernorm_key(&self, _layer: usize) -> Option<String> {
        None
    }
    fn post_feedforward_layernorm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    // Dense FFN: V4 routes everything through experts (256 routed +
    // 1 shared) — there's no non-MoE FFN block. These slots return
    // the shared-expert keys so name-lookup callers at least see a
    // valid tensor name; production code should use the explicit
    // shared-expert accessors below.
    fn ffn_gate_key(&self, layer: usize) -> String {
        format!("{}ffn_gate_shexp.weight", self.layer_prefix(layer))
    }
    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}ffn_up_shexp.weight", self.layer_prefix(layer))
    }
    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}ffn_down_shexp.weight", self.layer_prefix(layer))
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        true
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(256)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(6)
    }

    fn num_shared_experts(&self) -> usize {
        self.config.num_shared_experts.unwrap_or(1)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}ffn_gate_inp.weight", self.layer_prefix(layer)))
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        // V4 packs all 256 experts into a 3-D `ffn_gate_exps.weight`
        // tensor (`[hidden, intermediate, num_experts]`). The GGUF
        // loader (`loading/gguf.rs:753-776`) detects the 3-D shape and
        // emits per-expert HF-style aliases at
        // `layers.{L}.mlp.experts.{E}.{proj}_proj.weight` that share
        // the same Arc mmap data via `QuantTensor::expert_slice`.
        // Returning the alias here keeps consumers (vindex writer,
        // forward) ignorant of the 3-D packing.
        Some(format!(
            "{}mlp.experts.{expert_id}.gate_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.{expert_id}.up_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.{expert_id}.down_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_gate_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}ffn_gate_shexp.weight", self.layer_prefix(layer)))
    }

    fn shared_expert_up_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}ffn_up_shexp.weight", self.layer_prefix(layer)))
    }

    fn shared_expert_down_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}ffn_down_shexp.weight", self.layer_prefix(layer)))
    }

    // ── MLA ──
    //
    // V4 fuses KV down-projection into a single `attn_kv` tensor,
    // unlike V3 which split it into `attn_kv_a` (projection) +
    // `attn_kv_b` (per-head up-projection). The semantic shape is
    // the same — `kv_lora_rank` floats of joint KV latent — but the
    // tensor name differs.

    fn uses_mla(&self) -> bool {
        true
    }

    fn kv_lora_rank(&self) -> usize {
        // V4 GGUF reports this via `attention.key_length=512`. The
        // gguf loader forwards that into `config.kv_lora_rank` when
        // the explicit `attention.kv_lora_rank` key is absent.
        self.config.kv_lora_rank.unwrap_or(512)
    }

    fn q_lora_rank(&self) -> usize {
        self.config.q_lora_rank.unwrap_or(1024)
    }

    fn mla_kv_a_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn_kv.weight", self.layer_prefix(layer)))
    }

    fn mla_kv_b_key(&self, _layer: usize) -> Option<String> {
        // V4 doesn't split KV up — the joint latent stays
        // `kv_lora_rank` floats wide and the per-head up-projection
        // is computed implicitly via the indexer / sparse attention
        // path. No separate kv_b tensor exists.
        None
    }

    fn mla_q_a_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn_q_a.weight", self.layer_prefix(layer)))
    }

    fn mla_q_b_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn_q_b.weight", self.layer_prefix(layer)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn dsv4_config() -> ModelConfig {
        // Reproduce the GGUF metadata for the `antirez/deepseek-v4-gguf`
        // Q4KExperts variant (`general.architecture='deepseek4'`,
        // dump in scope-1 PR description). 43 layers, 256 routed
        // experts, 1 shared, top-6 routing, MLA with q_lora=1024 and
        // kv_latent=512.
        ModelConfig {
            model_type: "deepseek4".into(),
            num_layers: 43,
            hidden_size: 4096,
            intermediate_size: 2048,
            head_dim: 512,
            num_q_heads: 64,
            num_kv_heads: 1,
            vocab_size: Some(129280),
            rope_base: 10_000.0,
            rope_local_base: None,
            sliding_window: Some(128),
            num_experts: Some(256),
            num_experts_per_token: Some(6),
            num_shared_experts: Some(1),
            kv_lora_rank: Some(512),
            q_lora_rank: Some(1024),
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: None,
            residual_multiplier: None,
            attention_multiplier: None,
            logits_scaling: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: Some(2048),
            full_attention_interval: None,
            ssm_state_size: None,
            ssm_inner_size: None,
            ssm_dt_rank: None,
            ssm_group_count: None,
            ssm_conv_kernel: None,
            rope_dimension_sections: None,
        }
    }

    #[test]
    fn dsv4_family_and_basic_traits() {
        let arch = DeepSeekV4Arch::from_config(dsv4_config());
        assert_eq!(arch.family(), "deepseek4");
        assert!(arch.is_moe());
        assert!(arch.uses_mla());
        assert_eq!(arch.num_experts(), 256);
        assert_eq!(arch.num_experts_per_token(), 6);
        assert_eq!(arch.num_shared_experts(), 1);
        assert_eq!(arch.kv_lora_rank(), 512);
        assert_eq!(arch.q_lora_rank(), 1024);
    }

    #[test]
    fn dsv4_per_layer_keys_match_gguf_post_normalisation() {
        let arch = DeepSeekV4Arch::from_config(dsv4_config());
        // GGUF→HF rewrites `blk.7.` → `layers.7.`; V4-specific
        // suffixes pass through unchanged.
        assert_eq!(arch.input_layernorm_key(7), "layers.7.attn_norm.weight");
        assert_eq!(
            arch.post_attention_layernorm_key(7),
            "layers.7.ffn_norm.weight"
        );
        assert_eq!(
            arch.mla_q_a_key(7),
            Some("layers.7.attn_q_a.weight".to_string())
        );
        assert_eq!(
            arch.mla_q_b_key(7),
            Some("layers.7.attn_q_b.weight".to_string())
        );
        assert_eq!(
            arch.mla_kv_a_key(7),
            Some("layers.7.attn_kv.weight".to_string())
        );
        assert_eq!(arch.mla_kv_b_key(7), None);
    }

    #[test]
    fn dsv4_moe_keys_use_fused_per_expert_layout() {
        let arch = DeepSeekV4Arch::from_config(dsv4_config());
        assert_eq!(
            arch.moe_router_key(3),
            Some("layers.3.ffn_gate_inp.weight".to_string())
        );
        // V4 packs all 256 experts into a 3-D fused tensor; the GGUF
        // loader unpacks it into per-expert HF-style aliases that the
        // arch surfaces here, so the vindex writer can iterate
        // experts without knowing about the packing.
        assert_eq!(
            arch.expert_ffn_gate_key(3, 0),
            Some("layers.3.mlp.experts.0.gate_proj.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_gate_key(3, 255),
            Some("layers.3.mlp.experts.255.gate_proj.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_down_key(3, 42),
            Some("layers.3.mlp.experts.42.down_proj.weight".to_string())
        );
        assert_eq!(
            arch.shared_expert_gate_key(3),
            Some("layers.3.ffn_gate_shexp.weight".to_string())
        );
    }

    #[test]
    fn dsv4_embed_and_final_norm_match_normalised_gguf() {
        let arch = DeepSeekV4Arch::from_config(dsv4_config());
        // `token_embd.` → `embed_tokens.`, `output_norm.` → `norm.`.
        assert_eq!(arch.embed_key(), "embed_tokens.weight");
        assert_eq!(arch.final_norm_key(), "norm.weight");
    }
}
