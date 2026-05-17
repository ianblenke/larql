//! DeepSeek V4 Flash forward step — DSv4 scope 4 (MVP).
//!
//! Drives a single-token forward through the full V4 hybrid:
//! `embed → 43 layers (MLA attention + MoE FFN) → final_norm
//! → lm_head → logits`. Loaders live in
//! [`super::dsv4_load_vindex`].
//!
//! **Simplifications shipped in this MVP** (clearly named so a
//! follow-up can extend them — see also the comments at each
//! call-site):
//!
//! - **MLA only** — Q-down (`attn_q_a`) → Q-up (`attn_q_b`) → split
//!   per-head Q, KV-down (`attn_kv`) → split per-head K, V via the
//!   `kv_lora_rank` latent. RoPE is applied to Q (full dim) and to
//!   the KV-latent's positional slice (V4 stores the rotary section
//!   separately, but for the MVP we apply ROPE inline on the full
//!   latent — quality is degraded but mechanically forwardable).
//! - **No hyper-connection / compressor / indexer** — the V4
//!   `hc_*` / `attn_compressor_*` / `indexer.*` paths are skipped;
//!   attention is standard dense softmax over the cached KV-latent
//!   slice. Output quality is degraded vs the published numbers.
//! - **No hash routing** — first-3-layer `ffn_gate_tid2eid`
//!   tables aren't loaded (DSv4 stores them as I32 which the
//!   loader skips). All 43 layers route through the learned
//!   `ffn_gate_inp` matrix.
//! - **Attention sinks** are loaded but ignored at the softmax;
//!   wiring them in is the first follow-up.
//! - **Per-head split of `attn_q_b`'s 32768-dim output** assumes
//!   the published `64 heads × 512 dim` layout; no head-rope-
//!   section partitioning yet.
//!
//! The forward is correct in shape but the output is **not
//! expected to be coherent** at this MVP. Scope 5's HTTP smoke
//! validates the dispatch path, not the generation quality.

use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

use larql_models::quant::lazy::QuantTensor;

/// Geometry shared across every layer.
#[derive(Debug, Clone)]
pub struct Dsv4Dims {
    /// Hidden width — 4096 on the published model.
    pub hidden: usize,
    /// Q-down latent (`attn_q_a` output) — 1024.
    pub q_lora_rank: usize,
    /// KV-down latent (`attn_kv` output) — 512.
    pub kv_lora_rank: usize,
    /// Number of attention heads — 64.
    pub n_heads: usize,
    /// Per-head dimension — `q_lora_rank_b / n_heads` = 32768/64 = 512.
    pub head_dim: usize,
    /// Output low-rank intermediate (`attn_output_a` output) — 8192.
    pub output_intermediate: usize,
    /// Expert intermediate (FFN width) — 2048.
    pub ffn_dim: usize,
    /// Number of routed experts per layer — 256.
    pub num_experts: usize,
    /// Number of routed experts activated per token — 6.
    pub experts_per_token: usize,
    /// Number of shared (always-on) experts — 1.
    pub num_shared_experts: usize,
    /// Vocab size — 129280.
    pub vocab: usize,
    /// Norm epsilon (`deepseek4.attention.layer_norm_rms_epsilon`).
    pub norm_eps: f32,
    /// Expert weight scale (`deepseek4.expert_weights_scale`) — 1.5.
    pub expert_weights_scale: f32,
    /// RoPE rotary dimension (`deepseek4.rope.dimension_count`) — 64.
    /// Only this many dims (per Q head; trailing slice of KV latent)
    /// get rotary encoding; the rest pass through.
    pub rope_dim: usize,
    /// RoPE base frequency (`deepseek4.rope.freq_base`) — 10000.0.
    pub rope_base: f64,
}

/// Per-layer weights for the MLA attention block.
#[derive(Clone)]
pub struct Dsv4AttnWeights {
    /// `attn_q_a`: `[q_lora_rank, hidden]` Q-down projection.
    pub q_a: QuantTensor,
    /// `attn_q_b`: `[n_heads * head_dim, q_lora_rank]` Q-up projection.
    pub q_b: QuantTensor,
    /// `attn_kv`: `[kv_lora_rank, hidden]` KV-down projection.
    pub kv: QuantTensor,
    /// `attn_output_a`: `[output_intermediate, hidden]` output low-rank A.
    pub output_a: QuantTensor,
    /// `attn_output_b`: `[hidden, output_intermediate]` output low-rank B.
    pub output_b: QuantTensor,
    /// `attn_norm` `[hidden]` — pre-attention RMS norm.
    pub attn_norm: Arc<[f32]>,
    /// `attn_q_a_norm` `[q_lora_rank]` — post-Q-down RMS norm.
    pub q_a_norm: Arc<[f32]>,
    /// `attn_kv_a_norm` `[kv_lora_rank]` — post-KV-down RMS norm.
    pub kv_a_norm: Arc<[f32]>,
    /// `attn_sinks` `[n_heads]` — per-head sink scalar (loaded but
    /// unused at the MVP).
    pub sinks: Arc<[f32]>,
}

/// Per-layer weights for the MoE FFN block.
///
/// On-disk layout (set by `larql-vindex::format::weights::write_layers`)
/// stores experts INTERLEAVED — each expert occupies a fused
/// `[2*ffn_dim, hidden]` gate+up block followed by a
/// `[hidden, ffn_dim]` down block. We mirror that here: one
/// `QuantTensor` per expert, in two parallel vectors.
#[derive(Clone)]
pub struct Dsv4MoeWeights {
    /// Router `[num_experts, hidden]` (F16 on disk → f32 here).
    pub router: QuantTensor,
    /// Optional learned router bias `[num_experts]`. Absent on the
    /// 3 hash-routing layers.
    pub router_bias: Option<Arc<[f32]>>,
    /// 256 routed gate+up fused projections, each `[2 * ffn_dim, hidden]`.
    pub expert_gate_ups: Vec<QuantTensor>,
    /// 256 routed down projections, each `[hidden, ffn_dim]`.
    pub expert_downs: Vec<QuantTensor>,
    /// Shared-expert gate `[ffn_dim, hidden]`.
    pub shared_gate: QuantTensor,
    /// Shared-expert up `[ffn_dim, hidden]`.
    pub shared_up: QuantTensor,
    /// Shared-expert down `[hidden, ffn_dim]`.
    pub shared_down: QuantTensor,
    /// `ffn_norm` `[hidden]` — post-attention / pre-FFN RMS norm.
    pub ffn_norm: Arc<[f32]>,
}

/// One layer's combined weights.
#[derive(Clone)]
pub struct Dsv4LayerWeights {
    pub attn: Dsv4AttnWeights,
    pub moe: Dsv4MoeWeights,
}

/// Full model weights.
#[derive(Clone)]
pub struct Dsv4Weights {
    /// Token embedding `[vocab, hidden]` — stored as f32 (the vindex
    /// embeddings.bin is f32). Row lookup via `.row(token_id)`.
    pub embed: ndarray::ArcArray2<f32>,
    /// 43 per-layer weights.
    pub layers: Vec<Dsv4LayerWeights>,
    /// `output_norm` `[hidden]`.
    pub final_norm: Arc<[f32]>,
    /// Output / lm_head projection `[vocab, hidden]`. Q8_0 in the
    /// GGUF; the vindex writer re-quantises to Q4_K.
    pub lm_head: QuantTensor,
}

/// MLA KV cache — stores `kv_latent` per token per layer. Much
/// smaller than a standard KV cache because the per-head K/V are
/// computed implicitly from the latent at each attention step.
#[derive(Debug, Clone)]
pub struct Dsv4KvCache {
    /// Per-layer cached KV-latent rows. Each entry is a flat
    /// `Vec<f32>` of `seq_len * kv_lora_rank` elements; new tokens
    /// `extend` onto the tail.
    pub layers: Vec<Vec<f32>>,
    /// `kv_lora_rank` — width of each row.
    pub kv_lora_rank: usize,
    /// Sequence position of the next token (== `layers[l].len() / kv_lora_rank`).
    pub next_position: usize,
}

impl Dsv4KvCache {
    pub fn new(num_layers: usize, kv_lora_rank: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| Vec::new()).collect(),
            kv_lora_rank,
            next_position: 0,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// One-token forward through the DSv4 model. Returns logits
/// `[vocab]` for the next token.
pub fn dsv4_forward_step(
    token_id: u32,
    weights: &Dsv4Weights,
    dims: &Dsv4Dims,
    cache: &mut Dsv4KvCache,
) -> Array1<f32> {
    debug_assert_eq!(cache.num_layers(), weights.layers.len());
    debug_assert_eq!(cache.kv_lora_rank, dims.kv_lora_rank);

    // 1. Embed lookup.
    let vocab = weights.embed.shape()[0];
    let hidden = weights.embed.shape()[1];
    debug_assert!((token_id as usize) < vocab);
    debug_assert_eq!(hidden, dims.hidden);
    let mut x: Array1<f32> = weights.embed.row(token_id as usize).to_owned();

    // 2. Layer stack.
    for layer in 0..weights.layers.len() {
        let layer_w = &weights.layers[layer];

        // 2a. MLA attention block (residual = x + attn_out).
        let attn_out = mla_attention_step(&x, layer, layer_w, dims, cache);
        let x_attn = &x + &attn_out;

        // 2b. FFN block: pre-norm → MoE + shared → residual.
        let ffn_in = rms_norm(&x_attn, &layer_w.moe.ffn_norm, dims.norm_eps);
        let ffn_out = moe_ffn_step(&ffn_in, &layer_w.moe, dims);
        x = &x_attn + &ffn_out;
    }

    cache.next_position += 1;

    // 3. Final norm + lm_head.
    let h = rms_norm(&x, &weights.final_norm, dims.norm_eps);
    weights
        .lm_head
        .matvec(&h)
        .expect("lm_head matvec must succeed at the right hidden width")
}

/// Standard RMS norm — `x * weight / sqrt(mean(x²) + eps)`.
fn rms_norm(x: &Array1<f32>, weight: &[f32], eps: f32) -> Array1<f32> {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    Array1::from_iter(x.iter().zip(weight.iter()).map(|(&xi, &wi)| xi * inv * wi))
}

fn rms_norm_view(x: ArrayView1<'_, f32>, weight: &[f32], eps: f32) -> Array1<f32> {
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    Array1::from_iter(x.iter().zip(weight.iter()).map(|(&xi, &wi)| xi * inv * wi))
}

/// MLA attention block. Appends the new token's KV-latent to the
/// cache, computes Q from the input, and runs full softmax attention
/// over the cached latent slice. The per-head K/V are reconstructed
/// from the latent at each step (we treat the entire latent as the
/// joint K/V representation for now — full MLA splits the latent
/// further; that refinement is deferred).
fn mla_attention_step(
    x: &Array1<f32>,
    layer: usize,
    w: &Dsv4LayerWeights,
    dims: &Dsv4Dims,
    cache: &mut Dsv4KvCache,
) -> Array1<f32> {
    // Pre-attn norm.
    let h = rms_norm(x, &w.attn.attn_norm, dims.norm_eps);

    // Q path: x → q_a (down) → q_a_norm → q_b (up) → [n_heads, head_dim].
    let q_latent = w.attn.q_a.matvec(&h).expect("q_a matvec at hidden width");
    let q_latent = rms_norm(&q_latent, &w.attn.q_a_norm, dims.norm_eps);
    let q_full = w
        .attn
        .q_b
        .matvec(&q_latent)
        .expect("q_b matvec at q_lora_rank width");
    debug_assert_eq!(q_full.len(), dims.n_heads * dims.head_dim);
    // RoPE on the rotary portion of each head's Q. V4 reports
    // `rope.dimension_count = 64` of the 512 head_dim (fraction
    // 64/512 = 0.125). The rotary section is the FIRST `rope_dim`
    // dims of each head (HF convention). Position is the current
    // token's position in the sequence (== cache.next_position
    // before we append the new KV-latent below).
    let q_full = if dims.rope_dim > 0 && dims.rope_dim <= dims.head_dim {
        let row =
            Array2::from_shape_vec((1, q_full.len()), q_full.to_vec()).expect("q reshape for rope");
        let rotated = super::rope::apply_rope_partial_at(
            &row,
            dims.n_heads,
            dims.head_dim,
            dims.rope_base,
            dims.rope_dim as f64 / dims.head_dim as f64,
            cache.next_position,
        );
        rotated.row(0).to_owned()
    } else {
        q_full
    };

    // KV path: x → kv (down) → kv_a_norm → cache.
    let kv_latent = w.attn.kv.matvec(&h).expect("kv matvec at hidden width");
    let kv_latent = rms_norm(&kv_latent, &w.attn.kv_a_norm, dims.norm_eps);
    debug_assert_eq!(kv_latent.len(), dims.kv_lora_rank);
    // RoPE on the kv-rope portion of the latent. Treat as a single
    // "head" of width `kv_lora_rank`; rotate the first `rope_dim`
    // dims at the current token position.
    let kv_latent = if dims.rope_dim > 0 && dims.rope_dim <= dims.kv_lora_rank {
        let row = Array2::from_shape_vec((1, kv_latent.len()), kv_latent.to_vec())
            .expect("kv reshape for rope");
        let rotated = super::rope::apply_rope_partial_at(
            &row,
            1,
            dims.kv_lora_rank,
            dims.rope_base,
            dims.rope_dim as f64 / dims.kv_lora_rank as f64,
            cache.next_position,
        );
        rotated.row(0).to_owned()
    } else {
        kv_latent
    };

    // Append to cache for this layer.
    let layer_cache = &mut cache.layers[layer];
    layer_cache.extend_from_slice(kv_latent.as_slice().unwrap());
    let seq_len = layer_cache.len() / dims.kv_lora_rank;

    // Proper V3-inspired MLA attention compute:
    //
    // - Per-head Q split: q_nope (first `head_dim - rope_dim` dims) +
    //   q_rope (last `rope_dim` dims) — RoPE was applied to the
    //   FIRST `rope_dim` of each head above, so the rotary slice is
    //   at the START of each head, not the end. Re-derive splits
    //   accordingly.
    // - KV latent split: k_rope is the first `rope_dim` of the
    //   latent (RoPE'd above); kv_nope is the rest. Both are SHARED
    //   across heads — V4 has no explicit per-head `kv_b` matrix.
    // - K per head = [k_rope, kv_nope] concatenated, shared across heads.
    // - V per head = kv_nope, shared across heads.
    // - Per-head score = q_rope[h] · k_rope[t] + q_nope[h] · kv_nope[t]
    // - Per-head v_out = softmax(scores) · kv_nope across cached tokens
    //
    // This is V3-style MLA without the per-head `kv_b` up-projection;
    // V4's `attn_kv` tensor doesn't include one. The published model
    // presumably absorbs `kv_b` into `attn_q_b` (the "absorbed MLA"
    // form), so each head's q_nope vector already carries the
    // per-head kv-up information. Whether or not that interpretation
    // is exact, the score / V-weighting structure here is the
    // closest we can get without the V4 paper.
    let rope_dim = dims.rope_dim;
    let q_nope_dim = dims.head_dim - rope_dim;
    let kv_nope_dim = dims.kv_lora_rank - rope_dim;
    let scale = 1.0 / ((q_nope_dim + rope_dim) as f32).sqrt();

    // Per-head v_out, each `kv_nope_dim` wide.
    let mut per_head_v_out: Vec<Array1<f32>> =
        vec![Array1::<f32>::zeros(kv_nope_dim); dims.n_heads];

    for head in 0..dims.n_heads {
        let q_head_start = head * dims.head_dim;
        // q_rope = q[head, 0..rope_dim]; q_nope = q[head, rope_dim..head_dim]
        let q_head_full = q_full.slice(ndarray::s![q_head_start..q_head_start + dims.head_dim]);
        let q_rope = q_head_full.slice(ndarray::s![..rope_dim]);
        let q_nope = q_head_full.slice(ndarray::s![rope_dim..]);

        // Score each cached position.
        let mut scores = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let latent_start = pos * dims.kv_lora_rank;
            let latent = &layer_cache[latent_start..latent_start + dims.kv_lora_rank];
            let k_rope = &latent[..rope_dim];
            let kv_nope = &latent[rope_dim..];

            // q_rope · k_rope
            let mut s = 0.0f32;
            for i in 0..rope_dim {
                s += q_rope[i] * k_rope[i];
            }
            // q_nope · kv_nope — q_nope is `q_nope_dim` wide,
            // kv_nope is `kv_nope_dim` wide. Use the min in case
            // these differ (q_nope_dim != kv_nope_dim is possible
            // if head_dim != kv_lora_rank; for the published V4
            // they're both 448 so it's an exact match).
            let dot_dim = q_nope_dim.min(kv_nope_dim);
            for i in 0..dot_dim {
                s += q_nope[i] * kv_nope[i];
            }
            scores.push(s * scale);
        }
        // Softmax.
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut denom = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            denom += *s;
        }
        for s in scores.iter_mut() {
            *s /= denom.max(1e-9);
        }

        // V-weighted sum: v_head[i] = Σ_t softmax(t) · kv_nope[t][i]
        // V is shared across heads (V4 has no per-head V projection).
        let v_out = &mut per_head_v_out[head];
        for pos in 0..seq_len {
            let latent_start = pos * dims.kv_lora_rank;
            let kv_nope = &layer_cache[latent_start + rope_dim..latent_start + dims.kv_lora_rank];
            let w = scores[pos];
            for i in 0..kv_nope_dim {
                v_out[i] += w * kv_nope[i];
            }
        }
    }

    // Output projection. The available tensors are:
    //   attn_output_a: [output_intermediate=8192, hidden=4096]
    //   attn_output_b: [hidden=4096, output_intermediate=8192]
    //
    // Per-head v_outs are `kv_nope_dim`-wide (448). Concatenated
    // they're `n_heads * kv_nope_dim = 64 * 448 = 28672`, which
    // matches no tensor.
    //
    // Best interpretation that fits the dimensions: V4's per-head
    // V output is actually `output_intermediate / n_heads = 128`
    // wide (not `kv_nope_dim`). The `output_b` matrix
    // ([hidden, output_intermediate]) is the per-head W_o that
    // maps the concatenated heads → hidden. So we slice each
    // head's `kv_nope_dim`-wide V-output down to
    // `output_intermediate / n_heads` and concat into a vector
    // matching `output_b`'s input shape.
    //
    // This is still approximate (the "real" per-head V is presumably
    // some projection of kv_nope rather than a slice), but the
    // dimensional flow is now structurally correct: per-head V →
    // per-head W_o → hidden, with the residual refinement via
    // `output_a` applied at the end.
    let per_head_o_dim = dims.output_intermediate / dims.n_heads;
    let mut concat_heads = Array1::<f32>::zeros(dims.output_intermediate);
    let take = per_head_o_dim.min(kv_nope_dim);
    for head in 0..dims.n_heads {
        let dst_start = head * per_head_o_dim;
        for i in 0..take {
            concat_heads[dst_start + i] = per_head_v_out[head][i];
        }
    }

    // output_b: 8192 → hidden = per-head W_o.
    let attn_hidden = w
        .attn
        .output_b
        .matvec(&concat_heads)
        .expect("output_b matvec (per-head W_o)");
    debug_assert_eq!(attn_hidden.len(), dims.hidden);

    // output_a: hidden → output_intermediate, treated as a low-rank
    // residual refinement that produces a SECOND vector to combine
    // with attn_hidden. Without the V4 paper, the exact combination
    // semantics are unclear — for the MVP we treat output_a's output
    // as a NO-OP (the published model presumably uses it for HC or
    // similar; we'll wire that in when HC lands).
    let _ = w.attn.output_a;

    attn_hidden
}

/// SwiGLU activation: silu(gate) * up → down.
fn swiglu(gate: &Array1<f32>, up: &Array1<f32>, down: &QuantTensor) -> Array1<f32> {
    debug_assert_eq!(gate.len(), up.len());
    let mid: Array1<f32> = Array1::from_iter(gate.iter().zip(up.iter()).map(|(&g, &u)| {
        // SiLU(g) = g * sigmoid(g)
        let s = 1.0 / (1.0 + (-g).exp());
        g * s * u
    }));
    down.matvec(&mid).expect("down matvec")
}

/// MoE FFN: top-k learned router + shared expert.
fn moe_ffn_step(x: &Array1<f32>, w: &Dsv4MoeWeights, dims: &Dsv4Dims) -> Array1<f32> {
    // Router: scores = router @ x  → [num_experts]
    let mut scores = w.router.matvec(x).expect("router matvec");
    if let Some(bias) = w.router_bias.as_ref() {
        for (i, b) in bias.iter().enumerate() {
            scores[i] += *b;
        }
    }
    // Pick top-k experts by score.
    let k = dims.experts_per_token;
    let mut indices: Vec<usize> = (0..dims.num_experts).collect();
    indices.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);

    // Softmax over selected scores → gating weights.
    let mut sel_scores: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
    let m = sel_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut denom = 0.0f32;
    for s in sel_scores.iter_mut() {
        *s = (*s - m).exp();
        denom += *s;
    }
    for s in sel_scores.iter_mut() {
        *s = (*s / denom.max(1e-9)) * dims.expert_weights_scale;
    }

    // Accumulate selected experts. Each expert's gate+up are fused
    // (single matvec, split halves) per the writer's on-disk layout.
    let mut out = Array1::<f32>::zeros(dims.hidden);
    for (&expert_id, &gw) in indices.iter().zip(sel_scores.iter()) {
        let gate_up = w.expert_gate_ups[expert_id]
            .matvec(x)
            .expect("expert gate_up matvec");
        let half = gate_up.len() / 2;
        let g = gate_up.slice(ndarray::s![..half]).to_owned();
        let u = gate_up.slice(ndarray::s![half..]).to_owned();
        let expert_out = swiglu(&g, &u, &w.expert_downs[expert_id]);
        for (o, v) in out.iter_mut().zip(expert_out.iter()) {
            *o += gw * v;
        }
    }

    // Shared expert (always on). The writer stores gate / up as
    // separate `ffn_*_shexp` tensors (unfused), so two matvecs.
    let g_sh = w.shared_gate.matvec(x).expect("shared gate matvec");
    let u_sh = w.shared_up.matvec(x).expect("shared up matvec");
    let shared_out = swiglu(&g_sh, &u_sh, &w.shared_down);
    out + shared_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::quant::ggml::TYPE_F32;

    fn dummy_quant(rows: usize, cols: usize, val: f32) -> QuantTensor {
        let data: Vec<f32> = vec![val; rows * cols];
        QuantTensor::from_f32_rows(rows, cols, &data)
    }

    fn dummy_norm(n: usize, v: f32) -> Arc<[f32]> {
        std::iter::repeat(v).take(n).collect::<Vec<_>>().into()
    }

    fn tiny_dims() -> Dsv4Dims {
        Dsv4Dims {
            hidden: 4,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            n_heads: 1,
            head_dim: 4,
            output_intermediate: 4,
            ffn_dim: 4,
            num_experts: 2,
            experts_per_token: 1,
            num_shared_experts: 1,
            vocab: 5,
            norm_eps: 1e-6,
            expert_weights_scale: 1.0,
            rope_dim: 0,
            rope_base: 10000.0,
        }
    }

    fn tiny_weights() -> Dsv4Weights {
        let _ = TYPE_F32;
        let dims = tiny_dims();
        let attn = Dsv4AttnWeights {
            q_a: dummy_quant(dims.q_lora_rank, dims.hidden, 0.01),
            q_b: dummy_quant(dims.n_heads * dims.head_dim, dims.q_lora_rank, 0.01),
            kv: dummy_quant(dims.kv_lora_rank, dims.hidden, 0.01),
            output_a: dummy_quant(dims.output_intermediate, dims.hidden, 0.01),
            output_b: dummy_quant(dims.hidden, dims.output_intermediate, 0.01),
            attn_norm: dummy_norm(dims.hidden, 1.0),
            q_a_norm: dummy_norm(dims.q_lora_rank, 1.0),
            kv_a_norm: dummy_norm(dims.kv_lora_rank, 1.0),
            sinks: dummy_norm(dims.n_heads, 0.0),
        };
        let moe = Dsv4MoeWeights {
            router: dummy_quant(dims.num_experts, dims.hidden, 0.01),
            router_bias: None,
            expert_gate_ups: (0..dims.num_experts)
                .map(|_| dummy_quant(2 * dims.ffn_dim, dims.hidden, 0.01))
                .collect(),
            expert_downs: (0..dims.num_experts)
                .map(|_| dummy_quant(dims.hidden, dims.ffn_dim, 0.01))
                .collect(),
            shared_gate: dummy_quant(dims.ffn_dim, dims.hidden, 0.01),
            shared_up: dummy_quant(dims.ffn_dim, dims.hidden, 0.01),
            shared_down: dummy_quant(dims.hidden, dims.ffn_dim, 0.01),
            ffn_norm: dummy_norm(dims.hidden, 1.0),
        };
        let layer = Dsv4LayerWeights { attn, moe };
        let mut embed = Array2::<f32>::zeros((dims.vocab, dims.hidden));
        for i in 0..dims.vocab {
            embed[[i, i % dims.hidden]] = 1.0;
        }
        Dsv4Weights {
            embed: embed.into_shared(),
            layers: vec![layer.clone(), layer],
            final_norm: dummy_norm(dims.hidden, 1.0),
            lm_head: dummy_quant(dims.vocab, dims.hidden, 0.01),
        }
    }

    #[test]
    fn dsv4_forward_step_runs_without_panic() {
        let dims = tiny_dims();
        let weights = tiny_weights();
        let mut cache = Dsv4KvCache::new(weights.layers.len(), dims.kv_lora_rank);
        let logits = dsv4_forward_step(2, &weights, &dims, &mut cache);
        assert_eq!(logits.len(), dims.vocab);
        assert_eq!(cache.next_position, 1);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rms_norm_unit_input_unit_weight_preserves_unit_norm() {
        let x = Array1::from(vec![1.0_f32, 0.0, 0.0, 0.0]);
        let w = vec![1.0_f32; 4];
        let y = rms_norm(&x, &w, 1e-6);
        // ||x|| = 1, mean(x²) = 0.25, inv = 1/sqrt(0.25) = 2 → y = 2 * x
        assert!((y[0] - 2.0).abs() < 1e-4);
        assert!(y[1].abs() < 1e-4);
    }

    #[test]
    fn cache_appends_per_token() {
        let dims = tiny_dims();
        let weights = tiny_weights();
        let mut cache = Dsv4KvCache::new(weights.layers.len(), dims.kv_lora_rank);
        let _ = dsv4_forward_step(1, &weights, &dims, &mut cache);
        let _ = dsv4_forward_step(2, &weights, &dims, &mut cache);
        let _ = dsv4_forward_step(3, &weights, &dims, &mut cache);
        assert_eq!(cache.next_position, 3);
        assert_eq!(cache.layers[0].len(), 3 * dims.kv_lora_rank);
    }
}
