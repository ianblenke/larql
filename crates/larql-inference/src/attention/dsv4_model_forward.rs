//! DSv4 model-level forward — the full composition that produces logits
//! from token ids.
//!
//! Pipeline:
//! ```text
//! 1. embed: x = token_embd[token_ids]                  (n_tokens, n_embd)
//! 2. broadcast x into 4 streams: residual = x repeated (n_tokens, n_hc, n_embd)
//! 3. for layer in 0..n_layer:
//!        residual = dsv4_per_layer_forward(residual, ...)
//! 4. head: pooled = dsv4_hc_head(residual, ...)       (n_tokens, n_embd)
//! 5. norm: pooled = rms_norm(pooled, output_norm)
//! 6. lm_head: logits = pooled @ output_lm_head^T       (n_tokens, n_vocab)
//! ```
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:1098-1580`
//! (the model-level forward graph).

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use super::dsv4_mhc::weighted_sum;
use super::dsv4_per_layer::{dsv4_per_layer_forward, DsV4LayerParams, DsV4LayerWeights};

/// Output-head weights (the mHC head + final norm + lm_head).
#[derive(Clone, Copy)]
pub struct DsV4HeadWeights<'a> {
    /// `(n_hc, n_hc * n_embd)` — output mHC fn projection (note the
    /// smaller output dim vs per-layer mHC fn: only `n_hc`, since the
    /// head collapses 4-stream → 1-stream directly).
    pub output_hc_fn: ArrayView2<'a, f32>,
    /// `(1,)` — single-element scalar applied to the head pre-bias.
    pub output_hc_scale: f32,
    /// `(n_hc,)` — per-stream bias for the head sigmoid.
    pub output_hc_base: &'a [f32],
    /// `(n_embd,)` — final RMSNorm gain.
    pub output_norm: &'a [f32],
    /// `(n_vocab, n_embd)` — lm_head projection.
    pub output_lm_head: ArrayView2<'a, f32>,
}

/// Output-head scalar config.
#[derive(Clone, Copy, Debug)]
pub struct DsV4HeadParams {
    pub n_embd: usize,
    pub n_hc: usize,
    pub n_vocab: usize,
    pub norm_eps: f32,
    pub hc_eps: f32,
}

/// Construct the initial 4-stream residual from a 1-stream embedding
/// by broadcasting `x` across all `n_hc` streams. This is the most
/// common DSv4 initial-state convention — alternative weighting via
/// `output_hc_base` is sometimes used at the head, not the entry.
pub fn broadcast_to_streams(x: ArrayView2<f32>, n_hc: usize) -> Array3<f32> {
    let n_tokens = x.shape()[0];
    let n_embd = x.shape()[1];
    let mut out = Array3::<f32>::zeros((n_tokens, n_hc, n_embd));
    for t in 0..n_tokens {
        for h in 0..n_hc {
            for d in 0..n_embd {
                out[[t, h, d]] = x[[t, d]];
            }
        }
    }
    out
}

/// Apply the output mHC head: 4-stream residual → 1-stream pooled output.
///
/// `residual`: `(n_tokens, n_hc, n_embd)`.
/// Returns `(n_tokens, n_embd)`.
pub fn dsv4_hc_head(
    residual: ArrayView3<f32>,
    w: &DsV4HeadWeights,
    p: &DsV4HeadParams,
) -> Array2<f32> {
    let n_tokens = residual.shape()[0];
    let n_hc = p.n_hc;
    let n_embd = p.n_embd;
    assert_eq!(
        residual.shape(),
        &[n_tokens, n_hc, n_embd],
        "residual shape"
    );
    let hc_dim = n_hc * n_embd;
    assert_eq!(
        w.output_hc_fn.shape(),
        &[n_hc, hc_dim],
        "output_hc_fn shape"
    );
    assert_eq!(w.output_hc_base.len(), n_hc, "output_hc_base length");

    // 1. Flatten residual to (n_tokens, hc_dim).
    let mut flat = Array2::<f32>::zeros((n_tokens, hc_dim));
    for t in 0..n_tokens {
        for h in 0..n_hc {
            let off = h * n_embd;
            for d in 0..n_embd {
                flat[[t, off + d]] = residual[[t, h, d]];
            }
        }
    }
    // 2. RMSNorm per token over the flat axis.
    for t in 0..n_tokens {
        let mut sumsq = 0.0_f32;
        for k in 0..hc_dim {
            let v = flat[[t, k]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / hc_dim as f32 + p.norm_eps).sqrt();
        for k in 0..hc_dim {
            flat[[t, k]] *= inv;
        }
    }
    // 3. pre = flat @ output_hc_fn^T → (n_tokens, n_hc)
    let mut pre = Array2::<f32>::zeros((n_tokens, n_hc));
    for t in 0..n_tokens {
        for h in 0..n_hc {
            let mut acc = 0.0_f32;
            for k in 0..hc_dim {
                acc += flat[[t, k]] * w.output_hc_fn[[h, k]];
            }
            pre[[t, h]] = acc;
        }
    }
    // 4. pre *= output_hc_scale (broadcast across all)
    for v in pre.iter_mut() {
        *v *= w.output_hc_scale;
    }
    // 5. pre += output_hc_base (broadcast across n_tokens)
    for t in 0..n_tokens {
        for h in 0..n_hc {
            pre[[t, h]] += w.output_hc_base[h];
        }
    }
    // 6. pre = sigmoid(pre) + hc_eps
    for v in pre.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp()) + p.hc_eps;
    }
    // 7. weighted_sum(residual, pre) → (n_tokens, n_embd)
    weighted_sum(residual, pre.view())
}

/// Embed token ids into the n_embd space via lookup.
///
/// `token_ids`: `(n_tokens,)`.
/// `token_embd`: `(n_vocab, n_embd)`.
/// Returns `(n_tokens, n_embd)`.
pub fn embed_tokens(token_ids: &[u32], token_embd: ArrayView2<f32>) -> Array2<f32> {
    let n_tokens = token_ids.len();
    let n_vocab = token_embd.shape()[0];
    let n_embd = token_embd.shape()[1];
    let mut out = Array2::<f32>::zeros((n_tokens, n_embd));
    for t in 0..n_tokens {
        let tid = token_ids[t] as usize;
        assert!(
            tid < n_vocab,
            "token id {tid} out of range for vocab_size {n_vocab}"
        );
        for d in 0..n_embd {
            out[[t, d]] = token_embd[[tid, d]];
        }
    }
    out
}

/// Run the full DSv4 model forward: token_ids → logits.
///
/// All slices must have the same length `n_layer`. `position_offset`
/// is 0 for prefill from scratch.
pub fn dsv4_model_forward<'a, 'p>(
    token_ids: &[u32],
    token_embd: ArrayView2<'a, f32>,
    layers: &[DsV4LayerWeights<'a, 'p>],
    layer_params: &[DsV4LayerParams],
    head: &DsV4HeadWeights<'a>,
    head_params: &DsV4HeadParams,
    position_offset: usize,
) -> Array2<f32> {
    assert_eq!(
        layers.len(),
        layer_params.len(),
        "layers / layer_params length mismatch"
    );
    let n_tokens = token_ids.len();
    let n_embd = head_params.n_embd;
    let n_hc = head_params.n_hc;

    // 1. Embed token ids.
    let x = embed_tokens(token_ids, token_embd);
    assert_eq!(x.shape(), &[n_tokens, n_embd]);

    // 2. Broadcast to 4-stream initial residual.
    let mut residual = broadcast_to_streams(x.view(), n_hc);

    // 3. Walk N layers.
    for (layer_w, layer_p) in layers.iter().zip(layer_params.iter()) {
        residual = dsv4_per_layer_forward(
            residual.view(),
            layer_w,
            layer_p,
            Some(token_ids),
            position_offset,
        );
    }

    // 4. mHC head → 1-stream pooled.
    let mut pooled = dsv4_hc_head(residual.view(), head, head_params);

    // 5. Final RMSNorm.
    for t in 0..n_tokens {
        let mut sumsq = 0.0_f32;
        for d in 0..n_embd {
            let v = pooled[[t, d]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n_embd as f32 + head_params.norm_eps).sqrt();
        for d in 0..n_embd {
            pooled[[t, d]] *= inv * head.output_norm[d];
        }
    }

    // 6. lm_head projection.
    let n_vocab = head_params.n_vocab;
    let mut logits = Array2::<f32>::zeros((n_tokens, n_vocab));
    for t in 0..n_tokens {
        for v in 0..n_vocab {
            let mut acc = 0.0_f32;
            for d in 0..n_embd {
                acc += pooled[[t, d]] * head.output_lm_head[[v, d]];
            }
            logits[[t, v]] = acc;
        }
    }
    logits
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_attn_block::{DsV4AttnBlockParams, DsV4AttnBlockWeights};
    use super::super::dsv4_attn_dispatch::DsV4AttnLayer;
    use super::super::dsv4_ffn_block::{Dsv4FfnParams, Dsv4FfnWeights, SharedExpertWeights};
    use super::super::dsv4_mhc_bookend::{MhcParams, MhcWeights};
    use super::super::dsv4_moe_dispatch::MoeExpertWeights;
    use super::super::dsv4_rope_tail::DsV4RopeMode;
    use super::*;
    use ndarray::{Array1, Array3};

    /// Embed lookup: one-hot vocab entries produce the matching row.
    #[test]
    fn embed_picks_vocab_row() {
        let token_embd = Array2::<f32>::from_shape_fn((4, 3), |(t, d)| (t * 100 + d * 10) as f32);
        let tids = vec![2u32, 0, 3];
        let out = embed_tokens(&tids, token_embd.view());
        assert_eq!(out[[0, 0]], 200.0);
        assert_eq!(out[[0, 1]], 210.0);
        assert_eq!(out[[1, 0]], 0.0);
        assert_eq!(out[[2, 1]], 310.0);
    }

    /// broadcast_to_streams: each stream is a copy of the input.
    #[test]
    fn broadcast_to_streams_replicates() {
        let x = Array2::<f32>::from_shape_fn((2, 3), |(t, d)| (t + d) as f32);
        let r = broadcast_to_streams(x.view(), 4);
        assert_eq!(r.shape(), &[2, 4, 3]);
        for t in 0..2 {
            for h in 0..4 {
                for d in 0..3 {
                    assert_eq!(r[[t, h, d]], x[[t, d]]);
                }
            }
        }
    }

    /// Output head: random weights → finite output of expected shape.
    #[test]
    fn hc_head_shape_and_finite() {
        let n_tokens = 3;
        let n_hc = 4;
        let n_embd = 8;
        let residual = Array3::<f32>::from_shape_fn((n_tokens, n_hc, n_embd), |(t, h, d)| {
            ((t + h + d) as f32 * 0.013).sin()
        });
        let hc_fn = Array2::<f32>::from_shape_fn((n_hc, n_hc * n_embd), |(h, k)| {
            ((h + k) as f32 * 0.01).cos() * 0.1
        });
        let hc_base = vec![0.0_f32; n_hc];
        let output_norm = vec![1.0_f32; n_embd];
        let lm_head = Array2::<f32>::from_shape_fn((10, n_embd), |(v, d)| {
            ((v + d) as f32 * 0.015).sin() * 0.1
        });
        let head_w = DsV4HeadWeights {
            output_hc_fn: hc_fn.view(),
            output_hc_scale: 1.0,
            output_hc_base: &hc_base,
            output_norm: &output_norm,
            output_lm_head: lm_head.view(),
        };
        let head_p = DsV4HeadParams {
            n_embd,
            n_hc,
            n_vocab: 10,
            norm_eps: 1e-5,
            hc_eps: 1e-5,
        };
        let pooled = dsv4_hc_head(residual.view(), &head_w, &head_p);
        assert_eq!(pooled.shape(), &[n_tokens, n_embd]);
        assert!(pooled.iter().all(|v| v.is_finite()));
    }

    /// Full model forward with 2 layers and small dims. Output shape:
    /// `(n_tokens, n_vocab)`. Output finite and non-trivial.
    #[test]
    fn model_forward_two_layers_smoke() {
        let n_tokens = 3;
        let n_vocab = 16;
        let n_hc = 4;
        let n_embd = 64;
        let n_head = 4;
        let head_dim = 64;
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;
        let n_expert = 6;
        let n_expert_used = 2;
        let n_ff_exp = 8;
        let hidden_shared = 4;
        let n_layer = 2;

        // Token embeddings.
        let token_embd = Array2::<f32>::from_shape_fn((n_vocab, n_embd), |(v, d)| {
            ((v + d) as f32 * 0.005).sin() * 0.1
        });
        let token_ids = vec![1u32, 7, 11];

        // Per-layer weights: synthesize n_layer identical copies of the
        // 8h-3d-2 smoke setup, with slightly varied seeds.
        let attn_params = DsV4AttnBlockParams {
            n_embd,
            n_head,
            head_dim,
            q_lora_rank,
            n_groups,
            o_lora_rank,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
        };
        let hc_dim = n_hc * n_embd;
        let hc_mix = (2 + n_hc) * n_hc;
        let hc_scale = [0.5_f32, 0.5, 0.5];
        let hc_base = vec![0.0_f32; hc_mix];
        let layer_p = DsV4LayerParams {
            mhc: MhcParams {
                n_embd,
                n_hc,
                sinkhorn_iters: 2,
                hc_eps: 1e-5,
                norm_eps: 1e-5,
            },
            ffn: Dsv4FfnParams {
                n_expert,
                n_expert_used,
                norm_eps: 1e-5,
                routed_swiglu_limit: 7.0,
                shared_swiglu_limit: 0.0,
                expert_weights_norm: true,
                expert_weights_scale: 1.0,
            },
        };
        let layer_params: Vec<DsV4LayerParams> = vec![layer_p; n_layer];

        // Owned per-layer weight buffers (we keep arrays around so
        // DsV4LayerWeights borrows live long enough).
        let attn_norm = vec![1.0_f32; n_embd];
        let wq_a = Array2::<f32>::from_shape_fn((q_lora_rank, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.01).sin()
        });
        let q_a_norm = vec![1.0_f32; q_lora_rank];
        let wq_b = Array2::<f32>::from_shape_fn((n_head * head_dim, q_lora_rank), |(i, j)| {
            ((i + j) as f32 * 0.013).cos() * 0.1
        });
        let wkv = Array2::<f32>::from_shape_fn((head_dim, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.007).sin() * 0.05
        });
        let kv_a_norm = vec![1.0_f32; head_dim];
        let wo_a = Array3::<f32>::from_shape_fn((n_groups, o_lora_rank, group_dim), |(g, r, j)| {
            ((g + r + j) as f32 * 0.005).cos() * 0.05
        });
        let wo_b = Array2::<f32>::from_shape_fn((n_embd, low_dim), |(i, j)| {
            ((i + j) as f32 * 0.011).sin() * 0.1
        });
        let attn_sinks = Array1::<f32>::from_elem(n_head, -1.0);
        let hc_fn_attn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.003).sin() * 0.1
        });
        let hc_fn_ffn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.004).cos() * 0.1
        });
        let ffn_norm = vec![1.0_f32; n_embd];
        let gate_inp = Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| {
            ((e + d) as f32 * 0.01).cos() * 0.1
        });
        let gate_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.011).sin() * 0.1
        });
        let up_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.013).cos() * 0.1
        });
        let down_exps = Array3::<f32>::from_shape_fn((n_expert, n_embd, n_ff_exp), |(e, d, i)| {
            ((e + d + i) as f32 * 0.009).sin() * 0.1
        });
        let gate_shexp = Array2::<f32>::from_shape_fn((hidden_shared, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.015).sin() * 0.1
        });
        let up_shexp = Array2::<f32>::from_shape_fn((hidden_shared, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.017).cos() * 0.1
        });
        let down_shexp = Array2::<f32>::from_shape_fn((n_embd, hidden_shared), |(d, i)| {
            ((d + i) as f32 * 0.019).sin() * 0.1
        });

        // Single attention weight set reused across the n_layer layers.
        let attn_w = DsV4AttnBlockWeights {
            attn_norm: &attn_norm,
            wq_a: wq_a.view(),
            q_a_norm: &q_a_norm,
            wq_b: wq_b.view(),
            wkv: wkv.view(),
            kv_a_norm: &kv_a_norm,
            wo_a: wo_a.view(),
            wo_b: wo_b.view(),
            attn_sinks: Some(attn_sinks.view()),
        };

        let layer_w_template = DsV4LayerWeights {
            mhc_attn: MhcWeights {
                hc_fn: hc_fn_attn.view(),
                hc_scale: &hc_scale,
                hc_base: &hc_base,
            },
            attn: DsV4AttnLayer::NoCompress {
                weights: attn_w,
                params: &attn_params,
            },
            mhc_ffn: MhcWeights {
                hc_fn: hc_fn_ffn.view(),
                hc_scale: &hc_scale,
                hc_base: &hc_base,
            },
            ffn: Dsv4FfnWeights {
                ffn_norm: &ffn_norm,
                gate_inp: gate_inp.view(),
                exp_probs_b: None,
                gate_tid2eid: None,
                moe: MoeExpertWeights {
                    gate_exps: gate_exps.view(),
                    up_exps: up_exps.view(),
                    down_exps: down_exps.view(),
                },
                shared: SharedExpertWeights {
                    gate_shexp: gate_shexp.view(),
                    up_shexp: up_shexp.view(),
                    down_shexp: down_shexp.view(),
                },
            },
        };
        // n_layer copies. DsV4LayerWeights is a regular struct — we
        // can clone it because all interior fields are Copy or borrowed.
        let mut layers: Vec<DsV4LayerWeights> = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            layers.push(DsV4LayerWeights {
                mhc_attn: layer_w_template.mhc_attn,
                attn: match &layer_w_template.attn {
                    DsV4AttnLayer::NoCompress { weights, params } => DsV4AttnLayer::NoCompress {
                        weights: *weights,
                        params,
                    },
                    _ => unreachable!(),
                },
                mhc_ffn: layer_w_template.mhc_ffn,
                ffn: layer_w_template.ffn,
            });
        }

        // ── Head weights ──
        let output_hc_fn = Array2::<f32>::from_shape_fn((n_hc, hc_dim), |(h, k)| {
            ((h + k) as f32 * 0.005).sin() * 0.1
        });
        let output_hc_base = vec![0.0_f32; n_hc];
        let output_norm = vec![1.0_f32; n_embd];
        let lm_head = Array2::<f32>::from_shape_fn((n_vocab, n_embd), |(v, d)| {
            ((v + d) as f32 * 0.005).cos() * 0.1
        });
        let head_w = DsV4HeadWeights {
            output_hc_fn: output_hc_fn.view(),
            output_hc_scale: 1.0,
            output_hc_base: &output_hc_base,
            output_norm: &output_norm,
            output_lm_head: lm_head.view(),
        };
        let head_p = DsV4HeadParams {
            n_embd,
            n_hc,
            n_vocab,
            norm_eps: 1e-5,
            hc_eps: 1e-5,
        };

        let logits = dsv4_model_forward(
            &token_ids,
            token_embd.view(),
            &layers,
            &layer_params,
            &head_w,
            &head_p,
            0,
        );
        assert_eq!(logits.shape(), &[n_tokens, n_vocab]);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        let total: f32 = logits.iter().map(|v| v.abs()).sum();
        assert!(total > 1e-3, "logits collapsed (sum={total})");

        // Different token positions should produce different logit rows
        // (proves the layer loop actually distinguishes them).
        let mut row_diff = 0.0_f32;
        for v in 0..n_vocab {
            row_diff += (logits[[0, v]] - logits[[1, v]]).abs();
        }
        assert!(
            row_diff > 1e-3,
            "row 0 and row 1 logits are identical (loop not running per-token)"
        );
    }
}
