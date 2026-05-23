//! DSv4 FFN block — pre-norm + (routed MoE + shared expert) sum.
//!
//! The full per-layer FFN composed of:
//!
//! 1. **RMSNorm** with the layer's `ffn_norm` weight.
//! 2. **Routing**: regular sqrt-softplus + bias (Stage 8h-3a) OR hash
//!    (first `n_hash_layers` layers; reuses the same routing fn with
//!    a pre-built table).
//! 3. **Routed MoE dispatch** (Stage 8h-3b).
//! 4. **Shared expert FFN**: a single dense SwiGLU FFN over the
//!    concatenated shared-expert hidden width.
//! 5. **Sum** routed + shared → final FFN output.
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:1522-1559`
//! (the inline FFN construction inside the per-layer forward).

use ndarray::{Array2, ArrayView1, ArrayView2};

use super::dsv4_moe_dispatch::{dsv4_moe_dispatch, MoeExpertWeights};
use super::dsv4_moe_ops::clamped_swiglu_row;
use super::dsv4_moe_routing::{compute_router_topk, hash_route_topk, RoutingResult};

/// Shared-expert FFN weights — a single dense gate/up/down with hidden
/// width `n_ff_exp * n_expert_shared` (DSv4 concatenates the shared
/// experts into one big FFN).
#[derive(Clone, Copy)]
pub struct SharedExpertWeights<'a> {
    /// `(hidden, n_embd)` — concatenated shared-expert gate matrix.
    pub gate_shexp: ArrayView2<'a, f32>,
    /// `(hidden, n_embd)` — concatenated shared-expert up matrix.
    pub up_shexp: ArrayView2<'a, f32>,
    /// `(n_embd, hidden)` — concatenated shared-expert down matrix.
    pub down_shexp: ArrayView2<'a, f32>,
}

/// Run the shared-expert FFN over every token. Plain SwiGLU (no
/// clamping) — pass `swiglu_limit = 0.0` to mean "no clamping".
pub fn dsv4_shared_expert_ffn(
    x: ArrayView2<f32>,
    w: &SharedExpertWeights,
    swiglu_limit: f32,
) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    let n_embd = x.shape()[1];
    let hidden = w.gate_shexp.shape()[0];
    assert_eq!(w.gate_shexp.shape(), &[hidden, n_embd], "gate_shexp shape");
    assert_eq!(w.up_shexp.shape(), &[hidden, n_embd], "up_shexp shape");
    assert_eq!(w.down_shexp.shape(), &[n_embd, hidden], "down_shexp shape");

    let mut out = Array2::<f32>::zeros((n_tokens, n_embd));
    let mut gate_out = ndarray::Array1::<f32>::zeros(hidden);
    let mut up_out = ndarray::Array1::<f32>::zeros(hidden);
    for t in 0..n_tokens {
        // gate_out[i] = sum_j gate_shexp[i, j] * x[t, j]
        // up_out[i]   = sum_j up_shexp[i, j] * x[t, j]
        for i in 0..hidden {
            let mut g = 0.0_f32;
            let mut u = 0.0_f32;
            for j in 0..n_embd {
                let xj = x[[t, j]];
                g += w.gate_shexp[[i, j]] * xj;
                u += w.up_shexp[[i, j]] * xj;
            }
            gate_out[i] = g;
            up_out[i] = u;
        }
        // act = swiglu(gate_out, up_out)
        let activated = clamped_swiglu_row(gate_out.view(), up_out.view(), swiglu_limit);
        // out[t, d] = sum_i down_shexp[d, i] * act[i]
        for d in 0..n_embd {
            let mut acc = 0.0_f32;
            for i in 0..hidden {
                acc += w.down_shexp[[d, i]] * activated[i];
            }
            out[[t, d]] = acc;
        }
    }
    out
}

/// All weights for the DSv4 FFN block (pre-norm + routed + shared).
#[derive(Clone, Copy)]
pub struct Dsv4FfnWeights<'a> {
    /// `(n_embd,)` — RMSNorm gain.
    pub ffn_norm: &'a [f32],
    /// `(n_expert, n_embd)` — routing gate projection.
    pub gate_inp: ArrayView2<'a, f32>,
    /// Optional `(n_expert,)` per-expert selection bias.
    pub exp_probs_b: Option<ArrayView1<'a, f32>>,
    /// Optional `(n_vocab, n_expert_used)` hash-routing table. When
    /// `Some(_)` AND `token_ids` is provided to [`dsv4_ffn_block`],
    /// the hash routing path runs instead of sqrt-softplus.
    pub gate_tid2eid: Option<ArrayView2<'a, i32>>,
    /// Routed expert weights.
    pub moe: MoeExpertWeights<'a>,
    /// Shared expert weights.
    pub shared: SharedExpertWeights<'a>,
}

/// FFN-block scalar config.
#[derive(Clone, Copy, Debug)]
pub struct Dsv4FfnParams {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub norm_eps: f32,
    /// SwiGLU clamp magnitude for the **routed** experts. Pass `0.0`
    /// to disable clamping (plain SwiGLU).
    pub routed_swiglu_limit: f32,
    /// SwiGLU clamp magnitude for the **shared** expert. DSv4 spec
    /// uses plain SwiGLU (no clamp) — pass `0.0`.
    pub shared_swiglu_limit: f32,
    /// Normalize per-token routing weights to sum to 1.
    pub expert_weights_norm: bool,
    /// Post-normalization scalar applied to routing weights.
    pub expert_weights_scale: f32,
}

/// Run the full DSv4 FFN block. `token_ids` is required only when
/// `weights.gate_tid2eid` is `Some(_)` (hash-routing layer).
pub fn dsv4_ffn_block(
    x: ArrayView2<f32>,
    w: &Dsv4FfnWeights,
    p: &Dsv4FfnParams,
    token_ids: Option<&[u32]>,
) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    let n_embd = x.shape()[1];

    // 1. Pre-norm.
    let cur = rms_norm_2d(x, w.ffn_norm, p.norm_eps);

    // 2. Routing.
    let routing: RoutingResult = if let Some(table) = w.gate_tid2eid {
        let tids = token_ids.expect("dsv4_ffn_block: gate_tid2eid present but token_ids is None");
        assert_eq!(tids.len(), n_tokens, "token_ids length must equal n_tokens");
        hash_route_topk(tids, table, p.n_expert_used)
    } else {
        compute_router_topk(
            cur.view(),
            w.gate_inp,
            w.exp_probs_b,
            p.n_expert,
            p.n_expert_used,
            p.expert_weights_norm,
            p.expert_weights_scale,
        )
    };

    // 3. Routed MoE dispatch.
    let moe_out = dsv4_moe_dispatch(cur.view(), &routing, &w.moe, p.routed_swiglu_limit);

    // 4. Shared expert FFN.
    let shared_out = dsv4_shared_expert_ffn(cur.view(), &w.shared, p.shared_swiglu_limit);

    // 5. Sum.
    let mut out = Array2::<f32>::zeros((n_tokens, n_embd));
    for t in 0..n_tokens {
        for d in 0..n_embd {
            out[[t, d]] = moe_out[[t, d]] + shared_out[[t, d]];
        }
    }
    out
}

/// RMSNorm along axis 1 (per-row) with a learned per-feature gain weight.
fn rms_norm_2d(x: ArrayView2<f32>, weight: &[f32], eps: f32) -> Array2<f32> {
    let n_rows = x.shape()[0];
    let n_dims = x.shape()[1];
    assert_eq!(weight.len(), n_dims);
    let mut out = Array2::<f32>::zeros((n_rows, n_dims));
    for t in 0..n_rows {
        let mut sumsq = 0.0_f32;
        for d in 0..n_dims {
            let v = x[[t, d]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n_dims as f32 + eps).sqrt();
        for d in 0..n_dims {
            out[[t, d]] = x[[t, d]] * inv * weight[d];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    /// Shared-expert FFN: with identity-ish weights it produces finite
    /// non-zero output and matches a hand-rolled SwiGLU compute.
    #[test]
    fn shared_expert_matches_direct_compute() {
        let n_tokens = 2;
        let n_embd = 4;
        let hidden = 6;
        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| (t + d) as f32 * 0.1);
        let gate_shexp =
            Array2::<f32>::from_shape_fn((hidden, n_embd), |(i, j)| ((i + j) as f32 * 0.03).sin());
        let up_shexp =
            Array2::<f32>::from_shape_fn((hidden, n_embd), |(i, j)| ((i + j) as f32 * 0.05).cos());
        let down_shexp =
            Array2::<f32>::from_shape_fn((n_embd, hidden), |(d, i)| ((d + i) as f32 * 0.07).sin());
        let w = SharedExpertWeights {
            gate_shexp: gate_shexp.view(),
            up_shexp: up_shexp.view(),
            down_shexp: down_shexp.view(),
        };
        let out = dsv4_shared_expert_ffn(x.view(), &w, 0.0);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));

        // Direct compute for t=0.
        let mut gate_vec = vec![0.0_f32; hidden];
        let mut up_vec = vec![0.0_f32; hidden];
        for i in 0..hidden {
            for j in 0..n_embd {
                gate_vec[i] += gate_shexp[[i, j]] * x[[0, j]];
                up_vec[i] += up_shexp[[i, j]] * x[[0, j]];
            }
        }
        let act = clamped_swiglu_row(
            ndarray::ArrayView1::from(&gate_vec[..]),
            ndarray::ArrayView1::from(&up_vec[..]),
            0.0,
        );
        for d in 0..n_embd {
            let mut acc = 0.0_f32;
            for i in 0..hidden {
                acc += down_shexp[[d, i]] * act[i];
            }
            assert!(
                (out[[0, d]] - acc).abs() < 1e-5,
                "d={d}: dispatch={} direct={acc}",
                out[[0, d]]
            );
        }
    }

    /// Full FFN block with regular routing: pre-norm → routed + shared →
    /// finite non-zero output.
    #[test]
    fn ffn_block_regular_routing_smoke() {
        let n_tokens = 4;
        let n_embd = 16;
        let n_expert = 6;
        let n_expert_used = 2;
        let n_ff_exp = 8;
        let hidden_shared = 4; // n_ff_exp * n_expert_shared, picked small

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 7 + d) as f32 * 0.013).sin()
        });
        let ffn_norm = vec![1.0_f32; n_embd];
        let gate_inp = Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| {
            ((e + d) as f32 * 0.01).cos() * 0.1
        });
        let exp_probs_b = Array1::<f32>::zeros(n_expert);
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

        let w = Dsv4FfnWeights {
            ffn_norm: &ffn_norm,
            gate_inp: gate_inp.view(),
            exp_probs_b: Some(exp_probs_b.view()),
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
        };
        let p = Dsv4FfnParams {
            n_expert,
            n_expert_used,
            norm_eps: 1e-5,
            routed_swiglu_limit: 7.0,
            shared_swiglu_limit: 0.0,
            expert_weights_norm: true,
            expert_weights_scale: 1.0,
        };
        let out = dsv4_ffn_block(x.view(), &w, &p, None);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 1e-4, "output collapsed (sum={total})");
    }

    /// Hash routing path: passing a gate_tid2eid table + token_ids
    /// routes through the hash path instead of sqrt-softplus.
    #[test]
    fn ffn_block_hash_routing_smoke() {
        let n_tokens = 3;
        let n_embd = 8;
        let n_expert = 4;
        let n_expert_used = 2;
        let n_ff_exp = 4;
        let hidden_shared = 4;
        let n_vocab = 16;

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t + d) as f32 * 0.05).sin()
        });
        let ffn_norm = vec![1.0_f32; n_embd];
        // gate_inp shouldn't be consulted in the hash path — fill with
        // garbage to detect accidental fallthrough.
        let gate_inp = Array2::<f32>::from_elem((n_expert, n_embd), f32::NAN);
        let gate_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.01).sin() * 0.1
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
        let tid2eid = Array2::<i32>::from_shape_fn((n_vocab, n_expert_used), |(t, k)| {
            ((t + k) as i32) % n_expert as i32
        });

        let w = Dsv4FfnWeights {
            ffn_norm: &ffn_norm,
            gate_inp: gate_inp.view(),
            exp_probs_b: None,
            gate_tid2eid: Some(tid2eid.view()),
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
        };
        let p = Dsv4FfnParams {
            n_expert,
            n_expert_used,
            norm_eps: 1e-5,
            routed_swiglu_limit: 0.0,
            shared_swiglu_limit: 0.0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
        };
        let token_ids = vec![5u32, 0, 11];
        let out = dsv4_ffn_block(x.view(), &w, &p, Some(&token_ids));
        // gate_inp was NaN; hash path must not consult it. Output finite
        // proves we didn't fall through to sqrt-softplus.
        assert!(
            out.iter().all(|v| v.is_finite()),
            "NaN gate_inp leaked into hash-routed output"
        );
    }
}
