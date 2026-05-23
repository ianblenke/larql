//! DSv4 MoE expert dispatch — weighted sum of clamped-SwiGLU experts.
//!
//! Takes the routing decisions from Stage 8h-3a and runs the selected
//! experts per token. For each `(token, expert)` pair in the routing:
//!
//! ```text
//! gate_out[i] = sum_j gate_exps[e, i, j] * x[t, j]
//! up_out[i]   = sum_j up_exps  [e, i, j] * x[t, j]
//! activated   = clamped_swiglu(gate_out, up_out, limit)
//! expert_out[d] = sum_i down_exps[e, d, i] * activated[i]
//! out[t, d]  += weight * expert_out[d]
//! ```
//!
//! Reference: llama.cpp `build_moe_ffn`. Expert weight layout follows
//! the DSv4 GGUF convention (`{n_embd, n_ff_exp, n_expert}` for gate/up
//! and `{n_ff_exp, n_embd, n_expert}` for down, ggml-order), which
//! transposes in our ndarray C-order to `(n_expert, n_ff_exp, n_embd)`
//! for gate/up and `(n_expert, n_embd, n_ff_exp)` for down.

use ndarray::{Array2, ArrayView2, ArrayView3};

use super::dsv4_moe_ops::clamped_swiglu_row;
use super::dsv4_moe_routing::RoutingResult;

/// Per-layer MoE expert weight views.
#[derive(Clone, Copy)]
pub struct MoeExpertWeights<'a> {
    /// `(n_expert, n_ff_exp, n_embd)`
    pub gate_exps: ArrayView3<'a, f32>,
    /// `(n_expert, n_ff_exp, n_embd)`
    pub up_exps: ArrayView3<'a, f32>,
    /// `(n_expert, n_embd, n_ff_exp)`
    pub down_exps: ArrayView3<'a, f32>,
}

/// Run weighted MoE expert dispatch.
///
/// - `x`: `(n_tokens, n_embd)` — pre-norm residual.
/// - `routing`: per-token expert indices + weights from Stage 8h-3a.
/// - `w`: expert weight views.
/// - `swiglu_limit`: clamp magnitude for the SwiGLU activation.
///
/// Output: `(n_tokens, n_embd)`.
pub fn dsv4_moe_dispatch(
    x: ArrayView2<f32>,
    routing: &RoutingResult,
    w: &MoeExpertWeights,
    swiglu_limit: f32,
) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    let n_embd = x.shape()[1];
    let n_expert = w.gate_exps.shape()[0];
    let n_ff_exp = w.gate_exps.shape()[1];

    assert_eq!(
        w.gate_exps.shape(),
        &[n_expert, n_ff_exp, n_embd],
        "gate_exps shape"
    );
    assert_eq!(
        w.up_exps.shape(),
        &[n_expert, n_ff_exp, n_embd],
        "up_exps shape"
    );
    assert_eq!(
        w.down_exps.shape(),
        &[n_expert, n_embd, n_ff_exp],
        "down_exps shape"
    );
    let n_expert_used = routing.indices.shape()[1];
    assert_eq!(
        routing.indices.shape(),
        &[n_tokens, n_expert_used],
        "routing.indices shape"
    );
    assert_eq!(
        routing.weights.shape(),
        &[n_tokens, n_expert_used],
        "routing.weights shape"
    );

    let mut out = Array2::<f32>::zeros((n_tokens, n_embd));
    // Reused buffers for per-token compute.
    let mut gate_out = ndarray::Array1::<f32>::zeros(n_ff_exp);
    let mut up_out = ndarray::Array1::<f32>::zeros(n_ff_exp);
    for t in 0..n_tokens {
        for k in 0..n_expert_used {
            let e = routing.indices[[t, k]];
            assert!(
                e < n_expert,
                "routing index {e} out of bounds (n_expert={n_expert}) at t={t} k={k}"
            );
            let weight = routing.weights[[t, k]];
            if weight == 0.0 {
                continue;
            }

            // gate_out[i] = sum_j gate_exps[e, i, j] * x[t, j]
            // up_out[i]   = sum_j up_exps[e, i, j] * x[t, j]
            for i in 0..n_ff_exp {
                let mut g = 0.0_f32;
                let mut u = 0.0_f32;
                for j in 0..n_embd {
                    let xj = x[[t, j]];
                    g += w.gate_exps[[e, i, j]] * xj;
                    u += w.up_exps[[e, i, j]] * xj;
                }
                gate_out[i] = g;
                up_out[i] = u;
            }

            // activated = clamped_swiglu(gate_out, up_out, limit)
            let activated = clamped_swiglu_row(gate_out.view(), up_out.view(), swiglu_limit);

            // expert_out[d] = sum_i down_exps[e, d, i] * activated[i]
            // out[t, d] += weight * expert_out[d]
            for d in 0..n_embd {
                let mut acc = 0.0_f32;
                for i in 0..n_ff_exp {
                    acc += w.down_exps[[e, d, i]] * activated[i];
                }
                out[[t, d]] += weight * acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_moe_routing::compute_router_topk;
    use super::*;
    use ndarray::{Array2, Array3};

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Single expert (`n_expert_used=1`, weight=1) on a fixed token
    /// matches the direct gate→swiglu→down compute.
    #[test]
    fn single_expert_unit_weight_matches_direct_compute() {
        let n_expert = 3;
        let n_ff_exp = 4;
        let n_embd = 5;
        let n_tokens = 1;
        let n_expert_used = 1;
        let swiglu_limit = 7.0;

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(_, d)| (d as f32) * 0.1);
        let gate_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e * 100 + i * 10 + j) as f32 * 0.01).sin()
        });
        let up_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e * 100 + i * 10 + j) as f32 * 0.013).cos()
        });
        let down_exps = Array3::<f32>::from_shape_fn((n_expert, n_embd, n_ff_exp), |(e, d, i)| {
            ((e * 100 + d * 10 + i) as f32 * 0.011).sin()
        });

        // Hand-route to expert 2 with weight 1.0.
        let mut indices = ndarray::Array2::<usize>::zeros((n_tokens, n_expert_used));
        indices[[0, 0]] = 2;
        let weights = Array2::<f32>::from_elem((n_tokens, n_expert_used), 1.0);
        let routing = RoutingResult { indices, weights };

        let w = MoeExpertWeights {
            gate_exps: gate_exps.view(),
            up_exps: up_exps.view(),
            down_exps: down_exps.view(),
        };
        let out = dsv4_moe_dispatch(x.view(), &routing, &w, swiglu_limit);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);

        // Direct compute for expert 2.
        let e = 2;
        let mut gate_out = vec![0.0_f32; n_ff_exp];
        let mut up_out = vec![0.0_f32; n_ff_exp];
        for i in 0..n_ff_exp {
            for j in 0..n_embd {
                gate_out[i] += gate_exps[[e, i, j]] * x[[0, j]];
                up_out[i] += up_exps[[e, i, j]] * x[[0, j]];
            }
        }
        let activated = clamped_swiglu_row(
            ndarray::ArrayView1::from(&gate_out[..]),
            ndarray::ArrayView1::from(&up_out[..]),
            swiglu_limit,
        );
        for d in 0..n_embd {
            let mut acc = 0.0_f32;
            for i in 0..n_ff_exp {
                acc += down_exps[[e, d, i]] * activated[i];
            }
            assert!(
                approx_eq(out[[0, d]], acc, 1e-5),
                "d={d}: dispatch={} direct={acc}",
                out[[0, d]]
            );
        }
    }

    /// Two experts with weight=0.5 each — output equals the average of
    /// each expert's solo contribution (linearity of the weighted sum).
    #[test]
    fn weighted_sum_linearity() {
        let n_expert = 4;
        let n_ff_exp = 3;
        let n_embd = 4;
        let n_tokens = 1;
        let n_expert_used = 2;
        let swiglu_limit = 7.0;

        let x = Array2::<f32>::from_elem((n_tokens, n_embd), 0.3);
        let gate_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.07).sin()
        });
        let up_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.05).cos()
        });
        let down_exps = Array3::<f32>::from_shape_fn((n_expert, n_embd, n_ff_exp), |(e, d, i)| {
            ((e + d + i) as f32 * 0.09).sin()
        });
        let w = MoeExpertWeights {
            gate_exps: gate_exps.view(),
            up_exps: up_exps.view(),
            down_exps: down_exps.view(),
        };

        // Solo dispatch through expert 1 (weight 1.0).
        let solo1 = {
            let mut indices = ndarray::Array2::<usize>::zeros((n_tokens, 1));
            indices[[0, 0]] = 1;
            let weights = Array2::<f32>::from_elem((n_tokens, 1), 1.0);
            dsv4_moe_dispatch(
                x.view(),
                &RoutingResult { indices, weights },
                &w,
                swiglu_limit,
            )
        };
        // Solo dispatch through expert 3 (weight 1.0).
        let solo3 = {
            let mut indices = ndarray::Array2::<usize>::zeros((n_tokens, 1));
            indices[[0, 0]] = 3;
            let weights = Array2::<f32>::from_elem((n_tokens, 1), 1.0);
            dsv4_moe_dispatch(
                x.view(),
                &RoutingResult { indices, weights },
                &w,
                swiglu_limit,
            )
        };
        // Both, each weight 0.5.
        let combined = {
            let mut indices = ndarray::Array2::<usize>::zeros((n_tokens, 2));
            indices[[0, 0]] = 1;
            indices[[0, 1]] = 3;
            let weights = Array2::<f32>::from_elem((n_tokens, 2), 0.5);
            dsv4_moe_dispatch(
                x.view(),
                &RoutingResult { indices, weights },
                &w,
                swiglu_limit,
            )
        };

        for d in 0..n_embd {
            let want = 0.5 * solo1[[0, d]] + 0.5 * solo3[[0, d]];
            assert!(
                approx_eq(combined[[0, d]], want, 1e-5),
                "d={d}: combined={} want={want}",
                combined[[0, d]]
            );
        }
    }

    /// Zero weights → zero contribution: setting all routing weights
    /// to 0 produces all-zero output (no compute happens).
    #[test]
    fn zero_weights_produces_zero_output() {
        let n_expert = 3;
        let n_ff_exp = 2;
        let n_embd = 3;
        let n_tokens = 2;
        let n_expert_used = 2;

        let x = Array2::<f32>::from_elem((n_tokens, n_embd), 1.0);
        let gate_exps = Array3::<f32>::from_elem((n_expert, n_ff_exp, n_embd), 1.0);
        let up_exps = Array3::<f32>::from_elem((n_expert, n_ff_exp, n_embd), 1.0);
        let down_exps = Array3::<f32>::from_elem((n_expert, n_embd, n_ff_exp), 1.0);
        let routing = RoutingResult {
            indices: ndarray::Array2::<usize>::zeros((n_tokens, n_expert_used)),
            weights: Array2::<f32>::zeros((n_tokens, n_expert_used)),
        };
        let w = MoeExpertWeights {
            gate_exps: gate_exps.view(),
            up_exps: up_exps.view(),
            down_exps: down_exps.view(),
        };
        let out = dsv4_moe_dispatch(x.view(), &routing, &w, 7.0);
        for v in out.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    /// Composition with the Stage 8h-3a router: route → dispatch
    /// produces finite non-zero output on a small synthetic MoE.
    #[test]
    fn route_then_dispatch_finite_nonzero() {
        let n_tokens = 4;
        let n_embd = 8;
        let n_expert = 6;
        let n_ff_exp = 4;
        let n_expert_used = 2;

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 7 + d) as f32 * 0.013).sin()
        });
        let gate_inp = Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| {
            ((e + d) as f32 * 0.01).cos() * 0.1
        });
        let routing = compute_router_topk(
            x.view(),
            gate_inp.view(),
            None,
            n_expert,
            n_expert_used,
            true,
            1.0,
        );

        let gate_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.011).sin() * 0.1
        });
        let up_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.013).cos() * 0.1
        });
        let down_exps = Array3::<f32>::from_shape_fn((n_expert, n_embd, n_ff_exp), |(e, d, i)| {
            ((e + d + i) as f32 * 0.009).sin() * 0.1
        });
        let w = MoeExpertWeights {
            gate_exps: gate_exps.view(),
            up_exps: up_exps.view(),
            down_exps: down_exps.view(),
        };
        let out = dsv4_moe_dispatch(x.view(), &routing, &w, 7.0);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        // Synthetic weights are intentionally tiny (~1e-1 scale → tiny
        // product after gate→swiglu→down). 1e-4 is a safe non-zero floor.
        assert!(total > 1e-4, "output collapsed (sum={total})");
    }
}
