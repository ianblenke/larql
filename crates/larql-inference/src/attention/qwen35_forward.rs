//! End-to-end `qwen35_forward` glue (Phase C.4c of
//! `inference-qwen35-deltanet`).
//!
//! Single-token forward through the full Qwen 3.6 hybrid model:
//! embed → 64 layers (each is a hybrid linear / full-attn block with
//! a Gemma-2-style pre+post-norm sandwich and a SwiGLU FFN
//! suffix) → final_norm → lm_head → logits.
//!
//! Per design.md §6: the **FFN residual adds to the pre-post-norm
//! tensor**. So the layer flow is:
//!
//! ```text
//! attn_out = block(x)              # block does its own pre-norm
//! residual = x + attn_out          # residual add 1
//! ffn_in   = attn_post_norm(residual)  # pre-FFN norm
//! ffn_out  = swiglu(ffn_in)        # silu(gate @ x) * (up @ x) → down
//! final    = residual + ffn_out    # residual add 2 (to PRE-post-norm)
//! ```
//!
//! This is the load-bearing structural difference vs vanilla LLaMA
//! pre-norm-only layers; the FFN residual on the post-normed tensor
//! is a common bug per llama.cpp `qwen35.cpp:141–152`.
//!
//! Weight loading from a vindex / GGUF lives in Phase C.4d; this
//! module only provides the borrow-based forward function. Tests
//! exercise it on tiny synthetic 2-layer models.

use ndarray::{ArcArray2, Array1, Array2};
use std::sync::Arc;

use super::deltanet_block::{rms_norm_1d_pub, DeltaNetDims};
use super::deltanet_state::sigmoid;
use super::qwen35_block::{
    hybrid_layer_step, DeltaNetHybridCache, Qwen35AttentionDims, Qwen35LayerWeights,
};

/// Per-layer MoE FFN weights for `Qwen35MoeArch` (e.g.
/// Qwen3.6-35B-A3B). When a `Qwen35FullLayerWeights` has `moe =
/// Some(...)`, the forward step routes through `swiglu_moe_lazy`
/// instead of the dense `swiglu_ffn_lazy`.
///
/// Tensor layouts (matching the GGUF probe from F.0):
///   - `router`        f32, `[num_experts, hidden]`
///   - `gate_exps`     Q4_K, `[num_experts * expert_ffn_dim, hidden]`
///   - `up_exps`       Q4_K, `[num_experts * expert_ffn_dim, hidden]`
///   - `down_exps`     Q5_K, `[num_experts * hidden, expert_ffn_dim]`
///   - shexp_*         optional shared expert (always-on). Same
///                     layouts as a dense SwiGLU but with `expert_ffn_dim`
///                     (Qwen3.6-35B-A3B has one).
#[derive(Clone)]
pub struct Qwen35MoeFfnWeights {
    pub router: larql_models::quant::lazy::QuantTensor,
    pub gate_exps: larql_models::quant::lazy::QuantTensor,
    pub up_exps: larql_models::quant::lazy::QuantTensor,
    pub down_exps: larql_models::quant::lazy::QuantTensor,
    pub shexp_gate: Option<larql_models::quant::lazy::QuantTensor>,
    pub shexp_up: Option<larql_models::quant::lazy::QuantTensor>,
    pub shexp_down: Option<larql_models::quant::lazy::QuantTensor>,
    pub num_experts: usize,
    pub top_k: usize,
}

/// Full weights for one layer (both kinds): the block weights
/// (linear or attention) PLUS the post-attention norm + SwiGLU FFN.
#[derive(Clone)]
pub struct Qwen35FullLayerWeights {
    /// The block-level weights — DeltaNet for linear layers,
    /// attention for full-attn layers. Use `Qwen35LayerWeights::Linear(...)`
    /// or `::Attention(...)` to construct.
    pub block: Qwen35LayerWeights,
    /// Post-attention RMSNorm weight `[hidden]`. Applied to
    /// (x + block_out) and fed into the FFN as its pre-norm. The
    /// FFN residual still uses the pre-post-norm tensor.
    pub attn_post_norm: Arc<[f32]>,
    /// SwiGLU gate projection `[ffn_dim, hidden]`.
    pub ffn_gate: ArcArray2<f32>,
    /// SwiGLU up projection `[ffn_dim, hidden]`.
    pub ffn_up: ArcArray2<f32>,
    /// SwiGLU down projection `[hidden, ffn_dim]`.
    pub ffn_down: ArcArray2<f32>,
    /// Optional lazy-quantised gate / up / down. When `Some`, the
    /// matvec dispatches through `QuantTensor::matvec` and the
    /// corresponding dense `ffn_*` field is unused (typically a 0×0
    /// placeholder). Populated by `load_qwen35_weights_lazy_ffn`.
    pub ffn_gate_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub ffn_up_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub ffn_down_quant: Option<larql_models::quant::lazy::QuantTensor>,
    /// MoE FFN. When `Some(_)`, the forward step dispatches through
    /// `swiglu_moe_lazy` and the dense `ffn_*` / `ffn_*_quant` fields
    /// are unused. Populated by `load_qwen35_moe_weights` for
    /// `Qwen35MoeArch` GGUFs.
    pub moe: Option<Qwen35MoeFfnWeights>,
}

/// Full Qwen 3.6 model weights — embed, every layer's weights, and
/// the output head.
#[derive(Clone)]
pub struct Qwen35Weights {
    /// Token embedding matrix `[vocab, hidden]`. The lookup uses
    /// `embed.row(token_id)`. May be a 0×0 placeholder when
    /// `embed_quant` is populated and the caller does row-lookup
    /// via `QuantTensor::row_to_f32`.
    pub embed: ArcArray2<f32>,
    /// Lazy-quantised embed (Q4_K typical). When `Some`, the
    /// forward uses `embed_quant.row_to_f32(token_id)` instead of
    /// the dense `embed` field. Saves ~4 GiB on Qwen3.6-27B.
    pub embed_quant: Option<larql_models::quant::lazy::QuantTensor>,
    /// Per-layer full weights, indexed 0..n_layer.
    pub layers: Vec<Qwen35FullLayerWeights>,
    /// Final RMSNorm weight `[hidden]`.
    pub final_norm: Arc<[f32]>,
    /// LM head projection `[vocab, hidden]`. Often tied to `embed`
    /// (the caller decides; this struct just holds whichever).
    /// May be a 0×0 placeholder when `lm_head_quant` is populated and
    /// the caller is using the lazy-quantised matvec path.
    pub lm_head: ArcArray2<f32>,
    /// Lazy-quantised LM head, if loaded via
    /// `larql_models::load_gguf_lazy_lm_head`. When `Some`, the
    /// forward dispatches the final logits matvec through
    /// `QuantTensor::matvec` and `lm_head` is unused.
    pub lm_head_quant: Option<larql_models::quant::lazy::QuantTensor>,
    /// FFN intermediate dim (for shape assertions in tests).
    pub ffn_dim: usize,
    /// Optional GPU (or other) quant-matvec backend. Phase E.1
    /// of `qwen35-gpu-forward`: when set, lazy-quant tensors
    /// route through `backend.quant_matvec(...)` before falling
    /// back to the CPU rayon path. None keeps the all-CPU
    /// behaviour shipped in Phase 2a-2d.
    pub backend: Option<std::sync::Arc<dyn larql_compute::ComputeBackend>>,
}

/// One-token forward through the entire Qwen 3.6 model.
///
/// Returns the logits `[vocab]` for the next token after the input.
/// Mutates `hybrid_cache` (appends K/V or updates DeltaNet state per
/// layer) and advances `hybrid_cache.next_position`.
pub fn qwen35_forward_step(
    token_id: u32,
    weights: &Qwen35Weights,
    dn_dims: &DeltaNetDims,
    attn_dims: &Qwen35AttentionDims,
    hybrid_cache: &mut DeltaNetHybridCache,
    eps: f32,
) -> Array1<f32> {
    let n_layers = weights.layers.len();
    debug_assert_eq!(n_layers, hybrid_cache.num_layers());

    // Per-session token counter for the LARQL_QWEN35_DUMP_LAYER_BOUNDARY
    // diagnostic. Reset by setting `LARQL_QWEN35_DUMP_TOKEN_RESET=1`
    // before the first forward call.
    if std::env::var("LARQL_QWEN35_DUMP_LAYER_BOUNDARY").is_ok() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static TOK_COUNTER: AtomicUsize = AtomicUsize::new(0);
        if std::env::var("LARQL_QWEN35_DUMP_TOKEN_RESET").is_ok() {
            TOK_COUNTER.store(0, Ordering::Relaxed);
            std::env::remove_var("LARQL_QWEN35_DUMP_TOKEN_RESET");
        }
        let tok = TOK_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::set_var("LARQL_QWEN35_DUMP_TOKEN_TAG", format!("tok{tok}"));
    }

    use crate::attention::fine_profile::{self, *};
    use crate::time_section;

    // 1. Embed lookup. Lazy-quant path takes precedence when set;
    //    falls back to dense `embed.row(token_id)` otherwise.
    let mut x: Array1<f32> = time_section!(
        EMBED,
        if let Some(qt) = weights.embed_quant.as_ref() {
            let [vocab, hidden] = qt.shape();
            debug_assert!((token_id as usize) < vocab);
            debug_assert_eq!(hidden, dn_dims.hidden);
            qt.row_to_f32(token_id as usize)
                .expect("embed_quant row_to_f32")
        } else {
            let vocab = weights.embed.shape()[0];
            let hidden = weights.embed.shape()[1];
            debug_assert!((token_id as usize) < vocab);
            debug_assert_eq!(hidden, dn_dims.hidden);
            weights.embed.row(token_id as usize).to_owned()
        }
    );

    // Optional per-layer trace, enabled via `LARQL_QWEN35_TRACE=1`.
    // Prints residual stream stats to stderr at each layer so a
    // single forward run can be bisected for "where does the
    // stream blow up / collapse." Cheap when disabled (the env
    // lookup is a thread-local cache hit) so leaving it in the
    // production forward is fine.
    let trace = std::env::var("LARQL_QWEN35_TRACE").is_ok();
    if trace {
        eprintln!(
            "    embed:                  l2={:.3} max_abs={:.3}",
            l2(&x),
            max_abs(&x),
        );
    }

    // 2. Hybrid layer stack.
    for layer in 0..n_layers {
        let layer_w = &weights.layers[layer];
        // 2a. Block forward (linear or full-attn). The block does
        // its own pre-norm internally (`attn_norm`).
        let backend = weights.backend.as_deref();
        // Linear-block timing is attributed by the per-section
        // accumulators inside `deltanet_block_step`; full-attn time
        // lands in ATTN_BLOCK so the totals reflect the actual block
        // kind. Match on the discriminant cheaply.
        let is_full_attn = matches!(
            &layer_w.block,
            crate::attention::qwen35_block::Qwen35LayerWeights::Attention(_)
        );
        let block_out = if is_full_attn {
            time_section!(
                ATTN_BLOCK,
                hybrid_layer_step(
                    layer,
                    &x,
                    &layer_w.block,
                    dn_dims,
                    attn_dims,
                    hybrid_cache,
                    backend,
                    hybrid_cache.next_position,
                )
            )
        } else {
            hybrid_layer_step(
                layer,
                &x,
                &layer_w.block,
                dn_dims,
                attn_dims,
                hybrid_cache,
                backend,
                hybrid_cache.next_position,
            )
        };
        // 2b. Residual add 1 — `residual = x + block_out`.
        let residual: Array1<f32> = time_section!(RESIDUAL_ADDS, &x + &block_out);
        // 2c. Post-attention RMSNorm (only on `has_post_norms`
        // architectures — Qwen 3.6 is one of them).
        let ffn_in = time_section!(
            FFN_RMS_NORM,
            rms_norm_1d_pub(&residual, &layer_w.attn_post_norm, eps)
        );
        // 2d. SwiGLU FFN. MoE arch takes precedence; then the
        //     lazy-quant dense path; then dense ndarray dot fallback.
        let ffn_out = if let Some(moe) = layer_w.moe.as_ref() {
            swiglu_moe_lazy(
                &ffn_in,
                &moe.router,
                &moe.gate_exps,
                &moe.up_exps,
                &moe.down_exps,
                moe.shexp_gate.as_ref(),
                moe.shexp_up.as_ref(),
                moe.shexp_down.as_ref(),
                moe.num_experts,
                moe.top_k,
                weights.backend.as_deref(),
            )
        } else if layer_w.ffn_gate_quant.is_some()
            || layer_w.ffn_up_quant.is_some()
            || layer_w.ffn_down_quant.is_some()
        {
            swiglu_ffn_lazy(
                &ffn_in,
                layer_w.ffn_gate_quant.as_ref(),
                layer_w.ffn_up_quant.as_ref(),
                layer_w.ffn_down_quant.as_ref(),
                &layer_w.ffn_gate,
                &layer_w.ffn_up,
                &layer_w.ffn_down,
                weights.backend.as_deref(),
            )
        } else {
            swiglu_ffn(
                &ffn_in,
                layer_w.ffn_gate.view(),
                layer_w.ffn_up.view(),
                layer_w.ffn_down.view(),
            )
        };
        // 2e. Residual add 2 — `x = residual + ffn_out` (NOT
        // `ffn_in + ffn_out`; the FFN residual bypasses the
        // post-norm per design.md §6).
        x = time_section!(RESIDUAL_ADDS, &residual + &ffn_out);

        // Optional per-layer binary tensor dump for elementwise parity
        // diff vs llama-eval-callback (`LLAMA_DUMP_BIN_DIR` on llama.cpp
        // side). Triggered by `LARQL_QWEN35_DUMP_LAYER_BOUNDARY=<dir>` —
        // dumps `block_out_lN.bin`, `attn_residual_lN.bin`,
        // `ffn_in_lN.bin`, `ffn_out_lN.bin`, `x_after_lN.bin` for every
        // layer. Each call is per-token; the caller is responsible for
        // dispatching the right token (or pre-pending the token-id).
        if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_LAYER_BOUNDARY") {
            let _ = std::fs::create_dir_all(&dir);
            let write = |name: &str, data: &[f32]| {
                use std::io::Write;
                let path = std::path::Path::new(&dir).join(name);
                if let Ok(mut f) = std::fs::File::create(&path) {
                    let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
                    for n in ne {
                        f.write_all(&n.to_le_bytes()).ok();
                    }
                    for v in data {
                        f.write_all(&v.to_le_bytes()).ok();
                    }
                }
            };
            // Use token-prefixed dir so successive prompt tokens don't
            // overwrite each other. The harness sets an outer env that
            // names the token index, e.g. `dir/tok0/`, `dir/tok1/`, ...
            let token_subdir =
                std::env::var("LARQL_QWEN35_DUMP_TOKEN_TAG").unwrap_or_else(|_| "tok".to_string());
            let layer_dir = format!("{}/{}", dir, token_subdir);
            let _ = std::fs::create_dir_all(&layer_dir);
            let path_for = |name: &str| format!("{}/{}", layer_dir, name);
            let write_at = |name: &str, data: &[f32]| {
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create(path_for(name)) {
                    let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
                    for n in ne {
                        f.write_all(&n.to_le_bytes()).ok();
                    }
                    for v in data {
                        f.write_all(&v.to_le_bytes()).ok();
                    }
                }
            };
            let _ = write; // suppress unused; we now use write_at
            write_at(
                &format!("block_out_l{layer:02}.bin"),
                block_out.as_slice().unwrap_or(&[]),
            );
            write_at(
                &format!("residual_l{layer:02}.bin"),
                residual.as_slice().unwrap_or(&[]),
            );
            write_at(
                &format!("ffn_out_l{layer:02}.bin"),
                ffn_out.as_slice().unwrap_or(&[]),
            );
            write_at(
                &format!("x_after_l{layer:02}.bin"),
                x.as_slice().unwrap_or(&[]),
            );
        }

        if trace {
            let kind = matches!(
                layer_w.block,
                crate::attention::qwen35_block::Qwen35LayerWeights::Linear(_)
            );
            eprintln!(
                "  L{layer:02} {} block_l2={:.3} block_max={:.3}  ffn_l2={:.3} ffn_max={:.3}  x_l2={:.3} x_max={:.3}",
                if kind { "lin" } else { "atn" },
                l2(&block_out),
                max_abs(&block_out),
                l2(&ffn_out),
                max_abs(&ffn_out),
                l2(&x),
                max_abs(&x),
            );
        }
    }

    // 3. Final norm + lm_head.
    let x_final = time_section!(FINAL_NORM, rms_norm_1d_pub(&x, &weights.final_norm, eps));
    // Optional final-norm + logits dump for elementwise parity diff
    // vs llama.cpp's `result_norm.bin` / `result_output.bin`. Triggered
    // by `LARQL_QWEN35_DUMP_FINAL_BIN_DIR=<dir>` (single-shot, dumps
    // each call but the harness will only call once for the last
    // prompt token). Caller is responsible for invoking at the right
    // step.
    if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_FINAL_BIN_DIR") {
        let _ = std::fs::create_dir_all(&dir);
        let tag =
            std::env::var("LARQL_QWEN35_DUMP_TOKEN_TAG").unwrap_or_else(|_| "tok".to_string());
        let write = |name: &str, data: &[f32]| {
            use std::io::Write;
            let path = std::path::Path::new(&dir).join(format!("{tag}_{name}"));
            if let Ok(mut f) = std::fs::File::create(&path) {
                let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
                for n in ne {
                    f.write_all(&n.to_le_bytes()).ok();
                }
                for v in data {
                    f.write_all(&v.to_le_bytes()).ok();
                }
            }
        };
        write("x_final.bin", x_final.as_slice().unwrap_or(&[]));
    }
    if std::env::var("LARQL_QWEN35_DUMP_FINAL").is_ok() {
        let n = x_final.len();
        eprintln!(
            "x_final: first-3 = [{:.4}, {:.4}, {:.4}]  last-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}",
            x_final[0],
            x_final[1],
            x_final[2],
            x_final[n - 3],
            x_final[n - 2],
            x_final[n - 1],
            x_final.iter().map(|v| v * v).sum::<f32>().sqrt(),
        );
    }
    // Final logits matvec. Prefer the lazy-quantised path when
    // populated; falls back to dense f32 dot otherwise. When a
    // GPU backend is attached, the lazy path routes through
    // `matvec_with_backend` for cuBLAS-style GEMV.
    let logits = time_section!(
        LM_HEAD,
        if let Some(qt) = weights.lm_head_quant.as_ref() {
            // Phase E.7: per-class GPU residency. `LARQL_QWEN35_GPU_NO_LM_HEAD=1`
            // forces the final logits matvec onto CPU rayon.
            let lm_backend = crate::attention::gpu_tier::backend_for(
                crate::attention::gpu_tier::GpuClass::LmHead,
                weights.backend.as_deref(),
            );
            crate::attention::quant_dispatch::matvec_with_backend(qt, &x_final, lm_backend)
        } else {
            weights.lm_head.dot(&x_final)
        }
    );
    if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_FINAL_BIN_DIR") {
        let tag =
            std::env::var("LARQL_QWEN35_DUMP_TOKEN_TAG").unwrap_or_else(|_| "tok".to_string());
        let path = std::path::Path::new(&dir).join(format!("{tag}_logits.bin"));
        if let Ok(mut f) = std::fs::File::create(&path) {
            use std::io::Write;
            let data = logits.as_slice().unwrap_or(&[]);
            let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
            for n in ne {
                f.write_all(&n.to_le_bytes()).ok();
            }
            for v in data {
                f.write_all(&v.to_le_bytes()).ok();
            }
        }
    }

    if trace {
        eprintln!(
            "    final_norm(x):          l2={:.3} max_abs={:.3}",
            l2(&x_final),
            max_abs(&x_final),
        );
        eprintln!(
            "    logits:                 l2={:.3} max_abs={:.3}",
            l2(&logits),
            max_abs(&logits),
        );
    }

    // 4. Advance position for the next token's RoPE.
    hybrid_cache.next_position += 1;

    if fine_profile::enabled() {
        fine_profile::dump_and_reset();
    }

    logits
}

#[inline]
fn l2(a: &Array1<f32>) -> f32 {
    a.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

#[inline]
fn max_abs(a: &Array1<f32>) -> f32 {
    a.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()))
}

/// SwiGLU FFN: `down @ (silu(gate @ x) * (up @ x))`.
///
/// - `gate`: `[ffn_dim, hidden]`
/// - `up`:   `[ffn_dim, hidden]`
/// - `down`: `[hidden, ffn_dim]`
fn swiglu_ffn(
    x: &Array1<f32>,
    gate: ndarray::ArrayView2<f32>,
    up: ndarray::ArrayView2<f32>,
    down: ndarray::ArrayView2<f32>,
) -> Array1<f32> {
    let g = gate.dot(x); // [ffn_dim]
    let u = up.dot(x); // [ffn_dim]
    let mut inter = Array1::<f32>::zeros(g.len());
    for i in 0..g.len() {
        inter[i] = (g[i] * sigmoid(g[i])) * u[i]; // silu(g) * u
    }
    down.dot(&inter) // [hidden]
}

/// SwiGLU FFN with per-projection lazy-quantised fallback. Each of
/// the three projections takes the lazy form when `Some`, falling
/// back to the dense `ArcArray2<f32>` when `None`. Allows the bench
/// to compare configurations where only some projections are lazy
/// (or to drop in AVX2-accelerated `QuantTensor::matvec` in a
/// follow-up without touching the call sites).
#[allow(clippy::too_many_arguments)]
/// MoE FFN forward for Qwen3.6-35B-A3B and similar architectures.
///
/// Layout matches the Qwen35MoeArch GGUFs:
///   - `router`        `[num_experts, hidden]` f32 — produces a
///                     `[num_experts]` vector of routing logits.
///   - `gate_exps`     `[num_experts * ffn_dim, hidden]` Q4_K, 3D-packed
///   - `up_exps`       same shape, same quant
///   - `down_exps`     `[num_experts * hidden, ffn_dim]` Q5_K, 3D-packed
///   - shared expert (optional): the always-on dense SwiGLU with its
///     own `[ffn_dim, hidden]` gate/up and `[hidden, ffn_dim]` down.
///     If `shexp_*` is `None` this term is dropped (some Qwen MoE
///     variants don't have a shared expert).
///
/// Algorithm (per token):
///   1. `logits = router @ x`                                      // [num_experts]
///   2. `(idx, top_logits) = top_k(logits, top_k)`
///   3. `w = softmax(top_logits)`                                  // [top_k]
///   4. for each chosen expert `e_i`:
///        `y_i = swiglu(gate_exps[e_i] @ x, up_exps[e_i] @ x) @ down_exps[e_i]`
///   5. `y_moe = sum_i (w_i * y_i)`
///   6. `y_shared = swiglu(shexp_gate @ x, shexp_up @ x) @ shexp_down`
///   7. return `y_moe + y_shared`
///
/// Each expert dispatch reuses the existing `swiglu_ffn_lazy` path
/// (which already plumbs through gpu_tier::Ffn / paired matvec / the
/// device-resident fused block when the backend supports it). The
/// per-expert weight slicing is via `QuantTensor::expert_slice` —
/// zero-copy `Arc<[u8]>` views, F.1.
#[allow(clippy::too_many_arguments)]
fn swiglu_moe_lazy(
    x: &Array1<f32>,
    router: &larql_models::quant::lazy::QuantTensor,
    gate_exps: &larql_models::quant::lazy::QuantTensor,
    up_exps: &larql_models::quant::lazy::QuantTensor,
    down_exps: &larql_models::quant::lazy::QuantTensor,
    shexp_gate: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_up: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_down: Option<&larql_models::quant::lazy::QuantTensor>,
    num_experts: usize,
    top_k: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Array1<f32> {
    use crate::attention::gpu_tier::{self, GpuClass};
    use crate::attention::quant_dispatch::matvec_with_backend;
    use ndarray::ArcArray2;

    // The FFN GPU residency knob covers MoE too — pushing FFN to CPU
    // means all expert dispatches run CPU rayon (which is the
    // VRAM-minimal mode larql's value prop targets).
    let backend = gpu_tier::backend_for(GpuClass::Ffn, backend);

    // 1. Router logits.
    let logits = matvec_with_backend(router, x, backend);
    debug_assert_eq!(logits.len(), num_experts);

    // 2. Top-K selection on host. num_experts is typically 128-256 so a
    //    full sort is cheap; top_k is 8 so we don't bother with a
    //    proper partial-sort heap.
    let mut idx_logit: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx_logit.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<(usize, f32)> = idx_logit.into_iter().take(top_k).collect();

    // 3. Softmax over the top-K logits (numerically stable).
    let max_logit = top
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = top.iter().map(|&(_, l)| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let weights: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    // 4. Per-expert SwiGLU. Each expert_slice is a zero-copy view of
    //    the 3D-packed parent tensor (F.1).
    let hidden = x.len();
    let mut y_moe = Array1::<f32>::zeros(hidden);
    // Cheap empty placeholders for the dense-fallback slots in
    // swiglu_ffn_lazy — we never use them when the lazy form is set.
    let empty_2d: ArcArray2<f32> =
        ArcArray2::from_shape_vec((0, 0), vec![]).expect("0x0 array is always valid");

    // Phase G.1: pre-slice all top-K experts and issue
    // MADV_WILLNEED hints up front. The kernel starts paging in
    // every selected expert's gate/up/down byte range while we
    // dispatch the first expert's compute. No-op for heap-backed
    // tensors; real win for mmap-backed views (the load_gguf_lazy
    // path). Plus the shared expert, which is always-on.
    let expert_slices: Vec<(
        usize,
        larql_models::quant::lazy::QuantTensor,
        larql_models::quant::lazy::QuantTensor,
        larql_models::quant::lazy::QuantTensor,
    )> = top
        .iter()
        .map(|&(expert_id, _)| {
            let gate_e = gate_exps
                .expert_slice(expert_id, num_experts)
                .expect("expert_slice gate");
            let up_e = up_exps
                .expert_slice(expert_id, num_experts)
                .expect("expert_slice up");
            let down_e = down_exps
                .expert_slice(expert_id, num_experts)
                .expect("expert_slice down");
            gate_e.prefetch_willneed();
            up_e.prefetch_willneed();
            down_e.prefetch_willneed();
            (expert_id, gate_e, up_e, down_e)
        })
        .collect();
    if let (Some(g), Some(u), Some(d)) = (shexp_gate, shexp_up, shexp_down) {
        g.prefetch_willneed();
        u.prefetch_willneed();
        d.prefetch_willneed();
    }

    for (i, (_expert_id, gate_e, up_e, down_e)) in expert_slices.iter().enumerate() {
        let y_i = swiglu_ffn_lazy(
            x,
            Some(gate_e),
            Some(up_e),
            Some(down_e),
            &empty_2d,
            &empty_2d,
            &empty_2d,
            backend,
        );
        let w = weights[i];
        for h in 0..hidden {
            y_moe[h] += w * y_i[h];
        }
    }

    // 5. Shared expert (always-on, no routing weight). Some Qwen MoE
    //    variants ship without one — in which case the parameters are
    //    `None` and we skip.
    if let (Some(g), Some(u), Some(d)) = (shexp_gate, shexp_up, shexp_down) {
        let y_shared = swiglu_ffn_lazy(
            x,
            Some(g),
            Some(u),
            Some(d),
            &empty_2d,
            &empty_2d,
            &empty_2d,
            backend,
        );
        for h in 0..hidden {
            y_moe[h] += y_shared[h];
        }
    }

    y_moe
}

fn swiglu_ffn_lazy(
    x: &Array1<f32>,
    gate_q: Option<&larql_models::quant::lazy::QuantTensor>,
    up_q: Option<&larql_models::quant::lazy::QuantTensor>,
    down_q: Option<&larql_models::quant::lazy::QuantTensor>,
    gate_dense: &ArcArray2<f32>,
    up_dense: &ArcArray2<f32>,
    down_dense: &ArcArray2<f32>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Array1<f32> {
    use crate::attention::fine_profile::*;
    use crate::attention::gpu_tier::{self, GpuClass};
    use crate::attention::quant_dispatch::{ggml_type_to_quant_format, matvec_with_backend};
    use crate::time_section;
    use larql_compute::QuantFormat;

    // Phase E.7: per-class GPU residency. When the FFN class is
    // disabled (`LARQL_QWEN35_GPU_NO_FFN=1`), drop the backend so
    // every matvec in this block falls back to the CPU rayon path.
    let backend = gpu_tier::backend_for(GpuClass::Ffn, backend);

    // Fastest path: fully device-resident FFN block. Requires all
    // three projections (gate, up, down) to be lazy-quant in a
    // GPU-supported format. On Qwen3.6-27B-Q4_K_S `gate`/`up` are
    // Q4_K and `down` is Q5_K, so the backend must accept a mixed
    // pair. Saves 2 dtohs (g, u to host), 1 htod (inter back to
    // device), and 1 CPU silu loop per FFN layer.
    if let (Some(b), Some(gq), Some(uq), Some(dq)) = (backend, gate_q, up_q, down_q) {
        let gfmt = ggml_type_to_quant_format(gq.tensor_type());
        let ufmt = ggml_type_to_quant_format(uq.tensor_type());
        let dfmt = ggml_type_to_quant_format(dq.tensor_type());
        let gpu_ok =
            |f: Option<QuantFormat>| matches!(f, Some(QuantFormat::Q4_K | QuantFormat::Q5_K));
        if gpu_ok(gfmt) && gpu_ok(ufmt) && gpu_ok(dfmt) {
            let x_slice = x.as_slice().expect("Array1 contiguous");
            let g_shape = gq.shape();
            let u_shape = uq.shape();
            let d_shape = dq.shape();
            let block = time_section!(
                FFN_GATE_UP_PAIR,
                b.qwen35_ffn_lazy_block(
                    x_slice,
                    gq.raw_bytes(),
                    gfmt.unwrap(),
                    g_shape[0],
                    uq.raw_bytes(),
                    ufmt.unwrap(),
                    u_shape[0],
                    dq.raw_bytes(),
                    dfmt.unwrap(),
                    d_shape[0],
                    g_shape[1],
                )
            );
            if let Some(out) = block {
                return Array1::from(out);
            }
        }
    }

    let (g, u) = time_section!(FFN_GATE_UP_PAIR, {
        let paired = match (backend, gate_q, up_q) {
            (Some(b), Some(gq), Some(uq)) => {
                let gfmt = ggml_type_to_quant_format(gq.tensor_type());
                let ufmt = ggml_type_to_quant_format(uq.tensor_type());
                if matches!(gfmt, Some(QuantFormat::Q4_K))
                    && matches!(ufmt, Some(QuantFormat::Q4_K))
                {
                    let x_slice = x.as_slice().expect("Array1 contiguous");
                    let g_shape = gq.shape();
                    let u_shape = uq.shape();
                    b.qwen35_paired_q4k_matvec(
                        gq.raw_bytes(),
                        g_shape[0],
                        uq.raw_bytes(),
                        u_shape[0],
                        x_slice,
                        g_shape[1],
                    )
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some((gv, uv)) = paired {
            (Array1::from(gv), Array1::from(uv))
        } else {
            let g = match gate_q {
                Some(q) => matvec_with_backend(q, x, backend),
                None => gate_dense.dot(x),
            };
            let u = match up_q {
                Some(q) => matvec_with_backend(q, x, backend),
                None => up_dense.dot(x),
            };
            (g, u)
        }
    });
    let inter = time_section!(FFN_SILU_LOOP, {
        let mut inter = Array1::<f32>::zeros(g.len());
        for i in 0..g.len() {
            inter[i] = (g[i] * sigmoid(g[i])) * u[i];
        }
        inter
    });
    time_section!(
        FFN_DOWN,
        match down_q {
            Some(q) => matvec_with_backend(q, &inter, backend),
            None => down_dense.dot(&inter),
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::deltanet_block::DeltaNetLayerWeights;
    use crate::attention::qwen35_block::{DeltaNetHybridCache, Qwen35AttentionLayerWeights};
    use ndarray::Array2;

    fn tiny_dn_dims() -> DeltaNetDims {
        DeltaNetDims {
            hidden: 4,
            head_v_dim: 2,
            n_v_heads: 1,
            n_k_heads: 1,
            d_conv: 2,
            eps: 1e-6,
        }
    }

    fn tiny_attn_dims() -> Qwen35AttentionDims {
        Qwen35AttentionDims {
            hidden: 4,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 2,
            rotary_dim: 2,
            rope_base: 10_000.0,
            eps: 1e-6,
        }
    }

    /// Build a constant-filled tiny DeltaNet weights bundle. ArcArray
    /// + Arc<[f32]> are cheap-to-clone refcounted owners.
    fn make_dn_weights(dn_dims: &DeltaNetDims) -> DeltaNetLayerWeights {
        let conv_dim = dn_dims.conv_dim();
        let value_dim = dn_dims.value_dim();
        DeltaNetLayerWeights {
            attn_norm: Arc::from(vec![1.0_f32; dn_dims.hidden].as_slice()),
            attn_qkv: Array2::from_elem((conv_dim, dn_dims.hidden), 0.1_f32).into_shared(),
            attn_gate: Array2::from_elem((value_dim, dn_dims.hidden), 0.1_f32).into_shared(),
            ssm_conv1d: Array2::from_elem((dn_dims.d_conv, conv_dim), 0.5_f32).into_shared(),
            ssm_dt: Arc::from(vec![0.0_f32; dn_dims.n_v_heads].as_slice()),
            ssm_a: Arc::from(vec![-1.0_f32; dn_dims.n_v_heads].as_slice()),
            ssm_beta: Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32).into_shared(),
            ssm_alpha: Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32)
                .into_shared(),
            ssm_norm: Arc::from(vec![1.0_f32; dn_dims.head_v_dim].as_slice()),
            ssm_out: Array2::from_elem((dn_dims.hidden, value_dim), 0.5_f32).into_shared(),
            attn_qkv_quant: None,
            attn_gate_quant: None,
            ssm_out_quant: None,
        }
    }

    fn make_attn_weights(attn_dims: &Qwen35AttentionDims) -> Qwen35AttentionLayerWeights {
        Qwen35AttentionLayerWeights {
            attn_norm: Arc::from(vec![1.0_f32; attn_dims.hidden].as_slice()),
            attn_q: Array2::from_elem((attn_dims.fused_q_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_k: Array2::from_elem((attn_dims.kv_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_v: Array2::from_elem((attn_dims.kv_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_q_norm: Arc::from(vec![1.0_f32; attn_dims.head_dim].as_slice()),
            attn_k_norm: Arc::from(vec![1.0_f32; attn_dims.head_dim].as_slice()),
            attn_output: Array2::from_elem((attn_dims.hidden, attn_dims.q_dim()), 0.5_f32)
                .into_shared(),
            attn_q_quant: None,
            attn_k_quant: None,
            attn_v_quant: None,
            attn_output_quant: None,
        }
    }

    /// Build a per-layer suffix (attn_post_norm + SwiGLU FFN trio).
    fn make_suffix(
        hidden: usize,
        ffn_dim: usize,
    ) -> (
        Arc<[f32]>,
        ndarray::ArcArray2<f32>,
        ndarray::ArcArray2<f32>,
        ndarray::ArcArray2<f32>,
    ) {
        (
            Arc::from(vec![1.0_f32; hidden].as_slice()),
            Array2::from_elem((ffn_dim, hidden), 0.1_f32).into_shared(),
            Array2::from_elem((ffn_dim, hidden), 0.1_f32).into_shared(),
            Array2::from_elem((hidden, ffn_dim), 0.5_f32).into_shared(),
        )
    }

    #[test]
    fn swiglu_ffn_zero_input_zero_output() {
        let hidden = 4;
        let ffn_dim = 8;
        let x = Array1::<f32>::zeros(hidden);
        let gate = Array2::from_elem((ffn_dim, hidden), 1.0_f32);
        let up = Array2::from_elem((ffn_dim, hidden), 1.0_f32);
        let down = Array2::from_elem((hidden, ffn_dim), 1.0_f32);
        let y = swiglu_ffn(&x, gate.view(), up.view(), down.view());
        // gate @ 0 = 0, silu(0)=0, up @ 0 = 0, inter = 0, down @ 0 = 0.
        for &v in y.iter() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn swiglu_ffn_matches_definition_unit_vectors() {
        // hidden=2, ffn_dim=2. gate = I (identity), up = I, down = I.
        // y = down @ (silu(gate @ x) * (up @ x))
        //   = silu(x) * x
        //
        // For x = [1, 0]: silu(1)*1 + silu(0)*0 = silu(1) at index 0.
        let gate = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let up = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let down = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let x = ndarray::array![1.0_f32, 0.0];
        let y = swiglu_ffn(&x, gate.view(), up.view(), down.view());
        let expected = 1.0 * sigmoid(1.0); // silu(1)
        assert!((y[0] - expected).abs() < 1e-5);
        assert!(y[1].abs() < 1e-5);
    }

    /// Parity test for `swiglu_moe_lazy` against a hand-rolled scalar
    /// reference. Tiny setup: hidden=3, ffn_dim=4, num_experts=4,
    /// top_k=2. F32-typed QuantTensors so the GPU fast-paths
    /// fall through to the per-call `matvec_with_backend` fallback
    /// (which routes f32 through `QuantTensor::matvec`).
    #[test]
    fn swiglu_moe_lazy_matches_reference() {
        use larql_models::quant::lazy::QuantTensor;

        let hidden = 3usize;
        let ffn_dim = 4usize;
        let num_experts = 4usize;
        let top_k = 2usize;

        // Reproducible synthetic weights — each tensor gets a distinct
        // per-element pattern so the reference and the impl exercise
        // every slot.
        let mk = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * 0.07 + seed).sin() * 0.3)
                .collect()
        };
        let router_v = mk(num_experts * hidden, 0.1);
        let gate_exps_v = mk(num_experts * ffn_dim * hidden, 0.2);
        let up_exps_v = mk(num_experts * ffn_dim * hidden, 0.3);
        let down_exps_v = mk(num_experts * hidden * ffn_dim, 0.4);
        let shexp_g_v = mk(ffn_dim * hidden, 0.5);
        let shexp_u_v = mk(ffn_dim * hidden, 0.6);
        let shexp_d_v = mk(hidden * ffn_dim, 0.7);

        let router = QuantTensor::from_f32_rows(num_experts, hidden, &router_v);
        // 3D-packed layout per F.1: rows = num_experts * out_dim.
        let gate_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &gate_exps_v);
        let up_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &up_exps_v);
        let down_exps = QuantTensor::from_f32_rows(num_experts * hidden, ffn_dim, &down_exps_v);
        let shexp_g = QuantTensor::from_f32_rows(ffn_dim, hidden, &shexp_g_v);
        let shexp_u = QuantTensor::from_f32_rows(ffn_dim, hidden, &shexp_u_v);
        let shexp_d = QuantTensor::from_f32_rows(hidden, ffn_dim, &shexp_d_v);

        let x = Array1::from(vec![0.4_f32, -0.2, 0.15]);

        // --- Reference (scalar) ---
        // Router logits.
        let mut logits = vec![0.0_f32; num_experts];
        for e in 0..num_experts {
            let mut acc = 0.0_f32;
            for c in 0..hidden {
                acc += router_v[e * hidden + c] * x[c];
            }
            logits[e] = acc;
        }
        // Top-K (full sort is fine here).
        let mut idx: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<(usize, f32)> = idx.into_iter().take(top_k).collect();
        // Softmax.
        let max_l = top
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|&(_, l)| (l - max_l).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let w: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

        let ref_swiglu = |gate: &[f32], up: &[f32], down: &[f32]| -> Vec<f32> {
            // gate @ x and up @ x : [ffn_dim]
            let mut g = vec![0.0_f32; ffn_dim];
            let mut u = vec![0.0_f32; ffn_dim];
            for r in 0..ffn_dim {
                let mut ag = 0.0;
                let mut au = 0.0;
                for c in 0..hidden {
                    ag += gate[r * hidden + c] * x[c];
                    au += up[r * hidden + c] * x[c];
                }
                g[r] = ag;
                u[r] = au;
            }
            // SiLU(g) * u
            let inter: Vec<f32> = g
                .iter()
                .zip(u.iter())
                .map(|(&gi, &ui)| (gi * sigmoid(gi)) * ui)
                .collect();
            // down @ inter : [hidden]
            let mut y = vec![0.0_f32; hidden];
            for r in 0..hidden {
                let mut a = 0.0;
                for c in 0..ffn_dim {
                    a += down[r * ffn_dim + c] * inter[c];
                }
                y[r] = a;
            }
            y
        };

        let mut expected = vec![0.0_f32; hidden];
        let bytes_gate_per_expert = ffn_dim * hidden;
        let bytes_down_per_expert = hidden * ffn_dim;
        for (i, &(e, _)) in top.iter().enumerate() {
            let g0 = e * bytes_gate_per_expert;
            let g1 = g0 + bytes_gate_per_expert;
            let d0 = e * bytes_down_per_expert;
            let d1 = d0 + bytes_down_per_expert;
            let y_i = ref_swiglu(
                &gate_exps_v[g0..g1],
                &up_exps_v[g0..g1],
                &down_exps_v[d0..d1],
            );
            for h in 0..hidden {
                expected[h] += w[i] * y_i[h];
            }
        }
        // Shared expert.
        let y_sh = ref_swiglu(&shexp_g_v, &shexp_u_v, &shexp_d_v);
        for h in 0..hidden {
            expected[h] += y_sh[h];
        }

        // --- Impl ---
        let got = swiglu_moe_lazy(
            &x,
            &router,
            &gate_exps,
            &up_exps,
            &down_exps,
            Some(&shexp_g),
            Some(&shexp_u),
            Some(&shexp_d),
            num_experts,
            top_k,
            None, // CPU path through QuantTensor::matvec
        );

        assert_eq!(got.len(), hidden);
        for h in 0..hidden {
            assert!(
                (got[h] - expected[h]).abs() < 1e-5,
                "moe[{h}] got={} expected={}",
                got[h],
                expected[h],
            );
        }
    }

    /// Shared expert is optional — if all three shexp params are
    /// `None`, the output is just the weighted top-K sum.
    #[test]
    fn swiglu_moe_lazy_without_shared_expert() {
        use larql_models::quant::lazy::QuantTensor;
        let hidden = 2usize;
        let ffn_dim = 2usize;
        let num_experts = 2usize;
        let top_k = 1usize;

        // Router heavily favours expert 0 for x = [1, 0].
        let router_v = vec![
            1.0_f32, 0.0, // expert 0 row
            -1.0, 0.0, // expert 1 row
        ];
        // Expert 0: identity-ish. Expert 1: distinct but should be unused with top_k=1.
        let gate_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let up_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let down_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let router = QuantTensor::from_f32_rows(num_experts, hidden, &router_v);
        let gate_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &gate_exps_v);
        let up_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &up_exps_v);
        let down_exps = QuantTensor::from_f32_rows(num_experts * hidden, ffn_dim, &down_exps_v);
        let x = Array1::from(vec![1.0_f32, 0.0]);

        let got = swiglu_moe_lazy(
            &x,
            &router,
            &gate_exps,
            &up_exps,
            &down_exps,
            None,
            None,
            None,
            num_experts,
            top_k,
            None,
        );

        // top_k=1 → softmax over one logit = [1.0]. Expert 0 SwiGLU
        // with identity gate/up/down: y = silu(x) * x.
        // x = [1, 0] → y = [silu(1)*1, 0].
        let expected_y0 = sigmoid(1.0); // 1.0 * sigmoid(1.0)
        assert!(
            (got[0] - expected_y0).abs() < 1e-5,
            "got={} exp={}",
            got[0],
            expected_y0
        );
        assert!(got[1].abs() < 1e-5, "got[1]={}", got[1]);
    }

    #[test]
    fn qwen35_forward_step_returns_vocab_shape() {
        // 2-layer model: layer 0 linear, layer 1 attention.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        let layer_kinds = vec![true, false];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let dn_weights = make_dn_weights(&dn_dims);
        let attn_weights = make_attn_weights(&attn_dims);
        let (post0, gate0, up0, down0) = make_suffix(hidden, ffn_dim);
        let (post1, gate1, up1, down1) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            embed_quant: None,
            layers: vec![
                Qwen35FullLayerWeights {
                    block: Qwen35LayerWeights::Linear(dn_weights),
                    attn_post_norm: post0,
                    ffn_gate: gate0,
                    ffn_up: up0,
                    ffn_down: down0,
                    ffn_gate_quant: None,
                    ffn_up_quant: None,
                    ffn_down_quant: None,
                    moe: None,
                },
                Qwen35FullLayerWeights {
                    block: Qwen35LayerWeights::Attention(attn_weights),
                    attn_post_norm: post1,
                    ffn_gate: gate1,
                    ffn_up: up1,
                    ffn_down: down1,
                    ffn_gate_quant: None,
                    ffn_up_quant: None,
                    ffn_down_quant: None,
                    moe: None,
                },
            ],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            lm_head_quant: None,
            ffn_dim,
            backend: None,
        };

        let token = 2u32;
        let logits = qwen35_forward_step(token, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(cache.next_position, 1);
    }

    #[test]
    fn qwen35_forward_step_two_calls_advances_position_and_grows_cache() {
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        // Single full-attention layer so we can observe KV growth.
        let layer_kinds = vec![false];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let attn_weights = make_attn_weights(&attn_dims);
        let (post, gate, up, down) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            embed_quant: None,
            layers: vec![Qwen35FullLayerWeights {
                block: Qwen35LayerWeights::Attention(attn_weights),
                attn_post_norm: post,
                ffn_gate: gate,
                ffn_up: up,
                ffn_down: down,
                ffn_gate_quant: None,
                ffn_up_quant: None,
                ffn_down_quant: None,
                moe: None,
            }],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            lm_head_quant: None,
            ffn_dim,
            backend: None,
        };

        let _ = qwen35_forward_step(0, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(cache.next_position, 1);
        let (k_after_1, _) = cache.kv_layers[0].as_ref().unwrap();
        assert_eq!(k_after_1.shape()[0], 1);

        let _ = qwen35_forward_step(1, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(cache.next_position, 2);
        let (k_after_2, _) = cache.kv_layers[0].as_ref().unwrap();
        assert_eq!(k_after_2.shape()[0], 2);
    }

    #[test]
    fn qwen35_forward_step_dn_state_carries_across_calls() {
        // Single linear layer — verify DeltaNet state moves off zero
        // and accumulates across calls.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        let layer_kinds = vec![true];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let dn_weights = make_dn_weights(&dn_dims);
        let (post, gate, up, down) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            embed_quant: None,
            layers: vec![Qwen35FullLayerWeights {
                block: Qwen35LayerWeights::Linear(dn_weights),
                attn_post_norm: post,
                ffn_gate_quant: None,
                ffn_up_quant: None,
                ffn_down_quant: None,
                ffn_gate: gate,
                ffn_up: up,
                ffn_down: down,
                moe: None,
            }],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            lm_head_quant: None,
            ffn_dim,
            backend: None,
        };

        let _ = qwen35_forward_step(2, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        let mass_after_1: f32 = cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_after_1 > 0.0,
            "recurrent_state should be non-zero after one forward step"
        );

        let _ = qwen35_forward_step(2, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        let mass_after_2: f32 = cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_after_2 >= 0.0,
            "recurrent_state still finite after two steps"
        );
    }
}
