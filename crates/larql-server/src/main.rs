//! larql-server — HTTP server for vindex knowledge queries.
//!
//! Thin binary entry point: parse `Cli`, install tracing, hand off to
//! `bootstrap::serve`. Boot orchestration (vindex loading, warmups, listener
//! setup, grid announce) lives in `larql_server::bootstrap` so that
//! integration tests can drive the same code path without going through
//! `clap::Parser::parse_from`.

use clap::Parser;

use larql_server::bootstrap::{self, normalize_serve_alias, BoxError, Cli};

// OpenBLAS thread-control entry point. Only declared on Linux/Windows
// where `openblas-src` provides the symbol. macOS links Accelerate.framework
// (no `openblas_set_num_threads`); we rely on env vars there.
#[cfg(not(target_os = "macos"))]
extern "C" {
    fn openblas_set_num_threads(num_threads: std::os::raw::c_int);
}

/// Pin OpenBLAS / OpenMP thread count to 1 so BLAS doesn't compete with
/// the rayon-parallel Q4_K × Q8_K matvec kernels (#143 / #144). On a
/// 48-core EPYC host this was the dominant remaining decode-step cost —
/// OpenBLAS spawning 48 threads per `ndarray::Array2::dot` thrashed
/// against rayon's 48 workers for the same cores. With BLAS single-
/// threaded, end-to-end Gemma 3 4B at 150 tokens went 4.18 → 9.12 tok/s
/// (~65% of llama.cpp's 14.1 reference). All hot-path matvecs (FFN,
/// attention Q/K/V/O, lm_head) flow through rayon; the small BLAS calls
/// remaining in `gqa_attention_decode_step` per-head dots are
/// per-call-overhead-dominated at single-thread.
///
/// Respect user-supplied env vars so callers who explicitly want
/// multi-threaded BLAS (e.g., for prefill-only benchmarks) can override
/// with `OPENBLAS_NUM_THREADS=N`.
fn configure_blas_threads_for_rayon() {
    for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        if std::env::var_os(var).is_none() {
            // SAFETY: single-threaded at this point — called before the
            // tokio runtime spawns any worker. No torn-env hazard.
            unsafe {
                std::env::set_var(var, "1");
            }
        }
    }
    // Belt-and-suspenders: the env var catches OpenBLAS's lazy-init path
    // AND any OpenMP variant. The FFI call catches the eager-init path
    // where the env var was already read at process load. macOS uses
    // Accelerate which honours `VECLIB_MAXIMUM_THREADS` instead (set
    // below for completeness).
    #[cfg(target_os = "macos")]
    if std::env::var_os("VECLIB_MAXIMUM_THREADS").is_none() {
        // SAFETY: single-threaded at this point.
        unsafe {
            std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
        }
    }
    #[cfg(not(target_os = "macos"))]
    // SAFETY: link-time guarantee on Linux/Windows where openblas-src
    // provides the symbol. Idempotent — safe to call when the env var
    // was already 1.
    unsafe {
        openblas_set_num_threads(1);
    }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    configure_blas_threads_for_rayon();

    // Accept both `larql-server <path>` and `larql-server serve <path>`.
    let cli = Cli::parse_from(normalize_serve_alias(std::env::args().collect()));

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    bootstrap::serve(cli).await
}
