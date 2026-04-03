#[cfg(target_os = "macos")]
pub mod context;
#[cfg(target_os = "macos")]
pub mod dispatch;
pub mod error;
// turboquant: pure-math parts always compile; Metal dispatch only on macOS
pub mod turboquant;

#[cfg(target_os = "macos")]
pub use context::MetalContext;
pub use error::{KernelError, Result};

/// Supported kernel operations — used for documentation and future dynamic dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOp {
    /// General matrix-vector multiply (decode path, batch_size=1)
    Gemv,
    /// General matrix-matrix multiply (prefill path, batch_size>1)
    Gemm,
    /// FlashAttention tiled attention (GQA-aware)
    FlashAttention,
    /// Rotary position embeddings (in-place on Q or K)
    RoPE,
    /// RMS normalisation (no mean subtraction)
    RmsNorm,
    /// Softmax (standalone, used in sampler)
    Softmax,
    /// Q4_K_M dequantisation (Phase 4)
    DequantQ4K,
    /// Elementwise add (residual connections)
    Add,
    /// SiLU activation + multiply (SwiGLU FFN gate)
    SiluMul,
    /// Elementwise multiply (standalone)
    Mul,
    /// TurboQuant: RHT + Lloyd-Max key encode (Phase 5)
    TqEncodeKey,
    /// TurboQuant: per-group value encode (Phase 5)
    TqEncodeVal,
    /// TurboQuant: key decode + QJL correction (Phase 5)
    TqDecodeKey,
    /// TurboQuant: per-group value decode (Phase 5)
    TqDecodeVal,
}

/// Path to the compiled .metallib (set by build.rs at compile time).
pub fn metallib_path() -> Option<&'static str> {
    option_env!("METALLIB_PATH")
}
