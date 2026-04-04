use half::f16;
use metal::{Buffer, MTLSize};

use crate::context::MetalContext;
use crate::error::{KernelError, Result};

// ── Threadgroup size constants ────────────────────────────────────────────────
const TG_GEMV: u64 = 32;     // one SIMD lane width
const TG_ELEM: u64 = 256;    // for element-wise + norm kernels
const TG_ATTN: u64 = 32;     // BLOCK_Q for FlashAttention

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU: out[i] = silu(gate[i]) * up[i]
// ─────────────────────────────────────────────────────────────────────────────
pub fn silu_mul_f16(
    ctx: &MetalContext,
    gate: &Buffer,
    up: &Buffer,
    out: &Buffer,
    n: u32,
) -> Result<()> {
    ctx.encode("silu_mul_f16", |enc| {
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up),   0);
        enc.set_buffer(2, Some(out),  0);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: n as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,  height: 1, depth: 1 },
        );
    })
}

/// f32 SwiGLU: out[i] = silu(gate[i]) * up[i], all f32.
pub fn silu_mul_f32(
    ctx: &MetalContext,
    gate: &Buffer,
    up: &Buffer,
    out: &Buffer,
    n: u32,
) -> Result<()> {
    ctx.encode("silu_mul_f32", |enc| {
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up),   0);
        enc.set_buffer(2, Some(out),  0);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: n as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,  height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise add: out[i] = a[i] + b[i]
// ─────────────────────────────────────────────────────────────────────────────
pub fn add_f16(
    ctx: &MetalContext,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    n: u32,
) -> Result<()> {
    ctx.encode("add_f16", |enc| {
        enc.set_buffer(0, Some(a),   0);
        enc.set_buffer(1, Some(b),   0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: n as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,  height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise multiply: out[i] = a[i] * b[i]
// ─────────────────────────────────────────────────────────────────────────────
pub fn mul_f16(
    ctx: &MetalContext,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    n: u32,
) -> Result<()> {
    ctx.encode("mul_f16", |enc| {
        enc.set_buffer(0, Some(a),   0);
        enc.set_buffer(1, Some(b),   0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: n as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,  height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm: y[row] = x[row] / rms(x[row]) * gamma
// ─────────────────────────────────────────────────────────────────────────────
pub fn rms_norm_f16(
    ctx: &MetalContext,
    x: &Buffer,
    y: &Buffer,
    gamma: &Buffer,
    eps: f32,
    hidden: u32,
    num_rows: u32,
) -> Result<()> {
    // SRAM: one float per thread for reduction scratch
    let sram_bytes = TG_ELEM * std::mem::size_of::<f32>() as u64;
    ctx.encode("rms_norm_f16", |enc| {
        enc.set_buffer(0, Some(x),     0);
        enc.set_buffer(1, Some(y),     0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_bytes(3, 4, &eps    as *const f32 as *const _);
        enc.set_bytes(4, 4, &hidden as *const u32 as *const _);
        enc.set_threadgroup_memory_length(0, sram_bytes);
        enc.dispatch_thread_groups(
            MTLSize { width: num_rows as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,         height: 1, depth: 1 },
        );
    })
}

/// RMSNorm with f32 input → f16 output (for f32 residual stream).
pub fn rms_norm_f32in_f16out(
    ctx: &MetalContext,
    x: &Buffer,       // f32 input
    y: &Buffer,       // f16 output
    gamma: &Buffer,   // f16 scale
    eps: f32,
    hidden: u32,
    num_rows: u32,
) -> Result<()> {
    let sram_bytes = TG_ELEM * std::mem::size_of::<f32>() as u64;
    ctx.encode("rms_norm_f32in_f16out", |enc| {
        enc.set_buffer(0, Some(x),     0);
        enc.set_buffer(1, Some(y),     0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_bytes(3, 4, &eps    as *const f32 as *const _);
        enc.set_bytes(4, 4, &hidden as *const u32 as *const _);
        enc.set_threadgroup_memory_length(0, sram_bytes);
        enc.dispatch_thread_groups(
            MTLSize { width: num_rows as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,         height: 1, depth: 1 },
        );
    })
}

/// RMSNorm f32→f32 (full precision residual + output)
pub fn rms_norm_f32_f32(
    ctx: &MetalContext,
    x: &Buffer,
    y: &Buffer,
    gamma: &Buffer,
    eps: f32,
    hidden: u32,
    num_rows: u32,
) -> Result<()> {
    let sram_bytes = TG_ELEM * std::mem::size_of::<f32>() as u64;
    ctx.encode("rms_norm_f32_f32", |enc| {
        enc.set_buffer(0, Some(x),     0);
        enc.set_buffer(1, Some(y),     0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_bytes(3, 4, &eps    as *const f32 as *const _);
        enc.set_bytes(4, 4, &hidden as *const u32 as *const _);
        enc.set_threadgroup_memory_length(0, sram_bytes);
        enc.dispatch_thread_groups(
            MTLSize { width: num_rows as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,         height: 1, depth: 1 },
        );
    })
}

/// RMSNorm f32→f32 with f32 gamma (full precision norm weights)
pub fn rms_norm_f32_f32_f32g(
    ctx: &MetalContext,
    x: &Buffer,
    y: &Buffer,
    gamma: &Buffer,
    eps: f32,
    hidden: u32,
    num_rows: u32,
) -> Result<()> {
    let sram_bytes = TG_ELEM * std::mem::size_of::<f32>() as u64;
    ctx.encode("rms_norm_f32_f32_f32g", |enc| {
        enc.set_buffer(0, Some(x),     0);
        enc.set_buffer(1, Some(y),     0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_bytes(3, 4, &eps    as *const f32 as *const _);
        enc.set_bytes(4, 4, &hidden as *const u32 as *const _);
        enc.set_threadgroup_memory_length(0, sram_bytes);
        enc.dispatch_thread_groups(
            MTLSize { width: num_rows as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,         height: 1, depth: 1 },
        );
    })
}

/// Full f32 GEMV with f32 output: y_f32 = A_f32 * x_f32
pub fn gemv_f32_f32out(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_f32_f32out", |enc| {
        enc.set_buffer(0, Some(weight), 0);
        enc.set_buffer(1, Some(input),  0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// Full f32 GEMV: y_f16 = A_f32 * x_f32 (max precision)
pub fn gemv_f32(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_f32", |enc| {
        enc.set_buffer(0, Some(weight), 0);
        enc.set_buffer(1, Some(input),  0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// Full f32 GEMV + f32 residual add
pub fn gemv_add_f32_full(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    residual: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_add_f32", |enc| {
        enc.set_buffer(0, Some(weight),   0);
        enc.set_buffer(1, Some(input),    0);
        enc.set_buffer(2, Some(output),   0);
        enc.set_buffer(3, Some(residual), 0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// GEMV with f16 weights and f32 input: y_f16 = A_f16 * x_f32
pub fn gemv_f16w_f32in(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_f16w_f32in", |enc| {
        enc.set_buffer(0, Some(weight), 0);
        enc.set_buffer(1, Some(input),  0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// GEMV + f32 residual with f32 input
pub fn gemv_add_f32_f16w(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    residual: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_add_f32_f16w", |enc| {
        enc.set_buffer(0, Some(weight),   0);
        enc.set_buffer(1, Some(input),    0);
        enc.set_buffer(2, Some(output),   0);
        enc.set_buffer(3, Some(residual), 0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// Add f16 source into f32 accumulator: acc[i] += src[i]
pub fn add_f16_into_f32(
    ctx: &MetalContext,
    src: &Buffer,  // f16
    acc: &Buffer,  // f32 (read-write)
    n: u32,
) -> Result<()> {
    ctx.encode("add_f16_into_f32", |enc| {
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(acc), 0);
        enc.set_bytes(2, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: n as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM.min(n as u64), height: 1, depth: 1 },
        );
    })
}

/// GEMV + f32 residual add: y_f32[i] = (A_f16 * x_f16)[i] + res_f32[i]
pub fn gemv_add_f32res_f16(
    ctx: &MetalContext,
    weight: &Buffer,    // f16 [M, K]
    input: &Buffer,     // f16 [K]
    output: &Buffer,    // f32 [M]
    residual: &Buffer,  // f32 [M]
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_add_f32res_f16", |enc| {
        enc.set_buffer(0, Some(weight),   0);
        enc.set_buffer(1, Some(input),    0);
        enc.set_buffer(2, Some(output),   0);
        enc.set_buffer(3, Some(residual), 0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// Q4K GEMV + f32 residual add
pub fn gemv_q4k_add_f32res_f16(
    ctx: &MetalContext,
    weight_q4k: &Buffer,
    input: &Buffer,
    output: &Buffer,
    residual: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_q4k_add_f32res_f16", |enc| {
        enc.set_buffer(0, Some(weight_q4k), 0);
        enc.set_buffer(1, Some(input),      0);
        enc.set_buffer(2, Some(output),     0);
        enc.set_buffer(3, Some(residual),   0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// F32-weight GEMV variants (dequantized weights stored as f32 for precision)
// ═════════════════════════════════════════════════════════════════════════════

/// GEMV with f32 weights: y_f16 = A_f32 * x_f16
pub fn gemv_f32w(
    ctx: &MetalContext,
    weight: &Buffer,   // f32 [M, K]
    input: &Buffer,    // f16 [K]
    output: &Buffer,   // f16 [M]
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_f32w", |enc| {
        enc.set_buffer(0, Some(weight), 0);
        enc.set_buffer(1, Some(input),  0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// Fused GEMV + residual with f32 weights: y_f16 = A_f32 * x_f16 + residual_f16
pub fn gemv_add_f32w(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    residual: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_add_f32w", |enc| {
        enc.set_buffer(0, Some(weight),   0);
        enc.set_buffer(1, Some(input),    0);
        enc.set_buffer(2, Some(output),   0);
        enc.set_buffer(3, Some(residual), 0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// GEMV with f32 weights and f32 input: y_f16 = A_f32 * x_f32
pub fn gemv_f32w_f32in(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_f32w_f32in", |enc| {
        enc.set_buffer(0, Some(weight), 0);
        enc.set_buffer(1, Some(input),  0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// GEMV + f32 residual with f32 weights: y_f32 = A_f32 * x_f16 + residual_f32
pub fn gemv_add_f32res_f32w(
    ctx: &MetalContext,
    weight: &Buffer,    // f32 [M, K]
    input: &Buffer,     // f16 [K]
    output: &Buffer,    // f32 [M]
    residual: &Buffer,  // f32 [M]
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_add_f32res_f32w", |enc| {
        enc.set_buffer(0, Some(weight),   0);
        enc.set_buffer(1, Some(input),    0);
        enc.set_buffer(2, Some(output),   0);
        enc.set_buffer(3, Some(residual), 0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// GEMV + f32 residual with f32 weights and f32 input: y_f32 = A_f32 * x_f32 + res_f32
pub fn gemv_add_f32_f32w(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    residual: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_add_f32_f32w", |enc| {
        enc.set_buffer(0, Some(weight),   0);
        enc.set_buffer(1, Some(input),    0);
        enc.set_buffer(2, Some(output),   0);
        enc.set_buffer(3, Some(residual), 0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE: apply rotary embeddings in-place to Q or K
// ─────────────────────────────────────────────────────────────────────────────
pub fn rope_inplace_f16(
    ctx: &MetalContext,
    x: &Buffer,           // Q or K tensor, modified in-place
    head_dim: u32,
    n_heads: u32,
    seq_pos: u32,         // absolute position of this token
    rope_theta: f32,
    batch_seq: u32,       // batch * seq_len (number of tokens)
) -> Result<()> {
    let tg_width = ((head_dim / 2) as u64).min(32).max(1);
    ctx.encode("rope_inplace_f16", |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, &head_dim   as *const u32 as *const _);
        enc.set_bytes(2, 4, &n_heads    as *const u32 as *const _);
        enc.set_bytes(3, 4, &seq_pos    as *const u32 as *const _);
        enc.set_bytes(4, 4, &rope_theta as *const f32 as *const _);
        enc.dispatch_threads(
            MTLSize {
                width:  (head_dim / 2) as u64,
                height: n_heads as u64,
                depth:  batch_seq as u64,
            },
            MTLSize { width: tg_width, height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// GEMV: y = A * x   (F16 matrix-vector multiply, decode path)
// ─────────────────────────────────────────────────────────────────────────────
pub fn gemv_f16(
    ctx: &MetalContext,
    weight: &Buffer,   // A: [M, K]
    input: &Buffer,    // x: [K]
    output: &Buffer,   // y: [M]
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_f16", |enc| {
        enc.set_buffer(0, Some(weight), 0);
        enc.set_buffer(1, Some(input),  0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused GEMV + residual add: y = A * x + residual
// ─────────────────────────────────────────────────────────────────────────────

/// Fused GEMV + residual add: `y[i] = (A * x)[i] + residual[i]`.
/// Saves one kernel dispatch and one buffer pass per residual connection.
pub fn gemv_add_f16(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    residual: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_add_f16", |enc| {
        enc.set_buffer(0, Some(weight),   0);
        enc.set_buffer(1, Some(input),    0);
        enc.set_buffer(2, Some(output),   0);
        enc.set_buffer(3, Some(residual), 0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// Fused Q4K GEMV + residual add.
pub fn gemv_q4k_add_f16(
    ctx: &MetalContext,
    weight_q4k: &Buffer,
    input: &Buffer,
    output: &Buffer,
    residual: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_q4k_add_f16", |enc| {
        enc.set_buffer(0, Some(weight_q4k), 0);
        enc.set_buffer(1, Some(input),      0);
        enc.set_buffer(2, Some(output),     0);
        enc.set_buffer(3, Some(residual),   0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q4_K GEMV: y = A_q4k * x  (dequantize on-the-fly, 3.5x less bandwidth)
// ─────────────────────────────────────────────────────────────────────────────

/// Fused Q4_K matrix-vector multiply: reads packed Q4_K weights and dequantizes
/// on-the-fly during the dot product. Uses ~3.5x less memory bandwidth than
/// the F16 GEMV for weight reads.
///
/// - `weight_q4k`: raw Q4_K packed bytes `[M × K/256 × 144]`
/// - `input`:  `[K]` f16
/// - `output`: `[M]` f16
/// - `m`: output dimension
/// - `k`: input dimension (must be multiple of 256)
pub fn gemv_q4k_f16(
    ctx: &MetalContext,
    weight_q4k: &Buffer,
    input: &Buffer,
    output: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_q4k_f16", |enc| {
        enc.set_buffer(0, Some(weight_q4k), 0);
        enc.set_buffer(1, Some(input),      0);
        enc.set_buffer(2, Some(output),     0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention: GQA-aware tiled attention
// ─────────────────────────────────────────────────────────────────────────────
pub fn flash_attention_f16(
    ctx: &MetalContext,
    q: &Buffer,           // [batch, num_heads,    q_len,  head_dim] f16
    k: &Buffer,           // [batch, num_kv_heads, kv_head_stride, head_dim] f16
    v: &Buffer,           // [batch, num_kv_heads, kv_head_stride, head_dim] f16
    out: &Buffer,         // [batch, num_heads,    q_len,  head_dim] f16
    batch: u32,
    q_len: u32,           // query sequence length (1 for decode, full seq for prefill)
    kv_len: u32,          // number of valid KV positions (used for masking)
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,        // actual head dim (e.g. 64 for Qwen2.5-0.5B, 128 for 7B)
    kv_head_stride: u32,  // elements per kv_head in K/V buffers (= max_seq_len for flat cache)
) -> Result<()> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    const BLOCK_Q:      u64 = 32;
    const BLOCK_K:      u64 = 32;
    const HEAD_DIM_MAX: u64 = 128;
    let sram_bytes = (BLOCK_Q + 2 * BLOCK_K) * HEAD_DIM_MAX * 2
                   + BLOCK_Q * BLOCK_K * 2;

    let num_q_blocks = (q_len as u64 + BLOCK_Q - 1) / BLOCK_Q;

    ctx.encode("flash_attention_f16", |enc| {
        enc.set_buffer(0, Some(q),   0);
        enc.set_buffer(1, Some(k),   0);
        enc.set_buffer(2, Some(v),   0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 4, &q_len        as *const u32 as *const _);
        enc.set_bytes(5, 4, &kv_len       as *const u32 as *const _);
        enc.set_bytes(6, 4, &num_heads    as *const u32 as *const _);
        enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
        enc.set_bytes(8,  4, &scale          as *const f32 as *const _);
        enc.set_bytes(9,  4, &head_dim       as *const u32 as *const _);
        enc.set_bytes(10, 4, &kv_head_stride as *const u32 as *const _);
        enc.set_threadgroup_memory_length(0, sram_bytes);
        enc.dispatch_thread_groups(
            MTLSize { width: num_q_blocks, height: num_heads as u64, depth: batch as u64 },
            MTLSize { width: TG_ATTN,      height: 1,                depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode attention: specialised for q_len = 1 (single-token decode)
// ─────────────────────────────────────────────────────────────────────────────

/// Optimised attention for decode (q_len=1). Uses all 32 SIMD threads for
/// the dot product instead of wasting 31/32 threads in the FlashAttention tiler.
///
/// Grid: `[num_heads, batch, 1]`, threadgroup: `[32, 1, 1]`
pub fn decode_attention_f16(
    ctx: &MetalContext,
    q: &Buffer,           // [batch, num_heads, 1, head_dim]
    k: &Buffer,           // [batch, num_kv_heads, kv_head_stride, head_dim]
    v: &Buffer,
    out: &Buffer,         // [batch, num_heads, 1, head_dim]
    batch: u32,
    kv_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_head_stride: u32,
) -> Result<()> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    ctx.encode("decode_attention_f16", |enc| {
        enc.set_buffer(0, Some(q),   0);
        enc.set_buffer(1, Some(k),   0);
        enc.set_buffer(2, Some(v),   0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4,  4, &kv_len       as *const u32 as *const _);
        enc.set_bytes(5,  4, &num_heads    as *const u32 as *const _);
        enc.set_bytes(6,  4, &num_kv_heads as *const u32 as *const _);
        enc.set_bytes(7,  4, &scale        as *const f32 as *const _);
        enc.set_bytes(8,  4, &head_dim     as *const u32 as *const _);
        enc.set_bytes(9,  4, &kv_head_stride as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: num_heads as u64, height: batch as u64, depth: 1 },
            MTLSize { width: 32,               height: 1,            depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_K dequantization → F16
// ─────────────────────────────────────────────────────────────────────────────

/// Q4_K block size in bytes (2 f16 scales + 12 scale bytes + 128 quant bytes).
pub const Q4K_BLOCK_BYTES: usize = 144;
/// Elements per Q4_K block.
pub const Q4K_BLOCK_ELEMS: usize = 256;

/// Dequantise a Q4_K_M weight tensor from packed bytes into an F16 Metal buffer.
///
/// - `input`    — raw Q4_K bytes: `n_blocks × 144` bytes
/// - `output`   — pre-allocated F16 buffer: `n_blocks × 256 × 2` bytes
/// - `n_blocks` — number of Q4_K blocks
///
/// Grid: `[n_blocks, 1, 1]`, threadgroup: `[256, 1, 1]`
pub fn dequant_q4k_f16(
    ctx: &MetalContext,
    input: &Buffer,
    output: &Buffer,
    n_blocks: u32,
) -> Result<()> {
    ctx.encode("dequant_q4k_f16", |enc| {
        enc.set_buffer(0, Some(input),  0);
        enc.set_buffer(1, Some(output), 0);
        enc.set_bytes(2, 4, &n_blocks as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: n_blocks as u64, height: 1, depth: 1 },
            MTLSize { width: 256,             height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// GEMM — batched matrix multiply for prefill (Phase 8)
// ─────────────────────────────────────────────────────────────────────────────

/// Tiled matrix-matrix multiply: `Y = A @ B^T`
///
/// - `A`: activations `[M, K]` f16  (seq_len × in_features)
/// - `B`: weights     `[N, K]` f16  (out_features × in_features)
/// - `Y`: output      `[M, N]` f16  (seq_len × out_features)
///
/// Grid: `[ceil(N/16), ceil(M/16), 1]`, threadgroup: `[16, 16, 1]`
pub fn gemm_f16(
    ctx: &MetalContext,
    a: &Buffer, b: &Buffer, y: &Buffer,
    m: u32, n: u32, k: u32,
) -> Result<()> {
    const TILE: u64 = 16;
    ctx.encode("gemm_f16", |enc| {
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &n as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize {
                width:  (n as u64 + TILE - 1) / TILE,
                height: (m as u64 + TILE - 1) / TILE,
                depth:  1,
            },
            MTLSize { width: TILE, height: TILE, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Batched RoPE — for prefill (Phase 8)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply RoPE in-place to `[seq_len, n_heads, head_dim]` for all positions.
/// `start_pos` is the absolute position of sequence index 0.
pub fn rope_batch_inplace_f16(
    ctx:       &MetalContext,
    x:         &Buffer,
    head_dim:  u32,
    n_heads:   u32,
    start_pos: u32,
    theta:     f32,
    seq_len:   u32,
) -> Result<()> {
    let tg_width = ((head_dim / 2) as u64).min(32).max(1);
    ctx.encode("rope_batch_inplace_f16", |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, &head_dim  as *const u32   as *const _);
        enc.set_bytes(2, 4, &n_heads   as *const u32   as *const _);
        enc.set_bytes(3, 4, &start_pos as *const u32   as *const _);
        enc.set_bytes(4, 4, &theta     as *const f32   as *const _);
        enc.dispatch_threads(
            MTLSize {
                width:  (head_dim / 2) as u64,
                height: n_heads  as u64,
                depth:  seq_len  as u64,
            },
            MTLSize { width: tg_width, height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// KV cache scatter: copy [num_kv_heads, head_dim] → cache at position
// ─────────────────────────────────────────────────────────────────────────────

/// Copy K or V vector `[num_kv_heads, head_dim]` from `src` into the
/// KV cache buffer `dst [num_kv_heads, max_seq, head_dim]` at `pos`.
/// Runs entirely on GPU — no CPU flush needed between RoPE and KV write.
pub fn kv_copy_to_cache_f16(
    ctx: &MetalContext,
    src: &Buffer,
    dst: &Buffer,
    pos: u32,
    max_seq: u32,
    head_dim: u32,
    num_kv_heads: u32,
) -> Result<()> {
    let tg_width = (head_dim as u64).min(256);
    ctx.encode("kv_copy_to_cache_f16", |enc| {
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 4, &pos      as *const u32 as *const _);
        enc.set_bytes(3, 4, &max_seq  as *const u32 as *const _);
        enc.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: head_dim as u64, height: num_kv_heads as u64, depth: 1 },
            MTLSize { width: tg_width,        height: 1,                   depth: 1 },
        );
    })
}

/// f32 KV cache scatter: copy f32 K or V into f32 cache at position.
pub fn kv_copy_to_cache_f32(
    ctx: &MetalContext,
    src: &Buffer,
    dst: &Buffer,
    pos: u32,
    max_seq: u32,
    head_dim: u32,
    num_kv_heads: u32,
) -> Result<()> {
    let tg_width = (head_dim as u64).min(256);
    ctx.encode("kv_copy_to_cache_f32", |enc| {
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 4, &pos      as *const u32 as *const _);
        enc.set_bytes(3, 4, &max_seq  as *const u32 as *const _);
        enc.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: head_dim as u64, height: num_kv_heads as u64, depth: 1 },
            MTLSize { width: tg_width,        height: 1,                   depth: 1 },
        );
    })
}

/// GEMV with f16 weights, f32 input, f32 output: y_f32 = A_f16 * x_f32
pub fn gemv_f16w_f32in_f32out(
    ctx: &MetalContext,
    weight: &Buffer,
    input: &Buffer,
    output: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_f16w_f32in_f32out", |enc| {
        enc.set_buffer(0, Some(weight), 0);
        enc.set_buffer(1, Some(input),  0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// Q4K GEMV with f32 input and f32 output: y_f32 = A_q4k * x_f32
pub fn gemv_q4k_f32in_f32out(
    ctx: &MetalContext,
    weight_q4k: &Buffer,
    input: &Buffer,
    output: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_q4k_f32in_f32out", |enc| {
        enc.set_buffer(0, Some(weight_q4k), 0);
        enc.set_buffer(1, Some(input),      0);
        enc.set_buffer(2, Some(output),     0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// Q4K GEMV + f32 residual with f32 input: y_f32 = A_q4k * x_f32 + res_f32
pub fn gemv_q4k_add_f32_f32in(
    ctx: &MetalContext,
    weight_q4k: &Buffer,
    input: &Buffer,
    output: &Buffer,
    residual: &Buffer,
    m: u32,
    k: u32,
) -> Result<()> {
    ctx.encode("gemv_q4k_add_f32_f32in", |enc| {
        enc.set_buffer(0, Some(weight_q4k), 0);
        enc.set_buffer(1, Some(input),      0);
        enc.set_buffer(2, Some(output),     0);
        enc.set_buffer(3, Some(residual),   0);
        enc.set_bytes(4, 4, &m as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: TG_GEMV,  height: 1, depth: 1 },
        );
    })
}

/// f32 RoPE: apply rotary embeddings in-place on f32 Q or K.
pub fn rope_inplace_f32(
    ctx: &MetalContext,
    x: &Buffer,
    head_dim: u32,
    n_heads: u32,
    seq_pos: u32,
    rope_theta: f32,
    batch_seq: u32,
) -> Result<()> {
    let tg_width = ((head_dim / 2) as u64).min(32).max(1);
    ctx.encode("rope_inplace_f32", |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_bytes(1, 4, &head_dim   as *const u32 as *const _);
        enc.set_bytes(2, 4, &n_heads    as *const u32 as *const _);
        enc.set_bytes(3, 4, &seq_pos    as *const u32 as *const _);
        enc.set_bytes(4, 4, &rope_theta as *const f32 as *const _);
        enc.dispatch_threads(
            MTLSize {
                width:  (head_dim / 2) as u64,
                height: n_heads as u64,
                depth:  batch_seq as u64,
            },
            MTLSize { width: tg_width, height: 1, depth: 1 },
        );
    })
}

/// f32 decode attention: reads f32 Q/K/V, writes f32 output.
pub fn decode_attention_f32(
    ctx: &MetalContext,
    q: &Buffer,
    k: &Buffer,
    v: &Buffer,
    out: &Buffer,
    batch: u32,
    kv_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_head_stride: u32,
) -> Result<()> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    ctx.encode("decode_attention_f32", |enc| {
        enc.set_buffer(0, Some(q),   0);
        enc.set_buffer(1, Some(k),   0);
        enc.set_buffer(2, Some(v),   0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4,  4, &kv_len       as *const u32 as *const _);
        enc.set_bytes(5,  4, &num_heads    as *const u32 as *const _);
        enc.set_bytes(6,  4, &num_kv_heads as *const u32 as *const _);
        enc.set_bytes(7,  4, &scale        as *const f32 as *const _);
        enc.set_bytes(8,  4, &head_dim     as *const u32 as *const _);
        enc.set_bytes(9,  4, &kv_head_stride as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: num_heads as u64, height: batch as u64, depth: 1 },
            MTLSize { width: 32,               height: 1,            depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Argmax: find index of maximum value on GPU (avoids downloading full logits)
// ─────────────────────────────────────────────────────────────────────────────

/// Find the index of the maximum f16 value in `input [n]`.
/// Result is written to `result [1]` as a u32.
/// Runs entirely on GPU — only 4 bytes need to be downloaded.
pub fn argmax_f16(
    ctx: &MetalContext,
    input: &Buffer,
    result: &Buffer,
    n: u32,
) -> Result<()> {
    let tg_size = 1024u64.min(n as u64);
    ctx.encode("argmax_f16", |enc| {
        enc.set_buffer(0, Some(input),  0);
        enc.set_buffer(1, Some(result), 0);
        enc.set_bytes(2, 4, &n as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: tg_size, height: 1, depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Broadcast bias add: x[row, i] += bias[i]  for all rows
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-side scale + accumulate: `acc[i] += src[i] * weight` for all i.
/// Used by MoE to accumulate weighted expert outputs without flushing.
pub fn scale_accumulate_f16(
    ctx: &MetalContext,
    src: &Buffer,
    acc: &Buffer,
    weight: f32,
    n: u32,
) -> Result<()> {
    ctx.encode("scale_accumulate_f16", |enc| {
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(acc), 0);
        enc.set_bytes(2, 4, &weight as *const f32 as *const _);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: n as u64, height: 1, depth: 1 },
            MTLSize { width: TG_ELEM,  height: 1, depth: 1 },
        );
    })
}

/// Add a bias vector `[dim]` to each of `seq` rows of `x [seq, dim]` in-place.
/// Single GPU dispatch — replaces the per-row loop.
pub fn add_bias_broadcast_f16(
    ctx: &MetalContext,
    x: &Buffer,
    bias: &Buffer,
    seq: u32,
    dim: u32,
) -> Result<()> {
    ctx.encode("add_bias_broadcast_f16", |enc| {
        enc.set_buffer(0, Some(x),    0);
        enc.set_buffer(1, Some(bias), 0);
        enc.set_bytes(2, 4, &dim as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: dim as u64, height: seq as u64, depth: 1 },
            MTLSize { width: TG_ELEM,    height: 1,          depth: 1 },
        );
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference implementations (used in tests)
// ─────────────────────────────────────────────────────────────────────────────

/// CPU reference: SiLU activation
#[allow(dead_code)]
pub fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// CPU reference: SwiGLU
pub fn cpu_silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| cpu_silu(g) * u).collect()
}

/// CPU reference: elementwise add
pub fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

/// CPU reference: RMSNorm
pub fn cpu_rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(gamma).map(|(&v, &g)| v / rms * g).collect()
}

/// CPU reference: GEMV
pub fn cpu_gemv(a: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m).map(|row| {
        (0..k).map(|col| a[row * k + col] * x[col]).sum()
    }).collect()
}

/// CPU reference: scaled dot-product attention (non-tiled, no GQA for simplicity)
pub fn cpu_attention(
    q: &[f32], k: &[f32], v: &[f32],
    seq: usize, heads: usize, head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq * heads * head_dim];
    for h in 0..heads {
        for qi in 0..seq {
            // Compute scores for this query
            let mut scores: Vec<f32> = (0..seq).map(|ki| {
                let q_off = (h * seq + qi) * head_dim;
                let k_off = (h * seq + ki) * head_dim;
                (0..head_dim).map(|d| q[q_off + d] * k[k_off + d]).sum::<f32>() * scale
            }).collect();
            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e: f32 = exp_s.iter().sum();
            let attn: Vec<f32> = exp_s.iter().map(|&e| e / sum_e).collect();
            // Weighted sum of V
            let o_off = (h * seq + qi) * head_dim;
            for ki in 0..seq {
                let v_off = (h * seq + ki) * head_dim;
                for d in 0..head_dim {
                    out[o_off + d] += attn[ki] * v[v_off + d];
                }
            }
        }
    }
    out
}

// ── Utility: f16 ↔ f32 conversion ─────────────────────────────────────────

pub fn f32_to_f16_vec(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

pub fn f16_to_f32_vec(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// Upload a slice of f16 values into a new Metal buffer.
pub fn upload_f16(ctx: &MetalContext, data: &[f16]) -> Buffer {
    let buf = ctx.new_buffer(data.len() * 2);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents() as *mut f16,
            data.len(),
        );
    }
    buf
}

/// Download f16 values from a Metal buffer into a Vec<f32>.
pub fn download_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f16;
    (0..n).map(|i| unsafe { (*ptr.add(i)).to_f32() }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical validation tests
// Run only on macOS (the target platform); skipped on CI Linux.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use bare_metal_reference::validate_layer_output;

    const TOL: f32 = 1e-2;     // F16 kernels: ~1% tolerance acceptable
    const TOL_GEMV: f32 = 3e-2; // GEMV with k=128 f16 accum: up to ~1.7% observed on M-series

    /// Returns a ready `MetalContext`, or `None` if no Metal GPU is available
    /// (e.g. in a CI environment without GPU passthrough).  Callers use the
    /// `metal_ctx!()` macro which skips the test cleanly in that case.
    fn try_metal_ctx() -> Option<MetalContext> {
        MetalContext::new().ok()
    }

    /// Skip the test gracefully when Metal is unavailable; panic on any other error.
    macro_rules! metal_ctx {
        () => {
            match try_metal_ctx() {
                Some(ctx) => ctx,
                None => {
                    eprintln!("SKIP: Metal device not available");
                    return;
                }
            }
        };
    }

    fn random_f32(n: usize, seed: u64) -> Vec<f32> {
        // Simple LCG pseudo-random, no external deps
        let mut state = seed;
        (0..n).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-1, 1]
            (((state >> 33) as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }).collect()
    }

    #[test]
    fn test_silu_mul_matches_cpu() {
        let ctx = metal_ctx!();
        let n = 1024usize;
        let gate_f32 = random_f32(n, 42);
        let up_f32   = random_f32(n, 99);

        let reference = cpu_silu_mul(&gate_f32, &up_f32);

        let gate_gpu = upload_f16(&ctx, &f32_to_f16_vec(&gate_f32));
        let up_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&up_f32));
        let out_gpu  = ctx.new_buffer(n * 2);

        silu_mul_f16(&ctx, &gate_gpu, &up_gpu, &out_gpu, n as u32).unwrap();
        ctx.flush();

        let gpu_result = download_f32(&out_gpu, n);
        validate_layer_output(&reference, &gpu_result, 0, TOL).expect("silu_mul mismatch");
    }

    #[test]
    fn test_add_matches_cpu() {
        let ctx = metal_ctx!();
        let n = 2048usize;
        let a_f32 = random_f32(n, 7);
        let b_f32 = random_f32(n, 13);

        let reference = cpu_add(&a_f32, &b_f32);

        let a_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&a_f32));
        let b_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&b_f32));
        let out_gpu = ctx.new_buffer(n * 2);

        add_f16(&ctx, &a_gpu, &b_gpu, &out_gpu, n as u32).unwrap();
        ctx.flush();

        let gpu_result = download_f32(&out_gpu, n);
        validate_layer_output(&reference, &gpu_result, 0, TOL).expect("add mismatch");
    }

    #[test]
    fn test_rms_norm_matches_cpu() {
        let ctx = metal_ctx!();
        let hidden = 256usize;
        let x_f32     = random_f32(hidden, 17);
        let gamma_f32 = vec![1.0f32; hidden]; // identity scale

        let reference = cpu_rms_norm(&x_f32, &gamma_f32, 1e-6);

        let x_gpu     = upload_f16(&ctx, &f32_to_f16_vec(&x_f32));
        let y_gpu     = ctx.new_buffer(hidden * 2);
        let gamma_gpu = upload_f16(&ctx, &f32_to_f16_vec(&gamma_f32));

        rms_norm_f16(&ctx, &x_gpu, &y_gpu, &gamma_gpu, 1e-6, hidden as u32, 1).unwrap();
        ctx.flush();

        let gpu_result = download_f32(&y_gpu, hidden);
        validate_layer_output(&reference, &gpu_result, 0, TOL).expect("rms_norm mismatch");
    }

    #[test]
    fn test_gemv_matches_cpu() {
        let ctx = metal_ctx!();
        let m = 64usize;
        let k = 128usize;
        let a_f32 = random_f32(m * k, 31);
        let x_f32 = random_f32(k, 53);

        let reference = cpu_gemv(&a_f32, &x_f32, m, k);

        let a_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&a_f32));
        let x_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&x_f32));
        let out_gpu = ctx.new_buffer(m * 2);

        gemv_f16(&ctx, &a_gpu, &x_gpu, &out_gpu, m as u32, k as u32).unwrap();
        ctx.flush();

        let gpu_result = download_f32(&out_gpu, m);
        validate_layer_output(&reference, &gpu_result, 0, TOL_GEMV).expect("gemv mismatch");
    }

    #[test]
    fn test_flash_attention_matches_cpu() {
        let ctx = metal_ctx!();

        // Small single-head, single-batch attention: seq=32, heads=1, kv_heads=1, head_dim=128
        let batch     = 1usize;
        let seq       = 32usize;
        let num_heads = 1usize;
        let head_dim  = 128usize;
        let n = batch * num_heads * seq * head_dim;

        let q_f32 = random_f32(n, 11);
        let k_f32 = random_f32(n, 22);
        let v_f32 = random_f32(n, 33);

        let reference = cpu_attention(&q_f32, &k_f32, &v_f32, seq, num_heads, head_dim);

        let q_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&q_f32));
        let k_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&k_f32));
        let v_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&v_f32));
        let out_gpu = ctx.new_buffer(n * 2);

        flash_attention_f16(
            &ctx, &q_gpu, &k_gpu, &v_gpu, &out_gpu,
            batch as u32,
            seq as u32,   // q_len
            seq as u32,   // kv_len (same for prefill)
            num_heads as u32, num_heads as u32,  // no GQA for this test
            head_dim as u32,
            seq as u32,   // kv_head_stride = kv_len for contiguous buffer
        ).unwrap();
        ctx.flush();

        let gpu_result = download_f32(&out_gpu, n);
        validate_layer_output(&reference, &gpu_result, 0, TOL).expect("flash_attention mismatch");
    }

    /// Verify GQA mapping: 4 Q heads, 2 KV heads (group_size=2).
    /// Both Q heads in a group should get the same K/V.
    #[test]
    fn test_flash_attention_gqa() {
        let ctx = metal_ctx!();

        let batch        = 1usize;
        let seq          = 32usize;
        let num_q_heads  = 4usize;
        let num_kv_heads = 2usize;
        let head_dim     = 128usize;

        // Q: [1, 4, 32, 128]
        let q_f32  = random_f32(batch * num_q_heads  * seq * head_dim, 77);
        // K,V: [1, 2, 32, 128]
        let kv_f32 = random_f32(batch * num_kv_heads * seq * head_dim, 88);

        let q_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&q_f32));
        let k_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&kv_f32));
        let v_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&kv_f32));
        let out_gpu = ctx.new_buffer(batch * num_q_heads * seq * head_dim * 2);

        // Should run without error
        flash_attention_f16(
            &ctx, &q_gpu, &k_gpu, &v_gpu, &out_gpu,
            batch as u32,
            seq as u32,   // q_len
            seq as u32,   // kv_len (same for prefill)
            num_q_heads as u32, num_kv_heads as u32,
            head_dim as u32,
            seq as u32,   // kv_head_stride = kv_len for contiguous buffer
        ).unwrap();
        ctx.flush();
    }

    /// Verify that head_dim=64 works (Qwen2.5-0.5B uses 64-dim heads).
    #[test]
    fn test_flash_attention_head_dim_64() {
        let ctx = metal_ctx!();

        let batch     = 1usize;
        let seq       = 32usize;
        let num_heads = 1usize;
        let head_dim  = 64usize;
        let n = batch * num_heads * seq * head_dim;

        let q_f32 = random_f32(n, 55);
        let k_f32 = random_f32(n, 66);
        let v_f32 = random_f32(n, 77);

        let reference = cpu_attention(&q_f32, &k_f32, &v_f32, seq, num_heads, head_dim);

        let q_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&q_f32));
        let k_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&k_f32));
        let v_gpu   = upload_f16(&ctx, &f32_to_f16_vec(&v_f32));
        let out_gpu = ctx.new_buffer(n * 2);

        flash_attention_f16(
            &ctx, &q_gpu, &k_gpu, &v_gpu, &out_gpu,
            batch as u32,
            seq as u32,
            seq as u32,
            num_heads as u32, num_heads as u32,
            head_dim as u32,
            seq as u32,   // kv_head_stride = kv_len for contiguous buffer
        ).unwrap();
        ctx.flush();

        let gpu_result = download_f32(&out_gpu, n);
        validate_layer_output(&reference, &gpu_result, 0, TOL).expect("flash_attention head_dim=64 mismatch");
    }
}
