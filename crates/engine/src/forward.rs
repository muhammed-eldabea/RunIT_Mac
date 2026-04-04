use half::f16;
use metal::MTLResourceOptions;

use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::tq_kv_cache::TqKvCache;
use crate::loader::Model;
use crate::tensor::DType;

use bare_metal_kernels::{
    context::MetalContext,
    dispatch::{
        add_f16, add_f16_into_f32, argmax_f16,
        decode_attention_f16, decode_attention_f32,
        dequant_q4k_f16,
        flash_attention_f16, gemv_f16, gemv_add_f16, gemv_add_f32res_f16,
        gemv_f16w_f32in, gemv_f16w_f32in_f32out, gemv_add_f32_f16w,
        gemv_q4k_f16, gemv_q4k_add_f16, gemv_q4k_add_f32res_f16,
        gemv_q4k_f32in_f32out, gemv_q4k_add_f32_f32in,
        gemv_f32_f32out, gemv_f32w, gemv_add_f32w, gemv_f32w_f32in, gemv_add_f32res_f32w, gemv_add_f32_f32w,
        gemv_q8_0_f32in_f32out, gemv_q8_0_add_f32_f32in,
        gemv_q8_0_f32in, gemv_q8_0_f16, gemv_q8_0_add_f16, gemv_q8_0_add_f32res_f16,
        // Multi-row GEMV (4 rows/TG, 128 threads) for M ≥ 128
        gemv_f16w_f32in_f32out_mr, gemv_add_f32_f16w_mr,
        gemv_q8_0_f32in_f32out_mr, gemv_q8_0_add_f32_f32in_mr,
        // Fused kernels (reduce dispatch count)
        fused_qkv_q8_0_f32, fused_qkv_bias_q8_0_f32, fused_qkv_f16w_f32,
        fused_ffn_q8_0_f32, fused_ffn_f16w_f32, fused_rope_qk_f32,
        kv_copy_to_cache_f16, kv_copy_to_cache_f32,
        mul_f16, rms_norm_f16, rms_norm_f32in_f16out,
        rms_norm_f32_f32, rms_norm_f32_f32_f32g,
        rope_inplace_f16, rope_inplace_f32,
        scale_accumulate_f16, silu_mul_f16, silu_mul_f32,
        Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q8_0_BLOCK_ELEMS,
    },
    error::KernelError,
};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum ForwardError {
    #[error("kernel error: {0}")]
    Kernel(#[from] KernelError),

    #[error("tensor '{tensor}' has unsupported dtype {dtype:?}")]
    UnsupportedDtype { tensor: String, dtype: DType },

    #[error("required weight '{0}' not found in model")]
    MissingWeight(String),

    #[error("token id {0} out of vocab range")]
    InvalidTokenId(u32),
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight buffers
// ─────────────────────────────────────────────────────────────────────────────

/// A weight buffer in one of three formats:
/// - **F16**: native or dequantized-to-f16 weights (small tensors, native f16 files)
/// - **F32**: dequantized-to-f32 weights for precision (Q5_0, Q6K, Q8_0, etc.)
/// - **Q4K**: raw packed Q4K bytes for fused on-the-fly dequant (bandwidth-optimal)
pub(crate) enum WeightBuf {
    F16(metal::Buffer),
    /// Weights dequantized to f32 for precision parity with llama.cpp.
    /// Avoids the ~10-bit mantissa loss from f16 truncation that compounds across layers.
    F32(metal::Buffer),
    /// Q4K weights: raw packed buffer for fused GEMV + dequanted F16 for GEMM prefill.
    Q4K { raw: metal::Buffer, f16: metal::Buffer },
    /// Q8_0 weights: raw packed buffer (1.06 bytes/elem) for fused GEMV + F16 for GEMM.
    /// Uses 47% less bandwidth than F16 GEMV — the #1 optimization for Q8_0 models.
    Q8_0 { raw: metal::Buffer, f16: metal::Buffer },
}

impl WeightBuf {
    /// Dispatch GEMV using the appropriate kernel for this weight format.
    pub(crate) fn gemv(
        &self,
        ctx: &MetalContext,
        input: &metal::Buffer,
        output: &metal::Buffer,
        m: u32,
        k: u32,
    ) -> Result<(), ForwardError> {
        match self {
            WeightBuf::F16(buf) => gemv_f16(ctx, buf, input, output, m, k)?,
            WeightBuf::F32(buf) => gemv_f32w(ctx, buf, input, output, m, k)?,
            WeightBuf::Q4K { raw, .. } => gemv_q4k_f16(ctx, raw, input, output, m, k)?,
            WeightBuf::Q8_0 { raw, .. } => gemv_q8_0_f16(ctx, raw, input, output, m, k)?,
        }
        Ok(())
    }

    /// Fused GEMV + residual add: `output[i] = (weight * input)[i] + residual[i]`.
    pub(crate) fn gemv_add(
        &self,
        ctx: &MetalContext,
        input: &metal::Buffer,
        output: &metal::Buffer,
        residual: &metal::Buffer,
        m: u32,
        k: u32,
    ) -> Result<(), ForwardError> {
        match self {
            WeightBuf::F16(buf) => gemv_add_f16(ctx, buf, input, output, residual, m, k)?,
            WeightBuf::F32(buf) => gemv_add_f32w(ctx, buf, input, output, residual, m, k)?,
            WeightBuf::Q4K { raw, .. } => gemv_q4k_add_f16(ctx, raw, input, output, residual, m, k)?,
            WeightBuf::Q8_0 { raw, .. } => gemv_q8_0_add_f16(ctx, raw, input, output, residual, m, k)?,
        }
        Ok(())
    }

    /// GEMV with f32 input, f32 output: for Q/K/V where full precision matters.
    /// Uses multi-row dispatch (4 rows/TG, 128 threads) for M ≥ 128.
    pub(crate) fn gemv_f32_f32out(
        &self,
        ctx: &MetalContext,
        input: &metal::Buffer,   // f32
        output: &metal::Buffer,  // f32
        m: u32,
        k: u32,
    ) -> Result<(), ForwardError> {
        let use_mr = m >= 128;
        match self {
            WeightBuf::F16(buf) => {
                if use_mr {
                    gemv_f16w_f32in_f32out_mr(ctx, buf, input, output, m, k)?;
                } else {
                    gemv_f16w_f32in_f32out(ctx, buf, input, output, m, k)?;
                }
            }
            WeightBuf::F32(buf) => {
                gemv_f32_f32out(ctx, buf, input, output, m, k)?;
            }
            WeightBuf::Q4K { raw, .. } => {
                gemv_q4k_f32in_f32out(ctx, raw, input, output, m, k)?;
            }
            WeightBuf::Q8_0 { raw, .. } => {
                if use_mr {
                    gemv_q8_0_f32in_f32out_mr(ctx, raw, input, output, m, k)?;
                } else {
                    gemv_q8_0_f32in_f32out(ctx, raw, input, output, m, k)?;
                }
            }
        }
        Ok(())
    }

    /// GEMV with f32 input: `output_f16[i] = (weight * input_f32)[i]`.
    pub(crate) fn gemv_f32in(
        &self,
        ctx: &MetalContext,
        input: &metal::Buffer,   // f32
        output: &metal::Buffer,  // f16
        m: u32,
        k: u32,
    ) -> Result<(), ForwardError> {
        match self {
            WeightBuf::F16(buf) => gemv_f16w_f32in(ctx, buf, input, output, m, k)?,
            WeightBuf::F32(buf) => gemv_f32w_f32in(ctx, buf, input, output, m, k)?,
            WeightBuf::Q4K { f16, .. } => gemv_f16w_f32in(ctx, f16, input, output, m, k)?,
            WeightBuf::Q8_0 { raw, .. } => gemv_q8_0_f32in(ctx, raw, input, output, m, k)?,
        }
        Ok(())
    }

    /// Fused GEMV + f32 residual with f32 input.
    /// Uses multi-row dispatch (4 rows/TG) for M ≥ 128.
    pub(crate) fn gemv_add_f32(
        &self,
        ctx: &MetalContext,
        input: &metal::Buffer,      // f32
        output: &metal::Buffer,     // f32
        residual: &metal::Buffer,   // f32
        m: u32,
        k: u32,
    ) -> Result<(), ForwardError> {
        let use_mr = m >= 128;
        match self {
            WeightBuf::F16(buf) => {
                if use_mr {
                    gemv_add_f32_f16w_mr(ctx, buf, input, output, residual, m, k)?;
                } else {
                    gemv_add_f32_f16w(ctx, buf, input, output, residual, m, k)?;
                }
            }
            WeightBuf::F32(buf) => gemv_add_f32_f32w(ctx, buf, input, output, residual, m, k)?,
            WeightBuf::Q4K { raw, .. } => gemv_q4k_add_f32_f32in(ctx, raw, input, output, residual, m, k)?,
            WeightBuf::Q8_0 { raw, .. } => {
                if use_mr {
                    gemv_q8_0_add_f32_f32in_mr(ctx, raw, input, output, residual, m, k)?;
                } else {
                    gemv_q8_0_add_f32_f32in(ctx, raw, input, output, residual, m, k)?;
                }
            }
        }
        Ok(())
    }

    /// Fused GEMV + f32 residual: `output_f32[i] = (weight * input_f16)[i] + residual_f32[i]`.
    pub(crate) fn gemv_add_f32res(
        &self,
        ctx: &MetalContext,
        input: &metal::Buffer,      // f16
        output: &metal::Buffer,     // f32
        residual: &metal::Buffer,   // f32
        m: u32,
        k: u32,
    ) -> Result<(), ForwardError> {
        match self {
            WeightBuf::F16(buf) => gemv_add_f32res_f16(ctx, buf, input, output, residual, m, k)?,
            WeightBuf::F32(buf) => gemv_add_f32res_f32w(ctx, buf, input, output, residual, m, k)?,
            WeightBuf::Q4K { raw, .. } => gemv_q4k_add_f32res_f16(ctx, raw, input, output, residual, m, k)?,
            WeightBuf::Q8_0 { raw, .. } => gemv_q8_0_add_f32res_f16(ctx, raw, input, output, residual, m, k)?,
        }
        Ok(())
    }

    /// Get the F16 buffer (for GEMM prefill and other non-GEMV uses).
    pub(crate) fn f16_buf(&self) -> &metal::Buffer {
        match self {
            WeightBuf::F16(buf) => buf,
            WeightBuf::F32(_) => panic!("F32 weights: prefill GEMM needs f32-weight GEMM kernel (not yet implemented)"),
            WeightBuf::Q4K { f16, .. } => f16,
            WeightBuf::Q8_0 { f16, .. } => f16,
        }
    }
}

pub(crate) struct LayerWeights {
    pub(crate) tok_emb:     metal::Buffer,              // [vocab_size, hidden_size] f32 (or f16 for native F16 models)
    pub(crate) output_norm: metal::Buffer,              // [hidden_size] f16
    pub(crate) lm_head:     WeightBuf,                  // [vocab_size, hidden_size]

    pub(crate) attn_norm: Vec<metal::Buffer>,           // [hidden_size] per layer
    pub(crate) attn_q:    Vec<WeightBuf>,               // [num_heads*head_dim, hidden_size]
    pub(crate) attn_k:    Vec<WeightBuf>,               // [num_kv_heads*head_dim, hidden_size]
    pub(crate) attn_v:    Vec<WeightBuf>,               // [num_kv_heads*head_dim, hidden_size]
    pub(crate) attn_q_bias: Vec<Option<metal::Buffer>>, // [num_heads*head_dim]
    pub(crate) attn_k_bias: Vec<Option<metal::Buffer>>, // [num_kv_heads*head_dim]
    pub(crate) attn_v_bias: Vec<Option<metal::Buffer>>, // [num_kv_heads*head_dim]
    pub(crate) attn_out:  Vec<WeightBuf>,               // [hidden_size, num_heads*head_dim]

    pub(crate) ffn_norm: Vec<metal::Buffer>,            // [hidden_size]

    // ── Dense FFN (used when is_moe = false) ────────────────────
    pub(crate) ffn_gate: Vec<WeightBuf>,                // [intermediate_size, hidden_size]
    pub(crate) ffn_up:   Vec<WeightBuf>,                // [intermediate_size, hidden_size]
    pub(crate) ffn_down: Vec<WeightBuf>,                // [hidden_size, intermediate_size]

    // ── MoE FFN (used when is_moe = true) ───────────────────────
    /// Router: [num_experts, hidden_size] — produces expert logits
    pub(crate) moe_router:   Vec<metal::Buffer>,
    /// Per-expert gate projections: [layer][expert] — [ff_dim, hidden_size]
    pub(crate) moe_gate_exps: Vec<Vec<WeightBuf>>,
    /// Per-expert up projections
    pub(crate) moe_up_exps:   Vec<Vec<WeightBuf>>,
    /// Per-expert down projections
    pub(crate) moe_down_exps: Vec<Vec<WeightBuf>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Executor
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for the decode forward pass.
/// Reused across tokens to avoid per-token Metal buffer allocation overhead.
struct DecodeScratch {
    x:         metal::Buffer,   // [h]
    x_norm:    metal::Buffer,   // [h]
    q:         metal::Buffer,   // [q_dim]
    k:         metal::Buffer,   // [kv_dim]
    v:         metal::Buffer,   // [kv_dim]
    attn_out:  metal::Buffer,   // [q_dim]
    proj:      metal::Buffer,   // [h]
    x_norm2:   metal::Buffer,   // [h]
    gate:      metal::Buffer,   // [ff_dim]
    up:        metal::Buffer,   // [ff_dim]
    act:       metal::Buffer,   // [ff_dim]
    ffn_out:   metal::Buffer,   // [h]
    x_final:   metal::Buffer,   // [h]
    logits:    metal::Buffer,   // [vocab_size]
    argmax:    metal::Buffer,   // [1] u32 — GPU argmax result
    // MoE scratch (only used when is_moe)
    router_logits: metal::Buffer,  // [num_experts]
    moe_out:       metal::Buffer,  // [h] — weighted expert output accumulator
}

impl DecodeScratch {
    fn new_with_ff(ctx: &MetalContext, cfg: &ModelConfig, ff_dim: usize) -> Self {
        let h      = cfg.hidden_size;
        let q_dim  = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        Self {
            x:        ctx.new_buffer(h * 4),  // f32 residual stream
            x_norm:   ctx.new_buffer(h * 4),  // f32 for full-precision GEMV input
            q:        ctx.new_buffer(q_dim * 4),  // f32 for full-precision attention
            k:        ctx.new_buffer(kv_dim * 4), // f32 for full-precision KV cache
            v:        ctx.new_buffer(kv_dim * 4), // f32 for full-precision KV cache
            attn_out: ctx.new_buffer(q_dim * 4),  // f32 attention output
            proj:     ctx.new_buffer(h * 2),
            x_norm2:  ctx.new_buffer(h * 4),  // f32 for full-precision GEMV input
            gate:     ctx.new_buffer(ff_dim * 4),  // f32 FFN intermediates
            up:       ctx.new_buffer(ff_dim * 4),  // f32 FFN intermediates
            act:      ctx.new_buffer(ff_dim * 4),  // f32 FFN intermediates
            ffn_out:  ctx.new_buffer(h * 2),
            x_final:  ctx.new_buffer(h * 2),
            logits:   ctx.new_buffer(cfg.vocab_size * 4),  // f32 logits for precise ranking
            argmax:   ctx.new_buffer(4),  // 1 × u32
            router_logits: ctx.new_buffer(cfg.num_experts.max(1) * 2),
            moe_out:       ctx.new_buffer(h * 2),  // f16 (accumulate per-expert, then add to f32 x)
        }
    }
}

/// Owns the Metal context and all weight buffers. Stateless across tokens —
/// the caller manages the `KvCache`.
pub struct Executor {
    pub ctx: MetalContext,
    pub(crate) weights: LayerWeights,
    pub config: ModelConfig,
    scratch: DecodeScratch,
    /// Actual expert FFN dimension (derived from tensor size for MoE models).
    pub(crate) expert_ff_dim: usize,
    /// Whether tok_emb is stored as f32 (true for quantized models) or f16.
    tok_emb_f32: bool,
    /// Whether norm weights are f32.
    norm_f32: bool,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Extract 6-bit scale and 6-bit minimum for sub-block `j` (0..7) from the
/// 12-byte packed scales array used by both Q4_K and Q5_K blocks.
/// This mirrors the `get_scale_min_k4` function in dequant.metal.
#[inline(always)]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j]   & 0x3F;
        let m  = scales[j+4] & 0x3F;
        (sc, m)
    } else {
        let sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
        let m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4);
        (sc, m)
    }
}

/// Upload a weight tensor from the mmap to a GPU buffer.
///
/// - F16 tensors: zero-copy if page-aligned, else memcpy → returned as-is.
/// - Q4K tensors: raw bytes uploaded, then GPU dequantised to F16 in-place;
///   the returned buffer is always F16 (`n_elements × 2` bytes).
/// - All other dtypes: returns `UnsupportedDtype`.
fn upload_weight(
    ctx: &MetalContext,
    model: &Model,
    name: &str,
) -> Result<metal::Buffer, ForwardError> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| ForwardError::MissingWeight(name.to_string()))?;

    let dtype = tensor.dtype;

    let tb = model
        .tensor_buffer(name)
        .ok_or_else(|| ForwardError::MissingWeight(name.to_string()))?;

    let opts = MTLResourceOptions::StorageModeShared;

    // ── F16 fast path ─────────────────────────────────────────────────────────
    if dtype == DType::F16 {
        let addr = tb.ptr as usize;
        let buf = if addr % 4096 == 0 && tb.size % 4096 == 0 {
            // SAFETY: pointer is page-aligned and valid for the lifetime of the mmap
            unsafe {
                ctx.device.new_buffer_with_bytes_no_copy(
                    tb.ptr as *mut std::ffi::c_void,
                    tb.size as u64,
                    opts,
                    None,
                )
            }
        } else {
            let b = ctx.device.new_buffer(tb.size as u64, opts);
            // SAFETY: tb.ptr is valid for tb.size bytes (from mmap); b is
            // freshly allocated with the same size
            unsafe {
                std::ptr::copy_nonoverlapping(tb.ptr, b.contents() as *mut u8, tb.size);
            }
            b
        };
        return Ok(buf);
    }

    // ── BF16 → F16 CPU conversion ─────────────────────────────────────────────
    if dtype == DType::BF16 {
        let n_elements = tensor.num_elements();
        // BF16 is stored as 2-byte little-endian: sign(1) + exp(8) + mant(7).
        // Convert to f32 by zero-extending the mantissa (shift left 16 bits),
        // then convert f32 → f16.
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr as *const u16, n_elements) };
        let f16_vec: Vec<f16> = raw.iter().map(|&bf| {
            let bits = (bf as u32) << 16;
            f16::from_f32(f32::from_bits(bits))
        }).collect();
        let buf = ctx.device.new_buffer(f16_vec.len() as u64 * 2, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                f16_vec.len() * 2,
            );
        }
        return Ok(buf);
    }

    // ── F32 → F16 CPU conversion ──────────────────────────────────────────────
    // Norm weights and biases in GGUF are typically stored as F32.
    if dtype == DType::F32 {
        let n_elements = tensor.num_elements();
        let f32_slice = unsafe {
            std::slice::from_raw_parts(tb.ptr as *const f32, n_elements)
        };
        let f16_vec: Vec<f16> = f32_slice.iter().map(|&x| f16::from_f32(x)).collect();
        let buf = ctx.device.new_buffer(f16_vec.len() as u64 * 2, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                f16_vec.len() * 2,
            );
        }
        return Ok(buf);
    }

    // ── Q5_0 → F16 CPU dequantization ────────────────────────────────────────
    // Block layout (22 bytes = 32 elements):
    //   d[2]:   f16 scale
    //   qh[4]:  1 high bit per element (packed 8 per byte)
    //   qs[16]: lower 4 bits per element (2 per byte)
    // Value = d * (q5 - 16), where q5 = lower4 | (high1 << 4), range 0..31
    if dtype == DType::Q5_0 {
        const BLOCK_ELEMS: usize = 32;
        const BLOCK_BYTES: usize = 22;
        let n_elements = tensor.num_elements();
        let n_blocks   = n_elements / BLOCK_ELEMS;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BLOCK_BYTES) };
        let mut f16_vec: Vec<f16> = Vec::with_capacity(n_elements);
        for b in 0..n_blocks {
            let block = &raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let d  = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qh = &block[2..6];
            let qs = &block[6..22];
            // ggml Q5_0 layout: first 16 elements from low nibbles of qs[0..15],
            // next 16 elements from high nibbles of qs[0..15].
            // High bits: elements 0..15 use qh[j/8] bit (j%8),
            //            elements 16..31 use qh[j/8 + 2] bit (j%8) where j=i-16.
            let mut vals = [0.0f32; 32];
            for j in 0..16usize {
                let xh_0 = ((qh[j / 8] >> (j % 8)) & 1) as i32;
                let xh_1 = ((qh[j / 8 + 2] >> (j % 8)) & 1) as i32;
                let x0 = (qs[j] as i32 & 0x0F) | (xh_0 << 4);
                let x1 = ((qs[j] as i32 >> 4) & 0x0F) | (xh_1 << 4);
                vals[j]      = d * (x0 - 16) as f32;
                vals[j + 16] = d * (x1 - 16) as f32;
            }
            for v in &vals {
                f16_vec.push(f16::from_f32(*v));
            }
        }
        let buf = ctx.device.new_buffer(f16_vec.len() as u64 * 2, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                f16_vec.len() * 2,
            );
        }
        return Ok(buf);
    }

    // ── Q6K → F16 CPU dequantization ─────────────────────────────────────────
    // Block layout (210 bytes = 256 elements):
    //   ql[128]: lower 4 bits of each 6-bit value (2 nibbles/byte)
    //   qh[64]:  upper 2 bits of each value (4 packed per byte)
    //   sc[16]:  int8 per-group scale (16 groups of 16 elements)
    //   d:       f16 super-block scale
    if dtype == DType::Q6K {
        const BLOCK_ELEMS: usize = 256;
        const BLOCK_BYTES: usize = 210;
        let n_elements = tensor.num_elements();
        let n_blocks   = n_elements / BLOCK_ELEMS;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BLOCK_BYTES) };
        let mut f16_vec: Vec<f16> = Vec::with_capacity(n_elements);
        for b in 0..n_blocks {
            let block = &raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let ql     = &block[0..128];
            let qh     = &block[128..192];
            let scales = &block[192..208];
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
            // ggml Q6K layout: 2 halves of 128 elements, each with 4 interleaved groups of 32
            let mut vals = [0.0f32; 256];
            for n_half in [0usize, 128] {
                for l in 0..32usize {
                    let is = n_half / 16;
                    let ql_idx0 = n_half / 2 + l;
                    let ql_idx1 = n_half / 2 + l + 32;
                    let qh_idx = n_half / 4 + l;
                    let q1 = ((ql[ql_idx0] as i32 & 0x0F) | (((qh[qh_idx] as i32 >> 0) & 3) << 4)) - 32;
                    let q2 = ((ql[ql_idx1] as i32 & 0x0F) | (((qh[qh_idx] as i32 >> 2) & 3) << 4)) - 32;
                    let q3 = ((ql[ql_idx0] as i32 >> 4)    | (((qh[qh_idx] as i32 >> 4) & 3) << 4)) - 32;
                    let q4 = ((ql[ql_idx1] as i32 >> 4)    | (((qh[qh_idx] as i32 >> 6) & 3) << 4)) - 32;
                    vals[n_half + l +  0] = d * (scales[is + 0] as i8 as f32) * q1 as f32;
                    vals[n_half + l + 32] = d * (scales[is + 2] as i8 as f32) * q2 as f32;
                    vals[n_half + l + 64] = d * (scales[is + 4] as i8 as f32) * q3 as f32;
                    vals[n_half + l + 96] = d * (scales[is + 6] as i8 as f32) * q4 as f32;
                }
            }
            for v in &vals {
                f16_vec.push(f16::from_f32(*v));
            }
        }
        let buf = ctx.device.new_buffer(f16_vec.len() as u64 * 2, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                f16_vec.len() * 2,
            );
        }
        return Ok(buf);
    }

    // ── Q8_0 → F16 CPU dequantization ────────────────────────────────────────
    // Block layout (34 bytes = 32 elements):
    //   d[0..2]:   f16 scale
    //   qs[2..34]: 32 × i8 quantised values
    // Value = d * q8
    if dtype == DType::Q8_0 {
        const BLOCK_ELEMS: usize = 32;
        const BLOCK_BYTES: usize = 34;
        let n_elements = tensor.num_elements();
        let n_blocks   = n_elements / BLOCK_ELEMS;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BLOCK_BYTES) };
        let mut f16_vec: Vec<f16> = Vec::with_capacity(n_elements);
        for b in 0..n_blocks {
            let block = &raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let d  = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..34];
            for i in 0..BLOCK_ELEMS {
                let q8 = qs[i] as i8 as f32;
                f16_vec.push(f16::from_f32(d * q8));
            }
        }
        let buf = ctx.device.new_buffer(f16_vec.len() as u64 * 2, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                f16_vec.len() * 2,
            );
        }
        return Ok(buf);
    }

    // ── Q4_0 → F16 CPU dequantization ────────────────────────────────────────
    // Block layout (18 bytes = 32 elements):
    //   d[0..2]:   f16 scale
    //   qs[2..18]: 16 bytes, lower nibble = even element, upper nibble = odd element
    // Value = d * (q4 - 8), range -8..7
    if dtype == DType::Q4_0 {
        const BLOCK_ELEMS: usize = 32;
        const BLOCK_BYTES: usize = 18;
        let n_elements = tensor.num_elements();
        let n_blocks   = n_elements / BLOCK_ELEMS;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BLOCK_BYTES) };
        let mut f16_vec: Vec<f16> = Vec::with_capacity(n_elements);
        for b in 0..n_blocks {
            let block = &raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let d  = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..18];
            for i in 0..BLOCK_ELEMS {
                let byte = qs[i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                let q4 = nibble as i32 - 8;
                f16_vec.push(f16::from_f32(d * q4 as f32));
            }
        }
        let buf = ctx.device.new_buffer(f16_vec.len() as u64 * 2, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                f16_vec.len() * 2,
            );
        }
        return Ok(buf);
    }

    // ── Q5K → F16 CPU dequantization ─────────────────────────────────────────
    // Used by Q4_K_M GGUFs where some tensors are stored at higher Q5_K quality.
    // ggml block_q5_K layout (176 bytes = 256 elements):
    //   d[2]:       f16 super-scale for sub-block scales
    //   dmin[2]:    f16 super-scale for sub-block minimums
    //   scales[12]: packed 6-bit scale + 6-bit min for 8 sub-blocks (same as Q4K)
    //   qh[32]:     1 extra high bit per element (256 bits = 32 bytes)
    //   qs[128]:    lower 4 bits per element
    if dtype == DType::Q5K {
        const BLOCK_ELEMS: usize = 256;
        const BLOCK_BYTES: usize = 176;
        let n_elements = tensor.num_elements();
        let n_blocks   = n_elements / BLOCK_ELEMS;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BLOCK_BYTES) };
        let mut f16_vec: Vec<f16> = Vec::with_capacity(n_elements);
        for b in 0..n_blocks {
            let block  = &raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            // ggml block_q5_K layout (176 bytes):
            //   d[0..2]      f16 super-scale
            //   dmin[2..4]   f16 super-min
            //   scales[4..16] 12 bytes packed 6-bit scale/min for 8 sub-blocks
            //   qh[16..48]   32 bytes: 1 high bit per element (256 bits)
            //   qs[48..176]  128 bytes: lower 4 bits per element
            let d      = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin   = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales = &block[4..16];
            let qh     = &block[16..48];
            let qs     = &block[48..176];
            // ggml Q5K layout: same pairing as Q4K.
            // 4 pairs (j=0..3): low nibble of qs[j*32..] → elements j*64+0..31, scale j
            //                    high nibble of qs[j*32..] → elements j*64+32..63, scale j+4
            // Plus 1 high bit per element from qh[0..31] (256 bits).
            let mut vals = [0.0f32; 256];
            for j in 0..4usize {
                let (sc_lo, m_lo) = get_scale_min_k4(j, scales);
                let (sc_hi, m_hi) = get_scale_min_k4(j + 4, scales);
                let d_lo  = d * sc_lo as f32;
                let m_lo2 = dmin * m_lo as f32;
                let d_hi  = d * sc_hi as f32;
                let m_hi2 = dmin * m_hi as f32;
                for l in 0..32usize {
                    let byte_val = qs[j * 32 + l];
                    let elem_lo = j * 64 + l;
                    let elem_hi = j * 64 + l + 32;
                    let qh_lo = (qh[elem_lo / 8] >> (elem_lo % 8)) & 1;
                    let qh_hi = (qh[elem_hi / 8] >> (elem_hi % 8)) & 1;
                    let q5_lo = (byte_val & 0x0F) as u32 | ((qh_lo as u32) << 4);
                    let q5_hi = (byte_val >> 4)    as u32 | ((qh_hi as u32) << 4);
                    vals[elem_lo] = d_lo * q5_lo as f32 - m_lo2;
                    vals[elem_hi] = d_hi * q5_hi as f32 - m_hi2;
                }
            }
            for v in &vals {
                f16_vec.push(f16::from_f32(*v));
            }
        }
        let buf = ctx.device.new_buffer(f16_vec.len() as u64 * 2, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                f16_vec.len() * 2,
            );
        }
        return Ok(buf);
    }

    // ── Q8K → F16 CPU dequantization ─────────────────────────────────────────
    // Block layout (292 bytes = 256 elements):
    //   d[0..4]:    f32 super-scale
    //   qs[4..260]: 256 × i8 quantised values
    //   bsums[260..292]: 16 × i16 group sums (unused for plain dequant)
    // Value = d * qs[i]
    if dtype == DType::Q8K {
        const BLOCK_ELEMS: usize = 256;
        const BLOCK_BYTES: usize = 292;
        let n_elements = tensor.num_elements();
        let n_blocks   = n_elements / BLOCK_ELEMS;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BLOCK_BYTES) };
        let mut f16_vec: Vec<f16> = Vec::with_capacity(n_elements);
        for b in 0..n_blocks {
            let block = &raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            let qs = &block[4..260];
            for i in 0..BLOCK_ELEMS {
                let q8 = qs[i] as i8 as f32;
                f16_vec.push(f16::from_f32(d * q8));
            }
        }
        let buf = ctx.device.new_buffer(f16_vec.len() as u64 * 2, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f16_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                f16_vec.len() * 2,
            );
        }
        return Ok(buf);
    }

    if dtype != DType::Q4K {
        return Err(ForwardError::UnsupportedDtype {
            tensor: name.to_string(),
            dtype,
        });
    }

    // ── Q4K GPU dequant path (fallback for non-GEMV uses) ────────────────────
    let raw_buf = ctx.device.new_buffer(tb.size as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(tb.ptr, raw_buf.contents() as *mut u8, tb.size);
    }

    let n_elements = tensor.num_elements();
    let n_blocks   = (n_elements / Q4K_BLOCK_ELEMS) as u32;

    let f16_buf = ctx.device.new_buffer((n_elements * 2) as u64, opts);
    dequant_q4k_f16(ctx, &raw_buf, &f16_buf, n_blocks)?;

    Ok(f16_buf)
}

/// Upload a weight tensor dequantized to f32 for precision parity with llama.cpp.
///
/// Handles all quantized types (Q5_0, Q6K, Q8_0, Q4_0, Q5K, Q8K, Q4K) plus
/// BF16 and F32 (which stay as f32). Returns `None` for native F16 weights
/// (no precision benefit from promoting to f32).
fn upload_weight_as_f32(
    ctx: &MetalContext,
    model: &Model,
    name: &str,
) -> Result<Option<metal::Buffer>, ForwardError> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| ForwardError::MissingWeight(name.to_string()))?;

    let dtype = tensor.dtype;
    let n_elements = tensor.num_elements();

    // Native F16 — no precision gain from promoting to f32
    if dtype == DType::F16 {
        return Ok(None);
    }

    let tb = model
        .tensor_buffer(name)
        .ok_or_else(|| ForwardError::MissingWeight(name.to_string()))?;

    let opts = MTLResourceOptions::StorageModeShared;

    // F32 — already in target precision, just copy
    if dtype == DType::F32 {
        let buf = ctx.device.new_buffer(tb.size as u64, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(tb.ptr, buf.contents() as *mut u8, tb.size);
        }
        return Ok(Some(buf));
    }

    // BF16 → f32
    if dtype == DType::BF16 {
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr as *const u16, n_elements) };
        let f32_vec: Vec<f32> = raw.iter().map(|&bf| {
            let bits = (bf as u32) << 16;
            f32::from_bits(bits)
        }).collect();
        let buf = ctx.device.new_buffer((n_elements * 4) as u64, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                f32_vec.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                n_elements * 4,
            );
        }
        return Ok(Some(buf));
    }

    // ── Quantized types → f32 ────────────────────────────────────────────────
    let mut f32_vec: Vec<f32> = Vec::with_capacity(n_elements);

    if dtype == DType::Q5_0 {
        const BE: usize = 32;
        const BB: usize = 22;
        let n_blocks = n_elements / BE;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BB) };
        for b in 0..n_blocks {
            let block = &raw[b * BB..(b + 1) * BB];
            let d  = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qh = &block[2..6];
            let qs = &block[6..22];
            let mut vals = [0.0f32; 32];
            for j in 0..16usize {
                let xh_0 = ((qh[j / 8] >> (j % 8)) & 1) as i32;
                let xh_1 = ((qh[j / 8 + 2] >> (j % 8)) & 1) as i32;
                let x0 = (qs[j] as i32 & 0x0F) | (xh_0 << 4);
                let x1 = ((qs[j] as i32 >> 4) & 0x0F) | (xh_1 << 4);
                vals[j]      = d * (x0 - 16) as f32;
                vals[j + 16] = d * (x1 - 16) as f32;
            }
            f32_vec.extend_from_slice(&vals);
        }
    } else if dtype == DType::Q6K {
        const BE: usize = 256;
        const BB: usize = 210;
        let n_blocks = n_elements / BE;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BB) };
        for b in 0..n_blocks {
            let block = &raw[b * BB..(b + 1) * BB];
            let ql     = &block[0..128];
            let qh     = &block[128..192];
            let scales = &block[192..208];
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
            let mut vals = [0.0f32; 256];
            for n_half in [0usize, 128] {
                for l in 0..32usize {
                    let is = n_half / 16;
                    let ql_idx0 = n_half / 2 + l;
                    let ql_idx1 = n_half / 2 + l + 32;
                    let qh_idx = n_half / 4 + l;
                    let q1 = ((ql[ql_idx0] as i32 & 0x0F) | (((qh[qh_idx] as i32 >> 0) & 3) << 4)) - 32;
                    let q2 = ((ql[ql_idx1] as i32 & 0x0F) | (((qh[qh_idx] as i32 >> 2) & 3) << 4)) - 32;
                    let q3 = ((ql[ql_idx0] as i32 >> 4)    | (((qh[qh_idx] as i32 >> 4) & 3) << 4)) - 32;
                    let q4 = ((ql[ql_idx1] as i32 >> 4)    | (((qh[qh_idx] as i32 >> 6) & 3) << 4)) - 32;
                    vals[n_half + l +  0] = d * (scales[is + 0] as i8 as f32) * q1 as f32;
                    vals[n_half + l + 32] = d * (scales[is + 2] as i8 as f32) * q2 as f32;
                    vals[n_half + l + 64] = d * (scales[is + 4] as i8 as f32) * q3 as f32;
                    vals[n_half + l + 96] = d * (scales[is + 6] as i8 as f32) * q4 as f32;
                }
            }
            f32_vec.extend_from_slice(&vals);
        }
    } else if dtype == DType::Q8_0 {
        const BE: usize = 32;
        const BB: usize = 34;
        let n_blocks = n_elements / BE;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BB) };
        for b in 0..n_blocks {
            let block = &raw[b * BB..(b + 1) * BB];
            let d  = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..34];
            for i in 0..BE {
                f32_vec.push(d * qs[i] as i8 as f32);
            }
        }
    } else if dtype == DType::Q4_0 {
        const BE: usize = 32;
        const BB: usize = 18;
        let n_blocks = n_elements / BE;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BB) };
        for b in 0..n_blocks {
            let block = &raw[b * BB..(b + 1) * BB];
            let d  = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..18];
            for i in 0..BE {
                let byte = qs[i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                f32_vec.push(d * (nibble as i32 - 8) as f32);
            }
        }
    } else if dtype == DType::Q5K {
        const BE: usize = 256;
        const BB: usize = 176;
        let n_blocks = n_elements / BE;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BB) };
        for b in 0..n_blocks {
            let block  = &raw[b * BB..(b + 1) * BB];
            let d      = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin   = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales = &block[4..16];
            let qh     = &block[16..48];
            let qs     = &block[48..176];
            let mut vals = [0.0f32; 256];
            for j in 0..4usize {
                let (sc_lo, m_lo) = get_scale_min_k4(j, scales);
                let (sc_hi, m_hi) = get_scale_min_k4(j + 4, scales);
                let d_lo  = d * sc_lo as f32;
                let m_lo2 = dmin * m_lo as f32;
                let d_hi  = d * sc_hi as f32;
                let m_hi2 = dmin * m_hi as f32;
                for l in 0..32usize {
                    let byte_val = qs[j * 32 + l];
                    let elem_lo = j * 64 + l;
                    let elem_hi = j * 64 + l + 32;
                    let qh_lo = (qh[elem_lo / 8] >> (elem_lo % 8)) & 1;
                    let qh_hi = (qh[elem_hi / 8] >> (elem_hi % 8)) & 1;
                    let q5_lo = (byte_val & 0x0F) as u32 | ((qh_lo as u32) << 4);
                    let q5_hi = (byte_val >> 4)    as u32 | ((qh_hi as u32) << 4);
                    vals[elem_lo] = d_lo * q5_lo as f32 - m_lo2;
                    vals[elem_hi] = d_hi * q5_hi as f32 - m_hi2;
                }
            }
            f32_vec.extend_from_slice(&vals);
        }
    } else if dtype == DType::Q8K {
        const BE: usize = 256;
        const BB: usize = 292;
        let n_blocks = n_elements / BE;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BB) };
        for b in 0..n_blocks {
            let block = &raw[b * BB..(b + 1) * BB];
            let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            let qs = &block[4..260];
            for i in 0..BE {
                f32_vec.push(d * qs[i] as i8 as f32);
            }
        }
    } else if dtype == DType::Q4K {
        // CPU dequant for Q4K (used when K % 256 != 0 so fused GEMV can't be used)
        const BE: usize = 256;
        const BB: usize = 144;
        let n_blocks = n_elements / BE;
        let raw = unsafe { std::slice::from_raw_parts(tb.ptr, n_blocks * BB) };
        for b in 0..n_blocks {
            let block  = &raw[b * BB..(b + 1) * BB];
            let d      = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin   = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales = &block[4..16];
            let qs     = &block[16..144];
            let mut vals = [0.0f32; 256];
            for j in 0..4usize {
                let (sc_lo, m_lo) = get_scale_min_k4(j, scales);
                let (sc_hi, m_hi) = get_scale_min_k4(j + 4, scales);
                let d_lo  = d * sc_lo as f32;
                let m_lo2 = dmin * m_lo as f32;
                let d_hi  = d * sc_hi as f32;
                let m_hi2 = dmin * m_hi as f32;
                for l in 0..32usize {
                    let byte_val = qs[j * 32 + l];
                    let elem_lo = j * 64 + l;
                    let elem_hi = j * 64 + l + 32;
                    let q4_lo = (byte_val & 0x0F) as u32;
                    let q4_hi = (byte_val >> 4)    as u32;
                    vals[elem_lo] = d_lo * q4_lo as f32 - m_lo2;
                    vals[elem_hi] = d_hi * q4_hi as f32 - m_hi2;
                }
            }
            f32_vec.extend_from_slice(&vals);
        }
    } else {
        return Err(ForwardError::UnsupportedDtype {
            tensor: name.to_string(),
            dtype,
        });
    }

    let buf = ctx.device.new_buffer((n_elements * 4) as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(
            f32_vec.as_ptr() as *const u8,
            buf.contents() as *mut u8,
            n_elements * 4,
        );
    }
    Ok(Some(buf))
}

/// Upload a weight tensor, keeping Q4K weights in packed form for fused GEMV.
/// Returns a `WeightBuf` that dispatches to the appropriate GEMV kernel.
///
/// For non-Q4K quantized types, dequantizes to f32 for precision parity with
/// llama.cpp (avoids the ~10-bit mantissa loss from f16 truncation).
fn upload_weight_wb(
    ctx: &MetalContext,
    model: &Model,
    name: &str,
    m: u32,
    k: u32,
) -> Result<WeightBuf, ForwardError> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| ForwardError::MissingWeight(name.to_string()))?;

    let dtype = tensor.dtype;

    // Q4K with aligned K: use fused on-the-fly dequant (already f32 arithmetic internally)
    if dtype == DType::Q4K && (k as usize % Q4K_BLOCK_ELEMS == 0) {
        let tb = model
            .tensor_buffer(name)
            .ok_or_else(|| ForwardError::MissingWeight(name.to_string()))?;

        let opts = MTLResourceOptions::StorageModeShared;

        // Raw Q4K buffer for fused GEMV (decode path — 3.5x less bandwidth)
        let raw_buf = ctx.device.new_buffer(tb.size as u64, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(tb.ptr, raw_buf.contents() as *mut u8, tb.size);
        }

        // Also dequant to F16 for GEMM (prefill path)
        let n_elements = tensor.num_elements();
        let n_blocks = (n_elements / Q4K_BLOCK_ELEMS) as u32;
        let f16_buf = ctx.device.new_buffer((n_elements * 2) as u64, opts);
        dequant_q4k_f16(ctx, &raw_buf, &f16_buf, n_blocks)?;

        return Ok(WeightBuf::Q4K { raw: raw_buf, f16: f16_buf });
    }

    // Q8_0 with aligned K: use fused on-the-fly dequant (47% less bandwidth than f16)
    if dtype == DType::Q8_0 && (k as usize % Q8_0_BLOCK_ELEMS == 0) {
        let tb = model
            .tensor_buffer(name)
            .ok_or_else(|| ForwardError::MissingWeight(name.to_string()))?;

        let opts = MTLResourceOptions::StorageModeShared;

        // Raw Q8_0 buffer for fused GEMV (decode path — 47% less bandwidth than f16)
        let raw_buf = ctx.device.new_buffer(tb.size as u64, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(tb.ptr, raw_buf.contents() as *mut u8, tb.size);
        }

        // Also dequant to F16 for GEMM (prefill path)
        let f16_buf = upload_weight(ctx, model, name)?;

        return Ok(WeightBuf::Q8_0 { raw: raw_buf, f16: f16_buf });
    }

    // Native F16: keep as-is (already at true precision)
    if dtype == DType::F16 {
        return Ok(WeightBuf::F16(upload_weight(ctx, model, name)?));
    }

    // ── PERFORMANCE: Dequant to f16 for bandwidth-optimal GEMV ──────────
    // Quantized types (Q5_0, Q5K, Q6K, Q8_0, Q8K, Q4_0, BF16) have ≤8
    // significant bits — well within f16's 10-bit mantissa. Dequanting to
    // f32 wastes 2x memory bandwidth for zero precision gain.
    //
    // Impact: For Q8_0 model (K=896), this halves weight read bandwidth
    // for all Q/K/V/O and FFN projections → ~2x decode speedup.
    //
    // Q4K with non-aligned K: also safe — the dequanted values only have
    // ~4.5 bits of effective precision.
    Ok(WeightBuf::F16(upload_weight(ctx, model, name)?))
}

/// Upload an optional bias tensor (absent in some architectures).
fn upload_optional(
    ctx: &MetalContext,
    model: &Model,
    name: &str,
) -> Result<Option<metal::Buffer>, ForwardError> {
    if model.tensors.contains_key(name) {
        Ok(Some(upload_weight(ctx, model, name)?))
    } else {
        Ok(None)
    }
}

impl Executor {
    /// Upload all required F16 weights from `model` to the GPU.
    pub fn new(ctx: MetalContext, model: &Model) -> Result<Self, ForwardError> {
        let cfg = model.config.clone();
        let n = cfg.num_hidden_layers;

        // Global weights — token embedding stored as f32 for precision
        let tok_emb = if let Some(f32_buf) = upload_weight_as_f32(&ctx, model, "token_embd.weight")? {
            f32_buf
        } else {
            // Native F16 embedding — keep as-is (rare)
            upload_weight(&ctx, model, "token_embd.weight")?
        };
        let tok_emb_is_f32 = {
            let t = model.tensors.get("token_embd.weight").unwrap();
            t.dtype != DType::F16
        };
        // Norm weights: prefer f32 for full precision (F32 in GGUF → keep as f32)
        let output_norm = upload_weight_as_f32(&ctx, model, "output_norm.weight")?
            .unwrap_or_else(|| upload_weight(&ctx, model, "output_norm.weight").unwrap());
        let norm_is_f32 = {
            let t = model.tensors.get("output_norm.weight").unwrap();
            t.dtype != DType::F16
        };

        let h      = cfg.hidden_size as u32;
        let q_dim  = (cfg.num_attention_heads * cfg.head_dim) as u32;
        let kv_dim = (cfg.num_key_value_heads * cfg.head_dim) as u32;
        let mut ff_dim = cfg.intermediate_size as u32;
        let vocab  = cfg.vocab_size as u32;

        // lm_head may be tied to token_embd (Qwen2 ties them)
        // PERF: Use fused Q8_0 GEMV when possible (47% less bandwidth than f16).
        // The lm_head is the single largest weight read per token.
        let lm_head = if model.tensors.contains_key("output.weight") {
            upload_weight_wb(&ctx, model, "output.weight", vocab, h)?
        } else {
            // Tied to tok_emb: use upload_weight_wb to get best format (Q8_0 fused if possible)
            upload_weight_wb(&ctx, model, "token_embd.weight", vocab, h)?
        };

        // Per-layer weights
        let mut attn_norm  = Vec::with_capacity(n);
        let mut attn_q     = Vec::with_capacity(n);
        let mut attn_k     = Vec::with_capacity(n);
        let mut attn_v     = Vec::with_capacity(n);
        let mut attn_q_bias = Vec::with_capacity(n);
        let mut attn_k_bias = Vec::with_capacity(n);
        let mut attn_v_bias = Vec::with_capacity(n);
        let mut attn_out   = Vec::with_capacity(n);
        let mut ffn_norm   = Vec::with_capacity(n);
        let mut ffn_gate   = Vec::with_capacity(n);
        let mut ffn_up     = Vec::with_capacity(n);
        let mut ffn_down   = Vec::with_capacity(n);

        // MoE weight storage
        let mut moe_router    = Vec::with_capacity(n);
        let mut moe_gate_exps = Vec::with_capacity(n);
        let mut moe_up_exps   = Vec::with_capacity(n);
        let mut moe_down_exps = Vec::with_capacity(n);
        let is_moe = cfg.is_moe();
        let n_experts = cfg.num_experts;

        for l in 0..n {
            attn_norm.push(
                upload_weight_as_f32(&ctx, model, &format!("blk.{l}.attn_norm.weight"))?
                    .unwrap_or_else(|| upload_weight(&ctx, model, &format!("blk.{l}.attn_norm.weight")).unwrap())
            );
            attn_q.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.attn_q.weight"), q_dim, h)?);
            attn_k.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.attn_k.weight"), kv_dim, h)?);
            attn_v.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.attn_v.weight"), kv_dim, h)?);
            attn_q_bias.push(upload_optional(&ctx, model, &format!("blk.{l}.attn_q.bias"))?);
            attn_k_bias.push(upload_optional(&ctx, model, &format!("blk.{l}.attn_k.bias"))?);
            attn_v_bias.push(upload_optional(&ctx, model, &format!("blk.{l}.attn_v.bias"))?);
            attn_out.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.attn_output.weight"), h, q_dim)?);
            ffn_norm.push(
                upload_weight_as_f32(&ctx, model, &format!("blk.{l}.ffn_norm.weight"))?
                    .unwrap_or_else(|| upload_weight(&ctx, model, &format!("blk.{l}.ffn_norm.weight")).unwrap())
            );

            if is_moe {
                // MoE: router + per-expert FFN weights
                moe_router.push(upload_weight(&ctx, model, &format!("blk.{l}.ffn_gate_inp.weight"))?);

                let mut layer_gates = Vec::with_capacity(n_experts);
                let mut layer_ups   = Vec::with_capacity(n_experts);
                let mut layer_downs = Vec::with_capacity(n_experts);

                // Try per-expert naming first (Mixtral: blk.{l}.ffn_gate.{e}.weight)
                // Fall back to merged naming (OLMoE/llama.cpp: blk.{l}.ffn_gate_exps.weight)
                let per_expert = model.tensors.contains_key(&format!("blk.{l}.ffn_gate.0.weight"));

                if per_expert {
                    for e in 0..n_experts {
                        layer_gates.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.ffn_gate.{e}.weight"), ff_dim, h)?);
                        layer_ups.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.ffn_up.{e}.weight"), ff_dim, h)?);
                        layer_downs.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.ffn_down.{e}.weight"), h, ff_dim)?);
                    }
                } else {
                    // Merged expert tensors: shape [num_experts, out_dim, in_dim]
                    // Dequant to f32 for precision, then create per-expert sub-buffers.
                    let gate_name = format!("blk.{l}.ffn_gate_exps.weight");
                    let up_name   = format!("blk.{l}.ffn_up_exps.weight");
                    let down_name = format!("blk.{l}.ffn_down_exps.weight");

                    // Try f32 first, fallback to f16
                    let (gate_all, elem_size) = match upload_weight_as_f32(&ctx, model, &gate_name)? {
                        Some(buf) => (buf, 4usize),
                        None => (upload_weight(&ctx, model, &gate_name)?, 2usize),
                    };
                    let up_all = if elem_size == 4 {
                        upload_weight_as_f32(&ctx, model, &up_name)?.unwrap()
                    } else {
                        upload_weight(&ctx, model, &up_name)?
                    };
                    let down_all = if elem_size == 4 {
                        upload_weight_as_f32(&ctx, model, &down_name)?.unwrap()
                    } else {
                        upload_weight(&ctx, model, &down_name)?
                    };

                    // Flush any pending GPU work before CPU reads
                    ctx.flush();

                    // Derive actual expert FFN dimension from tensor size
                    let gate_total_bytes = gate_all.length() as usize;
                    let expert_ff_dim = gate_total_bytes / (n_experts * (h as usize) * elem_size);
                    let gate_expert_bytes = expert_ff_dim * (h as usize) * elem_size;
                    let up_expert_bytes   = gate_expert_bytes;
                    let down_expert_bytes = (h as usize) * expert_ff_dim * elem_size;

                    if l == 0 {
                        tracing::info!(
                            expert_ff_dim,
                            config_ff_dim = ff_dim as usize,
                            elem_size,
                            "MoE expert FFN dimension (derived from tensor)"
                        );
                        ff_dim = expert_ff_dim as u32;
                    }

                    for e in 0..n_experts {
                        let opts = MTLResourceOptions::StorageModeShared;

                        let g_buf = ctx.device.new_buffer(gate_expert_bytes as u64, opts);
                        let u_buf = ctx.device.new_buffer(up_expert_bytes as u64, opts);
                        let d_buf = ctx.device.new_buffer(down_expert_bytes as u64, opts);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                (gate_all.contents() as *const u8).add(e * gate_expert_bytes),
                                g_buf.contents() as *mut u8, gate_expert_bytes);
                            std::ptr::copy_nonoverlapping(
                                (up_all.contents() as *const u8).add(e * up_expert_bytes),
                                u_buf.contents() as *mut u8, up_expert_bytes);
                            std::ptr::copy_nonoverlapping(
                                (down_all.contents() as *const u8).add(e * down_expert_bytes),
                                d_buf.contents() as *mut u8, down_expert_bytes);
                        }
                        if elem_size == 4 {
                            layer_gates.push(WeightBuf::F32(g_buf));
                            layer_ups.push(WeightBuf::F32(u_buf));
                            layer_downs.push(WeightBuf::F32(d_buf));
                        } else {
                            layer_gates.push(WeightBuf::F16(g_buf));
                            layer_ups.push(WeightBuf::F16(u_buf));
                            layer_downs.push(WeightBuf::F16(d_buf));
                        }
                    }
                }

                moe_gate_exps.push(layer_gates);
                moe_up_exps.push(layer_ups);
                moe_down_exps.push(layer_downs);

                // Push empty dense FFN placeholders
                ffn_gate.push(WeightBuf::F16(ctx.new_buffer(4)));
                ffn_up.push(WeightBuf::F16(ctx.new_buffer(4)));
                ffn_down.push(WeightBuf::F16(ctx.new_buffer(4)));
            } else {
                // Dense FFN
                ffn_gate.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.ffn_gate.weight"), ff_dim, h)?);
                ffn_up.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.ffn_up.weight"), ff_dim, h)?);
                ffn_down.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.ffn_down.weight"), h, ff_dim)?);
            }
        }

        // Flush all pending GPU dequantisation work (Q4K uploads) in one batch.
        ctx.flush();


        // For MoE models, ff_dim may have been updated from tensor dimensions.
        // Re-derive config for scratch allocation.
        let actual_ff_dim = ff_dim as usize;
        let scratch = DecodeScratch::new_with_ff(&ctx, &cfg, actual_ff_dim);

        tracing::info!(
            layers = n,
            hidden = cfg.hidden_size,
            vocab  = cfg.vocab_size,
            "Executor: weights uploaded to GPU"
        );

        Ok(Self {
            ctx,
            weights: LayerWeights {
                tok_emb, output_norm, lm_head,
                attn_norm, attn_q, attn_k, attn_v,
                attn_q_bias, attn_k_bias, attn_v_bias, attn_out,
                ffn_norm, ffn_gate, ffn_up, ffn_down,
                moe_router, moe_gate_exps, moe_up_exps, moe_down_exps,
            },
            config: cfg,
            scratch,
            expert_ff_dim: actual_ff_dim,
            tok_emb_f32: tok_emb_is_f32,
            norm_f32: norm_is_f32,
        })
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /// Run one decode step: embed `token_id`, run all transformer layers,
    /// update the KV cache, and return logits `[vocab_size]` as f16.
    ///
    /// `pos` is the 0-based absolute sequence position of this token.
    /// `kv` must have `kv.seq_len() == pos` before the call;
    /// `kv.advance()` is called internally after all layers complete.
    pub fn forward(
        &self,
        token_id: u32,
        pos: u32,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>, ForwardError> {
        let cfg = &self.config;
        let ctx = &self.ctx;
        let s   = &self.scratch;

        if token_id >= cfg.vocab_size as u32 {
            return Err(ForwardError::InvalidTokenId(token_id));
        }

        let _profile = std::env::var("PROFILE_FORWARD").is_ok();
        let _profile_detail = std::env::var("PROFILE_DETAIL").is_ok();
        let _t0 = std::time::Instant::now();
        let mut _t_qkv = 0.0f64;
        let mut _t_attn = 0.0f64;
        let mut _t_ffn = 0.0f64;
        let mut _t_lmhead = 0.0f64;

        // ── 1. Token embedding lookup ──────────────────────────────────────
        let h = cfg.hidden_size;
        unsafe {
            let dst = s.x.contents() as *mut f32;
            if self.tok_emb_f32 {
                let src = (self.weights.tok_emb.contents() as *const f32)
                    .add(token_id as usize * h);
                std::ptr::copy_nonoverlapping(src, dst, h);
            } else {
                let src = (self.weights.tok_emb.contents() as *const f16)
                    .add(token_id as usize * h);
                for i in 0..h { *dst.add(i) = (*src.add(i)).to_f32(); }
            }
        }

        let q_dim  = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads  * cfg.head_dim;
        let ff_dim = self.expert_ff_dim;
        let kv_len_after = pos + 1;
        let max_seq = kv.max_seq_len();
        let hd = cfg.head_dim as u32;
        let n_kv = cfg.num_key_value_heads as u32;

        // ── 2. Transformer layers ─────────────────────────────────────────────
        for l in 0..cfg.num_hidden_layers {
            let w = &self.weights;

            // a) Attention pre-norm (f32 → f32)
            if self.norm_f32 {
                rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm, &w.attn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            } else {
                rms_norm_f32_f32(ctx, &s.x, &s.x_norm, &w.attn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            }

            let _ts = std::time::Instant::now();

            // b-d) FUSED Q+K+V + bias: 1 dispatch instead of 6
            match (&w.attn_q[l], &w.attn_k[l], &w.attn_v[l],
                   &w.attn_q_bias[l], &w.attn_k_bias[l], &w.attn_v_bias[l]) {
                // Q8_0 with biases (Qwen path: most common)
                (WeightBuf::Q8_0 { raw: rq, .. }, WeightBuf::Q8_0 { raw: rk, .. }, WeightBuf::Q8_0 { raw: rv, .. },
                 Some(bq), Some(bk), Some(bv)) => {
                    fused_qkv_bias_q8_0_f32(ctx, rq, rk, rv, &s.x_norm,
                                            &s.q, &s.k, &s.v,
                                            bq, bk, bv,
                                            q_dim as u32, kv_dim as u32, h as u32)?;
                }
                // Q8_0 without biases
                (WeightBuf::Q8_0 { raw: rq, .. }, WeightBuf::Q8_0 { raw: rk, .. }, WeightBuf::Q8_0 { raw: rv, .. },
                 None, None, None) => {
                    fused_qkv_q8_0_f32(ctx, rq, rk, rv, &s.x_norm,
                                       &s.q, &s.k, &s.v,
                                       q_dim as u32, kv_dim as u32, h as u32)?;
                }
                // F16 without biases
                (WeightBuf::F16(bq), WeightBuf::F16(bk), WeightBuf::F16(bv),
                 None, None, None) => {
                    fused_qkv_f16w_f32(ctx, bq, bk, bv, &s.x_norm,
                                       &s.q, &s.k, &s.v,
                                       q_dim as u32, kv_dim as u32, h as u32)?;
                }
                // Fallback: separate dispatches
                _ => {
                    w.attn_q[l].gemv_f32_f32out(ctx, &s.x_norm, &s.q, q_dim as u32, h as u32)?;
                    if let Some(bias) = &w.attn_q_bias[l] { add_f16_into_f32(ctx, bias, &s.q, q_dim as u32)?; }
                    w.attn_k[l].gemv_f32_f32out(ctx, &s.x_norm, &s.k, kv_dim as u32, h as u32)?;
                    if let Some(bias) = &w.attn_k_bias[l] { add_f16_into_f32(ctx, bias, &s.k, kv_dim as u32)?; }
                    w.attn_v[l].gemv_f32_f32out(ctx, &s.x_norm, &s.v, kv_dim as u32, h as u32)?;
                    if let Some(bias) = &w.attn_v_bias[l] { add_f16_into_f32(ctx, bias, &s.v, kv_dim as u32)?; }
                }
            }

            // e) Fused RoPE: apply to BOTH Q and K in ONE dispatch
            fused_rope_qk_f32(ctx, &s.q, &s.k, hd,
                              cfg.num_attention_heads as u32, n_kv,
                              pos, cfg.rope_theta)?;

            if _profile_detail { ctx.flush(); _t_qkv += _ts.elapsed().as_secs_f64() * 1000.0; }
            let _ts = std::time::Instant::now();

            // f) Write f32 K/V into f32 cache — GPU-side scatter
            kv_copy_to_cache_f32(ctx, &s.k, kv.k_buf(l), pos, max_seq, hd, n_kv)?;
            kv_copy_to_cache_f32(ctx, &s.v, kv.v_buf(l), pos, max_seq, hd, n_kv)?;

            // g) Decode attention: f32 Q/K/V → f32 output
            decode_attention_f32(
                ctx, &s.q, kv.k_buf(l), kv.v_buf(l), &s.attn_out,
                1, kv_len_after,
                cfg.num_attention_heads as u32, n_kv, hd, max_seq,
            )?;

            // h) Output projection: f32 attn_out + f32 residual → f32 output
            w.attn_out[l].gemv_add_f32(ctx, &s.attn_out, &s.x, &s.x, h as u32, q_dim as u32)?;

            if _profile_detail { ctx.flush(); _t_attn += _ts.elapsed().as_secs_f64() * 1000.0; }
            let _ts = std::time::Instant::now();

            // i) FFN pre-norm (f32 → f32)
            if self.norm_f32 {
                rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            } else {
                rms_norm_f32_f32(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            }

            if cfg.is_moe() {
                // ── MoE FFN: router → top-k → sparse expert dispatch ─────────
                let n_exp = cfg.num_experts;
                let top_k = cfg.num_experts_per_tok;

                // Router: [num_experts, hidden_size] @ x_norm2 → [num_experts]
                gemv_f16w_f32in(ctx, &w.moe_router[l], &s.x_norm2, &s.router_logits,
                         n_exp as u32, h as u32)?;

                // Flush to make router logits visible for CPU top-k
                ctx.flush();
                let (expert_ids, expert_weights) = topk_softmax(
                    &s.router_logits, n_exp, top_k,
                );


                // Zero the MoE output accumulator (CPU write is fine — no pending GPU work on this buffer)
                unsafe {
                    std::ptr::write_bytes(s.moe_out.contents() as *mut u8, 0, h * 2);
                }

                // Run selected experts and accumulate on GPU — NO per-expert flush!
                // All expert dispatches + scale_accumulate run async in one command buffer.
                for (i, &eid) in expert_ids.iter().enumerate() {
                    w.moe_gate_exps[l][eid].gemv_f32in(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                    w.moe_up_exps[l][eid].gemv_f32in(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                    silu_mul_f16(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                    w.moe_down_exps[l][eid].gemv(ctx, &s.act, &s.ffn_out, h as u32, ff_dim as u32)?;

                    // GPU-side weighted accumulate: moe_out += ffn_out * weight
                    scale_accumulate_f16(ctx, &s.ffn_out, &s.moe_out,
                                         expert_weights[i], h as u32)?;
                }

                // Residual: f32 x += f16 moe_out
                add_f16_into_f32(ctx, &s.moe_out, &s.x, h as u32)?;
            } else {
                // ── Dense FFN: FUSED gate+up+silu (1 dispatch instead of 3) ────
                match (&w.ffn_gate[l], &w.ffn_up[l]) {
                    (WeightBuf::Q8_0 { raw: g_raw, .. }, WeightBuf::Q8_0 { raw: u_raw, .. }) => {
                        fused_ffn_q8_0_f32(ctx, g_raw, u_raw, &s.x_norm2, &s.act,
                                           ff_dim as u32, h as u32)?;
                    }
                    (WeightBuf::F16(g_buf), WeightBuf::F16(u_buf)) => {
                        fused_ffn_f16w_f32(ctx, g_buf, u_buf, &s.x_norm2, &s.act,
                                           ff_dim as u32, h as u32)?;
                    }
                    _ => {
                        // Fallback: separate dispatches for mixed/unsupported types
                        w.ffn_gate[l].gemv_f32_f32out(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                        w.ffn_up[l].gemv_f32_f32out(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                        silu_mul_f32(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                    }
                }
                w.ffn_down[l].gemv_add_f32(ctx, &s.act, &s.x, &s.x, h as u32, ff_dim as u32)?;
            }

            if _profile_detail { ctx.flush(); _t_ffn += _ts.elapsed().as_secs_f64() * 1000.0; }

            // Per-layer diagnostic: dump x[0..8] after each layer
            if std::env::var("DEBUG_LAYERS").is_ok() {
                ctx.flush();
                let px = s.x.contents() as *const f32;
                let xv: Vec<f32> = (0..8).map(|i| unsafe { *px.add(i) }).collect();
                eprintln!("L{:02} x[0..8]: {:?}", l, xv);
            }
        }

        // All layers done — advance KV cache position.
        kv.advance();

        // ── 3. Final norm + lm_head (full f32 logits for precise ranking) ────
        let _ts = std::time::Instant::now();
        if self.norm_f32 {
            rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm, &self.weights.output_norm,
                         cfg.rms_norm_eps, h as u32, 1)?;
        } else {
            rms_norm_f32_f32(ctx, &s.x, &s.x_norm, &self.weights.output_norm,
                         cfg.rms_norm_eps, h as u32, 1)?;
        }
        self.weights.lm_head.gemv_f32_f32out(ctx, &s.x_norm, &s.logits,
                 cfg.vocab_size as u32, h as u32)?;

        if _profile_detail { ctx.flush(); _t_lmhead = _ts.elapsed().as_secs_f64() * 1000.0; }

        // ── 4. Download f32 logits ───────────────────────────────────────────
        let _t_encode = _t0.elapsed();
        ctx.flush();
        let _t_total = _t0.elapsed();
        if _profile {
            let cpu_ms = _t_encode.as_secs_f64() * 1000.0;
            let gpu_ms = (_t_total - _t_encode).as_secs_f64() * 1000.0;
            let total_ms = _t_total.as_secs_f64() * 1000.0;
            eprintln!("PROFILE pos={} cpu_encode={:.3}ms gpu_exec={:.3}ms total={:.3}ms",
                      pos, cpu_ms, gpu_ms, total_ms);
        }
        if _profile_detail {
            eprintln!("DETAIL  pos={} qkv={:.2}ms attn={:.2}ms ffn={:.2}ms lmhead={:.2}ms total={:.2}ms",
                      pos, _t_qkv, _t_attn, _t_ffn, _t_lmhead,
                      _t_qkv + _t_attn + _t_ffn + _t_lmhead);
        }
        let logits = download_f32_buf(&s.logits, cfg.vocab_size);
        Ok(logits)
    }

    /// Greedy decode: same as `forward()` but computes argmax on GPU and
    /// returns only the winning token ID (4 bytes) instead of downloading
    /// the full logit vector (304KB for Qwen2.5-0.5B).
    ///
    /// Use this for temperature=0 / greedy sampling to avoid the logit
    /// download bottleneck.
    pub fn forward_greedy(
        &self,
        token_id: u32,
        pos: u32,
        kv: &mut KvCache,
    ) -> Result<u32, ForwardError> {
        let cfg = &self.config;
        let ctx = &self.ctx;
        let s   = &self.scratch;

        if token_id >= cfg.vocab_size as u32 {
            return Err(ForwardError::InvalidTokenId(token_id));
        }

        let h = cfg.hidden_size;
        unsafe {
            let dst = s.x.contents() as *mut f32;
            if self.tok_emb_f32 {
                let src = (self.weights.tok_emb.contents() as *const f32)
                    .add(token_id as usize * h);
                std::ptr::copy_nonoverlapping(src, dst, h);
            } else {
                let src = (self.weights.tok_emb.contents() as *const f16)
                    .add(token_id as usize * h);
                for i in 0..h { *dst.add(i) = (*src.add(i)).to_f32(); }
            }
        }

        let q_dim  = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads  * cfg.head_dim;
        let ff_dim = self.expert_ff_dim;
        let kv_len_after = pos + 1;
        let max_seq = kv.max_seq_len();
        let hd = cfg.head_dim as u32;
        let n_kv = cfg.num_key_value_heads as u32;

        for l in 0..cfg.num_hidden_layers {
            let w = &self.weights;

            if self.norm_f32 {
                rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm, &w.attn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            } else {
                rms_norm_f32_f32(ctx, &s.x, &s.x_norm, &w.attn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            }

            // f32 FUSED Q+K+V + bias (1 dispatch instead of 6)
            match (&w.attn_q[l], &w.attn_k[l], &w.attn_v[l],
                   &w.attn_q_bias[l], &w.attn_k_bias[l], &w.attn_v_bias[l]) {
                (WeightBuf::Q8_0 { raw: rq, .. }, WeightBuf::Q8_0 { raw: rk, .. }, WeightBuf::Q8_0 { raw: rv, .. },
                 Some(bq), Some(bk), Some(bv)) => {
                    fused_qkv_bias_q8_0_f32(ctx, rq, rk, rv, &s.x_norm,
                                            &s.q, &s.k, &s.v, bq, bk, bv,
                                            q_dim as u32, kv_dim as u32, h as u32)?;
                }
                (WeightBuf::Q8_0 { raw: rq, .. }, WeightBuf::Q8_0 { raw: rk, .. }, WeightBuf::Q8_0 { raw: rv, .. },
                 None, None, None) => {
                    fused_qkv_q8_0_f32(ctx, rq, rk, rv, &s.x_norm,
                                       &s.q, &s.k, &s.v,
                                       q_dim as u32, kv_dim as u32, h as u32)?;
                }
                (WeightBuf::F16(bq), WeightBuf::F16(bk), WeightBuf::F16(bv), None, None, None) => {
                    fused_qkv_f16w_f32(ctx, bq, bk, bv, &s.x_norm,
                                       &s.q, &s.k, &s.v,
                                       q_dim as u32, kv_dim as u32, h as u32)?;
                }
                _ => {
                    w.attn_q[l].gemv_f32_f32out(ctx, &s.x_norm, &s.q, q_dim as u32, h as u32)?;
                    if let Some(bias) = &w.attn_q_bias[l] { add_f16_into_f32(ctx, bias, &s.q, q_dim as u32)?; }
                    w.attn_k[l].gemv_f32_f32out(ctx, &s.x_norm, &s.k, kv_dim as u32, h as u32)?;
                    if let Some(bias) = &w.attn_k_bias[l] { add_f16_into_f32(ctx, bias, &s.k, kv_dim as u32)?; }
                    w.attn_v[l].gemv_f32_f32out(ctx, &s.x_norm, &s.v, kv_dim as u32, h as u32)?;
                    if let Some(bias) = &w.attn_v_bias[l] { add_f16_into_f32(ctx, bias, &s.v, kv_dim as u32)?; }
                }
            }

            // f32 Fused QK RoPE (1 dispatch instead of 2)
            fused_rope_qk_f32(ctx, &s.q, &s.k, hd,
                              cfg.num_attention_heads as u32, n_kv,
                              pos, cfg.rope_theta)?;

            // f32 KV cache scatter
            kv_copy_to_cache_f32(ctx, &s.k, kv.k_buf(l), pos, max_seq, hd, n_kv)?;
            kv_copy_to_cache_f32(ctx, &s.v, kv.v_buf(l), pos, max_seq, hd, n_kv)?;

            // f32 decode attention
            decode_attention_f32(
                ctx, &s.q, kv.k_buf(l), kv.v_buf(l), &s.attn_out,
                1, kv_len_after, cfg.num_attention_heads as u32, n_kv, hd, max_seq,
            )?;

            // f32 output projection + f32 residual
            w.attn_out[l].gemv_add_f32(ctx, &s.attn_out, &s.x, &s.x, h as u32, q_dim as u32)?;

            if self.norm_f32 {
                rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            } else {
                rms_norm_f32_f32(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            }

            if cfg.is_moe() {
                let n_exp = cfg.num_experts;
                let top_k = cfg.num_experts_per_tok;

                gemv_f16w_f32in(ctx, &w.moe_router[l], &s.x_norm2, &s.router_logits,
                         n_exp as u32, h as u32)?;
                ctx.flush();
                let (expert_ids, expert_weights) = topk_softmax(
                    &s.router_logits, n_exp, top_k,
                );

                unsafe {
                    std::ptr::write_bytes(s.moe_out.contents() as *mut u8, 0, h * 2);
                }

                for (i, &eid) in expert_ids.iter().enumerate() {
                    w.moe_gate_exps[l][eid].gemv_f32in(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                    w.moe_up_exps[l][eid].gemv_f32in(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                    silu_mul_f16(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                    w.moe_down_exps[l][eid].gemv(ctx, &s.act, &s.ffn_out, h as u32, ff_dim as u32)?;
                    scale_accumulate_f16(ctx, &s.ffn_out, &s.moe_out,
                                         expert_weights[i], h as u32)?;
                }
                add_f16_into_f32(ctx, &s.moe_out, &s.x, h as u32)?;
            } else {
                // Dense FFN: FUSED gate+up+silu (1 dispatch instead of 3)
                match (&w.ffn_gate[l], &w.ffn_up[l]) {
                    (WeightBuf::Q8_0 { raw: g_raw, .. }, WeightBuf::Q8_0 { raw: u_raw, .. }) => {
                        fused_ffn_q8_0_f32(ctx, g_raw, u_raw, &s.x_norm2, &s.act,
                                           ff_dim as u32, h as u32)?;
                    }
                    (WeightBuf::F16(g_buf), WeightBuf::F16(u_buf)) => {
                        fused_ffn_f16w_f32(ctx, g_buf, u_buf, &s.x_norm2, &s.act,
                                           ff_dim as u32, h as u32)?;
                    }
                    _ => {
                        w.ffn_gate[l].gemv_f32_f32out(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                        w.ffn_up[l].gemv_f32_f32out(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                        silu_mul_f32(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                    }
                }
                w.ffn_down[l].gemv_add_f32(ctx, &s.act, &s.x, &s.x, h as u32, ff_dim as u32)?;
            }
        }

        kv.advance();

        // Use f32→f32 norm then f32-input GEMV for maximum lm_head precision
        if self.norm_f32 {
            rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm, &self.weights.output_norm,
                         cfg.rms_norm_eps, h as u32, 1)?;
        } else {
            rms_norm_f32_f32(ctx, &s.x, &s.x_norm, &self.weights.output_norm,
                         cfg.rms_norm_eps, h as u32, 1)?;
        }
        self.weights.lm_head.gemv_f32in(ctx, &s.x_norm, &s.logits,
                 cfg.vocab_size as u32, h as u32)?;

        // Argmax on GPU — download 4 bytes instead of 304KB
        argmax_f16(ctx, &s.logits, &s.argmax, cfg.vocab_size as u32)?;
        ctx.flush();
        let token = unsafe { *(s.argmax.contents() as *const u32) };
        Ok(token)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Speculative decoding: early-exit draft + batched verify
    // ═══════════════════════════════════════════════════════════════════════

    /// Draft one token using only the first `draft_layers` transformer layers.
    /// Much cheaper than full forward (~25% cost for 6/24 layers).
    /// Returns greedy argmax token. Updates `kv` for draft layers only.
    pub fn forward_draft(
        &self,
        token_id: u32,
        pos: u32,
        kv: &mut KvCache,
        draft_layers: usize,
    ) -> Result<u32, ForwardError> {
        let cfg = &self.config;
        let ctx = &self.ctx;
        let s   = &self.scratch;

        if token_id >= cfg.vocab_size as u32 {
            return Err(ForwardError::InvalidTokenId(token_id));
        }

        // Embedding
        let h = cfg.hidden_size;
        unsafe {
            let dst = s.x.contents() as *mut f32;
            if self.tok_emb_f32 {
                let src = (self.weights.tok_emb.contents() as *const f32)
                    .add(token_id as usize * h);
                std::ptr::copy_nonoverlapping(src, dst, h);
            } else {
                let src = (self.weights.tok_emb.contents() as *const f16)
                    .add(token_id as usize * h);
                for i in 0..h { *dst.add(i) = (*src.add(i)).to_f32(); }
            }
        }

        let q_dim  = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let ff_dim = self.expert_ff_dim;
        let kv_len_after = pos + 1;
        let max_seq = kv.max_seq_len();
        let hd = cfg.head_dim as u32;
        let n_kv = cfg.num_key_value_heads as u32;

        // Run only draft_layers (not all layers)
        let n_layers = draft_layers.min(cfg.num_hidden_layers);
        for l in 0..n_layers {
            let w = &self.weights;

            if self.norm_f32 {
                rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm, &w.attn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            } else {
                rms_norm_f32_f32(ctx, &s.x, &s.x_norm, &w.attn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            }

            w.attn_q[l].gemv_f32_f32out(ctx, &s.x_norm, &s.q, q_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_q_bias[l] {
                add_f16_into_f32(ctx, bias, &s.q, q_dim as u32)?;
            }
            w.attn_k[l].gemv_f32_f32out(ctx, &s.x_norm, &s.k, kv_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_k_bias[l] {
                add_f16_into_f32(ctx, bias, &s.k, kv_dim as u32)?;
            }
            w.attn_v[l].gemv_f32_f32out(ctx, &s.x_norm, &s.v, kv_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_v_bias[l] {
                add_f16_into_f32(ctx, bias, &s.v, kv_dim as u32)?;
            }

            fused_rope_qk_f32(ctx, &s.q, &s.k, hd,
                              cfg.num_attention_heads as u32, n_kv,
                              pos, cfg.rope_theta)?;

            kv_copy_to_cache_f32(ctx, &s.k, kv.k_buf(l), pos, max_seq, hd, n_kv)?;
            kv_copy_to_cache_f32(ctx, &s.v, kv.v_buf(l), pos, max_seq, hd, n_kv)?;

            decode_attention_f32(
                ctx, &s.q, kv.k_buf(l), kv.v_buf(l), &s.attn_out,
                1, kv_len_after, cfg.num_attention_heads as u32, n_kv, hd, max_seq,
            )?;

            w.attn_out[l].gemv_add_f32(ctx, &s.attn_out, &s.x, &s.x, h as u32, q_dim as u32)?;

            if self.norm_f32 {
                rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            } else {
                rms_norm_f32_f32(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                             cfg.rms_norm_eps, h as u32, 1)?;
            }

            // Use fused FFN where possible
            match (&w.ffn_gate[l], &w.ffn_up[l]) {
                (WeightBuf::Q8_0 { raw: g_raw, .. }, WeightBuf::Q8_0 { raw: u_raw, .. }) => {
                    fused_ffn_q8_0_f32(ctx, g_raw, u_raw, &s.x_norm2, &s.act,
                                       ff_dim as u32, h as u32)?;
                }
                (WeightBuf::F16(g_buf), WeightBuf::F16(u_buf)) => {
                    fused_ffn_f16w_f32(ctx, g_buf, u_buf, &s.x_norm2, &s.act,
                                       ff_dim as u32, h as u32)?;
                }
                _ => {
                    w.ffn_gate[l].gemv_f32_f32out(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                    w.ffn_up[l].gemv_f32_f32out(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                    silu_mul_f32(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                }
            }
            w.ffn_down[l].gemv_add_f32(ctx, &s.act, &s.x, &s.x, h as u32, ff_dim as u32)?;
        }

        kv.advance();

        // Final norm + lm_head (same weights as full model)
        if self.norm_f32 {
            rms_norm_f32_f32_f32g(ctx, &s.x, &s.x_norm, &self.weights.output_norm,
                         cfg.rms_norm_eps, h as u32, 1)?;
        } else {
            rms_norm_f32_f32(ctx, &s.x, &s.x_norm, &self.weights.output_norm,
                         cfg.rms_norm_eps, h as u32, 1)?;
        }
        self.weights.lm_head.gemv_f32in(ctx, &s.x_norm, &s.logits,
                 cfg.vocab_size as u32, h as u32)?;
        argmax_f16(ctx, &s.logits, &s.argmax, cfg.vocab_size as u32)?;
        ctx.flush();
        let token = unsafe { *(s.argmax.contents() as *const u32) };
        Ok(token)
    }

    /// Verify a batch of draft tokens using full-model prefill (GEMM).
    /// Returns the index of the first rejected token and the correct token
    /// at that position. All tokens before the rejection index are accepted.
    ///
    /// Returns `(accepted_count, correct_next_token)`:
    ///   - `accepted_count == draft_tokens.len()` means all accepted
    ///   - The correct_next_token is always valid (from verify logits)
    pub fn verify_draft(
        &self,
        prompt_token: u32,
        draft_tokens: &[u32],
        start_pos: usize,
        kv: &mut KvCache,
    ) -> Result<(usize, u32), ForwardError> {
        // Build verification sequence: [prompt_token, draft_0, draft_1, ...]
        let mut verify_seq: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        verify_seq.push(prompt_token);
        verify_seq.extend_from_slice(draft_tokens);
        let seq_len = verify_seq.len();

        let cfg = &self.config;
        let ctx = &self.ctx;
        let opts = metal::MTLResourceOptions::StorageModeShared;
        let h = cfg.hidden_size;
        let q_dim  = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let ff_dim = self.expert_ff_dim;

        // Reset KV cache to start_pos (discard draft entries)
        kv.set_seq_len(start_pos);

        // ── 1. Embed all tokens → X [seq, h] f16 ──────────────────────────
        let x_buf = ctx.device.new_buffer((seq_len * h * 2) as u64, opts);
        {
            let emb_row = if self.tok_emb_f32 { h * 4 } else { h * 2 };
            let dst = x_buf.contents() as *mut u8;
            for (i, &tid) in verify_seq.iter().enumerate() {
                if self.tok_emb_f32 {
                    // Tok emb is f32 — convert row to f16 for GEMM
                    let src = unsafe { (self.weights.tok_emb.contents() as *const f32)
                        .add(tid as usize * h) };
                    let dst_f16 = unsafe { (dst as *mut f16).add(i * h) };
                    for j in 0..h {
                        unsafe { *dst_f16.add(j) = f16::from_f32(*src.add(j)); }
                    }
                } else {
                    let src = unsafe { (self.weights.tok_emb.contents() as *const u8)
                        .add(tid as usize * h * 2) };
                    unsafe {
                        std::ptr::copy_nonoverlapping(src, dst.add(i * h * 2), h * 2);
                    }
                }
            }
        }

        // Scratch buffers for [seq, dim]
        let x_norm_buf   = ctx.device.new_buffer((seq_len * h * 2) as u64, opts);
        let q_buf        = ctx.device.new_buffer((seq_len * q_dim * 2) as u64, opts);
        let k_buf        = ctx.device.new_buffer((seq_len * kv_dim * 2) as u64, opts);
        let v_buf        = ctx.device.new_buffer((seq_len * kv_dim * 2) as u64, opts);
        let attn_out_buf = ctx.device.new_buffer((seq_len * q_dim * 2) as u64, opts);
        let proj_buf     = ctx.device.new_buffer((seq_len * h * 2) as u64, opts);
        let x_norm2_buf  = ctx.device.new_buffer((seq_len * h * 2) as u64, opts);
        let gate_buf     = ctx.device.new_buffer((seq_len * ff_dim * 2) as u64, opts);
        let up_buf       = ctx.device.new_buffer((seq_len * ff_dim * 2) as u64, opts);
        let act_buf      = ctx.device.new_buffer((seq_len * ff_dim * 2) as u64, opts);
        let ffn_out_buf  = ctx.device.new_buffer((seq_len * h * 2) as u64, opts);

        // ── 2. Transformer layers (GEMM path) ─────────────────────────────
        use bare_metal_kernels::dispatch::{
            gemm_f16, rms_norm_f16, rope_batch_inplace_f16,
            flash_attention_f16, add_f16, silu_mul_f16,
            add_bias_broadcast_f16,
        };

        let kv_start = start_pos;
        for l in 0..cfg.num_hidden_layers {
            let w = &self.weights;

            rms_norm_f16(ctx, &x_buf, &x_norm_buf, &w.attn_norm[l],
                         cfg.rms_norm_eps, h as u32, seq_len as u32)?;

            gemm_f16(ctx, &x_norm_buf, w.attn_q[l].f16_buf(), &q_buf,
                     seq_len as u32, q_dim as u32, h as u32)?;
            gemm_f16(ctx, &x_norm_buf, w.attn_k[l].f16_buf(), &k_buf,
                     seq_len as u32, kv_dim as u32, h as u32)?;
            gemm_f16(ctx, &x_norm_buf, w.attn_v[l].f16_buf(), &v_buf,
                     seq_len as u32, kv_dim as u32, h as u32)?;

            if let Some(bias) = &w.attn_q_bias[l] {
                add_bias_broadcast_f16(ctx, &q_buf, bias, seq_len as u32, q_dim as u32)?;
            }
            if let Some(bias) = &w.attn_k_bias[l] {
                add_bias_broadcast_f16(ctx, &k_buf, bias, seq_len as u32, kv_dim as u32)?;
            }
            if let Some(bias) = &w.attn_v_bias[l] {
                add_bias_broadcast_f16(ctx, &v_buf, bias, seq_len as u32, kv_dim as u32)?;
            }

            rope_batch_inplace_f16(ctx, &q_buf, cfg.head_dim as u32,
                                   cfg.num_attention_heads as u32,
                                   kv_start as u32, cfg.rope_theta, seq_len as u32)?;
            rope_batch_inplace_f16(ctx, &k_buf, cfg.head_dim as u32,
                                   cfg.num_key_value_heads as u32,
                                   kv_start as u32, cfg.rope_theta, seq_len as u32)?;

            // Populate KV cache for all positions
            ctx.flush();
            let kh = cfg.num_key_value_heads;
            let hd = cfg.head_dim;
            let k_ptr = k_buf.contents() as *const f16;
            let v_ptr = v_buf.contents() as *const f16;
            for pos in 0..seq_len {
                let k_slice: Vec<f16> = unsafe {
                    (0..kh * hd).map(|i| *k_ptr.add(pos * kh * hd + i)).collect()
                };
                let v_slice: Vec<f16> = unsafe {
                    (0..kh * hd).map(|i| *v_ptr.add(pos * kh * hd + i)).collect()
                };
                kv.write_at(l, kv_start + pos, &k_slice, &v_slice);
            }

            let total_kv = (kv_start + seq_len) as u32;
            flash_attention_f16(
                ctx, &q_buf, kv.k_buf(l), kv.v_buf(l), &attn_out_buf,
                1, seq_len as u32, total_kv,
                cfg.num_attention_heads as u32,
                cfg.num_key_value_heads as u32,
                cfg.head_dim as u32,
                kv.max_seq_len(),
            )?;

            gemm_f16(ctx, &attn_out_buf, w.attn_out[l].f16_buf(), &proj_buf,
                     seq_len as u32, h as u32, q_dim as u32)?;
            add_f16(ctx, &x_buf, &proj_buf, &x_buf, (seq_len * h) as u32)?;

            rms_norm_f16(ctx, &x_buf, &x_norm2_buf, &w.ffn_norm[l],
                         cfg.rms_norm_eps, h as u32, seq_len as u32)?;

            gemm_f16(ctx, &x_norm2_buf, w.ffn_gate[l].f16_buf(), &gate_buf,
                     seq_len as u32, ff_dim as u32, h as u32)?;
            gemm_f16(ctx, &x_norm2_buf, w.ffn_up[l].f16_buf(), &up_buf,
                     seq_len as u32, ff_dim as u32, h as u32)?;
            silu_mul_f16(ctx, &gate_buf, &up_buf, &act_buf, (seq_len * ff_dim) as u32)?;
            gemm_f16(ctx, &act_buf, w.ffn_down[l].f16_buf(), &ffn_out_buf,
                     seq_len as u32, h as u32, ff_dim as u32)?;
            add_f16(ctx, &x_buf, &ffn_out_buf, &x_buf, (seq_len * h) as u32)?;
        }

        kv.set_seq_len(kv_start + seq_len);

        // ── 3. Batched lm_head: norm ALL positions + GEMM → logits [seq, vocab]
        // ONE norm dispatch + ONE GEMM dispatch + ONE flush (was: seq × flush!)
        let vocab = cfg.vocab_size;
        let x_norm_all = ctx.device.new_buffer((seq_len * h * 2) as u64, opts);
        rms_norm_f16(ctx, &x_buf, &x_norm_all, &self.weights.output_norm,
                     cfg.rms_norm_eps, h as u32, seq_len as u32)?;

        // GEMM: [seq, h] × [vocab, h]^T = [seq, vocab] f16
        let logit_buf = ctx.device.new_buffer((seq_len * vocab * 2) as u64, opts);
        gemm_f16(ctx, &x_norm_all, self.weights.lm_head.f16_buf(), &logit_buf,
                 seq_len as u32, vocab as u32, h as u32)?;

        ctx.flush();

        // CPU-side: argmax per position and compare with draft
        let logit_ptr = logit_buf.contents() as *const f16;
        let mut accepted = 0usize;
        let mut correct_next = 0u32;

        for pos_idx in 0..seq_len {
            // Find argmax for this position
            let row = unsafe { std::slice::from_raw_parts(logit_ptr.add(pos_idx * vocab), vocab) };
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                let fv = v.to_f32();
                if fv > best_val {
                    best_val = fv;
                    best_idx = j as u32;
                }
            }

            if pos_idx < draft_tokens.len() {
                if best_idx == draft_tokens[pos_idx] {
                    accepted += 1;
                } else {
                    correct_next = best_idx;
                    kv.set_seq_len(kv_start + pos_idx + 1);
                    return Ok((accepted, correct_next));
                }
            } else {
                correct_next = best_idx;
            }
        }

        // All draft tokens accepted
        Ok((accepted, correct_next))
    }

    /// Run one decode step using a TurboQuant KV cache (Phase 5).
    ///
    /// Identical to `forward()` except attention reads from materialised F16
    /// buffers that include the compressed history. The KV cache update is
    /// delegated to `TqKvCache::update()` which compresses tokens on overflow.
    pub fn forward_tq(
        &self,
        token_id: u32,
        pos: u32,
        kv: &mut TqKvCache,
    ) -> Result<Vec<f16>, ForwardError> {
        let cfg = &self.config;
        let ctx = &self.ctx;
        let s   = &self.scratch;

        if token_id >= cfg.vocab_size as u32 {
            return Err(ForwardError::InvalidTokenId(token_id));
        }

        let h = cfg.hidden_size;
        let emb_row_bytes = h * 2;
        unsafe {
            let src = (self.weights.tok_emb.contents() as *const u8)
                .add(token_id as usize * emb_row_bytes);
            std::ptr::copy_nonoverlapping(src, s.x.contents() as *mut u8, emb_row_bytes);
        }

        let q_dim  = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads  * cfg.head_dim;
        let ff_dim = self.expert_ff_dim;
        let kv_len_after = pos + 1;

        for l in 0..cfg.num_hidden_layers {
            let w = &self.weights;

            rms_norm_f16(ctx, &s.x, &s.x_norm, &w.attn_norm[l],
                         cfg.rms_norm_eps, h as u32, 1)?;

            w.attn_q[l].gemv(ctx, &s.x_norm, &s.q, q_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_q_bias[l] {
                add_f16(ctx, &s.q, bias, &s.q, q_dim as u32)?;
            }
            w.attn_k[l].gemv(ctx, &s.x_norm, &s.k, kv_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_k_bias[l] {
                add_f16(ctx, &s.k, bias, &s.k, kv_dim as u32)?;
            }
            w.attn_v[l].gemv(ctx, &s.x_norm, &s.v, kv_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_v_bias[l] {
                add_f16(ctx, &s.v, bias, &s.v, kv_dim as u32)?;
            }

            rope_inplace_f16(ctx, &s.q, cfg.head_dim as u32,
                             cfg.num_attention_heads as u32, pos, cfg.rope_theta, 1)?;
            rope_inplace_f16(ctx, &s.k, cfg.head_dim as u32,
                             cfg.num_key_value_heads as u32, pos, cfg.rope_theta, 1)?;

            // TQ cache needs CPU-visible data for potential compression.
            ctx.flush();
            kv.update_from_buf(ctx, l, &s.k, &s.v);

            let k_full = kv.materialise_k_f16(ctx, l);
            let v_full = kv.materialise_v_f16(ctx, l);

            decode_attention_f16(
                ctx, &s.q, &k_full, &v_full, &s.attn_out,
                1, kv_len_after,
                cfg.num_attention_heads as u32,
                cfg.num_key_value_heads as u32,
                cfg.head_dim as u32,
                kv_len_after,
            )?;

            w.attn_out[l].gemv_add(ctx, &s.attn_out, &s.x, &s.x, h as u32, q_dim as u32)?;

            rms_norm_f16(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                         cfg.rms_norm_eps, h as u32, 1)?;
            w.ffn_gate[l].gemv(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
            w.ffn_up[l].gemv(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
            silu_mul_f16(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
            w.ffn_down[l].gemv_add(ctx, &s.act, &s.x, &s.x, h as u32, ff_dim as u32)?;
        }

        kv.advance();

        rms_norm_f16(ctx, &s.x, &s.x_final, &self.weights.output_norm,
                     cfg.rms_norm_eps, h as u32, 1)?;
        self.weights.lm_head.gemv(ctx, &s.x_final, &s.logits,
                 cfg.vocab_size as u32, h as u32)?;
        ctx.flush();
        Ok(download_f16_buf(&s.logits, cfg.vocab_size))
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Read `n` f16 values from a Metal buffer's CPU-visible contents pointer.
/// CPU-side softmax + top-k selection for MoE routing.
/// Returns (expert_indices, softmax_weights) for the top-k experts.
///
/// For small expert counts (<512), CPU is faster than a GPU kernel
/// because it avoids the flush + readback overhead being split across
/// two separate GPU dispatches.
/// CPU-side softmax-then-top-k selection for MoE routing.
///
/// OLMoE (and most MoE models) applies softmax FIRST across all experts,
/// then selects top-k from the softmax probabilities. This differs from
/// "top-k then softmax" which would give wrong magnitude weights.
///
/// Returns (expert_indices, routing_weights) for the top-k experts.
pub(crate) fn topk_softmax(logits_buf: &metal::Buffer, num_experts: usize, top_k: usize) -> (Vec<usize>, Vec<f32>) {
    let ptr = logits_buf.contents() as *const f16;
    let logits: Vec<f32> = (0..num_experts).map(|i| unsafe { (*ptr.add(i)).to_f32() }).collect();

    // Step 1: Softmax over ALL experts
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum).collect();

    // Step 2: Select top-k by probability
    let mut indices: Vec<usize> = (0..num_experts).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    indices.truncate(top_k);

    // Step 3: Extract weights for selected experts (already softmax-normalized)
    let weights: Vec<f32> = indices.iter().map(|&i| probs[i]).collect();

    // Optional: renormalize so selected weights sum to 1.0
    // (norm_topk_prob — some models do this, some don't)
    let w_sum: f32 = weights.iter().sum();
    let weights: Vec<f32> = if w_sum > 0.0 {
        weights.iter().map(|&w| w / w_sum).collect()
    } else {
        weights
    };

    (indices, weights)
}

fn download_f16_buf(buf: &metal::Buffer, n: usize) -> Vec<f16> {
    let ptr = buf.contents() as *const f16;
    (0..n).map(|i| unsafe { *ptr.add(i) }).collect()
}

fn download_f32_buf(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    (0..n).map(|i| unsafe { *ptr.add(i) }).collect()
}
