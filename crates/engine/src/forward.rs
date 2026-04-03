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
        add_f16, argmax_f16, decode_attention_f16, dequant_q4k_f16,
        flash_attention_f16, gemv_f16, gemv_add_f16, gemv_q4k_f16, gemv_q4k_add_f16,
        kv_copy_to_cache_f16, mul_f16, rms_norm_f16,
        rope_inplace_f16, scale_accumulate_f16, silu_mul_f16,
        Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS,
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

/// A weight buffer that is either pre-dequantized F16 or raw Q4K packed bytes.
/// Q4K weights use the fused `gemv_q4k_f16` kernel that dequantizes on-the-fly,
/// reading ~3.5x less memory bandwidth than the F16 path.
pub(crate) enum WeightBuf {
    F16(metal::Buffer),
    /// Q4K weights: raw packed buffer for fused GEMV + dequanted F16 for GEMM prefill.
    Q4K { raw: metal::Buffer, f16: metal::Buffer },
}

impl WeightBuf {
    /// Dispatch GEMV using the appropriate kernel for this weight format.
    /// Q4K weights use the fused kernel that reads ~3.5x less bandwidth.
    pub(crate) fn gemv(
        &self,
        ctx: &MetalContext,
        input: &metal::Buffer,
        output: &metal::Buffer,
        m: u32,
        k: u32,
    ) -> Result<(), ForwardError> {
        match self {
            WeightBuf::F16(buf) => {
                gemv_f16(ctx, buf, input, output, m, k)?;
            }
            WeightBuf::Q4K { raw, .. } => {
                gemv_q4k_f16(ctx, raw, input, output, m, k)?;
            }
        }
        Ok(())
    }

    /// Fused GEMV + residual add: `output[i] = (weight * input)[i] + residual[i]`.
    /// Saves one kernel dispatch and one buffer pass vs separate gemv + add.
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
            WeightBuf::F16(buf) => {
                gemv_add_f16(ctx, buf, input, output, residual, m, k)?;
            }
            WeightBuf::Q4K { raw, .. } => {
                gemv_q4k_add_f16(ctx, raw, input, output, residual, m, k)?;
            }
        }
        Ok(())
    }

    /// Get the F16 buffer (for GEMM prefill and other non-GEMV uses).
    pub(crate) fn f16_buf(&self) -> &metal::Buffer {
        match self {
            WeightBuf::F16(buf) => buf,
            WeightBuf::Q4K { f16, .. } => f16,
        }
    }
}

pub(crate) struct LayerWeights {
    pub(crate) tok_emb:     metal::Buffer,              // [vocab_size, hidden_size] f16 (always F16 for embedding lookup)
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
            x:        ctx.new_buffer(h * 2),
            x_norm:   ctx.new_buffer(h * 2),
            q:        ctx.new_buffer(q_dim * 2),
            k:        ctx.new_buffer(kv_dim * 2),
            v:        ctx.new_buffer(kv_dim * 2),
            attn_out: ctx.new_buffer(q_dim * 2),
            proj:     ctx.new_buffer(h * 2),
            x_norm2:  ctx.new_buffer(h * 2),
            gate:     ctx.new_buffer(ff_dim * 2),
            up:       ctx.new_buffer(ff_dim * 2),
            act:      ctx.new_buffer(ff_dim * 2),
            ffn_out:  ctx.new_buffer(h * 2),
            x_final:  ctx.new_buffer(h * 2),
            logits:   ctx.new_buffer(cfg.vocab_size * 2),
            argmax:   ctx.new_buffer(4),  // 1 × u32
            router_logits: ctx.new_buffer(cfg.num_experts.max(1) * 2),
            moe_out:       ctx.new_buffer(h * 2),
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
    /// May differ from config.intermediate_size.
    pub(crate) expert_ff_dim: usize,
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
            for i in 0..BLOCK_ELEMS {
                let low4  = if i % 2 == 0 { qs[i / 2] & 0x0F } else { (qs[i / 2] >> 4) & 0x0F };
                let high1 = (qh[i / 8] >> (i % 8)) & 1;
                let q5    = (low4 as i32) | ((high1 as i32) << 4);
                f16_vec.push(f16::from_f32(d * (q5 - 16) as f32));
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
            for i in 0..256usize {
                let ql_byte = ql[i / 2];
                let q_low  = if i % 2 == 0 { ql_byte & 0xF } else { (ql_byte >> 4) & 0xF };
                let qh_byte = qh[i / 4];
                let q_high = (qh_byte >> ((i % 4) * 2)) & 0x3;
                let q6 = ((q_low as i32) | ((q_high as i32) << 4)) - 32;
                let group = i / 16;
                let sc = scales[group] as i8 as f32;
                f16_vec.push(f16::from_f32(d * sc * q6 as f32));
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
            for i in 0..BLOCK_ELEMS {
                let sub   = i / 32;
                // Get 6-bit scale and min for this sub-block (same packing as Q4K)
                let (sc, m) = get_scale_min_k4(sub, scales);
                let actual_scale = d    * sc as f32;
                let actual_min   = dmin * m  as f32;
                // Lower 4 bits from qs
                let byte_idx = (sub * 32 + i % 32) / 2;
                let nibble   = if i % 2 == 0 { qs[byte_idx] & 0x0F } else { qs[byte_idx] >> 4 };
                // High bit from qh
                let high_bit = (qh[i / 8] >> (i % 8)) & 1;
                let q5 = nibble as u32 | ((high_bit as u32) << 4);
                f16_vec.push(f16::from_f32(actual_scale * q5 as f32 - actual_min));
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

/// Upload a weight tensor, keeping Q4K weights in packed form for fused GEMV.
/// Returns a `WeightBuf` that dispatches to the appropriate GEMV kernel.
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

    // Only use fused Q4K path if K is a multiple of 256 (Q4K block size)
    if tensor.dtype == DType::Q4K && (k as usize % Q4K_BLOCK_ELEMS == 0) {
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

    // Fall back to pre-dequantized F16 for other dtypes
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

        // Global weights
        let tok_emb     = upload_weight(&ctx, model, "token_embd.weight")?;
        let output_norm = upload_weight(&ctx, model, "output_norm.weight")?;

        let h      = cfg.hidden_size as u32;
        let q_dim  = (cfg.num_attention_heads * cfg.head_dim) as u32;
        let kv_dim = (cfg.num_key_value_heads * cfg.head_dim) as u32;
        let mut ff_dim = cfg.intermediate_size as u32;
        let vocab  = cfg.vocab_size as u32;

        // lm_head may be tied to token_embd (Qwen2 ties them)
        let lm_head = if model.tensors.contains_key("output.weight") {
            upload_weight_wb(&ctx, model, "output.weight", vocab, h)?
        } else {
            WeightBuf::F16(upload_weight(&ctx, model, "token_embd.weight")?)
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
            attn_norm.push(upload_weight(&ctx, model, &format!("blk.{l}.attn_norm.weight"))?);
            attn_q.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.attn_q.weight"), q_dim, h)?);
            attn_k.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.attn_k.weight"), kv_dim, h)?);
            attn_v.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.attn_v.weight"), kv_dim, h)?);
            attn_q_bias.push(upload_optional(&ctx, model, &format!("blk.{l}.attn_q.bias"))?);
            attn_k_bias.push(upload_optional(&ctx, model, &format!("blk.{l}.attn_k.bias"))?);
            attn_v_bias.push(upload_optional(&ctx, model, &format!("blk.{l}.attn_v.bias"))?);
            attn_out.push(upload_weight_wb(&ctx, model, &format!("blk.{l}.attn_output.weight"), h, q_dim)?);
            ffn_norm.push(upload_weight(&ctx, model, &format!("blk.{l}.ffn_norm.weight"))?);

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
                    // Upload the full tensor as F16, then create per-expert views.
                    let gate_all = upload_weight(&ctx, model, &format!("blk.{l}.ffn_gate_exps.weight"))?;
                    let up_all   = upload_weight(&ctx, model, &format!("blk.{l}.ffn_up_exps.weight"))?;
                    let down_all = upload_weight(&ctx, model, &format!("blk.{l}.ffn_down_exps.weight"))?;

                    // Derive actual expert FFN dimension from tensor size rather than
                    // trusting config.intermediate_size (which may differ for MoE models).
                    // gate_all is [num_experts, expert_ff_dim, hidden_size] in F16.
                    let gate_total_f16_bytes = gate_all.length() as usize;
                    let expert_ff_dim = gate_total_f16_bytes / (n_experts * (h as usize) * 2);
                    let gate_expert_bytes = expert_ff_dim * (h as usize) * 2;
                    let up_expert_bytes   = gate_expert_bytes;
                    // down is [num_experts, hidden_size, expert_ff_dim]
                    let down_expert_bytes = (h as usize) * expert_ff_dim * 2;

                    if l == 0 {
                        tracing::info!(
                            expert_ff_dim,
                            config_ff_dim = ff_dim as usize,
                            "MoE expert FFN dimension (derived from tensor)"
                        );
                        // Update ff_dim for scratch buffer sizing and GEMV dispatches
                        ff_dim = expert_ff_dim as u32;
                    }

                    for e in 0..n_experts {
                        // Create per-expert sub-buffers by copying from the merged tensor.
                        // (Metal doesn't support buffer sub-ranges well, so we copy.)
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
                        layer_gates.push(WeightBuf::F16(g_buf));
                        layer_ups.push(WeightBuf::F16(u_buf));
                        layer_downs.push(WeightBuf::F16(d_buf));
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
    ) -> Result<Vec<f16>, ForwardError> {
        let cfg = &self.config;
        let ctx = &self.ctx;
        let s   = &self.scratch;

        if token_id >= cfg.vocab_size as u32 {
            return Err(ForwardError::InvalidTokenId(token_id));
        }

        // ── 1. Token embedding lookup ─────────────────────────────────────────
        let h = cfg.hidden_size;
        let emb_row_bytes = h * 2;
        unsafe {
            let src = (self.weights.tok_emb.contents() as *const u8)
                .add(token_id as usize * emb_row_bytes);
            std::ptr::copy_nonoverlapping(src, s.x.contents() as *mut u8, emb_row_bytes);
        }

        let q_dim  = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads  * cfg.head_dim;
        let ff_dim = self.expert_ff_dim; // use actual expert FFN dim (may differ from config for MoE)
        let kv_len_after = pos + 1;
        let max_seq = kv.max_seq_len();
        let hd = cfg.head_dim as u32;
        let n_kv = cfg.num_key_value_heads as u32;

        // ── 2. Transformer layers ─────────────────────────────────────────────
        for l in 0..cfg.num_hidden_layers {
            let w = &self.weights;

            // a) Attention pre-norm
            rms_norm_f16(ctx, &s.x, &s.x_norm, &w.attn_norm[l],
                         cfg.rms_norm_eps, h as u32, 1)?;

            // b) Q projection + optional bias
            w.attn_q[l].gemv(ctx, &s.x_norm, &s.q, q_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_q_bias[l] {
                add_f16(ctx, &s.q, bias, &s.q, q_dim as u32)?;
            }

            // c) K projection + optional bias
            w.attn_k[l].gemv(ctx, &s.x_norm, &s.k, kv_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_k_bias[l] {
                add_f16(ctx, &s.k, bias, &s.k, kv_dim as u32)?;
            }

            // d) V projection + optional bias
            w.attn_v[l].gemv(ctx, &s.x_norm, &s.v, kv_dim as u32, h as u32)?;
            if let Some(bias) = &w.attn_v_bias[l] {
                add_f16(ctx, &s.v, bias, &s.v, kv_dim as u32)?;
            }

            // e) RoPE in-place on Q and K
            rope_inplace_f16(ctx, &s.q, hd,
                             cfg.num_attention_heads as u32,
                             pos, cfg.rope_theta, 1)?;
            rope_inplace_f16(ctx, &s.k, hd, n_kv, pos, cfg.rope_theta, 1)?;

            // f) Write K/V into cache — GPU-side scatter (no CPU flush needed!)
            kv_copy_to_cache_f16(ctx, &s.k, kv.k_buf(l), pos, max_seq, hd, n_kv)?;
            kv_copy_to_cache_f16(ctx, &s.v, kv.v_buf(l), pos, max_seq, hd, n_kv)?;

            // g) Decode attention (q_len=1 specialised kernel)
            decode_attention_f16(
                ctx, &s.q, kv.k_buf(l), kv.v_buf(l), &s.attn_out,
                1, kv_len_after,
                cfg.num_attention_heads as u32, n_kv, hd, max_seq,
            )?;

            // h) Output projection + residual (fused: saves 1 dispatch + 1 buffer pass)
            w.attn_out[l].gemv_add(ctx, &s.attn_out, &s.x, &s.x, h as u32, q_dim as u32)?;

            // i) FFN pre-norm
            rms_norm_f16(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                         cfg.rms_norm_eps, h as u32, 1)?;

            if cfg.is_moe() {
                // ── MoE FFN: router → top-k → sparse expert dispatch ─────────
                let n_exp = cfg.num_experts;
                let top_k = cfg.num_experts_per_tok;

                // Router: [num_experts, hidden_size] @ x_norm2 → [num_experts]
                gemv_f16(ctx, &w.moe_router[l], &s.x_norm2, &s.router_logits,
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
                    w.moe_gate_exps[l][eid].gemv(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                    w.moe_up_exps[l][eid].gemv(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                    silu_mul_f16(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                    w.moe_down_exps[l][eid].gemv(ctx, &s.act, &s.ffn_out, h as u32, ff_dim as u32)?;

                    // GPU-side weighted accumulate: moe_out += ffn_out * weight
                    scale_accumulate_f16(ctx, &s.ffn_out, &s.moe_out,
                                         expert_weights[i], h as u32)?;
                }

                // Residual: x += moe_out (all async, no flush needed here)
                add_f16(ctx, &s.x, &s.moe_out, &s.x, h as u32)?;
            } else {
                // ── Dense FFN ─────────────────────────────────────────────────
                w.ffn_gate[l].gemv(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                w.ffn_up[l].gemv(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                silu_mul_f16(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                w.ffn_down[l].gemv_add(ctx, &s.act, &s.x, &s.x, h as u32, ff_dim as u32)?;
            }
        }

        // All layers done — advance KV cache position.
        kv.advance();

        // ── 3. Final norm + lm_head ───────────────────────────────────────────
        rms_norm_f16(ctx, &s.x, &s.x_final, &self.weights.output_norm,
                     cfg.rms_norm_eps, h as u32, 1)?;
        self.weights.lm_head.gemv(ctx, &s.x_final, &s.logits,
                 cfg.vocab_size as u32, h as u32)?;

        // ── 4. Download logits — flush for the entire token ───────────────────
        ctx.flush();
        let logits = download_f16_buf(&s.logits, cfg.vocab_size);
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
        let max_seq = kv.max_seq_len();
        let hd = cfg.head_dim as u32;
        let n_kv = cfg.num_key_value_heads as u32;

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

            rope_inplace_f16(ctx, &s.q, hd,
                             cfg.num_attention_heads as u32, pos, cfg.rope_theta, 1)?;
            rope_inplace_f16(ctx, &s.k, hd, n_kv, pos, cfg.rope_theta, 1)?;

            kv_copy_to_cache_f16(ctx, &s.k, kv.k_buf(l), pos, max_seq, hd, n_kv)?;
            kv_copy_to_cache_f16(ctx, &s.v, kv.v_buf(l), pos, max_seq, hd, n_kv)?;

            decode_attention_f16(
                ctx, &s.q, kv.k_buf(l), kv.v_buf(l), &s.attn_out,
                1, kv_len_after, cfg.num_attention_heads as u32, n_kv, hd, max_seq,
            )?;

            w.attn_out[l].gemv_add(ctx, &s.attn_out, &s.x, &s.x, h as u32, q_dim as u32)?;

            rms_norm_f16(ctx, &s.x, &s.x_norm2, &w.ffn_norm[l],
                         cfg.rms_norm_eps, h as u32, 1)?;

            if cfg.is_moe() {
                let n_exp = cfg.num_experts;
                let top_k = cfg.num_experts_per_tok;

                gemv_f16(ctx, &w.moe_router[l], &s.x_norm2, &s.router_logits,
                         n_exp as u32, h as u32)?;
                ctx.flush();
                let (expert_ids, expert_weights) = topk_softmax(
                    &s.router_logits, n_exp, top_k,
                );

                unsafe {
                    std::ptr::write_bytes(s.moe_out.contents() as *mut u8, 0, h * 2);
                }

                for (i, &eid) in expert_ids.iter().enumerate() {
                    w.moe_gate_exps[l][eid].gemv(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                    w.moe_up_exps[l][eid].gemv(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                    silu_mul_f16(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                    w.moe_down_exps[l][eid].gemv(ctx, &s.act, &s.ffn_out, h as u32, ff_dim as u32)?;
                    scale_accumulate_f16(ctx, &s.ffn_out, &s.moe_out,
                                         expert_weights[i], h as u32)?;
                }
                add_f16(ctx, &s.x, &s.moe_out, &s.x, h as u32)?;
            } else {
                w.ffn_gate[l].gemv(ctx, &s.x_norm2, &s.gate, ff_dim as u32, h as u32)?;
                w.ffn_up[l].gemv(ctx, &s.x_norm2, &s.up, ff_dim as u32, h as u32)?;
                silu_mul_f16(ctx, &s.gate, &s.up, &s.act, ff_dim as u32)?;
                w.ffn_down[l].gemv_add(ctx, &s.act, &s.x, &s.x, h as u32, ff_dim as u32)?;
            }
        }

        kv.advance();

        rms_norm_f16(ctx, &s.x, &s.x_final, &self.weights.output_norm,
                     cfg.rms_norm_eps, h as u32, 1)?;
        self.weights.lm_head.gemv(ctx, &s.x_final, &s.logits,
                 cfg.vocab_size as u32, h as u32)?;

        // Argmax on GPU — download 4 bytes instead of 304KB
        argmax_f16(ctx, &s.logits, &s.argmax, cfg.vocab_size as u32)?;
        ctx.flush();
        let token = unsafe { *(s.argmax.contents() as *const u32) };
        Ok(token)
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
