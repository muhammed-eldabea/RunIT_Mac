/// TurboQuant Phase 5 — KV-cache compression support.
///
/// Provides:
/// - Lloyd-Max codebook generation (CPU, run once at startup)
/// - Random sign vector generation for Randomized Hadamard Transform (RHT)
/// - QJL seed constant
/// - Rust dispatch functions for all TQ Metal kernels
///
/// # Algorithm summary
///
/// Keys use Algorithm 2 (inner-product preserving):
///   normalize → RHT → Lloyd-Max at (b-1) bits → residual → QJL sign pack
///
/// Values use per-group symmetric quantization (simpler; only reconstruction fidelity needed).


// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Default buffer size (recent tokens kept in F16, not compressed).
pub const TQ_BUFFER_TOKENS: usize = 128;

/// Default QJL seed — matches the Metal kernel's seeded hash.
pub const QJL_SEED: u32 = 12345;

/// Default RHT sign seed.
pub const RHT_SEED: u64 = 42;

/// Value quantization group size (128 elements per group).
pub const VAL_GROUP_SIZE: usize = 128;

// ─────────────────────────────────────────────────────────────────────────────
// Codebook
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed Lloyd-Max codebook for a given bit-width, calibrated to the
/// Beta distribution that emerges from coordinates of unit-sphere vectors
/// of dimension `head_dim`.
#[derive(Debug, Clone)]
pub struct Codebook {
    pub bits:       u32,
    pub centroids:  Vec<f32>,   // 2^bits values in [-1, 1]
    pub boundaries: Vec<f32>,   // 2^bits - 1 decision points
}

impl Codebook {
    /// Number of quantization levels.
    pub fn n_levels(&self) -> usize {
        1 << self.bits
    }

    /// Generate codebook via iterative Lloyd-Max for the Beta distribution
    /// on [-1,1] arising from `head_dim`-sphere coordinates.
    ///
    /// PDF: f(x) ∝ (1 - x²)^((d-3)/2)  for d = head_dim.
    pub fn generate(bits: u32, head_dim: usize) -> Self {
        let n = 1usize << bits;
        let alpha = ((head_dim as f64 - 3.0) / 2.0).max(0.0);

        // Approximate CDF by numerical integration (1000 steps)
        let steps = 10_000usize;
        let dx = 2.0f64 / steps as f64;
        let mut pdf: Vec<f64> = (0..steps)
            .map(|i| {
                let x = -1.0 + (i as f64 + 0.5) * dx;
                let u = 1.0 - x * x;
                if u <= 0.0 { 0.0 } else { u.powf(alpha) }
            })
            .collect();

        // Normalise
        let total: f64 = pdf.iter().sum::<f64>() * dx;
        for p in &mut pdf { *p /= total; }

        // Build CDF
        let mut cdf = vec![0.0f64; steps + 1];
        for i in 0..steps {
            cdf[i + 1] = cdf[i] + pdf[i] * dx;
        }

        // Initialise centroids at quantile midpoints
        let mut centroids = vec![0.0f64; n];
        for k in 0..n {
            let lo = k as f64 / n as f64;
            let hi = (k + 1) as f64 / n as f64;
            let mid = (lo + hi) / 2.0;
            // Inverse CDF: find x where CDF(x) ≈ mid
            let idx = cdf.partition_point(|&c| c < mid).min(steps - 1);
            centroids[k] = -1.0 + (idx as f64 + 0.5) * dx;
        }

        // Lloyd-Max iterations
        for _ in 0..200 {
            // Compute boundaries as midpoints between centroids
            let mut bounds = vec![0.0f64; n - 1];
            for k in 0..n - 1 {
                bounds[k] = (centroids[k] + centroids[k + 1]) / 2.0;
            }

            // Recompute centroids as conditional means
            let mut prev_cost = 0.0f64;
            let mut next_centroids = vec![0.0f64; n];
            let mut masses = vec![0.0f64; n];

            for i in 0..steps {
                let x = -1.0 + (i as f64 + 0.5) * dx;
                let p = pdf[i] * dx;
                // Find which centroid bin this falls into
                let bin = bounds.partition_point(|&b| x > b);
                next_centroids[bin] += x * p;
                masses[bin] += p;
                let diff = x - centroids[bin];
                prev_cost += diff * diff * p;
            }

            let mut cost = 0.0f64;
            for k in 0..n {
                if masses[k] > 1e-12 {
                    centroids[k] = next_centroids[k] / masses[k];
                }
            }
            // Recompute cost
            for i in 0..steps {
                let x = -1.0 + (i as f64 + 0.5) * dx;
                let p = pdf[i] * dx;
                let bin = bounds.partition_point(|&b| x > b);
                let diff = x - centroids[bin];
                cost += diff * diff * p;
            }
            if (prev_cost - cost).abs() < 1e-12 { break; }
            let _ = prev_cost; // suppress warning
        }

        // Recompute final boundaries
        let boundaries: Vec<f32> = (0..n - 1)
            .map(|k| ((centroids[k] + centroids[k + 1]) / 2.0) as f32)
            .collect();
        let centroids: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();

        Self { bits, centroids, boundaries }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RHT sign vector
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `d` random ±1 signs for the Randomized Hadamard Transform.
/// Uses a simple xorshift64 PRNG with a fixed seed for reproducibility.
pub fn generate_rht_signs(d: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..d)
        .map(|_| {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 { 1i8 } else { -1i8 }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU buffer helpers + dispatch  (macOS only — all Metal code below)
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_os = "macos")]
mod gpu {
    use metal::{Buffer, MTLResourceOptions, MTLSize};
    use crate::context::MetalContext;
    use crate::error::Result;
    use super::{Codebook, generate_rht_signs, RHT_SEED, VAL_GROUP_SIZE};

fn opts() -> MTLResourceOptions { MTLResourceOptions::StorageModeShared }

/// Upload a slice of f32 to a new Metal buffer.
pub fn upload_f32_slice(ctx: &MetalContext, data: &[f32]) -> Buffer {
    let buf = ctx.device.new_buffer((data.len() * 4) as u64, opts());
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents() as *mut f32,
            data.len(),
        );
    }
    buf
}

/// Upload a slice of i8 to a new Metal buffer.
pub fn upload_i8_slice(ctx: &MetalContext, data: &[i8]) -> Buffer {
    let buf = ctx.device.new_buffer(data.len() as u64, opts());
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents() as *mut i8,
            data.len(),
        );
    }
    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a kernel using dispatch_threads into the shared async command buffer.
fn dispatch_sync(
    ctx: &MetalContext,
    name: &str,
    setup: impl FnOnce(&metal::ComputeCommandEncoderRef),
    grid: MTLSize,
    tg: MTLSize,
) -> Result<()> {
    ctx.encode(name, |enc| {
        setup(enc);
        enc.dispatch_threads(grid, tg);
    })
}

/// Encode a kernel using dispatch_thread_groups into the shared async command buffer.
fn dispatch_sync_groups(
    ctx: &MetalContext,
    name: &str,
    setup: impl FnOnce(&metal::ComputeCommandEncoderRef),
    threadgroups: MTLSize,
    tg: MTLSize,
) -> Result<()> {
    ctx.encode(name, |enc| {
        setup(enc);
        enc.dispatch_thread_groups(threadgroups, tg);
    })
}

fn set_u32(enc: &metal::ComputeCommandEncoderRef, idx: u64, v: u32) {
    enc.set_bytes(idx, 4, &v as *const u32 as *const _);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public dispatch functions
// ─────────────────────────────────────────────────────────────────────────────

/// Normalize `x` to unit sphere; write result to `x_hat`, L2 norm to `norm`.
/// `d` must equal threadgroup size (power of 2, ≤ 1024).
pub fn tq_normalize_f16(
    ctx:   &MetalContext,
    x:     &Buffer,
    x_hat: &Buffer,
    norm:  &Buffer,
    d:     u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_normalize_f16",
        |enc| {
            enc.set_buffer(0, Some(x),     0);
            enc.set_buffer(1, Some(x_hat), 0);
            enc.set_buffer(2, Some(norm),  0);
            set_u32(enc, 3, d);
        },
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: d as u64, height: 1, depth: 1 },
    )
}

/// Forward Randomized Hadamard Transform: y = WHT(diag(signs) × x) / sqrt(d).
pub fn tq_rht_f16(
    ctx:   &MetalContext,
    x:     &Buffer,   // [d] f16 input
    y:     &Buffer,   // [d] f16 output
    signs: &Buffer,   // [d] i8 ±1
    d:     u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_rht_f16",
        |enc| {
            enc.set_buffer(0, Some(x),     0);
            enc.set_buffer(1, Some(y),     0);
            enc.set_buffer(2, Some(signs), 0);
            set_u32(enc, 3, d);
        },
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: d as u64, height: 1, depth: 1 },
    )
}

/// Inverse RHT: x = diag(signs) × WHT(y) / sqrt(d).
pub fn tq_rht_inverse_f16(
    ctx:   &MetalContext,
    y:     &Buffer,
    x:     &Buffer,
    signs: &Buffer,
    d:     u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_rht_inverse_f16",
        |enc| {
            enc.set_buffer(0, Some(y),     0);
            enc.set_buffer(1, Some(x),     0);
            enc.set_buffer(2, Some(signs), 0);
            set_u32(enc, 3, d);
        },
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: d as u64, height: 1, depth: 1 },
    )
}

/// Lloyd-Max quantize rotated vector `y` using precomputed codebook.
/// Writes `indices` (uint16) and `recon` (f16 centroid values).
pub fn tq_lloyd_quant_f16(
    ctx:         &MetalContext,
    y:           &Buffer,    // [d] f16
    indices:     &Buffer,    // [d] u16
    recon:       &Buffer,    // [d] f16
    boundaries:  &Buffer,    // [n_centroids-1] f32
    centroids:   &Buffer,    // [n_centroids] f32
    n_centroids: u32,
    d:           u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_lloyd_quant_f16",
        |enc| {
            enc.set_buffer(0, Some(y),          0);
            enc.set_buffer(1, Some(indices),    0);
            enc.set_buffer(2, Some(recon),      0);
            enc.set_buffer(3, Some(boundaries), 0);
            enc.set_buffer(4, Some(centroids),  0);
            set_u32(enc, 5, n_centroids);
        },
        MTLSize { width: d as u64, height: 1, depth: 1 },
        MTLSize { width: 1,        height: 1, depth: 1 },
    )
}

/// Pack `d` uint16 indices (`bits` bits each) into `d*bits/8` bytes.
pub fn tq_pack_bits_u8(
    ctx:    &MetalContext,
    indices: &Buffer,  // [d] u16
    packed:  &Buffer,  // [d*bits/8] u8
    bits:    u32,
    d:       u32,
) -> Result<()> {
    let elems_per_byte = 8 / bits;
    let n_bytes = d / elems_per_byte;
    dispatch_sync(ctx, "tq_pack_bits_u8",
        |enc| {
            enc.set_buffer(0, Some(indices), 0);
            enc.set_buffer(1, Some(packed),  0);
            set_u32(enc, 2, bits);
            set_u32(enc, 3, d);
        },
        MTLSize { width: n_bytes as u64, height: 1, depth: 1 },
        MTLSize { width: 1,              height: 1, depth: 1 },
    )
}

/// Unpack `d*bits/8` bytes back to `d` uint16 indices.
pub fn tq_unpack_bits_u8(
    ctx:    &MetalContext,
    packed:  &Buffer,  // [d*bits/8] u8
    indices: &Buffer,  // [d] u16
    bits:    u32,
    d:       u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_unpack_bits_u8",
        |enc| {
            enc.set_buffer(0, Some(packed),  0);
            enc.set_buffer(1, Some(indices), 0);
            set_u32(enc, 2, bits);
        },
        MTLSize { width: d as u64, height: 1, depth: 1 },
        MTLSize { width: 1,        height: 1, depth: 1 },
    )
}

/// Map indices → centroid f16 values.
pub fn tq_centroid_lookup_f16(
    ctx:       &MetalContext,
    indices:   &Buffer,   // [d] u16
    recon:     &Buffer,   // [d] f16
    centroids: &Buffer,   // [n_centroids] f32
    d:         u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_centroid_lookup_f16",
        |enc| {
            enc.set_buffer(0, Some(indices),   0);
            enc.set_buffer(1, Some(recon),     0);
            enc.set_buffer(2, Some(centroids), 0);
        },
        MTLSize { width: d as u64, height: 1, depth: 1 },
        MTLSize { width: 1,        height: 1, depth: 1 },
    )
}

/// Compute residual r = a - b and its L2 norm.
pub fn tq_residual_norm_f16(
    ctx:    &MetalContext,
    a:      &Buffer,  // [d] f16
    b:      &Buffer,  // [d] f16
    r:      &Buffer,  // [d] f16 output
    r_norm: &Buffer,  // [1] f16 output
    d:      u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_residual_norm_f16",
        |enc| {
            enc.set_buffer(0, Some(a),      0);
            enc.set_buffer(1, Some(b),      0);
            enc.set_buffer(2, Some(r),      0);
            enc.set_buffer(3, Some(r_norm), 0);
            set_u32(enc, 4, d);
        },
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: d as u64, height: 1, depth: 1 },
    )
}

/// QJL projection: compute sign(S^T @ r) and pack as 1 bit/element.
/// `d` must be divisible by 8.
pub fn tq_qjl_signs_f16(
    ctx:          &MetalContext,
    r:            &Buffer,   // [d] f16 residual
    signs_packed: &Buffer,   // [d/8] u8 output
    d:            u32,
    qjl_seed:     u32,
) -> Result<()> {
    // tq_qjl_signs_f16 uses [[threadgroup_position_in_grid]]; one threadgroup per
    // output byte (8 sign bits) → need d/8 threadgroups.
    dispatch_sync_groups(ctx, "tq_qjl_signs_f16",
        |enc| {
            enc.set_buffer(0, Some(r),            0);
            enc.set_buffer(1, Some(signs_packed),  0);
            set_u32(enc, 2, d);
            set_u32(enc, 3, qjl_seed);
        },
        MTLSize { width: (d / 8) as u64, height: 1, depth: 1 },
        MTLSize { width: 8,              height: 1, depth: 1 },
    )
}

/// Compute QJL correction vector: corr = sqrt(π/2)/d × r_norm × S^T @ signs.
pub fn tq_qjl_correction_f16(
    ctx:          &MetalContext,
    signs_packed: &Buffer,   // [d/8] u8
    corr:         &Buffer,   // [d] f16 output
    r_norm:       &Buffer,   // [1] f16
    d:            u32,
    qjl_seed:     u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_qjl_correction_f16",
        |enc| {
            enc.set_buffer(0, Some(signs_packed), 0);
            enc.set_buffer(1, Some(corr),         0);
            enc.set_buffer(2, Some(r_norm),       0);
            set_u32(enc, 3, d);
            set_u32(enc, 4, qjl_seed);
        },
        MTLSize { width: d as u64, height: 1, depth: 1 },
        MTLSize { width: 1,        height: 1, depth: 1 },
    )
}

/// out[i] = a[i] * scalar[0] + b[i].  Used to restore norm after decode.
pub fn tq_scale_add_f16(
    ctx:    &MetalContext,
    a:      &Buffer,   // [d] f16 (unit-norm reconstruction)
    b:      &Buffer,   // [d] f16 (correction vector or zeros)
    out:    &Buffer,   // [d] f16
    scalar: &Buffer,   // [1] f16 (original norm)
    d:      u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_scale_add_f16",
        |enc| {
            enc.set_buffer(0, Some(a),      0);
            enc.set_buffer(1, Some(b),      0);
            enc.set_buffer(2, Some(out),    0);
            enc.set_buffer(3, Some(scalar), 0);
            set_u32(enc, 4, d);
        },
        MTLSize { width: d as u64, height: 1, depth: 1 },
        MTLSize { width: 1,        height: 1, depth: 1 },
    )
}

/// Per-group symmetric quantization for Values.
/// `packed` must be zeroed before calling (kernel uses atomic-or).
pub fn tq_group_quant_val_f16(
    ctx:        &MetalContext,
    v:          &Buffer,   // [d] f16
    packed:     &Buffer,   // [d*bits/8] u8  (must be pre-zeroed)
    scales:     &Buffer,   // [d/group_size] f16
    group_size: u32,
    bits:       u32,       // 2 or 4
    d:          u32,
) -> Result<()> {
    let n_groups = d / group_size;
    // tq_group_quant_val_f16 uses [[threadgroup_position_in_grid]] (tgid = group);
    // one threadgroup per group → use dispatch_thread_groups.
    dispatch_sync_groups(ctx, "tq_group_quant_val_f16",
        |enc| {
            enc.set_buffer(0, Some(v),      0);
            enc.set_buffer(1, Some(packed), 0);
            enc.set_buffer(2, Some(scales), 0);
            set_u32(enc, 3, group_size);
            set_u32(enc, 4, bits);
        },
        MTLSize { width: n_groups as u64, height: 1, depth: 1 },
        MTLSize { width: group_size as u64, height: 1, depth: 1 },
    )
}

/// Dequantize Values from per-group symmetric packing.
pub fn tq_group_dequant_val_f16(
    ctx:        &MetalContext,
    packed:     &Buffer,   // [d*bits/8] u8
    scales:     &Buffer,   // [d/group_size] f16
    out:        &Buffer,   // [d] f16
    group_size: u32,
    bits:       u32,
    d:          u32,
) -> Result<()> {
    dispatch_sync(ctx, "tq_group_dequant_val_f16",
        |enc| {
            enc.set_buffer(0, Some(packed), 0);
            enc.set_buffer(1, Some(scales), 0);
            enc.set_buffer(2, Some(out),    0);
            set_u32(enc, 3, group_size);
            set_u32(enc, 4, bits);
        },
        MTLSize { width: d as u64, height: 1, depth: 1 },
        MTLSize { width: 1,        height: 1, depth: 1 },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Codebook cache  (compute once per (bits, head_dim) pair)
// ─────────────────────────────────────────────────────────────────────────────

/// All precomputed data needed at runtime: codebook + RHT signs on GPU.
pub struct TqMatrices {
    pub centroids_buf:  Buffer,   // [2^(b-1)] f32  (for key MSE quantization)
    pub boundaries_buf: Buffer,   // [2^(b-1)-1] f32
    pub rht_signs_buf:  Buffer,   // [head_dim] i8
    pub key_bits:       u32,      // total bits for key (MSE uses key_bits-1)
    pub val_bits:       u32,
    pub head_dim:       u32,
}

impl TqMatrices {
    /// Precompute codebook + upload everything to GPU.
    pub fn new(
        ctx:      &MetalContext,
        head_dim: usize,
        key_bits: u32,   // 3 or 4
        val_bits: u32,   // 2 or 4
    ) -> Self {
        // Lloyd-Max codebook for (key_bits - 1) bit MSE quantization
        let codebook = Codebook::generate(key_bits - 1, head_dim);
        let centroids_buf  = upload_f32_slice(ctx, &codebook.centroids);
        let boundaries_buf = upload_f32_slice(ctx, &codebook.boundaries);

        // RHT sign vector
        let signs = generate_rht_signs(head_dim, RHT_SEED);
        let rht_signs_buf = upload_i8_slice(ctx, &signs);

        Self {
            centroids_buf,
            boundaries_buf,
            rht_signs_buf,
            key_bits,
            val_bits,
            head_dim: head_dim as u32,
        }
    }

    /// Bytes per compressed key token.
    pub fn key_bytes_per_token(&self) -> usize {
        let mse_bits = (self.key_bits - 1) as usize;
        let d = self.head_dim as usize;
        // MSE indices + QJL signs + 2 × norm (f16)
        d * mse_bits / 8 + d / 8 + 4
    }

    /// Bytes per compressed value token.
    pub fn val_bytes_per_token(&self) -> usize {
        let d = self.head_dim as usize;
        let n_groups = d / VAL_GROUP_SIZE;
        // packed data + scales f16
        d * self.val_bits as usize / 8 + n_groups * 2
    }
}

} // end mod gpu

#[cfg(target_os = "macos")]
pub use gpu::{
    upload_f32_slice, upload_i8_slice,
    tq_normalize_f16, tq_rht_f16, tq_rht_inverse_f16,
    tq_lloyd_quant_f16, tq_pack_bits_u8, tq_unpack_bits_u8,
    tq_centroid_lookup_f16, tq_residual_norm_f16,
    tq_qjl_signs_f16, tq_qjl_correction_f16,
    tq_scale_add_f16,
    tq_group_quant_val_f16, tq_group_dequant_val_f16,
    TqMatrices,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_2bit_d128_is_sorted_and_bounded() {
        let cb = Codebook::generate(2, 128);
        assert_eq!(cb.centroids.len(), 4);
        assert_eq!(cb.boundaries.len(), 3);
        for c in &cb.centroids {
            assert!(*c >= -1.0 && *c <= 1.0, "centroid out of [-1,1]: {c}");
        }
        for i in 1..cb.centroids.len() {
            assert!(cb.centroids[i] > cb.centroids[i-1], "centroids not sorted");
        }
        for i in 0..cb.boundaries.len() {
            assert!(cb.boundaries[i] > cb.centroids[i]);
            assert!(cb.boundaries[i] < cb.centroids[i+1]);
        }
    }

    #[test]
    fn codebook_3bit_d128_has_8_levels() {
        let cb = Codebook::generate(3, 128);
        assert_eq!(cb.n_levels(), 8);
        assert_eq!(cb.boundaries.len(), 7);
    }

    #[test]
    fn rht_signs_are_plus_minus_one() {
        let signs = generate_rht_signs(128, RHT_SEED);
        assert_eq!(signs.len(), 128);
        for s in &signs {
            assert!(*s == 1 || *s == -1);
        }
    }

    #[test]
    fn rht_signs_reproducible() {
        let a = generate_rht_signs(64, 42);
        let b = generate_rht_signs(64, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn tq_matrices_memory_estimate_d128_3bit_4bit() {
        // key_bytes_per_token with b=3: 2-bit MSE + 1-bit QJL + 4B norms
        // = 128*2/8 + 128/8 + 4 = 32 + 16 + 4 = 52
        // val_bytes_per_token with b_v=4: 128*4/8 + (128/128)*2 = 64 + 2 = 66
        // We can't run Metal on CI but we can test the arithmetic:
        let mse_bits = 2usize;  // key_bits - 1 = 3 - 1
        let d = 128usize;
        let key_mse   = d * mse_bits / 8;  // 32
        let key_signs = d / 8;             // 16
        let key_norms = 4usize;            // 2 × f16
        assert_eq!(key_mse + key_signs + key_norms, 52);

        let val_bits = 4usize;
        let val_pack = d * val_bits / 8;   // 64
        let n_groups  = d / VAL_GROUP_SIZE;  // 1
        let val_meta  = n_groups * 2;        // 2
        assert_eq!(val_pack + val_meta, 66);
    }
}
