/// TurboQuant KV cache — Phase 5.
///
/// Hybrid design:
///   - Recent `buffer_tokens` positions stored as F16 (no compression overhead on the hot path).
///   - Older positions compressed: Keys via RHT + Lloyd-Max(b-1 bits) + QJL sign bit;
///     Values via per-group symmetric quantization.
///
/// # Memory layout
///
/// F16 buffer (per layer, per head):
///   k_f16_buf:  [num_kv_heads, buffer_tokens, head_dim]  f16
///   v_f16_buf:  same
///
/// Compressed store — grows one token at a time as buffer overflows.
/// Per token, per kv_head, we store:
///   key_mse:   [head_dim * (key_bits-1) / 8]  u8   — packed MSE indices
///   key_signs: [head_dim / 8]                 u8   — QJL sign bits
///   key_norms: [2]                            f16  — (||k||, ||r||)
///   val_pack:  [head_dim * val_bits / 8]      u8   — packed values
///   val_scales:[head_dim / group_size]        f16  — per-group scales

use half::f16;
use metal::MTLResourceOptions;

use crate::config::ModelConfig;
use bare_metal_kernels::{
    context::MetalContext,
    turboquant::{
        TqMatrices, QJL_SEED, VAL_GROUP_SIZE,
        tq_normalize_f16, tq_rht_f16, tq_rht_inverse_f16,
        tq_lloyd_quant_f16, tq_pack_bits_u8, tq_unpack_bits_u8,
        tq_centroid_lookup_f16, tq_residual_norm_f16,
        tq_qjl_signs_f16, tq_qjl_correction_f16,
        tq_scale_add_f16,
        tq_group_quant_val_f16, tq_group_dequant_val_f16,
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-layer compressed store
// ─────────────────────────────────────────────────────────────────────────────

/// Stores compressed KV data for all positions that have been evicted
/// from the F16 buffer zone.  One instance per transformer layer.
struct CompressedStore {
    // Keys — one contiguous buffer per field, length grows with seq
    key_mse:    Vec<u8>,   // [seq × kv_heads × key_mse_bytes_per_head]
    key_signs:  Vec<u8>,   // [seq × kv_heads × (head_dim/8)]
    key_norms:  Vec<f16>,  // [seq × kv_heads × 2]  (||k||, ||r||)
    // Values
    val_pack:   Vec<u8>,   // [seq × kv_heads × val_pack_bytes_per_head]
    val_scales: Vec<f16>,  // [seq × kv_heads × n_groups]
}

impl CompressedStore {
    fn new() -> Self {
        Self {
            key_mse: Vec::new(),
            key_signs: Vec::new(),
            key_norms: Vec::new(),
            val_pack: Vec::new(),
            val_scales: Vec::new(),
        }
    }

    fn n_tokens(&self, kv_heads: usize, key_mse_bytes: usize) -> usize {
        if key_mse_bytes == 0 || kv_heads == 0 { return 0; }
        self.key_mse.len() / (kv_heads * key_mse_bytes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TqKvCache
// ─────────────────────────────────────────────────────────────────────────────

pub struct TqKvCache {
    // F16 ring buffer (recent tokens)
    k_f16:        Vec<metal::Buffer>,  // [layers] → [kv_heads, buf_tokens, head_dim] f16
    v_f16:        Vec<metal::Buffer>,

    // CPU-side compressed store (older tokens)
    compressed:   Vec<CompressedStore>,  // [layers]

    // Precomputed matrices + codebooks (GPU buffers)
    pub matrices: TqMatrices,

    // Dimensions
    num_kv_heads:  usize,
    head_dim:      usize,
    buffer_tokens: usize,

    // How many token positions have been committed (after advance())
    filled: usize,
}

impl TqKvCache {
    /// Allocate the TurboQuant KV cache.
    ///
    /// - `key_bits`: 3 or 4  (total bits for key; MSE uses key_bits-1)
    /// - `val_bits`: 2 or 4
    /// - `buffer_tokens`: how many recent positions to keep in F16 (default 128)
    pub fn new(
        ctx:           &MetalContext,
        cfg:           &ModelConfig,
        key_bits:      u32,
        val_bits:      u32,
        buffer_tokens: usize,
    ) -> Self {
        let opts = MTLResourceOptions::StorageModeShared;
        let n = cfg.num_hidden_layers;
        let kv = cfg.num_key_value_heads;
        let d  = cfg.head_dim;

        let f16_bytes = (kv * buffer_tokens * d * 2) as u64;
        let mut k_f16 = Vec::with_capacity(n);
        let mut v_f16 = Vec::with_capacity(n);
        let mut compressed = Vec::with_capacity(n);

        for _ in 0..n {
            k_f16.push(ctx.device.new_buffer(f16_bytes, opts));
            v_f16.push(ctx.device.new_buffer(f16_bytes, opts));
            compressed.push(CompressedStore::new());
        }

        let matrices = TqMatrices::new(ctx, d, key_bits, val_bits);

        Self {
            k_f16, v_f16, compressed, matrices,
            num_kv_heads: kv,
            head_dim: d,
            buffer_tokens,
            filled: 0,
        }
    }

    /// Write the new K/V vectors for position `self.filled` in `layer`.
    /// Inputs are `[num_kv_heads, head_dim]` f16, row-major.
    ///
    /// If the F16 buffer is full, the oldest F16 entry is compressed and
    /// evicted to the CPU-side store before the new token is written.
    pub fn update(
        &mut self,
        ctx:   &MetalContext,
        layer: usize,
        new_k: &[f16],
        new_v: &[f16],
    ) {
        let kv = self.num_kv_heads;
        let d  = self.head_dim;
        assert_eq!(new_k.len(), kv * d);
        assert_eq!(new_v.len(), kv * d);

        let buf_pos = self.filled % self.buffer_tokens;

        // If buffer zone is full, compress the oldest slot before overwriting
        if self.filled >= self.buffer_tokens {
            self.compress_oldest(ctx, layer, buf_pos);
        }

        // Write into F16 buffer at slot `buf_pos`
        // Layout: [kv_head, buf_pos, head_dim]
        let k_ptr = self.k_f16[layer].contents() as *mut f16;
        let v_ptr = self.v_f16[layer].contents() as *mut f16;
        for h in 0..kv {
            let dst = (h * self.buffer_tokens + buf_pos) * d;
            let src = h * d;
            unsafe {
                std::ptr::copy_nonoverlapping(new_k.as_ptr().add(src), k_ptr.add(dst), d);
                std::ptr::copy_nonoverlapping(new_v.as_ptr().add(src), v_ptr.add(dst), d);
            }
        }
    }

    /// Write K/V directly from GPU buffers into the TQ cache.
    /// Source buffers must contain `[num_kv_heads, head_dim]` f16 values.
    /// **Caller must flush GPU work** before calling.
    pub fn update_from_buf(
        &mut self,
        ctx:   &MetalContext,
        layer: usize,
        k_src: &metal::Buffer,
        v_src: &metal::Buffer,
    ) {
        let kv = self.num_kv_heads;
        let d  = self.head_dim;

        let buf_pos = self.filled % self.buffer_tokens;

        // If buffer zone is full, compress the oldest slot before overwriting
        if self.filled >= self.buffer_tokens {
            self.compress_oldest(ctx, layer, buf_pos);
        }

        // Write into F16 buffer at slot `buf_pos` — direct buffer-to-buffer copy
        let k_src_ptr = k_src.contents() as *const f16;
        let v_src_ptr = v_src.contents() as *const f16;
        let k_ptr = self.k_f16[layer].contents() as *mut f16;
        let v_ptr = self.v_f16[layer].contents() as *mut f16;
        for h in 0..kv {
            let dst = (h * self.buffer_tokens + buf_pos) * d;
            let src = h * d;
            unsafe {
                std::ptr::copy_nonoverlapping(k_src_ptr.add(src), k_ptr.add(dst), d);
                std::ptr::copy_nonoverlapping(v_src_ptr.add(src), v_ptr.add(dst), d);
            }
        }
    }

    /// Advance position counter — call once after all layers have been updated.
    pub fn advance(&mut self) {
        self.filled += 1;
    }

    /// Reset the cache (new sequence).
    pub fn reset(&mut self) {
        self.filled = 0;
        for cs in &mut self.compressed {
            cs.key_mse.clear();
            cs.key_signs.clear();
            cs.key_norms.clear();
            cs.val_pack.clear();
            cs.val_scales.clear();
        }
    }

    /// Total F16 positions in the buffer zone (capped at filled).
    pub fn f16_len(&self) -> usize {
        self.filled.min(self.buffer_tokens)
    }

    /// Total compressed positions.
    pub fn compressed_len(&self) -> usize {
        self.filled.saturating_sub(self.buffer_tokens)
    }

    /// Total sequence length (compressed + F16 buffer).
    pub fn seq_len(&self) -> u32 {
        self.filled as u32
    }

    // ── Attention accessors ───────────────────────────────────────────────────

    /// Return the F16 K buffer for `layer` (for use by FlashAttention on
    /// the buffer zone only — recent `f16_len()` tokens).
    pub fn k_f16_buf(&self, layer: usize) -> &metal::Buffer {
        &self.k_f16[layer]
    }

    /// Return the F16 V buffer for `layer`.
    pub fn v_f16_buf(&self, layer: usize) -> &metal::Buffer {
        &self.v_f16[layer]
    }

    /// Materialise the full K cache for `layer` as F16 in a fresh GPU buffer.
    ///
    /// For a fully-F16 sequence (no compressed tokens yet) this just returns
    /// a view slice.  When compressed tokens exist, this dequantises them into
    /// a contiguous `[kv_heads, filled, head_dim]` buffer.
    pub fn materialise_k_f16(
        &self,
        ctx:   &MetalContext,
        layer: usize,
    ) -> metal::Buffer {
        let kv = self.num_kv_heads;
        let d  = self.head_dim;
        // `total` is filled+1 because this is called after `update()` (which writes
        // the current position into the ring buffer) but before `advance()` (which
        // increments `filled`). So `self.filled` is one behind the true sequence
        // length at this point in the decode loop.
        let total = self.filled + 1;
        let opts  = MTLResourceOptions::StorageModeShared;

        let out = ctx.device.new_buffer((kv * total * d * 2) as u64, opts);

        // 1. Copy compressed tokens (positions 0..compressed_len)
        let c_len = self.compressed_len();
        if c_len > 0 {
            self.dequant_compressed_to(ctx, layer, &out, 0, c_len);
        }

        // 2. Copy F16 buffer zone (positions c_len..total)
        // The F16 buffer is a ring; the oldest slot is at index `c_len % buffer_tokens`.
        // Since we evict the oldest when the buffer is full, the F16 buffer always
        // holds positions [c_len .. c_len + f16_len) in order starting at ring index
        // `c_len % buffer_tokens`.
        let f_len   = (self.filled + 1).min(self.buffer_tokens);
        let k_f16_ptr = self.k_f16[layer].contents() as *const f16;
        let out_ptr   = out.contents() as *mut f16;

        for h in 0..kv {
            for i in 0..f_len {
                let ring_slot = (c_len + i) % self.buffer_tokens;
                let src = (h * self.buffer_tokens + ring_slot) * d;
                let dst = (h * total + c_len + i) * d;
                unsafe {
                    std::ptr::copy_nonoverlapping(k_f16_ptr.add(src), out_ptr.add(dst), d);
                }
            }
        }

        out
    }

    /// Same as `materialise_k_f16` but for Values.
    pub fn materialise_v_f16(
        &self,
        ctx:   &MetalContext,
        layer: usize,
    ) -> metal::Buffer {
        let kv = self.num_kv_heads;
        let d  = self.head_dim;
        // See materialise_k_f16 for why we use filled+1 here.
        let total = self.filled + 1;
        let opts  = MTLResourceOptions::StorageModeShared;

        let out = ctx.device.new_buffer((kv * total * d * 2) as u64, opts);

        let c_len = self.compressed_len();
        if c_len > 0 {
            self.dequant_val_compressed_to(ctx, layer, &out, c_len);
        }

        let f_len     = (self.filled + 1).min(self.buffer_tokens);
        let v_f16_ptr = self.v_f16[layer].contents() as *const f16;
        let out_ptr   = out.contents() as *mut f16;

        for h in 0..kv {
            for i in 0..f_len {
                let ring_slot = (c_len + i) % self.buffer_tokens;
                let src = (h * self.buffer_tokens + ring_slot) * d;
                let dst = (h * total + c_len + i) * d;
                unsafe {
                    std::ptr::copy_nonoverlapping(v_f16_ptr.add(src), out_ptr.add(dst), d);
                }
            }
        }

        out
    }

    // ── Private: compression ─────────────────────────────────────────────────

    /// Compress the token at ring slot `buf_pos` in layer `layer` and append
    /// to the CPU-side compressed store.
    fn compress_oldest(
        &mut self,
        ctx:     &MetalContext,
        layer:   usize,
        buf_pos: usize,
    ) {
        let kv = self.num_kv_heads;
        let d  = self.head_dim as u32;
        let opts = MTLResourceOptions::StorageModeShared;

        let mse_bits     = self.matrices.key_bits - 1;
        let val_bits     = self.matrices.val_bits;
        let key_mse_bytes = d as usize * mse_bits as usize / 8;
        let key_sign_bytes = d as usize / 8;
        let val_pack_bytes = d as usize * val_bits as usize / 8;
        let n_groups       = d as usize / VAL_GROUP_SIZE;

        for h in 0..kv {
            let k_base = (h * self.buffer_tokens + buf_pos) * d as usize;
            let v_base = k_base;

            // Gather K vector into a GPU buffer
            let k_vec_buf = ctx.device.new_buffer((d * 2) as u64, opts);
            let v_vec_buf = ctx.device.new_buffer((d * 2) as u64, opts);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (self.k_f16[layer].contents() as *const f16).add(k_base),
                    k_vec_buf.contents() as *mut f16, d as usize,
                );
                std::ptr::copy_nonoverlapping(
                    (self.v_f16[layer].contents() as *const f16).add(v_base),
                    v_vec_buf.contents() as *mut f16, d as usize,
                );
            }

            // ── Key compression ────────────────────────────────────────────
            let x_hat_buf  = ctx.device.new_buffer((d * 2) as u64, opts);
            let norm_buf   = ctx.device.new_buffer(2, opts);           // 1 × f16
            let y_buf      = ctx.device.new_buffer((d * 2) as u64, opts);
            let indices_buf = ctx.device.new_buffer((d * 2) as u64, opts); // u16
            let recon_y_buf = ctx.device.new_buffer((d * 2) as u64, opts);
            let recon_x_buf = ctx.device.new_buffer((d * 2) as u64, opts);
            let residual_buf = ctx.device.new_buffer((d * 2) as u64, opts);
            let r_norm_buf  = ctx.device.new_buffer(2, opts);
            let signs_packed_buf = ctx.device.new_buffer((d / 8) as u64, opts);
            let packed_mse_buf   = ctx.device.new_buffer(key_mse_bytes as u64, opts);

            // 1. Normalize
            let _ = tq_normalize_f16(ctx, &k_vec_buf, &x_hat_buf, &norm_buf, d);
            // 2. Forward RHT
            let _ = tq_rht_f16(ctx, &x_hat_buf, &y_buf, &self.matrices.rht_signs_buf, d);
            // 3. Lloyd-Max quantize (b-1 bits)
            let n_cent = 1u32 << mse_bits;
            let _ = tq_lloyd_quant_f16(ctx, &y_buf, &indices_buf, &recon_y_buf,
                                        &self.matrices.boundaries_buf,
                                        &self.matrices.centroids_buf, n_cent, d);
            // 4. Inverse RHT on reconstruction
            let _ = tq_rht_inverse_f16(ctx, &recon_y_buf, &recon_x_buf,
                                        &self.matrices.rht_signs_buf, d);
            // 5. Pack MSE indices
            let _ = tq_pack_bits_u8(ctx, &indices_buf, &packed_mse_buf, mse_bits, d);
            // 6. Residual + norm
            let _ = tq_residual_norm_f16(ctx, &x_hat_buf, &recon_x_buf,
                                          &residual_buf, &r_norm_buf, d);
            // 7. QJL signs
            let _ = tq_qjl_signs_f16(ctx, &residual_buf, &signs_packed_buf, d, QJL_SEED);

            // Flush GPU work so compressed data is visible for CPU readback
            ctx.flush();

            // Readback compressed key data to CPU
            let cs = &mut self.compressed[layer];
            let mse_ptr   = packed_mse_buf.contents() as *const u8;
            let signs_ptr = signs_packed_buf.contents() as *const u8;
            let norms_ptr = norm_buf.contents() as *const f16;   // ||k||
            let rnorm_ptr = r_norm_buf.contents() as *const f16; // ||r||

            unsafe {
                let old_len = cs.key_mse.len();
                cs.key_mse.resize(old_len + key_mse_bytes, 0);
                std::ptr::copy_nonoverlapping(mse_ptr, cs.key_mse.as_mut_ptr().add(old_len), key_mse_bytes);

                let old_len = cs.key_signs.len();
                cs.key_signs.resize(old_len + key_sign_bytes, 0);
                std::ptr::copy_nonoverlapping(signs_ptr, cs.key_signs.as_mut_ptr().add(old_len), key_sign_bytes);

                let old_len = cs.key_norms.len();
                cs.key_norms.resize(old_len + 2, f16::ZERO);
                cs.key_norms[old_len]     = *norms_ptr;
                cs.key_norms[old_len + 1] = *rnorm_ptr;
            }

            // ── Value compression ──────────────────────────────────────────
            let packed_v_buf = ctx.device.new_buffer(val_pack_bytes as u64, opts);
            let scales_v_buf = ctx.device.new_buffer((n_groups * 2) as u64, opts);
            // Zero the packed buffer (atomic-or kernel requires it)
            unsafe { std::ptr::write_bytes(packed_v_buf.contents() as *mut u8, 0, val_pack_bytes); }

            let _ = tq_group_quant_val_f16(ctx, &v_vec_buf, &packed_v_buf, &scales_v_buf,
                                            VAL_GROUP_SIZE as u32, val_bits, d);

            // Flush GPU work so quantised values are visible for CPU readback
            ctx.flush();

            unsafe {
                let old_len = cs.val_pack.len();
                cs.val_pack.resize(old_len + val_pack_bytes, 0);
                std::ptr::copy_nonoverlapping(
                    packed_v_buf.contents() as *const u8,
                    cs.val_pack.as_mut_ptr().add(old_len),
                    val_pack_bytes,
                );

                let old_len = cs.val_scales.len();
                cs.val_scales.resize(old_len + n_groups, f16::ZERO);
                std::ptr::copy_nonoverlapping(
                    scales_v_buf.contents() as *const f16,
                    cs.val_scales.as_mut_ptr().add(old_len),
                    n_groups,
                );
            }
        }
    }

    // ── Private: dequantisation ───────────────────────────────────────────────

    /// Dequantise `c_len` compressed key tokens for `layer` into `out` at offset `dst_off`.
    fn dequant_compressed_to(
        &self,
        ctx:     &MetalContext,
        layer:   usize,
        out:     &metal::Buffer,
        dst_off: usize,
        c_len:   usize,
    ) {
        let kv = self.num_kv_heads;
        let d  = self.head_dim as u32;
        let opts = MTLResourceOptions::StorageModeShared;

        let mse_bits     = self.matrices.key_bits - 1;
        let key_mse_bytes = d as usize * mse_bits as usize / 8;
        let key_sign_bytes = d as usize / 8;

        let cs = &self.compressed[layer];
        let out_ptr = out.contents() as *mut f16;

        for tok in 0..c_len {
            for h in 0..kv {
                let flat_idx = tok * kv + h;
                let mse_off   = flat_idx * key_mse_bytes;
                let sign_off  = flat_idx * key_sign_bytes;
                let norm_off  = flat_idx * 2;

                // Upload compressed data to GPU
                let packed_buf  = ctx.device.new_buffer(key_mse_bytes as u64, opts);
                let signs_buf   = ctx.device.new_buffer(key_sign_bytes as u64, opts);
                let norm_buf    = ctx.device.new_buffer(2, opts);
                let rnorm_buf   = ctx.device.new_buffer(2, opts);

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        cs.key_mse.as_ptr().add(mse_off),
                        packed_buf.contents() as *mut u8, key_mse_bytes);
                    std::ptr::copy_nonoverlapping(
                        cs.key_signs.as_ptr().add(sign_off),
                        signs_buf.contents() as *mut u8, key_sign_bytes);
                    *(norm_buf.contents() as *mut f16)  = cs.key_norms[norm_off];
                    *(rnorm_buf.contents() as *mut f16) = cs.key_norms[norm_off + 1];
                }

                // Decode:
                let indices_buf  = ctx.device.new_buffer((d * 2) as u64, opts);
                let recon_y_buf  = ctx.device.new_buffer((d * 2) as u64, opts);
                let recon_x_buf  = ctx.device.new_buffer((d * 2) as u64, opts);
                let corr_buf     = ctx.device.new_buffer((d * 2) as u64, opts);
                let x_tq_buf     = ctx.device.new_buffer((d * 2) as u64, opts);
                let k_final_buf  = ctx.device.new_buffer((d * 2) as u64, opts);
                let zero_buf     = ctx.device.new_buffer((d * 2) as u64, opts);
                unsafe { std::ptr::write_bytes(zero_buf.contents() as *mut u8, 0, d as usize * 2); }

                let n_cent = 1u32 << mse_bits;
                // 1. Unpack indices
                let _ = tq_unpack_bits_u8(ctx, &packed_buf, &indices_buf, mse_bits, d);
                // 2. Centroid lookup
                let _ = tq_centroid_lookup_f16(ctx, &indices_buf, &recon_y_buf,
                                                &self.matrices.centroids_buf, d);
                let _ = n_cent; // used above
                // 3. Inverse RHT
                let _ = tq_rht_inverse_f16(ctx, &recon_y_buf, &recon_x_buf,
                                            &self.matrices.rht_signs_buf, d);
                // 4. QJL correction
                let _ = tq_qjl_correction_f16(ctx, &signs_buf, &corr_buf, &rnorm_buf, d, QJL_SEED);
                // 5. x_tq = recon_x + corr
                let _ = bare_metal_kernels::dispatch::add_f16(ctx, &recon_x_buf, &corr_buf, &x_tq_buf, d);
                // 6. Restore norm: k = x_tq * ||k||  (add zero vector)
                let _ = tq_scale_add_f16(ctx, &x_tq_buf, &zero_buf, &k_final_buf, &norm_buf, d);

                // Flush so dequantised key is visible for CPU readback
                ctx.flush();

                // Readback into out
                let dst = (h * self.filled + dst_off + tok) * d as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_final_buf.contents() as *const f16,
                        out_ptr.add(dst), d as usize,
                    );
                }
            }
        }
    }

    /// Dequantise compressed value tokens for `layer` into `out`.
    fn dequant_val_compressed_to(
        &self,
        ctx:   &MetalContext,
        layer: usize,
        out:   &metal::Buffer,
        c_len: usize,
    ) {
        let kv = self.num_kv_heads;
        let d  = self.head_dim as u32;
        let opts = MTLResourceOptions::StorageModeShared;

        let val_bits      = self.matrices.val_bits;
        let val_pack_bytes = d as usize * val_bits as usize / 8;
        let n_groups       = d as usize / VAL_GROUP_SIZE;

        let cs = &self.compressed[layer];
        let out_ptr = out.contents() as *mut f16;

        for tok in 0..c_len {
            for h in 0..kv {
                let flat_idx  = tok * kv + h;
                let pack_off  = flat_idx * val_pack_bytes;
                let scale_off = flat_idx * n_groups;

                let packed_buf = ctx.device.new_buffer(val_pack_bytes as u64, opts);
                let scales_buf = ctx.device.new_buffer((n_groups * 2) as u64, opts);
                let v_out_buf  = ctx.device.new_buffer((d * 2) as u64, opts);

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        cs.val_pack.as_ptr().add(pack_off),
                        packed_buf.contents() as *mut u8, val_pack_bytes);
                    std::ptr::copy_nonoverlapping(
                        cs.val_scales.as_ptr().add(scale_off),
                        scales_buf.contents() as *mut f16, n_groups);
                }

                let _ = tq_group_dequant_val_f16(ctx, &packed_buf, &scales_buf, &v_out_buf,
                                                   VAL_GROUP_SIZE as u32, val_bits, d);

                // Flush so dequantised value is visible for CPU readback
                ctx.flush();

                let dst = (h * self.filled + tok) * d as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        v_out_buf.contents() as *const f16,
                        out_ptr.add(dst), d as usize,
                    );
                }
            }
        }
    }

    /// Approximate GPU memory in bytes (F16 buffers only — compressed data is on CPU).
    pub fn gpu_memory_bytes(&self) -> usize {
        let n = self.k_f16.len();
        n * 2 * self.num_kv_heads * self.buffer_tokens * self.head_dim * 2
    }
}
