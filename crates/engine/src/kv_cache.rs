use half::f16;
use metal::MTLResourceOptions;

use crate::config::ModelConfig;
use bare_metal_kernels::context::MetalContext;

/// Flat (non-paged) KV cache for a single sequence.
///
/// Stores K and V tensors for every transformer layer in f32 precision
/// in `StorageModeShared` Metal buffers — visible from both CPU and GPU
/// on Apple Silicon unified memory without any explicit blit.
///
/// # Layout
///
/// Each layer owns two buffers:
/// ```text
/// K: [num_kv_heads, max_seq_len, head_dim]  f32
/// V: [num_kv_heads, max_seq_len, head_dim]  f32
/// ```
///
/// Element `(kv_head, pos, d)` is at flat index
/// `(kv_head * max_seq_len + pos) * head_dim + d`.
pub struct KvCache {
    k_bufs:       Vec<metal::Buffer>,
    v_bufs:       Vec<metal::Buffer>,
    num_kv_heads: usize,
    head_dim:     usize,
    max_seq_len:  usize,
    /// How many token positions have been written so far.
    filled:       usize,
}

impl KvCache {
    /// Allocate empty KV cache for all layers.
    ///
    /// Memory: `2 × num_layers × num_kv_heads × max_seq_len × head_dim × 4` bytes (f32).
    pub fn new(ctx: &MetalContext, cfg: &ModelConfig, max_seq_len: usize) -> Self {
        let buf_bytes = (cfg.num_key_value_heads * max_seq_len * cfg.head_dim * 4) as u64;
        let opts = MTLResourceOptions::StorageModeShared;

        let mut k_bufs = Vec::with_capacity(cfg.num_hidden_layers);
        let mut v_bufs = Vec::with_capacity(cfg.num_hidden_layers);

        for _ in 0..cfg.num_hidden_layers {
            k_bufs.push(ctx.device.new_buffer(buf_bytes, opts));
            v_bufs.push(ctx.device.new_buffer(buf_bytes, opts));
        }

        Self {
            k_bufs,
            v_bufs,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            max_seq_len,
            filled: 0,
        }
    }

    /// Write new K and V vectors (f16) for the current position into the f32 cache.
    /// Widens f16 → f32 on the fly. Used by prefill which still operates in f16.
    pub fn update(&mut self, layer: usize, new_k: &[f16], new_v: &[f16]) {
        let pos = self.filled;
        assert!(pos < self.max_seq_len, "KV cache full ({} tokens)", self.max_seq_len);
        assert_eq!(new_k.len(), self.num_kv_heads * self.head_dim);
        assert_eq!(new_v.len(), self.num_kv_heads * self.head_dim);

        let k_ptr = self.k_bufs[layer].contents() as *mut f32;
        let v_ptr = self.v_bufs[layer].contents() as *mut f32;

        for h in 0..self.num_kv_heads {
            let dst_off = (h * self.max_seq_len + pos) * self.head_dim;
            let src_off = h * self.head_dim;
            for d in 0..self.head_dim {
                unsafe {
                    *k_ptr.add(dst_off + d) = new_k[src_off + d].to_f32();
                    *v_ptr.add(dst_off + d) = new_v[src_off + d].to_f32();
                }
            }
        }
    }

    /// Write K/V directly from GPU f16 buffers into the f32 cache.
    /// Widens f16 → f32 on the fly. Used by the TQ cache path.
    pub fn update_from_buf(&mut self, layer: usize, k_src: &metal::Buffer, v_src: &metal::Buffer) {
        let pos = self.filled;
        assert!(pos < self.max_seq_len, "KV cache full ({} tokens)", self.max_seq_len);

        let k_src_ptr = k_src.contents() as *const f16;
        let v_src_ptr = v_src.contents() as *const f16;
        let k_dst_ptr = self.k_bufs[layer].contents() as *mut f32;
        let v_dst_ptr = self.v_bufs[layer].contents() as *mut f32;

        for h in 0..self.num_kv_heads {
            let dst_off = (h * self.max_seq_len + pos) * self.head_dim;
            let src_off = h * self.head_dim;
            for d in 0..self.head_dim {
                unsafe {
                    *k_dst_ptr.add(dst_off + d) = (*k_src_ptr.add(src_off + d)).to_f32();
                    *v_dst_ptr.add(dst_off + d) = (*v_src_ptr.add(src_off + d)).to_f32();
                }
            }
        }
    }

    /// Advance the filled position by 1 (call after `update` for each layer).
    pub fn advance(&mut self) {
        self.filled += 1;
    }

    /// K buffer for `layer`. Shape: `[1, num_kv_heads, filled, head_dim]` (batch=1).
    pub fn k_buf(&self, layer: usize) -> &metal::Buffer {
        &self.k_bufs[layer]
    }

    /// V buffer for `layer`. Shape: `[1, num_kv_heads, filled, head_dim]` (batch=1).
    pub fn v_buf(&self, layer: usize) -> &metal::Buffer {
        &self.v_bufs[layer]
    }

    /// Number of token positions written — the `kv_len` to pass to attention.
    pub fn seq_len(&self) -> u32 {
        self.filled as u32
    }

    /// Allocated sequence capacity — the `kv_head_stride` to pass to attention.
    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len as u32
    }

    /// Write K and V (f16) at an explicit position without advancing `filled`.
    /// Widens f16 → f32 on the fly. Used by prefill.
    pub fn write_at(&mut self, layer: usize, pos: usize, new_k: &[f16], new_v: &[f16]) {
        assert!(pos < self.max_seq_len, "KV cache full ({} tokens)", self.max_seq_len);
        assert_eq!(new_k.len(), self.num_kv_heads * self.head_dim);
        assert_eq!(new_v.len(), self.num_kv_heads * self.head_dim);

        let k_ptr = self.k_bufs[layer].contents() as *mut f32;
        let v_ptr = self.v_bufs[layer].contents() as *mut f32;

        for h in 0..self.num_kv_heads {
            let dst_off = (h * self.max_seq_len + pos) * self.head_dim;
            let src_off = h * self.head_dim;
            for d in 0..self.head_dim {
                unsafe {
                    *k_ptr.add(dst_off + d) = new_k[src_off + d].to_f32();
                    *v_ptr.add(dst_off + d) = new_v[src_off + d].to_f32();
                }
            }
        }
    }

    /// Set the logical sequence length without writing data (used after prefill).
    pub fn set_seq_len(&mut self, len: usize) {
        assert!(len <= self.max_seq_len);
        self.filled = len;
    }

    /// Reset the cache (clear `filled`; buffers are reused without zeroing).
    pub fn reset(&mut self) {
        self.filled = 0;
    }

    /// Total GPU memory allocated in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.k_bufs.len() * 2 * self.num_kv_heads * self.max_seq_len * self.head_dim * 4
    }
}
