use half::f16;
use metal::MTLResourceOptions;

use crate::config::ModelConfig;
use bare_metal_kernels::context::MetalContext;

/// Flat (non-paged) KV cache for a single sequence.
///
/// Stores K and V tensors for every transformer layer in
/// `StorageModeShared` Metal buffers — visible from both CPU and GPU
/// on Apple Silicon unified memory without any explicit blit.
///
/// # Layout
///
/// Each layer owns two buffers:
/// ```text
/// K: [num_kv_heads, max_seq_len, head_dim]  f16
/// V: [num_kv_heads, max_seq_len, head_dim]  f16
/// ```
///
/// Element `(kv_head, pos, d)` is at flat index
/// `(kv_head * max_seq_len + pos) * head_dim + d`.
///
/// This is the same layout expected by `flash_attention_f16` for its
/// `K` and `V` inputs when `batch = 1`.
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
    /// Memory: `2 × num_layers × num_kv_heads × max_seq_len × head_dim × 2` bytes.
    /// Qwen2.5-7B at max_seq_len=4096: 2 × 28 × 4 × 4096 × 128 × 2 ≈ 234 MB.
    pub fn new(ctx: &MetalContext, cfg: &ModelConfig, max_seq_len: usize) -> Self {
        let buf_bytes = (cfg.num_key_value_heads * max_seq_len * cfg.head_dim * 2) as u64;
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

    /// Write new K and V vectors for the current position (`self.filled`)
    /// into `layer`'s cache buffers via the CPU-visible unified memory pointer.
    ///
    /// `new_k` and `new_v` must each be exactly `num_kv_heads × head_dim` f16
    /// elements, laid out as `[num_kv_heads, head_dim]` row-major.
    pub fn update(&mut self, layer: usize, new_k: &[f16], new_v: &[f16]) {
        let pos = self.filled;
        assert!(pos < self.max_seq_len, "KV cache full ({} tokens)", self.max_seq_len);
        assert_eq!(new_k.len(), self.num_kv_heads * self.head_dim);
        assert_eq!(new_v.len(), self.num_kv_heads * self.head_dim);

        // For each kv_head, write head_dim f16 values at column `pos`.
        // Buffer layout: [kv_head, pos, d] → flat[(kv_head * max_seq_len + pos) * head_dim + d]
        let k_ptr = self.k_bufs[layer].contents() as *mut f16;
        let v_ptr = self.v_bufs[layer].contents() as *mut f16;

        for h in 0..self.num_kv_heads {
            let dst_off = (h * self.max_seq_len + pos) * self.head_dim;
            let src_off = h * self.head_dim;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    new_k.as_ptr().add(src_off),
                    k_ptr.add(dst_off),
                    self.head_dim,
                );
                std::ptr::copy_nonoverlapping(
                    new_v.as_ptr().add(src_off),
                    v_ptr.add(dst_off),
                    self.head_dim,
                );
            }
        }
    }

    /// Write K/V directly from GPU buffers into the cache at position `self.filled`.
    /// The source buffers must contain `[num_kv_heads, head_dim]` f16 values
    /// (i.e. `num_kv_heads * head_dim` contiguous f16 elements).
    ///
    /// This avoids the GPU→Vec→GPU round-trip of `update()`.
    /// **Caller must ensure GPU work is flushed** before calling (so that
    /// the source buffer contents are visible to the CPU).
    pub fn update_from_buf(&mut self, layer: usize, k_src: &metal::Buffer, v_src: &metal::Buffer) {
        let pos = self.filled;
        assert!(pos < self.max_seq_len, "KV cache full ({} tokens)", self.max_seq_len);

        let k_src_ptr = k_src.contents() as *const f16;
        let v_src_ptr = v_src.contents() as *const f16;
        let k_dst_ptr = self.k_bufs[layer].contents() as *mut f16;
        let v_dst_ptr = self.v_bufs[layer].contents() as *mut f16;

        for h in 0..self.num_kv_heads {
            let dst_off = (h * self.max_seq_len + pos) * self.head_dim;
            let src_off = h * self.head_dim;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    k_src_ptr.add(src_off),
                    k_dst_ptr.add(dst_off),
                    self.head_dim,
                );
                std::ptr::copy_nonoverlapping(
                    v_src_ptr.add(src_off),
                    v_dst_ptr.add(dst_off),
                    self.head_dim,
                );
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

    /// Number of token positions written — the `kv_len` to pass to FlashAttention.
    pub fn seq_len(&self) -> u32 {
        self.filled as u32
    }

    /// Allocated sequence capacity — the `kv_head_stride` to pass to FlashAttention.
    /// The K/V buffers have layout [num_kv_heads, max_seq_len, head_dim], so the
    /// stride between kv_heads in elements is `max_seq_len * head_dim`. Flash attention
    /// needs `max_seq_len` (not `kv_len`) to correctly address each kv_head's data.
    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len as u32
    }

    /// Write K and V at an explicit position without advancing `filled`.
    ///
    /// Used by `Executor::prefill` to populate all positions at once.
    /// After calling this for all positions, call `set_seq_len(seq)`.
    pub fn write_at(&mut self, layer: usize, pos: usize, new_k: &[f16], new_v: &[f16]) {
        assert!(pos < self.max_seq_len, "KV cache full ({} tokens)", self.max_seq_len);
        assert_eq!(new_k.len(), self.num_kv_heads * self.head_dim);
        assert_eq!(new_v.len(), self.num_kv_heads * self.head_dim);

        let k_ptr = self.k_bufs[layer].contents() as *mut f16;
        let v_ptr = self.v_bufs[layer].contents() as *mut f16;

        for h in 0..self.num_kv_heads {
            let dst_off = (h * self.max_seq_len + pos) * self.head_dim;
            let src_off = h * self.head_dim;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    new_k.as_ptr().add(src_off),
                    k_ptr.add(dst_off),
                    self.head_dim,
                );
                std::ptr::copy_nonoverlapping(
                    new_v.as_ptr().add(src_off),
                    v_ptr.add(dst_off),
                    self.head_dim,
                );
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
        self.k_bufs.len() * 2 * self.num_kv_heads * self.max_seq_len * self.head_dim * 2
    }
}
