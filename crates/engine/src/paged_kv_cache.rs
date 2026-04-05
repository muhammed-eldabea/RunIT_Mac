//! PagedAttention KV Cache — block-allocated memory for multi-sequence serving.
//!
//! Instead of allocating one large contiguous buffer per sequence, the paged
//! cache allocates fixed-size blocks (pages) on demand. This eliminates memory
//! fragmentation and enables:
//!   - Dynamic context length (no pre-allocated max_seq)
//!   - Multiple concurrent sequences sharing the same memory pool
//!   - Efficient memory reclamation when sequences complete
//!
//! # Architecture
//!
//! ```text
//! Block Pool: [block_0] [block_1] [block_2] ... [block_N]
//!             ↑ free     ↑ seq_0   ↑ seq_0       ↑ seq_1
//!
//! Page Table (per sequence):
//!   seq_0: [block_1, block_2, block_5, ...]  → logical pos 0..95
//!   seq_1: [block_3, block_7, ...]           → logical pos 0..31
//! ```
//!
//! Each block holds `BLOCK_SIZE` token positions (default: 16).
//! Block layout: `[num_kv_heads, BLOCK_SIZE, head_dim]` f32.
//!
//! The attention kernel receives a block table and gathers K/V from
//! non-contiguous blocks during the dot product.

use metal::MTLResourceOptions;
use bare_metal_kernels::context::MetalContext;
use crate::config::ModelConfig;

/// Number of token positions per block (page).
pub const BLOCK_SIZE: usize = 16;

/// A single block in the KV cache pool.
/// Holds K and V data for one layer for `BLOCK_SIZE` token positions.
struct KvBlock {
    /// K data: `[num_kv_heads, BLOCK_SIZE, head_dim]` f32
    k_buf: metal::Buffer,
    /// V data: same layout
    v_buf: metal::Buffer,
}

/// Per-sequence page table: maps logical token positions to physical blocks.
pub struct SequenceState {
    /// Block indices for each logical position chunk.
    /// `page_table[i]` is the index into the pool for positions `[i*BLOCK_SIZE .. (i+1)*BLOCK_SIZE)`.
    page_table: Vec<usize>,
    /// Number of tokens actually written.
    seq_len: usize,
    /// Unique sequence identifier.
    pub seq_id: u64,
}

impl SequenceState {
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn num_blocks(&self) -> usize {
        self.page_table.len()
    }
}

/// Block-allocated KV cache supporting multiple concurrent sequences.
pub struct PagedKvCache {
    /// Pool of pre-allocated blocks, indexed per layer.
    /// `blocks[layer][block_idx]` = one KvBlock.
    blocks: Vec<Vec<KvBlock>>,
    /// Free block indices (shared across all layers — blocks are allocated in lockstep).
    free_blocks: Vec<usize>,
    /// Active sequences.
    sequences: Vec<SequenceState>,
    /// Config
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Total number of blocks in the pool.
    total_blocks: usize,
    /// Next sequence ID.
    next_seq_id: u64,
}

impl PagedKvCache {
    /// Create a new paged KV cache with capacity for `max_tokens` total
    /// across all sequences.
    pub fn new(ctx: &MetalContext, cfg: &ModelConfig, max_tokens: usize) -> Self {
        let total_blocks = (max_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let opts = MTLResourceOptions::StorageModeShared;
        let block_bytes = (cfg.num_key_value_heads * BLOCK_SIZE * cfg.head_dim * 4) as u64;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            let mut layer_blocks = Vec::with_capacity(total_blocks);
            for _ in 0..total_blocks {
                layer_blocks.push(KvBlock {
                    k_buf: ctx.device.new_buffer(block_bytes, opts),
                    v_buf: ctx.device.new_buffer(block_bytes, opts),
                });
            }
            blocks.push(layer_blocks);
        }

        let free_blocks: Vec<usize> = (0..total_blocks).collect();

        tracing::info!(
            total_blocks,
            block_size = BLOCK_SIZE,
            layers = cfg.num_hidden_layers,
            memory_mb = (total_blocks * cfg.num_hidden_layers * 2) as f64
                * block_bytes as f64 / 1_048_576.0 / cfg.num_hidden_layers as f64,
            "PagedKvCache created"
        );

        Self {
            blocks,
            free_blocks,
            sequences: Vec::new(),
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            total_blocks,
            next_seq_id: 0,
        }
    }

    /// Allocate a new sequence. Returns the sequence ID.
    pub fn new_sequence(&mut self) -> u64 {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        self.sequences.push(SequenceState {
            page_table: Vec::new(),
            seq_len: 0,
            seq_id: id,
        });
        id
    }

    /// Free all blocks owned by a sequence and remove it.
    pub fn free_sequence(&mut self, seq_id: u64) {
        if let Some(idx) = self.sequences.iter().position(|s| s.seq_id == seq_id) {
            let seq = self.sequences.remove(idx);
            // Return blocks to free pool
            for block_idx in seq.page_table {
                self.free_blocks.push(block_idx);
            }
        }
    }

    /// Allocate a new block for a sequence. Returns `None` if pool is exhausted.
    fn alloc_block(&mut self, seq_id: u64) -> Option<usize> {
        let block_idx = self.free_blocks.pop()?;
        if let Some(seq) = self.sequences.iter_mut().find(|s| s.seq_id == seq_id) {
            seq.page_table.push(block_idx);
        }
        Some(block_idx)
    }

    /// Write K/V data for the next token position in a sequence.
    /// Automatically allocates new blocks as needed.
    pub fn append_token(
        &mut self,
        seq_id: u64,
        layer: usize,
        k_data: &[f32],   // [num_kv_heads * head_dim]
        v_data: &[f32],
    ) -> bool {
        let seq = match self.sequences.iter_mut().find(|s| s.seq_id == seq_id) {
            Some(s) => s,
            None => return false,
        };

        let pos = seq.seq_len;
        let block_local = pos / BLOCK_SIZE;
        let offset_in_block = pos % BLOCK_SIZE;

        // Allocate new block if needed
        if block_local >= seq.page_table.len() {
            let block_idx = match self.free_blocks.pop() {
                Some(idx) => idx,
                None => return false, // Pool exhausted
            };
            seq.page_table.push(block_idx);
        }

        let phys_block = seq.page_table[block_local];

        // Write K and V into the block at the correct offset
        let kv_size = self.num_kv_heads * self.head_dim;
        let block_k = &self.blocks[layer][phys_block].k_buf;
        let block_v = &self.blocks[layer][phys_block].v_buf;

        unsafe {
            let k_dst = (block_k.contents() as *mut f32)
                .add(offset_in_block * kv_size);
            let v_dst = (block_v.contents() as *mut f32)
                .add(offset_in_block * kv_size);
            std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_dst, kv_size);
            std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_dst, kv_size);
        }

        // Only advance seq_len on the last layer
        if layer == self.num_layers - 1 {
            seq.seq_len = pos + 1;
        }

        true
    }

    /// Get the page table for a sequence (for passing to the attention kernel).
    pub fn page_table(&self, seq_id: u64) -> Option<&[usize]> {
        self.sequences
            .iter()
            .find(|s| s.seq_id == seq_id)
            .map(|s| s.page_table.as_slice())
    }

    /// Get a block's K buffer for a specific layer.
    pub fn block_k(&self, layer: usize, block_idx: usize) -> &metal::Buffer {
        &self.blocks[layer][block_idx].k_buf
    }

    /// Get a block's V buffer for a specific layer.
    pub fn block_v(&self, layer: usize, block_idx: usize) -> &metal::Buffer {
        &self.blocks[layer][block_idx].v_buf
    }

    /// Number of free blocks remaining.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Number of active sequences.
    pub fn active_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Total GPU memory allocated in bytes.
    pub fn memory_bytes(&self) -> usize {
        let block_bytes = self.num_kv_heads * BLOCK_SIZE * self.head_dim * 4;
        self.total_blocks * self.num_layers * 2 * block_bytes
    }

    /// Materialize the KV cache for a sequence into contiguous buffers
    /// compatible with the existing flat attention kernel.
    /// This is a temporary bridge until the paged attention kernel is ready.
    pub fn materialize_flat(
        &self,
        ctx: &MetalContext,
        seq_id: u64,
        layer: usize,
    ) -> Option<(metal::Buffer, metal::Buffer)> {
        let seq = self.sequences.iter().find(|s| s.seq_id == seq_id)?;
        let seq_len = seq.seq_len;
        if seq_len == 0 { return None; }

        let kv_size = self.num_kv_heads * self.head_dim;
        let opts = MTLResourceOptions::StorageModeShared;
        // Flat layout: [num_kv_heads, seq_len, head_dim] f32
        // But attention expects [num_kv_heads, max_seq_stride, head_dim]
        let stride = seq_len;
        let buf_bytes = (self.num_kv_heads * stride * self.head_dim * 4) as u64;
        let k_flat = ctx.device.new_buffer(buf_bytes, opts);
        let v_flat = ctx.device.new_buffer(buf_bytes, opts);

        for pos in 0..seq_len {
            let block_local = pos / BLOCK_SIZE;
            let offset_in_block = pos % BLOCK_SIZE;
            let phys_block = seq.page_table[block_local];

            let block_k = &self.blocks[layer][phys_block].k_buf;
            let block_v = &self.blocks[layer][phys_block].v_buf;

            // Copy each KV head's data for this position
            for h in 0..self.num_kv_heads {
                let src_off = (offset_in_block * kv_size + h * self.head_dim) * 4;
                let dst_off = (h * stride * self.head_dim + pos * self.head_dim) * 4;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (block_k.contents() as *const u8).add(src_off),
                        (k_flat.contents() as *mut u8).add(dst_off),
                        self.head_dim * 4,
                    );
                    std::ptr::copy_nonoverlapping(
                        (block_v.contents() as *const u8).add(src_off),
                        (v_flat.contents() as *mut u8).add(dst_off),
                        self.head_dim * 4,
                    );
                }
            }
        }

        Some((k_flat, v_flat))
    }
}
