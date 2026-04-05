use std::cell::RefCell;
use std::collections::HashMap;

use metal::{
    CommandBuffer, CommandBufferRef, ComputeCommandEncoder, ComputeCommandEncoderRef,
    ComputePipelineState, CommandQueue, Device, Library,
    MTLResourceOptions,
};

use crate::error::{KernelError, Result};

/// All kernel function names compiled into the .metallib.
/// These must exactly match the `kernel void <name>(...)` identifiers in the .metal files.
pub const KERNEL_NAMES: &[&str] = &[
    // activation.metal
    "silu_mul_f16",
    "add_f16",
    "mul_f16",
    // activation.metal (broadcast bias)
    "add_bias_broadcast_f16",
    // activation.metal (KV cache scatter)
    "kv_copy_to_cache_f16",
    // activation.metal (argmax)
    "argmax_f16",
    // activation.metal (MoE scale + accumulate)
    "scale_accumulate_f16",
    // norm.metal
    "rms_norm_f16",
    // rope.metal
    "rope_inplace_f16",
    // gemv.metal
    "gemv_f16",
    "gemv_add_f16",
    // gemv_q4k.metal
    "gemv_q4k_f16",
    "gemv_q4k_add_f16",
    // attention.metal
    "flash_attention_f16",
    "decode_attention_f16",
    // dequant.metal
    "dequant_q4k_f16",
    // norm.metal (f32 variants)
    "rms_norm_f32in_f16out",
    "rms_norm_f32_f32",
    // norm.metal (f32 gamma)
    "rms_norm_f32_f32_f32g",
    // activation.metal (f32 residual)
    "add_f16_into_f32",
    // gemv.metal (f32 variants)
    "gemv_f32_f32out",
    "gemv_f32",
    "gemv_add_f32",
    "gemv_f16w_f32in",
    "gemv_add_f32res_f16",
    "gemv_add_f32_f16w",
    // gemv_q4k.metal (f32 residual)
    "gemv_q4k_add_f32res_f16",
    // gemv.metal (f32 weight variants)
    "gemv_f32w",
    "gemv_add_f32w",
    "gemv_f32w_f32in",
    "gemv_add_f32res_f32w",
    "gemv_add_f32_f32w",
    // gemv.metal (f16 weight, f32 in/out)
    "gemv_f16w_f32in_f32out",
    // gemv_q4k.metal (f32 in/out)
    "gemv_q4k_f32in_f32out",
    "gemv_q4k_add_f32_f32in",
    // activation.metal (f32 SwiGLU)
    "silu_mul_f32",
    // activation.metal (f32 KV cache scatter)
    "kv_copy_to_cache_f32",
    // rope.metal (f32)
    "rope_inplace_f32",
    // attention.metal (f32)
    "decode_attention_f32",
    // gemm.metal
    "gemm_f16",
    // rope.metal (batch)
    "rope_batch_inplace_f16",
    // gemv_q4_0.metal (fused Q4_0 dequant — 47% less BW than Q8_0, 72% less than f16)
    "gemv_q4_0_f32in_f32out",
    "gemv_q4_0_add_f32_f32in",
    "gemv_q4_0_f32in",
    "gemv_q4_0_f16",
    "gemv_q4_0_add_f16",
    "fused_ffn_q4_0_f32",
    // fused_qkv.metal (Q+K+V in one dispatch — 36× more TGs for better BW utilization)
    "fused_qkv_q8_0_f32",
    "fused_qkv_bias_q8_0_f32",
    "fused_qkv_f16w_f32",
    // fused_ffn.metal (gate+up+silu in one dispatch — saves 48 dispatches/token)
    "fused_ffn_q8_0_f32",
    "fused_ffn_f16w_f32",
    // fused_rope.metal (Q+K RoPE in one dispatch — saves 24 dispatches/token)
    "fused_rope_qk_f32",
    // gemm.metal (simdgroup_matrix AMX)
    "gemm_f16_simd",
    // gemv_multirow.metal (4 rows per TG, 128 threads)
    "gemv_f16_mr",
    "gemv_add_f16_mr",
    "gemv_f16w_f32in_f32out_mr",
    "gemv_add_f32_f16w_mr",
    "gemv_q8_0_f32in_f32out_mr",
    "gemv_q8_0_add_f32_f32in_mr",
    // gemv_q8_0_wide.metal (256 threads/row for better BW saturation)
    "gemv_q8_0_f32in_f32out_wide",
    "gemv_q8_0_add_f32_f32in_wide",
    "gemv_q8_0_f32in_wide",
    // gemv_q8_0.metal (fused Q8_0 dequant — 47% less bandwidth than f16)
    "gemv_q8_0_f32in_f32out",
    "gemv_q8_0_add_f32_f32in",
    "gemv_q8_0_f32in",
    "gemv_q8_0_f16",
    "gemv_q8_0_add_f16",
    "gemv_q8_0_add_f32res_f16",
    // turboquant.metal
    "tq_normalize_f16",
    "tq_rht_f16",
    "tq_rht_inverse_f16",
    "tq_lloyd_quant_f16",
    "tq_pack_bits_u8",
    "tq_unpack_bits_u8",
    "tq_centroid_lookup_f16",
    "tq_residual_norm_f16",
    "tq_qjl_signs_f16",
    "tq_qjl_correction_f16",
    "tq_scale_add_f16",
    "tq_group_quant_val_f16",
    "tq_group_dequant_val_f16",
];

/// Interior state for the pending command buffer + reusable encoder.
///
/// We keep a single `ComputeCommandEncoder` open and dispatch multiple
/// kernels on it (switching pipeline states via `set_compute_pipeline_state`).
/// This eliminates the ~15-30µs overhead of creating a new encoder per kernel.
/// With ~437 kernels per token, this saves ~6-13ms per token.
struct PendingCmdBuf {
    cmd: CommandBuffer,
    enc: ComputeCommandEncoder,
    encode_count: u32,
}

/// Holds the Metal device, command queue, compiled library, and
/// pre-built pipeline state objects for every kernel.
///
/// Also manages an **async command buffer**: kernels are encoded
/// into a shared command buffer and only committed + waited when
/// `flush()` is called (or when the buffer is full).
pub struct MetalContext {
    pub device:  Device,
    pub queue:   CommandQueue,
    library:     Library,
    pipelines:   HashMap<&'static str, ComputePipelineState>,
    /// The current pending command buffer. Wrapped in RefCell so that
    /// dispatch functions can borrow `&self` while mutating this.
    pending: RefCell<Option<PendingCmdBuf>>,
}

impl MetalContext {
    /// Initialise Metal: find device, load compiled .metallib, build all PSOs.
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(KernelError::NoDevice)?;
        let queue = device.new_command_queue();

        // Load the .metallib compiled by build.rs (path is baked in at compile time)
        let metallib_path = option_env!("METALLIB_PATH")
            .ok_or_else(|| KernelError::LibraryLoad {
                path: "<unknown>".into(),
                reason: "METALLIB_PATH not set — did build.rs find any .metal files?".into(),
            })?;

        let library = device
            .new_library_with_file(metallib_path)
            .map_err(|e| KernelError::LibraryLoad {
                path: metallib_path.into(),
                reason: e.to_string(),
            })?;

        // Pre-compile one PSO per kernel function
        let mut pipelines = HashMap::new();
        for &name in KERNEL_NAMES {
            let func = library
                .get_function(name, None)
                .map_err(|_| KernelError::PipelineNotFound(name.to_string()))?;

            let pso = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| KernelError::PipelineCreate {
                    name: name.to_string(),
                    reason: e.to_string(),
                })?;

            pipelines.insert(name, pso);
        }

        tracing::info!(
            device = device.name(),
            kernels = KERNEL_NAMES.len(),
            "Metal context initialised"
        );

        Ok(Self { device, queue, library, pipelines, pending: RefCell::new(None) })
    }

    /// Look up a pre-compiled pipeline state by kernel function name.
    pub fn pipeline(&self, name: &str) -> Result<&ComputePipelineState> {
        self.pipelines
            .get(name)
            .ok_or_else(|| KernelError::PipelineNotFound(name.to_string()))
    }

    /// Allocate a new GPU-accessible buffer of `size` bytes (for outputs/activations).
    pub fn new_buffer(&self, size: usize) -> metal::Buffer {
        self.device.new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Wrap an existing CPU pointer as a Metal buffer — zero-copy.
    ///
    /// # Safety
    /// `ptr` must be page-aligned (4096 bytes) and remain valid for the
    /// lifetime of the returned buffer. mmap pointers satisfy this automatically.
    pub unsafe fn buffer_from_ptr(
        &self,
        ptr: *mut std::ffi::c_void,
        size: usize,
    ) -> metal::Buffer {
        self.device.new_buffer_with_bytes_no_copy(
            ptr,
            size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }

    // ── Async command buffer API ─────────────────────────────────────────────

    /// Encode a compute command into the current pending command buffer.
    /// The command is NOT committed until `flush()` is called.
    ///
    /// `setup` receives a compute command encoder that is already configured
    /// with the named pipeline state. The caller must set buffers/bytes and
    /// dispatch threads/threadgroups within the closure.
    pub fn encode(
        &self,
        pipeline_name: &str,
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) -> Result<()> {
        let pso = self.pipeline(pipeline_name)?;
        let mut pending = self.pending.borrow_mut();

        // Lazily create command buffer + encoder on first use.
        // Auto-commit after 8192 encodes. A 7B model with 28 layers uses
        // ~437 kernels/token; 8192 allows ~18 tokens before forced flush.
        // Previous limit of 512 caused mid-token stalls on larger models.
        if pending.is_none() || pending.as_ref().map_or(false, |p| p.encode_count >= 8192) {
            if let Some(old) = pending.take() {
                old.enc.end_encoding();
                old.cmd.commit();
                old.cmd.wait_until_completed();
            }
            let cmd = self.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder().to_owned();
            *pending = Some(PendingCmdBuf {
                cmd,
                enc,
                encode_count: 0,
            });
        }

        let p = pending.as_mut().unwrap();
        // Reuse the open encoder — just switch pipeline state and dispatch.
        // This avoids the ~15-30µs cost of creating a new encoder per kernel.
        p.enc.set_compute_pipeline_state(pso);
        setup(&p.enc);
        p.encode_count += 1;

        Ok(())
    }

    /// Commit and wait for all pending GPU work to complete.
    /// After this call, all GPU buffer contents are visible to the CPU.
    ///
    /// This is a no-op if there is no pending work.
    pub fn flush(&self) {
        let mut pending = self.pending.borrow_mut();
        if let Some(p) = pending.take() {
            p.enc.end_encoding();
            p.cmd.commit();
            p.cmd.wait_until_completed();
        }
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        // Flush any remaining work
        self.flush();
    }
}
