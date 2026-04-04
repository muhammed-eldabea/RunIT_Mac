#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// KV cache scatter: copy [num_kv_heads, head_dim] from src into dst at a
// specific sequence position, given dst layout [num_kv_heads, max_seq, head_dim].
//
// This runs entirely on GPU, eliminating the CPU flush + memcpy round-trip.
//
// Grid  : [head_dim, num_kv_heads, 1]
// Group : [min(head_dim, 256), 1, 1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void kv_copy_to_cache_f16(
    device const half* src  [[buffer(0)]],   // [num_kv_heads, head_dim]
    device       half* dst  [[buffer(1)]],   // [num_kv_heads, max_seq, head_dim]
    constant     uint& pos       [[buffer(2)]],   // sequence position to write
    constant     uint& max_seq   [[buffer(3)]],   // max_seq_len
    constant     uint& head_dim  [[buffer(4)]],   // head_dim
    uint2 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint h = gid.y;
    if (d >= head_dim) return;
    uint src_idx = h * head_dim + d;
    uint dst_idx = (h * max_seq + pos) * head_dim + d;
    dst[dst_idx] = src[src_idx];
}

// f32 KV cache scatter: copy [num_kv_heads, head_dim] f32 into f32 cache.
kernel void kv_copy_to_cache_f32(
    device const float* src  [[buffer(0)]],
    device       float* dst  [[buffer(1)]],
    constant     uint&  pos      [[buffer(2)]],
    constant     uint&  max_seq  [[buffer(3)]],
    constant     uint&  head_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint h = gid.y;
    if (d >= head_dim) return;
    uint src_idx = h * head_dim + d;
    uint dst_idx = (h * max_seq + pos) * head_dim + d;
    dst[dst_idx] = src[src_idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// Argmax: find the index of the maximum element in a half buffer.
//
// Two-pass reduction:
//   1. Each threadgroup finds a local max + index, writes to partial results
//   2. Thread 0 of group 0 does a final scan over partial results
//
// Grid  : [1, 1, 1]   — single threadgroup
// Group : [1024, 1, 1] — max threadgroup size
// ─────────────────────────────────────────────────────────────────────────────
kernel void argmax_f16(
    device const half* input  [[buffer(0)]],   // [N] f16 values
    device       uint* result [[buffer(1)]],   // [1] uint — winning index
    constant     uint& N      [[buffer(2)]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    // Each thread scans a strided slice
    float best_val = -INFINITY;
    uint  best_idx = 0;
    for (uint i = lid; i < N; i += lsize) {
        float v = float(input[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // SIMD reduction within the 32-thread warp
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint  other_idx = simd_shuffle_down(best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Lane 0 of each SIMD group has the local winner.
    // Use threadgroup memory for cross-SIMD reduction.
    threadgroup float tg_vals[32];  // max 1024/32 = 32 SIMD groups
    threadgroup uint  tg_idxs[32];

    uint simd_id   = lid / 32;
    uint simd_lane = lid % 32;
    if (simd_lane == 0) {
        tg_vals[simd_id] = best_val;
        tg_idxs[simd_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 does the final reduction across SIMD groups
    if (lid == 0) {
        uint n_simds = (lsize + 31) / 32;
        float final_val = tg_vals[0];
        uint  final_idx = tg_idxs[0];
        for (uint s = 1; s < n_simds; s++) {
            if (tg_vals[s] > final_val) {
                final_val = tg_vals[s];
                final_idx = tg_idxs[s];
            }
        }
        result[0] = final_idx;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale + accumulate: acc[i] += src[i] * scalar
//
// Used by MoE to accumulate weighted expert outputs on GPU without flushing.
//
// Grid  : [N, 1, 1]
// Group : [256, 1, 1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void scale_accumulate_f16(
    device const half*  src    [[buffer(0)]],   // [N] source (expert output)
    device       half*  acc    [[buffer(1)]],   // [N] accumulator (modified in-place)
    constant     float& weight [[buffer(2)]],   // scalar routing weight
    constant     uint&  N      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    acc[gid] = acc[gid] + half(float(src[gid]) * weight);
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU: fused silu(gate) * up
//
// Used in Qwen2/LLaMA FFN:
//   FFN(x) = down_proj( silu(gate_proj(x)) ⊙ up_proj(x) )
//
// Grid  : [num_elements, 1, 1]   (each thread handles one element)
// Group : [256, 1, 1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void silu_mul_f16(
    device const half* gate [[buffer(0)]],   // gate_proj output [N]
    device const half* up   [[buffer(1)]],   // up_proj output   [N]
    device       half* out  [[buffer(2)]],   // output           [N]
    constant     uint& N    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;

    float g = float(gate[gid]);
    // SiLU: x * sigmoid(x)  =  x / (1 + exp(-x))
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = half(silu_g * float(up[gid]));
}

// f32 SwiGLU: reads f32 gate/up, writes f32 output.
kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint&  N    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    float g = gate[gid];
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = silu_g * up[gid];
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise add (residual connections)
//
// Grid  : [num_elements, 1, 1]
// Group : [256, 1, 1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void add_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device       half* out [[buffer(2)]],
    constant     uint& N   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    out[gid] = a[gid] + b[gid];
}

// Add f16 source into f32 accumulator: acc[i] += src[i]
kernel void add_f16_into_f32(
    device const half*  src [[buffer(0)]],
    device       float* acc [[buffer(1)]],
    constant     uint&  N   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    acc[gid] += float(src[gid]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise multiply
//
// Grid  : [num_elements, 1, 1]
// Group : [256, 1, 1]
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// Broadcast bias add: x[row, col] += bias[col]
//
// Used in prefill to add bias to all sequence positions in one dispatch.
//
// Grid  : [dim, seq, 1]   (width = dim, height = seq)
// Group : [256, 1, 1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void add_bias_broadcast_f16(
    device       half* x    [[buffer(0)]],   // [seq, dim] — modified in-place
    device const half* bias [[buffer(1)]],   // [dim]
    constant     uint& dim  [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= dim) return;
    x[row * dim + col] = x[row * dim + col] + bias[col];
}

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise multiply
//
// Grid  : [num_elements, 1, 1]
// Group : [256, 1, 1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void mul_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device       half* out [[buffer(2)]],
    constant     uint& N   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    out[gid] = a[gid] * b[gid];
}
