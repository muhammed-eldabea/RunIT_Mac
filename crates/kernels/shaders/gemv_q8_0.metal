#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q8_0 GEMV: y = A_q8 * x  (dequantize on-the-fly)
//
// Reads packed Q8_0 weights (1.0625 bytes/element) directly and dequantizes
// during the dot product. This uses ~47% less memory bandwidth than the
// dequanted F16 path (2 bytes/element).
//
// Q8_0 block layout (34 bytes = 32 elements):
//   bytes 0-1:   d    (half) — scale factor
//   bytes 2-33:  qs[32] (int8) — quantized values
//   Value[i] = d * qs[i]
//
// Grid  : [M, 1, 1]   — one threadgroup per output row
// Group : [32, 1, 1]   — one SIMD group
// ─────────────────────────────────────────────────────────────────────────────

#define Q8_0_BLOCK_BYTES 34
#define Q8_0_BLOCK_ELEMS 32

// ─── Inner dot: one Q8_0 block × f32 input ──────────────────────────────────
inline void q8_0_block_dot_f32(device const uchar* block,
                                device const float* x_vec,
                                uint x_base, thread float& acc) {
    float d = float(*reinterpret_cast<device const half*>(block));
    device const char* qs = reinterpret_cast<device const char*>(block + 2);

    // Process 32 elements in 8 iterations, vectorized x_vec reads
    for (uint i = 0; i < 32; i += 4) {
        float4 x4 = *(device const float4*)(x_vec + x_base + i);
        acc += d * (float(qs[i])   * x4.x + float(qs[i+1]) * x4.y +
                    float(qs[i+2]) * x4.z + float(qs[i+3]) * x4.w);
    }
}

// ─── Inner dot: one Q8_0 block × f16 input ──────────────────────────────────
inline void q8_0_block_dot_f16(device const uchar* block,
                                device const half* x_vec,
                                uint x_base, thread float& acc) {
    float d = float(*reinterpret_cast<device const half*>(block));
    device const char* qs = reinterpret_cast<device const char*>(block + 2);

    for (uint i = 0; i < 32; i += 4) {
        float4 x4 = float4(*(device const half4*)(x_vec + x_base + i));
        acc += d * (float(qs[i])   * x4.x + float(qs[i+1]) * x4.y +
                    float(qs[i+2]) * x4.z + float(qs[i+3]) * x4.w);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q8_0 GEMV: f32 input → f32 output  (decode path — Q/K/V/FFN projections)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q8_0_f32in_f32out(
    device const uchar* A_q8  [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q8_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BLOCK_BYTES;

    // Distribute blocks evenly across 32 threads
    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lid * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q8_0_block_dot_f32(row_data + blk * Q8_0_BLOCK_BYTES, x_vec,
                           blk * Q8_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q8_0 GEMV + f32 residual  (output proj + residual, FFN down + residual)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q8_0_add_f32_f32in(
    device const uchar* A_q8     [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q8_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lid * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q8_0_block_dot_f32(row_data + blk * Q8_0_BLOCK_BYTES, x_vec,
                           blk * Q8_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q8_0 GEMV: f32 input → f16 output  (for forward_greedy lm_head, MoE)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q8_0_f32in(
    device const uchar* A_q8  [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q8_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lid * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q8_0_block_dot_f32(row_data + blk * Q8_0_BLOCK_BYTES, x_vec,
                           blk * Q8_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q8_0 GEMV: f16 input → f16 output  (TurboQuant / f16 pipeline)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q8_0_f16(
    device const uchar* A_q8  [[buffer(0)]],
    device const half*  x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q8_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lid * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q8_0_block_dot_f16(row_data + blk * Q8_0_BLOCK_BYTES, x_vec,
                           blk * Q8_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q8_0 GEMV + f16 residual  (TQ path)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q8_0_add_f16(
    device const uchar* A_q8     [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       half*  y        [[buffer(2)]],
    device const half*  residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q8_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lid * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q8_0_block_dot_f16(row_data + blk * Q8_0_BLOCK_BYTES, x_vec,
                           blk * Q8_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result + float(residual[row]));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q8_0 GEMV + f32 residual with f16 input (MoE expert accumulate path)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q8_0_add_f32res_f16(
    device const uchar* A_q8     [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q8_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lid * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q8_0_block_dot_f16(row_data + blk * Q8_0_BLOCK_BYTES, x_vec,
                           blk * Q8_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}
