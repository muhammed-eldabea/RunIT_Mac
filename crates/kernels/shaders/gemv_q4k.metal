#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// HIGH-PERFORMANCE Fused Q4_K GEMV: y = A_q4k * x  (dequantize on-the-fly)
//
// Optimizations applied:
//   1. simd_sum() hardware reduction (replaces Kahan threadgroup reduction)
//   2. No Kahan compensation in inner loop (3 fewer ops per element)
//   3. Vectorized half4 reads of input vector (4x fewer loads)
//   4. 4-byte batch reads of quantized data
//
// Grid  : [M, 1, 1]   — one threadgroup per output row
// Group : [32, 1, 1]   — one SIMD group
// ─────────────────────────────────────────────────────────────────────────────

#define Q4K_BLOCK_BYTES 144
#define Q4K_BLOCK_ELEMS 256
#define Q4K_SCALES_OFF    4
#define Q4K_QS_OFF       16

inline void q4k_get_scale_min(uint j, device const uchar* scales,
                               thread float& sc, thread float& m) {
    uchar sc_u, m_u;
    if (j < 4) {
        sc_u = scales[j]   & 0x3Fu;
        m_u  = scales[j+4] & 0x3Fu;
    } else {
        sc_u = (scales[j+4] & 0x0Fu) | ((scales[j-4] >> 6u) << 4u);
        m_u  = (scales[j+4] >> 4u)   | ((scales[j]   >> 6u) << 4u);
    }
    sc = float(sc_u);
    m  = float(m_u);
}

// ─── Optimized Q4K block dot product with f16 input ─────────────────────────
// Processes 256 elements from one Q4K block. Uses vectorized half4 reads
// for the input vector.
inline void q4k_block_dot_f16_fast(device const uchar* block,
                                    device const half* x_vec,
                                    uint x_base, thread float& acc) {
    const float d    = float(*reinterpret_cast<device const half*>(block + 0));
    const float dmin = float(*reinterpret_cast<device const half*>(block + 2));
    device const uchar* scales = block + Q4K_SCALES_OFF;
    device const uchar* qs     = block + Q4K_QS_OFF;

    for (uint j = 0; j < 4; j++) {
        float sc_lo, m_lo, sc_hi, m_hi;
        q4k_get_scale_min(j,     scales, sc_lo, m_lo);
        q4k_get_scale_min(j + 4, scales, sc_hi, m_hi);
        const float d_lo = d * sc_lo, m_lo2 = dmin * m_lo;
        const float d_hi = d * sc_hi, m_hi2 = dmin * m_hi;
        const uint base_lo = x_base + j * 64;
        const uint base_hi = x_base + j * 64 + 32;

        // Process 32 bytes (64 elements) with 4-byte batch + half4 vectorized reads
        for (uint i = 0; i < 32; i += 4) {
            // Read 4 quantized bytes
            uchar bv0 = qs[j * 32 + i];
            uchar bv1 = qs[j * 32 + i + 1];
            uchar bv2 = qs[j * 32 + i + 2];
            uchar bv3 = qs[j * 32 + i + 3];

            // Read 4 input values at once (low nibble block)
            float4 x_lo = float4(*(device const half4*)(x_vec + base_lo + i));
            // Read 4 input values at once (high nibble block)
            float4 x_hi = float4(*(device const half4*)(x_vec + base_hi + i));

            // Dequant + multiply-accumulate (low nibbles)
            acc += (d_lo * float(bv0 & 0x0Fu) - m_lo2) * x_lo.x;
            acc += (d_lo * float(bv1 & 0x0Fu) - m_lo2) * x_lo.y;
            acc += (d_lo * float(bv2 & 0x0Fu) - m_lo2) * x_lo.z;
            acc += (d_lo * float(bv3 & 0x0Fu) - m_lo2) * x_lo.w;

            // Dequant + multiply-accumulate (high nibbles)
            acc += (d_hi * float(bv0 >> 4u) - m_hi2) * x_hi.x;
            acc += (d_hi * float(bv1 >> 4u) - m_hi2) * x_hi.y;
            acc += (d_hi * float(bv2 >> 4u) - m_hi2) * x_hi.z;
            acc += (d_hi * float(bv3 >> 4u) - m_hi2) * x_hi.w;
        }
    }
}

// ─── Optimized Q4K block dot product with f32 input ─────────────────────────
inline void q4k_block_dot_f32_fast(device const uchar* block,
                                    device const float* x_vec,
                                    uint x_base, thread float& acc) {
    const float d    = float(*reinterpret_cast<device const half*>(block + 0));
    const float dmin = float(*reinterpret_cast<device const half*>(block + 2));
    device const uchar* scales = block + Q4K_SCALES_OFF;
    device const uchar* qs     = block + Q4K_QS_OFF;

    for (uint j = 0; j < 4; j++) {
        float sc_lo, m_lo, sc_hi, m_hi;
        q4k_get_scale_min(j,     scales, sc_lo, m_lo);
        q4k_get_scale_min(j + 4, scales, sc_hi, m_hi);
        const float d_lo = d * sc_lo, m_lo2 = dmin * m_lo;
        const float d_hi = d * sc_hi, m_hi2 = dmin * m_hi;
        const uint base_lo = x_base + j * 64;
        const uint base_hi = x_base + j * 64 + 32;

        for (uint i = 0; i < 32; i += 4) {
            uchar bv0 = qs[j * 32 + i];
            uchar bv1 = qs[j * 32 + i + 1];
            uchar bv2 = qs[j * 32 + i + 2];
            uchar bv3 = qs[j * 32 + i + 3];

            float4 x_lo = *(device const float4*)(x_vec + base_lo + i);
            float4 x_hi = *(device const float4*)(x_vec + base_hi + i);

            acc += (d_lo * float(bv0 & 0x0Fu) - m_lo2) * x_lo.x;
            acc += (d_lo * float(bv1 & 0x0Fu) - m_lo2) * x_lo.y;
            acc += (d_lo * float(bv2 & 0x0Fu) - m_lo2) * x_lo.z;
            acc += (d_lo * float(bv3 & 0x0Fu) - m_lo2) * x_lo.w;

            acc += (d_hi * float(bv0 >> 4u) - m_hi2) * x_hi.x;
            acc += (d_hi * float(bv1 >> 4u) - m_hi2) * x_hi.y;
            acc += (d_hi * float(bv2 >> 4u) - m_hi2) * x_hi.z;
            acc += (d_hi * float(bv3 >> 4u) - m_hi2) * x_hi.w;
        }
    }
}

// ─── Q4K GEMV: f16 input → f16 output ──────────────────────────────────────

kernel void gemv_q4k_f16(
    device const uchar* A_q4k [[buffer(0)]],
    device const half*  x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    // Contiguous block assignment per thread
    uint blk_per_thread = n_blocks / 32;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == 31) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f16_fast(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                               blk * Q4K_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result);
}

// ─── Q4K GEMV + f16 residual ────────────────────────────────────────────────

kernel void gemv_q4k_add_f16(
    device const uchar* A_q4k    [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       half*  y        [[buffer(2)]],
    device const half*  residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    uint blk_per_thread = n_blocks / 32;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == 31) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f16_fast(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                               blk * Q4K_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result + float(residual[row]));
}

// ─── Q4K GEMV: f32 input → f32 output ──────────────────────────────────────

kernel void gemv_q4k_f32in_f32out(
    device const uchar* A_q4k [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    uint blk_per_thread = n_blocks / 32;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == 31) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f32_fast(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                               blk * Q4K_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = result;
}

// ─── Q4K GEMV + f32 residual with f32 input ────────────────────────────────

kernel void gemv_q4k_add_f32_f32in(
    device const uchar* A_q4k    [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    uint blk_per_thread = n_blocks / 32;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == 31) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f32_fast(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                               blk * Q4K_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}

// ─── Q4K GEMV + f32 residual: f16 input ────────────────────────────────────

kernel void gemv_q4k_add_f32res_f16(
    device const uchar* A_q4k    [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    uint blk_per_thread = n_blocks / 32;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == 31) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f16_fast(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                               blk * Q4K_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}
