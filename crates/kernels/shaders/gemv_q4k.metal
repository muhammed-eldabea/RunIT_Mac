#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q4_K GEMV: y = A_q4k * x  (dequantize on-the-fly)
//
// Instead of reading pre-dequantized F16 weights (2 bytes/element),
// reads packed Q4_K data (0.5625 bytes/element = 144 bytes / 256 elements).
// This reduces memory bandwidth by ~3.5x for weight reads.
//
// A_q4k: [M, K] stored as Q4_K blocks.  Each row = K/256 blocks × 144 bytes.
// x:     [K] f16 input vector (already dequantized activations).
// y:     [M] f16 output vector.
//
// Each threadgroup computes one output element y[row].
// Threads collaborate on the K reduction via simd_sum.
//
// Grid  : [M, 1, 1]   — one threadgroup per output row
// Group : [32, 1, 1]   — one SIMD group
// ─────────────────────────────────────────────────────────────────────────────

#define Q4K_BLOCK_BYTES 144
#define Q4K_BLOCK_ELEMS 256
#define Q4K_SCALES_OFF    4
#define Q4K_QS_OFF       16

// Extract 6-bit scale and min for sub-block j (0..7)
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

kernel void gemv_q4k_f16(
    device const uchar* A_q4k [[buffer(0)]],  // Q4_K packed weights [M rows × n_blocks_per_row blocks]
    device const half*  x_vec [[buffer(1)]],   // input vector [K] f16
    device       half*  y     [[buffer(2)]],   // output vector [M] f16
    constant     uint&  M     [[buffer(3)]],   // output dimension
    constant     uint&  K     [[buffer(4)]],   // input dimension (must be multiple of 256)
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;

    const uint n_blocks_per_row = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks_per_row * Q4K_BLOCK_BYTES;

    float acc = 0.0f;

    // Each thread processes a strided set of Q4K blocks
    for (uint blk = lid; blk < n_blocks_per_row; blk += lsize) {
        device const uchar* block = row_data + blk * Q4K_BLOCK_BYTES;

        // Super-block scales
        const float d    = float(*reinterpret_cast<device const half*>(block + 0));
        const float dmin = float(*reinterpret_cast<device const half*>(block + 2));

        device const uchar* scales = block + Q4K_SCALES_OFF;
        device const uchar* qs     = block + Q4K_QS_OFF;

        // Base index into x for this block
        const uint x_base = blk * Q4K_BLOCK_ELEMS;

        // Process 4 pairs of sub-blocks (j=0..3), matching ggml Q4K layout:
        //   Elements j*64+0..31:  low nibble of qs[j*32..], scale index j
        //   Elements j*64+32..63: high nibble of qs[j*32..], scale index j+4
        for (uint j = 0; j < 4; j++) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4k_get_scale_min(j,     scales, sc_lo, m_lo);
            q4k_get_scale_min(j + 4, scales, sc_hi, m_hi);

            const float d_lo  = d    * sc_lo;
            const float m_lo2 = dmin * m_lo;
            const float d_hi  = d    * sc_hi;
            const float m_hi2 = dmin * m_hi;

            const uint base_lo = j * 64;
            const uint base_hi = j * 64 + 32;

            for (uint i = 0; i < 32; i++) {
                const uchar byte_val = qs[j * 32 + i];

                const float q_lo = float(byte_val & 0x0Fu);
                acc += (d_lo * q_lo - m_lo2) * float(x_vec[x_base + base_lo + i]);

                const float q_hi = float(byte_val >> 4u);
                acc += (d_hi * q_hi - m_hi2) * float(x_vec[x_base + base_hi + i]);
            }
        }
    }

    // SIMD reduction across the 32-lane warp
    acc = simd_sum(acc);

    if (lid == 0) {
        y[row] = half(acc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q4K GEMV + residual add: y[i] = (A_q4k * x)[i] + residual[i]
// ─────────────────────────────────────────────────────────────────────────────
kernel void gemv_q4k_add_f16(
    device const uchar* A_q4k    [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       half*  y        [[buffer(2)]],
    device const half*  residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;

    const uint n_blocks_per_row = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks_per_row * Q4K_BLOCK_BYTES;

    float acc = 0.0f;

    for (uint blk = lid; blk < n_blocks_per_row; blk += lsize) {
        device const uchar* block = row_data + blk * Q4K_BLOCK_BYTES;

        const float d    = float(*reinterpret_cast<device const half*>(block + 0));
        const float dmin = float(*reinterpret_cast<device const half*>(block + 2));

        device const uchar* scales = block + Q4K_SCALES_OFF;
        device const uchar* qs     = block + Q4K_QS_OFF;
        const uint x_base = blk * Q4K_BLOCK_ELEMS;

        for (uint j = 0; j < 4; j++) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4k_get_scale_min(j,     scales, sc_lo, m_lo);
            q4k_get_scale_min(j + 4, scales, sc_hi, m_hi);

            const float d_lo  = d    * sc_lo;
            const float m_lo2 = dmin * m_lo;
            const float d_hi  = d    * sc_hi;
            const float m_hi2 = dmin * m_hi;

            const uint base_lo = j * 64;
            const uint base_hi = j * 64 + 32;

            for (uint i = 0; i < 32; i++) {
                const uchar byte_val = qs[j * 32 + i];

                const float q_lo = float(byte_val & 0x0Fu);
                acc += (d_lo * q_lo - m_lo2) * float(x_vec[x_base + base_lo + i]);

                const float q_hi = float(byte_val >> 4u);
                acc += (d_hi * q_hi - m_hi2) * float(x_vec[x_base + base_hi + i]);
            }
        }
    }

    acc = simd_sum(acc);

    if (lid == 0) {
        y[row] = half(acc + float(residual[row]));
    }
}

// Q4K GEMV with f32 input and f32 output: y_f32 = A_q4k * x_f32
// Uses Kahan compensated summation for precision.
kernel void gemv_q4k_f32in_f32out(
    device const uchar* A_q4k [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks_per_row = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks_per_row * Q4K_BLOCK_BYTES;
    float acc = 0.0f;
    float comp = 0.0f;
    for (uint blk = lid; blk < n_blocks_per_row; blk += lsize) {
        device const uchar* block = row_data + blk * Q4K_BLOCK_BYTES;
        const float d    = float(*reinterpret_cast<device const half*>(block + 0));
        const float dmin = float(*reinterpret_cast<device const half*>(block + 2));
        device const uchar* scales = block + Q4K_SCALES_OFF;
        device const uchar* qs     = block + Q4K_QS_OFF;
        const uint x_base = blk * Q4K_BLOCK_ELEMS;
        for (uint j = 0; j < 4; j++) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4k_get_scale_min(j,     scales, sc_lo, m_lo);
            q4k_get_scale_min(j + 4, scales, sc_hi, m_hi);
            const float d_lo = d*sc_lo, m_lo2 = dmin*m_lo;
            const float d_hi = d*sc_hi, m_hi2 = dmin*m_hi;
            const uint base_lo = j*64, base_hi = j*64+32;
            for (uint i = 0; i < 32; i++) {
                const uchar bv = qs[j*32+i];
                float p1 = (d_lo*float(bv&0x0Fu)-m_lo2) * x_vec[x_base+base_lo+i] - comp;
                float t1 = acc + p1; comp = (t1 - acc) - p1; acc = t1;
                float p2 = (d_hi*float(bv>>4u)-m_hi2) * x_vec[x_base+base_hi+i] - comp;
                float t2 = acc + p2; comp = (t2 - acc) - p2; acc = t2;
            }
        }
    }
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc;
}

// Q4K GEMV + f32 residual with f32 input: y_f32[i] = (A_q4k * x_f32)[i] + residual_f32[i]
// Uses Kahan compensated summation for precision.
kernel void gemv_q4k_add_f32_f32in(
    device const uchar* A_q4k    [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks_per_row = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks_per_row * Q4K_BLOCK_BYTES;
    float acc = 0.0f;
    float comp = 0.0f;
    for (uint blk = lid; blk < n_blocks_per_row; blk += lsize) {
        device const uchar* block = row_data + blk * Q4K_BLOCK_BYTES;
        const float d    = float(*reinterpret_cast<device const half*>(block + 0));
        const float dmin = float(*reinterpret_cast<device const half*>(block + 2));
        device const uchar* scales = block + Q4K_SCALES_OFF;
        device const uchar* qs     = block + Q4K_QS_OFF;
        const uint x_base = blk * Q4K_BLOCK_ELEMS;
        for (uint j = 0; j < 4; j++) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4k_get_scale_min(j,     scales, sc_lo, m_lo);
            q4k_get_scale_min(j + 4, scales, sc_hi, m_hi);
            const float d_lo = d*sc_lo, m_lo2 = dmin*m_lo;
            const float d_hi = d*sc_hi, m_hi2 = dmin*m_hi;
            const uint base_lo = j*64, base_hi = j*64+32;
            for (uint i = 0; i < 32; i++) {
                const uchar bv = qs[j*32+i];
                float p1 = (d_lo*float(bv&0x0Fu)-m_lo2) * x_vec[x_base+base_lo+i] - comp;
                float t1 = acc + p1; comp = (t1 - acc) - p1; acc = t1;
                float p2 = (d_hi*float(bv>>4u)-m_hi2) * x_vec[x_base+base_hi+i] - comp;
                float t2 = acc + p2; comp = (t2 - acc) - p2; acc = t2;
            }
        }
    }
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc + residual[row];
}

// Q4K GEMV + f32 residual: y_f32[i] = (A_q4k * x_f16)[i] + residual_f32[i]
kernel void gemv_q4k_add_f32res_f16(
    device const uchar* A_q4k    [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks_per_row = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks_per_row * Q4K_BLOCK_BYTES;
    float acc = 0.0f;
    for (uint blk = lid; blk < n_blocks_per_row; blk += lsize) {
        device const uchar* block = row_data + blk * Q4K_BLOCK_BYTES;
        const float d    = float(*reinterpret_cast<device const half*>(block + 0));
        const float dmin = float(*reinterpret_cast<device const half*>(block + 2));
        device const uchar* scales = block + Q4K_SCALES_OFF;
        device const uchar* qs     = block + Q4K_QS_OFF;
        const uint x_base = blk * Q4K_BLOCK_ELEMS;
        for (uint j = 0; j < 4; j++) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4k_get_scale_min(j,     scales, sc_lo, m_lo);
            q4k_get_scale_min(j + 4, scales, sc_hi, m_hi);
            const float d_lo = d*sc_lo, m_lo2 = dmin*m_lo;
            const float d_hi = d*sc_hi, m_hi2 = dmin*m_hi;
            const uint base_lo = j*64, base_hi = j*64+32;
            for (uint i = 0; i < 32; i++) {
                const uchar bv = qs[j*32+i];
                acc += (d_lo*float(bv&0x0Fu)-m_lo2) * float(x_vec[x_base+base_lo+i]);
                acc += (d_hi*float(bv>>4u)-m_hi2)   * float(x_vec[x_base+base_hi+i]);
            }
        }
    }
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc + residual[row];
}
