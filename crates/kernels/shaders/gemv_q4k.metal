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

        // Process all 8 sub-blocks (32 elements each)
        for (uint sub = 0; sub < 8; sub++) {
            float sc_f, m_f;
            q4k_get_scale_min(sub, scales, sc_f, m_f);

            const float actual_scale = d    * sc_f;
            const float actual_min   = dmin * m_f;

            const uint sub_offset = sub * 32;

            // Unroll inner loop — process 32 elements per sub-block
            // Read 16 bytes of qs (32 nibbles) and 32 f16 from x
            for (uint i = 0; i < 32; i += 2) {
                const uint byte_idx = (sub_offset + i) / 2;
                const uchar byte_val = qs[byte_idx];

                // Even element
                const float q0 = float(byte_val & 0x0Fu);
                const float w0 = actual_scale * q0 - actual_min;
                acc += w0 * float(x_vec[x_base + sub_offset + i]);

                // Odd element
                const float q1 = float(byte_val >> 4u);
                const float w1 = actual_scale * q1 - actual_min;
                acc += w1 * float(x_vec[x_base + sub_offset + i + 1]);
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

        for (uint sub = 0; sub < 8; sub++) {
            float sc_f, m_f;
            q4k_get_scale_min(sub, scales, sc_f, m_f);
            const float actual_scale = d    * sc_f;
            const float actual_min   = dmin * m_f;
            const uint sub_offset = sub * 32;

            for (uint i = 0; i < 32; i += 2) {
                const uint byte_idx = (sub_offset + i) / 2;
                const uchar byte_val = qs[byte_idx];

                const float q0 = float(byte_val & 0x0Fu);
                const float w0 = actual_scale * q0 - actual_min;
                acc += w0 * float(x_vec[x_base + sub_offset + i]);

                const float q1 = float(byte_val >> 4u);
                const float w1 = actual_scale * q1 - actual_min;
                acc += w1 * float(x_vec[x_base + sub_offset + i + 1]);
            }
        }
    }

    acc = simd_sum(acc);

    if (lid == 0) {
        y[row] = half(acc + float(residual[row]));
    }
}
