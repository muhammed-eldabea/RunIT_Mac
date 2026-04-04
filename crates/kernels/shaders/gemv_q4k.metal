#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q4_K GEMV: y = A_q4k * x  (dequantize on-the-fly)
//
// Uses contiguous block assignment + Kahan-compensated sequential reduction
// to match llama.cpp's accumulation order.
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

// Sequential Kahan reduction of partial sums in threadgroup memory.
inline float kahan_reduce_tg(threadgroup float* tg, uint lsize) {
    float sum = tg[0];
    float c = 0.0f;
    for (uint i = 1; i < lsize; i++) {
        float val = tg[i] - c;
        float t = sum + val;
        c = (t - sum) - val;
        sum = t;
    }
    return sum;
}

// Inner loop: accumulate one Q4K block with Kahan compensation.
// Processes 256 elements from the block, reading f16 input.
inline void q4k_block_dot_f16(device const uchar* block, device const half* x_vec,
                               uint x_base, thread float& acc, thread float& comp) {
    const float d    = float(*reinterpret_cast<device const half*>(block + 0));
    const float dmin = float(*reinterpret_cast<device const half*>(block + 2));
    device const uchar* scales = block + Q4K_SCALES_OFF;
    device const uchar* qs     = block + Q4K_QS_OFF;

    for (uint j = 0; j < 4; j++) {
        float sc_lo, m_lo, sc_hi, m_hi;
        q4k_get_scale_min(j,     scales, sc_lo, m_lo);
        q4k_get_scale_min(j + 4, scales, sc_hi, m_hi);
        const float d_lo = d*sc_lo, m_lo2 = dmin*m_lo;
        const float d_hi = d*sc_hi, m_hi2 = dmin*m_hi;
        const uint base_lo = j*64, base_hi = j*64+32;

        for (uint i = 0; i < 32; i++) {
            const uchar bv = qs[j*32+i];
            float p1 = (d_lo*float(bv&0x0Fu)-m_lo2) * float(x_vec[x_base+base_lo+i]) - comp;
            float t1 = acc + p1; comp = (t1 - acc) - p1; acc = t1;
            float p2 = (d_hi*float(bv>>4u)-m_hi2) * float(x_vec[x_base+base_hi+i]) - comp;
            float t2 = acc + p2; comp = (t2 - acc) - p2; acc = t2;
        }
    }
}

// Same as above but reads f32 input.
inline void q4k_block_dot_f32(device const uchar* block, device const float* x_vec,
                               uint x_base, thread float& acc, thread float& comp) {
    const float d    = float(*reinterpret_cast<device const half*>(block + 0));
    const float dmin = float(*reinterpret_cast<device const half*>(block + 2));
    device const uchar* scales = block + Q4K_SCALES_OFF;
    device const uchar* qs     = block + Q4K_QS_OFF;

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

// ─── Q4K GEMV: f16 input → f16 output ──────────────────────────────────────

kernel void gemv_q4k_f16(
    device const uchar* A_q4k [[buffer(0)]],
    device const half*  x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    // Contiguous block assignment
    uint blk_per_thread = n_blocks / lsize;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == lsize - 1) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f, comp = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f16(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                          blk * Q4K_BLOCK_ELEMS, acc, comp);
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) y[row] = half(kahan_reduce_tg(tg, lsize));
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
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    uint blk_per_thread = n_blocks / lsize;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == lsize - 1) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f, comp = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f16(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                          blk * Q4K_BLOCK_ELEMS, acc, comp);
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) y[row] = half(kahan_reduce_tg(tg, lsize) + float(residual[row]));
}

// ─── Q4K GEMV: f32 input → f32 output ──────────────────────────────────────

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
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    uint blk_per_thread = n_blocks / lsize;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == lsize - 1) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f, comp = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f32(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                          blk * Q4K_BLOCK_ELEMS, acc, comp);
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize);
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
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    uint blk_per_thread = n_blocks / lsize;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == lsize - 1) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f, comp = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f32(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                          blk * Q4K_BLOCK_ELEMS, acc, comp);
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize) + residual[row];
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
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= M) return;
    const uint n_blocks = K / Q4K_BLOCK_ELEMS;
    device const uchar* row_data = A_q4k + row * n_blocks * Q4K_BLOCK_BYTES;

    uint blk_per_thread = n_blocks / lsize;
    uint blk_start = lid * blk_per_thread;
    uint blk_end   = (lid == lsize - 1) ? n_blocks : blk_start + blk_per_thread;

    float acc = 0.0f, comp = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4k_block_dot_f16(row_data + blk * Q4K_BLOCK_BYTES, x_vec,
                          blk * Q4K_BLOCK_ELEMS, acc, comp);
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize) + residual[row];
}
