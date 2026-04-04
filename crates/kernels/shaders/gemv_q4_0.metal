#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Fused Q4_0 GEMV: y = A_q4 * x  (dequantize on-the-fly)
//
// Q4_0 reads only 0.5625 bytes per element — 47% less than Q8_0 (1.0625)
// and 72% less than F16 (2.0). This is the ULTIMATE bandwidth optimization.
//
// Q4_0 block layout (18 bytes = 32 elements):
//   bytes 0-1:   d (half) — scale factor
//   bytes 2-17:  qs[16] — 4-bit nibbles, 2 per byte
//   Low nibble  = element 2i     (qs[i] & 0x0F)
//   High nibble = element 2i+1   (qs[i] >> 4)
//   Value = d * (nibble - 8)
//
// Grid:  [ceil(M/4), 1, 1]  — multi-row (4 rows/TG)
// Group: [128, 1, 1]        — 4 SIMD groups × 32 lanes
// ─────────────────────────────────────────────────────────────────────────────

#define Q4_0_BLOCK_BYTES 18
#define Q4_0_BLOCK_ELEMS 32
#define Q4_0_MR 4

// ─── Inner dot: one Q4_0 block × f32 input ──────────────────────────────────
inline void q4_0_block_dot_f32(device const uchar* block,
                                device const float* x_vec,
                                uint x_base, thread float& acc) {
    float d = float(*reinterpret_cast<device const half*>(block));
    device const uchar* qs = block + 2;

    for (uint i = 0; i < 16; i += 2) {
        // Read 2 bytes = 4 elements
        uchar b0 = qs[i];
        uchar b1 = qs[i + 1];

        // 4 input values
        float x0 = x_vec[x_base + i*2];
        float x1 = x_vec[x_base + i*2 + 1];
        float x2 = x_vec[x_base + i*2 + 2];
        float x3 = x_vec[x_base + i*2 + 3];

        // Dequant: value = d * (nibble - 8)
        acc += d * (float((int)(b0 & 0x0F) - 8) * x0 +
                    float((int)(b0 >> 4)    - 8) * x1 +
                    float((int)(b1 & 0x0F) - 8) * x2 +
                    float((int)(b1 >> 4)    - 8) * x3);
    }
}

// ─── Inner dot: one Q4_0 block × f16 input ──────────────────────────────────
inline void q4_0_block_dot_f16(device const uchar* block,
                                device const half* x_vec,
                                uint x_base, thread float& acc) {
    float d = float(*reinterpret_cast<device const half*>(block));
    device const uchar* qs = block + 2;

    for (uint i = 0; i < 16; i += 2) {
        uchar b0 = qs[i];
        uchar b1 = qs[i + 1];

        float4 x4 = float4(*(device const half4*)(x_vec + x_base + i*2));

        acc += d * (float((int)(b0 & 0x0F) - 8) * x4.x +
                    float((int)(b0 >> 4)    - 8) * x4.y +
                    float((int)(b1 & 0x0F) - 8) * x4.z +
                    float((int)(b1 >> 4)    - 8) * x4.w);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q4_0 GEMV: f32 input → f32 output (decode path)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q4_0_f32in_f32out(
    device const uchar* A_q4  [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint row  = tg_id * Q4_0_MR + sg;
    if (row >= M) return;

    const uint n_blocks = K / Q4_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q4 + row * n_blocks * Q4_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4_0_block_dot_f32(row_data + blk * Q4_0_BLOCK_BYTES, x_vec,
                           blk * Q4_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lane == 0) y[row] = result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q4_0 GEMV + f32 residual (output proj + FFN down + residual)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q4_0_add_f32_f32in(
    device const uchar* A_q4     [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint row  = tg_id * Q4_0_MR + sg;
    if (row >= M) return;

    const uint n_blocks = K / Q4_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q4 + row * n_blocks * Q4_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4_0_block_dot_f32(row_data + blk * Q4_0_BLOCK_BYTES, x_vec,
                           blk * Q4_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lane == 0) y[row] = result + residual[row];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q4_0 GEMV: f32 input → f16 output (for forward_greedy lm_head)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q4_0_f32in(
    device const uchar* A_q4  [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint row  = tg_id * Q4_0_MR + sg;
    if (row >= M) return;

    const uint n_blocks = K / Q4_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q4 + row * n_blocks * Q4_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4_0_block_dot_f32(row_data + blk * Q4_0_BLOCK_BYTES, x_vec,
                           blk * Q4_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lane == 0) y[row] = half(result);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q4_0 GEMV: f16 input → f16 output (TQ / f16 path)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q4_0_f16(
    device const uchar* A_q4  [[buffer(0)]],
    device const half*  x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint row  = tg_id * Q4_0_MR + sg;
    if (row >= M) return;

    const uint n_blocks = K / Q4_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q4 + row * n_blocks * Q4_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4_0_block_dot_f16(row_data + blk * Q4_0_BLOCK_BYTES, x_vec,
                           blk * Q4_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lane == 0) y[row] = half(result);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q4_0 GEMV + f16 residual
// ═══════════════════════════════════════════════════════════════════════════════

kernel void gemv_q4_0_add_f16(
    device const uchar* A_q4     [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       half*  y        [[buffer(2)]],
    device const half*  residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint row  = tg_id * Q4_0_MR + sg;
    if (row >= M) return;

    const uint n_blocks = K / Q4_0_BLOCK_ELEMS;
    device const uchar* row_data = A_q4 + row * n_blocks * Q4_0_BLOCK_BYTES;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        q4_0_block_dot_f16(row_data + blk * Q4_0_BLOCK_BYTES, x_vec,
                           blk * Q4_0_BLOCK_ELEMS, acc);
    }

    float result = simd_sum(acc);
    if (lane == 0) y[row] = half(result + float(residual[row]));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fused Q4_0 gate+up+silu for FFN (1 dispatch instead of 3)
// ═══════════════════════════════════════════════════════════════════════════════

inline float silu_fn(float x) { return x / (1.0f + exp(-x)); }

kernel void fused_ffn_q4_0_f32(
    device const uchar* W_gate  [[buffer(0)]],
    device const uchar* W_up    [[buffer(1)]],
    device const float* x_vec   [[buffer(2)]],
    device       float* act_out [[buffer(3)]],
    constant     uint&  ff_dim  [[buffer(4)]],
    constant     uint&  K       [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint row  = tg_id * Q4_0_MR + sg;
    if (row >= ff_dim) return;

    const uint n_blocks = K / Q4_0_BLOCK_ELEMS;
    const uint row_bytes = n_blocks * Q4_0_BLOCK_BYTES;
    device const uchar* gate_row = W_gate + row * row_bytes;
    device const uchar* up_row   = W_up   + row * row_bytes;

    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (uint blk = blk_start; blk < blk_end; blk++) {
        device const uchar* g_blk = gate_row + blk * Q4_0_BLOCK_BYTES;
        device const uchar* u_blk = up_row   + blk * Q4_0_BLOCK_BYTES;

        float g_d = float(*reinterpret_cast<device const half*>(g_blk));
        float u_d = float(*reinterpret_cast<device const half*>(u_blk));
        device const uchar* g_qs = g_blk + 2;
        device const uchar* u_qs = u_blk + 2;

        uint x_base = blk * Q4_0_BLOCK_ELEMS;

        for (uint i = 0; i < 16; i += 2) {
            float x0 = x_vec[x_base + i*2];
            float x1 = x_vec[x_base + i*2 + 1];
            float x2 = x_vec[x_base + i*2 + 2];
            float x3 = x_vec[x_base + i*2 + 3];

            uchar gb0 = g_qs[i], gb1 = g_qs[i+1];
            uchar ub0 = u_qs[i], ub1 = u_qs[i+1];

            gate_acc += g_d * (float((int)(gb0&0xF)-8)*x0 + float((int)(gb0>>4)-8)*x1 +
                               float((int)(gb1&0xF)-8)*x2 + float((int)(gb1>>4)-8)*x3);
            up_acc   += u_d * (float((int)(ub0&0xF)-8)*x0 + float((int)(ub0>>4)-8)*x1 +
                               float((int)(ub1&0xF)-8)*x2 + float((int)(ub1>>4)-8)*x3);
        }
    }

    float gate_val = simd_sum(gate_acc);
    float up_val   = simd_sum(up_acc);
    if (lane == 0) act_out[row] = silu_fn(gate_val) * up_val;
}
