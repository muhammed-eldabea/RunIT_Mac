#include <metal_stdlib>
using namespace metal;

// FUSED QKV PROJECTION + BIAS: Q, K, V + bias-add in ONE dispatch
// Replaces 6 dispatches per layer (3 GEMVs + 3 bias-adds) with 1.
// Grid:  [ceil(M_total/4), 1, 1]  Group: [128, 1, 1]

#define QKV_MR 4
#define Q8_0_BB 34
#define Q8_0_BE 32

// Q8_0 weights + fused bias
kernel void fused_qkv_bias_q8_0_f32(
    device const uchar* W_q      [[buffer(0)]],
    device const uchar* W_k      [[buffer(1)]],
    device const uchar* W_v      [[buffer(2)]],
    device const float* x_vec    [[buffer(3)]],
    device       float* out_q    [[buffer(4)]],
    device       float* out_k    [[buffer(5)]],
    device       float* out_v    [[buffer(6)]],
    device const half*  bias_q   [[buffer(7)]],
    device const half*  bias_k   [[buffer(8)]],
    device const half*  bias_v   [[buffer(9)]],
    constant     uint&  q_dim    [[buffer(10)]],
    constant     uint&  kv_dim   [[buffer(11)]],
    constant     uint&  K        [[buffer(12)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint global_row = tg_id * QKV_MR + sg;
    const uint m_total = q_dim + kv_dim + kv_dim;
    if (global_row >= m_total) return;

    device const uchar* W_row;
    device float* out;
    device const half* bias;
    uint local_row;

    if (global_row < q_dim) {
        local_row = global_row; W_row = W_q; out = out_q; bias = bias_q;
    } else if (global_row < q_dim + kv_dim) {
        local_row = global_row - q_dim; W_row = W_k; out = out_k; bias = bias_k;
    } else {
        local_row = global_row - q_dim - kv_dim; W_row = W_v; out = out_v; bias = bias_v;
    }

    const uint n_blocks = K / Q8_0_BE;
    device const uchar* row_data = W_row + local_row * n_blocks * Q8_0_BB;
    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        device const uchar* block = row_data + blk * Q8_0_BB;
        float d = float(*reinterpret_cast<device const half*>(block));
        device const char* qs = reinterpret_cast<device const char*>(block + 2);
        uint x_base = blk * Q8_0_BE;
        for (uint i = 0; i < 32; i += 4) {
            float4 x4 = *(device const float4*)(x_vec + x_base + i);
            acc += d * (float(qs[i])*x4.x + float(qs[i+1])*x4.y +
                        float(qs[i+2])*x4.z + float(qs[i+3])*x4.w);
        }
    }
    float result = simd_sum(acc);
    if (lane == 0) out[local_row] = result + float(bias[local_row]);
}

// Q8_0 weights, no bias
kernel void fused_qkv_q8_0_f32(
    device const uchar* W_q      [[buffer(0)]],
    device const uchar* W_k      [[buffer(1)]],
    device const uchar* W_v      [[buffer(2)]],
    device const float* x_vec    [[buffer(3)]],
    device       float* out_q    [[buffer(4)]],
    device       float* out_k    [[buffer(5)]],
    device       float* out_v    [[buffer(6)]],
    constant     uint&  q_dim    [[buffer(7)]],
    constant     uint&  kv_dim   [[buffer(8)]],
    constant     uint&  K        [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint global_row = tg_id * QKV_MR + sg;
    const uint m_total = q_dim + kv_dim + kv_dim;
    if (global_row >= m_total) return;

    device const uchar* W_row; device float* out; uint local_row;
    if (global_row < q_dim) {
        local_row = global_row; W_row = W_q; out = out_q;
    } else if (global_row < q_dim + kv_dim) {
        local_row = global_row - q_dim; W_row = W_k; out = out_k;
    } else {
        local_row = global_row - q_dim - kv_dim; W_row = W_v; out = out_v;
    }
    const uint n_blocks = K / Q8_0_BE;
    device const uchar* row_data = W_row + local_row * n_blocks * Q8_0_BB;
    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);
    float acc = 0.0f;
    for (uint blk = blk_start; blk < blk_end; blk++) {
        device const uchar* block = row_data + blk * Q8_0_BB;
        float d = float(*reinterpret_cast<device const half*>(block));
        device const char* qs = reinterpret_cast<device const char*>(block + 2);
        uint x_base = blk * Q8_0_BE;
        for (uint i = 0; i < 32; i += 4) {
            float4 x4 = *(device const float4*)(x_vec + x_base + i);
            acc += d * (float(qs[i])*x4.x + float(qs[i+1])*x4.y +
                        float(qs[i+2])*x4.z + float(qs[i+3])*x4.w);
        }
    }
    float result = simd_sum(acc);
    if (lane == 0) out[local_row] = result;
}

// F16 weights, no bias
kernel void fused_qkv_f16w_f32(
    device const half*  W_q      [[buffer(0)]],
    device const half*  W_k      [[buffer(1)]],
    device const half*  W_v      [[buffer(2)]],
    device const float* x_vec    [[buffer(3)]],
    device       float* out_q    [[buffer(4)]],
    device       float* out_k    [[buffer(5)]],
    device       float* out_v    [[buffer(6)]],
    constant     uint&  q_dim    [[buffer(7)]],
    constant     uint&  kv_dim   [[buffer(8)]],
    constant     uint&  K        [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint global_row = tg_id * QKV_MR + sg;
    const uint m_total = q_dim + kv_dim + kv_dim;
    if (global_row >= m_total) return;

    device const half* W_row; device float* out; uint local_row;
    if (global_row < q_dim) {
        local_row = global_row; W_row = W_q; out = out_q;
    } else if (global_row < q_dim + kv_dim) {
        local_row = global_row - q_dim; W_row = W_k; out = out_k;
    } else {
        local_row = global_row - q_dim - kv_dim; W_row = W_v; out = out_v;
    }
    device const half* A_row = W_row + local_row * K;
    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lane; i < K4; i += 32) {
        acc += dot(float4(((device const half4*)A_row)[i]),
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lane; i < K; i += 32) {
        acc += float(A_row[i]) * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lane == 0) out[local_row] = result;
}
