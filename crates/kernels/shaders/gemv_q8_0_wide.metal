#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// WIDE Q8_0 GEMV: 256 threads per row (8 SIMD groups)
//
// Problem: Current Q8_0 GEMV uses 32 threads per row (1 SIMD group).
// For K=896, only 28 Q8_0 blocks per row → most threads do <1 block.
// This means only 1 memory request pipeline is active per row.
//
// Solution: Use 256 threads (8 SIMD groups) per row. Each SIMD group
// handles a STRIPE of the K dimension. The 8 concurrent SIMD groups
// create 8× more in-flight memory requests, hiding memory latency
// and pushing bandwidth utilization closer to the hardware peak.
//
// Reduction: simd_sum within each SIMD group → 8 partial sums →
// threadgroup memory tree reduction by thread 0.
//
// Grid:  [M, 1, 1]  — one threadgroup per output row
// Group: [256, 1, 1] — 8 SIMD groups × 32 lanes
// ─────────────────────────────────────────────────────────────────────────────

#define Q8_0_BB 34
#define Q8_0_BE 32
#define WIDE_TG 256
#define WIDE_SIMD_GROUPS 8

kernel void gemv_q8_0_f32in_f32out_wide(
    device const uchar* A_q8  [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    threadgroup  float* tg_partials [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;

    const uint sg_id   = lid / 32;    // SIMD group index (0..7)
    const uint sg_lane = lid % 32;    // lane within SIMD group

    const uint n_blocks = K / Q8_0_BE;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BB;

    // Each thread processes blocks with stride 256
    float acc = 0.0f;
    for (uint blk = lid; blk < n_blocks; blk += WIDE_TG) {
        device const uchar* block = row_data + blk * Q8_0_BB;
        float d = float(*reinterpret_cast<device const half*>(block));
        device const char* qs = reinterpret_cast<device const char*>(block + 2);
        uint x_base = blk * Q8_0_BE;

        for (uint i = 0; i < 32; i += 4) {
            float4 x4 = *(device const float4*)(x_vec + x_base + i);
            acc += d * (float(qs[i])   * x4.x + float(qs[i+1]) * x4.y +
                        float(qs[i+2]) * x4.z + float(qs[i+3]) * x4.w);
        }
    }

    // Level 1: simd_sum within each SIMD group
    float simd_result = simd_sum(acc);

    // Level 2: write to threadgroup memory, tree reduce
    if (sg_lane == 0) {
        tg_partials[sg_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float total = tg_partials[0];
        for (uint i = 1; i < WIDE_SIMD_GROUPS; i++) {
            total += tg_partials[i];
        }
        y[row] = total;
    }
}

// Wide Q8_0 GEMV + f32 residual
kernel void gemv_q8_0_add_f32_f32in_wide(
    device const uchar* A_q8     [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    threadgroup  float* tg_partials [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;

    const uint sg_id   = lid / 32;
    const uint sg_lane = lid % 32;
    const uint n_blocks = K / Q8_0_BE;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BB;

    float acc = 0.0f;
    for (uint blk = lid; blk < n_blocks; blk += WIDE_TG) {
        device const uchar* block = row_data + blk * Q8_0_BB;
        float d = float(*reinterpret_cast<device const half*>(block));
        device const char* qs = reinterpret_cast<device const char*>(block + 2);
        uint x_base = blk * Q8_0_BE;

        for (uint i = 0; i < 32; i += 4) {
            float4 x4 = *(device const float4*)(x_vec + x_base + i);
            acc += d * (float(qs[i])   * x4.x + float(qs[i+1]) * x4.y +
                        float(qs[i+2]) * x4.z + float(qs[i+3]) * x4.w);
        }
    }

    float simd_result = simd_sum(acc);
    if (sg_lane == 0) tg_partials[sg_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float total = tg_partials[0];
        for (uint i = 1; i < WIDE_SIMD_GROUPS; i++) total += tg_partials[i];
        y[row] = total + residual[row];
    }
}

// Wide Q8_0 GEMV: f32 in → f16 out
kernel void gemv_q8_0_f32in_wide(
    device const uchar* A_q8  [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    threadgroup  float* tg_partials [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= M) return;

    const uint sg_id   = lid / 32;
    const uint sg_lane = lid % 32;
    const uint n_blocks = K / Q8_0_BE;
    device const uchar* row_data = A_q8 + row * n_blocks * Q8_0_BB;

    float acc = 0.0f;
    for (uint blk = lid; blk < n_blocks; blk += WIDE_TG) {
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

    float simd_result = simd_sum(acc);
    if (sg_lane == 0) tg_partials[sg_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float total = tg_partials[0];
        for (uint i = 1; i < WIDE_SIMD_GROUPS; i++) total += tg_partials[i];
        y[row] = half(total);
    }
}
