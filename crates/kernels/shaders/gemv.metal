#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// HIGH-PERFORMANCE GEMV — Decode-path matrix-vector multiply
//
// Optimizations applied (vs previous Kahan-compensated version):
//   1. Vectorized half4/float4 loads — 4x fewer load instructions, perfect
//      coalescing across 32 SIMD lanes (256-byte reads per iteration)
//   2. simd_sum() hardware reduction — 1 instruction replaces 124-step
//      sequential Kahan summation in threadgroup memory
//   3. No Kahan compensation in inner loop — removes 3 extra ops per element
//      (for memory-bound GEMV, this was pure compute waste)
//   4. dot(float4, float4) — leverages Apple Silicon's fused dot-product ALU
//
// Correctness: Quantized weights (Q4K/Q5K/Q8_0 etc.) have ≤8 significant
// bits, well within f16/f32 accumulation accuracy. The Kahan overhead was
// unnecessary for the actual precision content of the data.
//
// Grid  : [M, 1, 1]   — one threadgroup per output row
// Group : [32, 1, 1]   — one SIMD lane width
// ─────────────────────────────────────────────────────────────────────────────

// ═══════════════════════════════════════════════════════════════════════════════
// F16 WEIGHT KERNELS
// ═══════════════════════════════════════════════════════════════════════════════

// y_f16 = A_f16 * x_f16
kernel void gemv_f16(
    device const half* A     [[buffer(0)]],
    device const half* x_vec [[buffer(1)]],
    device       half* y     [[buffer(2)]],
    constant     uint& M     [[buffer(3)]],
    constant     uint& K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(float4(((device const half4*)A_row)[i]),
                   float4(((device const half4*)x_vec)[i]));
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += float(A_row[i]) * float(x_vec[i]);
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result);
}

// y_f16 = A_f16 * x_f16 + residual_f16
kernel void gemv_add_f16(
    device const half* A        [[buffer(0)]],
    device const half* x_vec    [[buffer(1)]],
    device       half* y        [[buffer(2)]],
    device const half* residual [[buffer(3)]],
    constant     uint& M        [[buffer(4)]],
    constant     uint& K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(float4(((device const half4*)A_row)[i]),
                   float4(((device const half4*)x_vec)[i]));
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += float(A_row[i]) * float(x_vec[i]);
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result + float(residual[row]));
}

// y_f16 = A_f16 * x_f32
kernel void gemv_f16w_f32in(
    device const half*  A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(float4(((device const half4*)A_row)[i]),
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += float(A_row[i]) * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result);
}

// y_f32 = A_f16 * x_f32   (critical for decode Q/K/V projections)
kernel void gemv_f16w_f32in_f32out(
    device const half*  A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(float4(((device const half4*)A_row)[i]),
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += float(A_row[i]) * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = result;
}

// y_f32 = A_f16 * x_f16 + residual_f32
kernel void gemv_add_f32res_f16(
    device const half*  A        [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(float4(((device const half4*)A_row)[i]),
                   float4(((device const half4*)x_vec)[i]));
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += float(A_row[i]) * float(x_vec[i]);
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}

// y_f32 = A_f16 * x_f32 + residual_f32   (critical for output proj + residual)
kernel void gemv_add_f32_f16w(
    device const half*  A        [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(float4(((device const half4*)A_row)[i]),
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += float(A_row[i]) * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}

// ═══════════════════════════════════════════════════════════════════════════════
// F32 FULL-PRECISION KERNELS (f32 weight + f32 input)
// ═══════════════════════════════════════════════════════════════════════════════

// y_f32 = A_f32 * x_f32
kernel void gemv_f32_f32out(
    device const float* A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(((device const float4*)A_row)[i],
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += A_row[i] * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = result;
}

// y_f16 = A_f32 * x_f32
kernel void gemv_f32(
    device const float* A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(((device const float4*)A_row)[i],
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += A_row[i] * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result);
}

// y_f32 = A_f32 * x_f32 + residual_f32
kernel void gemv_add_f32(
    device const float* A        [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(((device const float4*)A_row)[i],
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += A_row[i] * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}

// ═══════════════════════════════════════════════════════════════════════════════
// F32-WEIGHT, F16-INPUT KERNELS (dequantized weights stored as f32)
// ═══════════════════════════════════════════════════════════════════════════════

// y_f16 = A_f32 * x_f16
kernel void gemv_f32w(
    device const float* A     [[buffer(0)]],
    device const half*  x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    float acc = 0.0f;
    for (uint i = lid; i < K; i += 32) {
        acc += A_row[i] * float(x_vec[i]);
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result);
}

// y_f16 = A_f32 * x_f16 + residual_f16
kernel void gemv_add_f32w(
    device const float* A        [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       half*  y        [[buffer(2)]],
    device const half*  residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    float acc = 0.0f;
    for (uint i = lid; i < K; i += 32) {
        acc += A_row[i] * float(x_vec[i]);
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result + float(residual[row]));
}

// y_f16 = A_f32 * x_f32
kernel void gemv_f32w_f32in(
    device const float* A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(((device const float4*)A_row)[i],
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += A_row[i] * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = half(result);
}

// y_f32 = A_f32 * x_f16 + residual_f32
kernel void gemv_add_f32res_f32w(
    device const float* A        [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    float acc = 0.0f;
    for (uint i = lid; i < K; i += 32) {
        acc += A_row[i] * float(x_vec[i]);
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}

// y_f32 = A_f32 * x_f32 + residual_f32
kernel void gemv_add_f32_f32w(
    device const float* A        [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    float acc = 0.0f;
    uint K4 = K >> 2;
    for (uint i = lid; i < K4; i += 32) {
        acc += dot(((device const float4*)A_row)[i],
                   ((device const float4*)x_vec)[i]);
    }
    for (uint i = (K4 << 2) + lid; i < K; i += 32) {
        acc += A_row[i] * x_vec[i];
    }
    float result = simd_sum(acc);
    if (lid == 0) y[row] = result + residual[row];
}
