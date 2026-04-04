#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// GEMV: y = A * x   (F16 matrix-vector multiply)
//
// Used in the decode path (batch_size = 1) where only one token is processed.
// Memory-bandwidth bound: ~2 bytes read per multiply-accumulate.
//
// Each threadgroup computes one output element y[row].
// Threads split the K reduction dimension and combine with simd_sum().
//
// Grid  : [M, 1, 1]   — one threadgroup per output element
// Group : [32, 1, 1]  — one SIMD lane width
//
// A layout: [M, K] row-major
// x layout: [K]
// y layout: [M]
// ─────────────────────────────────────────────────────────────────────────────
kernel void gemv_f16(
    device const half* A     [[buffer(0)]],  // weight matrix [M, K]
    device const half* x_vec [[buffer(1)]],  // input vector  [K]
    device       half* y     [[buffer(2)]],  // output vector [M]
    constant     uint& M     [[buffer(3)]],
    constant     uint& K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;

    device const half* A_row = A + row * K;

    // Each thread accumulates its slice of the dot product
    float acc = 0.0f;

    // Vectorised: load 4 halfs at a time where possible
    uint k4 = K / 4;
    uint k4_rem = K % 4;

    for (uint i = lid; i < k4; i += lsize) {
        half4 av = ((device const half4*)A_row)[i];
        half4 xv = ((device const half4*)x_vec)[i];
        acc += float(av.x) * float(xv.x)
             + float(av.y) * float(xv.y)
             + float(av.z) * float(xv.z)
             + float(av.w) * float(xv.w);
    }

    // Handle tail elements
    uint tail_start = k4 * 4;
    for (uint i = tail_start + lid; i < K; i += lsize) {
        acc += float(A_row[i]) * float(x_vec[i]);
    }

    // SIMD reduction across the 32-lane warp
    acc = simd_sum(acc);

    // Lane 0 writes the result
    if (lid == 0) {
        y[row] = half(acc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused GEMV + residual add: y[i] = (A * x)[i] + residual[i]
//
// Saves one kernel dispatch + one full buffer read/write per residual
// connection. Used for attention output projection and FFN down projection.
//
// Grid  : [M, 1, 1]   — one threadgroup per output element
// Group : [32, 1, 1]   — one SIMD lane width
// ─────────────────────────────────────────────────────────────────────────────
kernel void gemv_add_f16(
    device const half* A        [[buffer(0)]],  // weight matrix [M, K]
    device const half* x_vec    [[buffer(1)]],  // input vector  [K]
    device       half* y        [[buffer(2)]],  // output vector [M] — also accumulates residual
    device const half* residual [[buffer(3)]],  // residual [M] — added to GEMV result
    constant     uint& M        [[buffer(4)]],
    constant     uint& K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;

    device const half* A_row = A + row * K;
    float acc = 0.0f;

    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        half4 av = ((device const half4*)A_row)[i];
        half4 xv = ((device const half4*)x_vec)[i];
        acc += float(av.x) * float(xv.x)
             + float(av.y) * float(xv.y)
             + float(av.z) * float(xv.z)
             + float(av.w) * float(xv.w);
    }

    uint tail_start = k4 * 4;
    for (uint i = tail_start + lid; i < K; i += lsize) {
        acc += float(A_row[i]) * float(x_vec[i]);
    }

    acc = simd_sum(acc);

    if (lid == 0) {
        y[row] = half(acc + float(residual[row]));
    }
}

// Full f32 GEMV with f32 output: y_f32 = A_f32 * x_f32
kernel void gemv_f32_f32out(
    device const float* A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        float4 av = ((device const float4*)A_row)[i];
        float4 xv = ((device const float4*)x_vec)[i];
        acc += av.x*xv.x + av.y*xv.y + av.z*xv.z + av.w*xv.w;
    }
    for (uint i = k4*4 + lid; i < K; i += lsize)
        acc += A_row[i] * x_vec[i];
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc;
}

// Full f32 GEMV: y = A_f32 * x_f32 (maximum precision for dequanted weights)
kernel void gemv_f32(
    device const float* A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        float4 av = ((device const float4*)A_row)[i];
        float4 xv = ((device const float4*)x_vec)[i];
        acc += av.x*xv.x + av.y*xv.y + av.z*xv.z + av.w*xv.w;
    }
    for (uint i = k4*4 + lid; i < K; i += lsize)
        acc += A_row[i] * x_vec[i];
    acc = simd_sum(acc);
    if (lid == 0) y[row] = half(acc);
}

// Full f32 GEMV + f32 residual: y_f32 = A_f32 * x_f32 + residual_f32
kernel void gemv_add_f32(
    device const float* A        [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        float4 av = ((device const float4*)A_row)[i];
        float4 xv = ((device const float4*)x_vec)[i];
        acc += av.x*xv.x + av.y*xv.y + av.z*xv.z + av.w*xv.w;
    }
    for (uint i = k4*4 + lid; i < K; i += lsize)
        acc += A_row[i] * x_vec[i];
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc + residual[row];
}

// GEMV with f32 input: y = A_f16 * x_f32
kernel void gemv_f16w_f32in(
    device const half*  A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;
    float acc = 0.0f;
    for (uint i = lid; i < K; i += lsize)
        acc += float(A_row[i]) * x_vec[i];
    acc = simd_sum(acc);
    if (lid == 0) y[row] = half(acc);
}

// GEMV + f32 residual: y_f32[i] = (A_f16 * x_f16)[i] + residual_f32[i]
kernel void gemv_add_f32res_f16(
    device const half*  A        [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        half4 av = ((device const half4*)A_row)[i];
        half4 xv = ((device const half4*)x_vec)[i];
        acc += float(av.x)*float(xv.x) + float(av.y)*float(xv.y)
             + float(av.z)*float(xv.z) + float(av.w)*float(xv.w);
    }
    uint tail_start = k4 * 4;
    for (uint i = tail_start + lid; i < K; i += lsize)
        acc += float(A_row[i]) * float(x_vec[i]);
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc + residual[row];
}

// GEMV + f32 residual with f32 input: y_f32[i] = (A_f16 * x_f32)[i] + res_f32[i]
kernel void gemv_add_f32_f16w(
    device const half*  A        [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;
    float acc = 0.0f;
    for (uint i = lid; i < K; i += lsize)
        acc += float(A_row[i]) * x_vec[i];
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc + residual[row];
}

// GEMV with f16 weights, f32 input, f32 output: y_f32 = A_f16 * x_f32
// Fills the gap for Q/K/V projections that need full f32 output precision.
kernel void gemv_f16w_f32in_f32out(
    device const half*  A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       float* y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;
    float acc = 0.0f;
    for (uint i = lid; i < K; i += lsize)
        acc += float(A_row[i]) * x_vec[i];
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc;
}

// ═════════════════════════════════════════════════════════════════════════════
// F32-WEIGHT GEMV VARIANTS
//
// These kernels read weights as float (f32) instead of half (f16).
// Used when quantized weights (Q5_0, Q6K, Q8_0, etc.) are dequantized to
// f32 on the CPU to preserve precision. llama.cpp dequantizes to f32;
// truncating to f16 loses ~10 mantissa bits which compounds across layers.
// ═════════════════════════════════════════════════════════════════════════════

// GEMV with f32 weights: y_f16 = A_f32 * x_f16
kernel void gemv_f32w(
    device const float* A     [[buffer(0)]],
    device const half*  x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        float4 av = ((device const float4*)A_row)[i];
        half4  xv = ((device const half4*)x_vec)[i];
        acc += av.x * float(xv.x) + av.y * float(xv.y)
             + av.z * float(xv.z) + av.w * float(xv.w);
    }
    uint tail_start = k4 * 4;
    for (uint i = tail_start + lid; i < K; i += lsize)
        acc += A_row[i] * float(x_vec[i]);
    acc = simd_sum(acc);
    if (lid == 0) y[row] = half(acc);
}

// Fused GEMV + residual with f32 weights: y_f16 = A_f32 * x_f16 + residual_f16
kernel void gemv_add_f32w(
    device const float* A        [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       half*  y        [[buffer(2)]],
    device const half*  residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        float4 av = ((device const float4*)A_row)[i];
        half4  xv = ((device const half4*)x_vec)[i];
        acc += av.x * float(xv.x) + av.y * float(xv.y)
             + av.z * float(xv.z) + av.w * float(xv.w);
    }
    uint tail_start = k4 * 4;
    for (uint i = tail_start + lid; i < K; i += lsize)
        acc += A_row[i] * float(x_vec[i]);
    acc = simd_sum(acc);
    if (lid == 0) y[row] = half(acc + float(residual[row]));
}

// GEMV with f32 weights and f32 input: y_f16 = A_f32 * x_f32
kernel void gemv_f32w_f32in(
    device const float* A     [[buffer(0)]],
    device const float* x_vec [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        float4 av = ((device const float4*)A_row)[i];
        float4 xv = ((device const float4*)x_vec)[i];
        acc += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
    }
    uint tail_start = k4 * 4;
    for (uint i = tail_start + lid; i < K; i += lsize)
        acc += A_row[i] * x_vec[i];
    acc = simd_sum(acc);
    if (lid == 0) y[row] = half(acc);
}

// GEMV + f32 residual with f32 weights: y_f32 = A_f32 * x_f16 + residual_f32
kernel void gemv_add_f32res_f32w(
    device const float* A        [[buffer(0)]],
    device const half*  x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        float4 av = ((device const float4*)A_row)[i];
        half4  xv = ((device const half4*)x_vec)[i];
        acc += av.x * float(xv.x) + av.y * float(xv.y)
             + av.z * float(xv.z) + av.w * float(xv.w);
    }
    uint tail_start = k4 * 4;
    for (uint i = tail_start + lid; i < K; i += lsize)
        acc += A_row[i] * float(x_vec[i]);
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc + residual[row];
}

// GEMV + f32 residual with f32 weights and f32 input: y_f32 = A_f32 * x_f32 + res_f32
kernel void gemv_add_f32_f32w(
    device const float* A        [[buffer(0)]],
    device const float* x_vec    [[buffer(1)]],
    device       float* y        [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant     uint&  M        [[buffer(4)]],
    constant     uint&  K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;
    float acc = 0.0f;
    uint k4 = K / 4;
    for (uint i = lid; i < k4; i += lsize) {
        float4 av = ((device const float4*)A_row)[i];
        float4 xv = ((device const float4*)x_vec)[i];
        acc += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
    }
    uint tail_start = k4 * 4;
    for (uint i = tail_start + lid; i < K; i += lsize)
        acc += A_row[i] * x_vec[i];
    acc = simd_sum(acc);
    if (lid == 0) y[row] = acc + residual[row];
}
