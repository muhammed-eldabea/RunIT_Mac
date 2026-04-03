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
