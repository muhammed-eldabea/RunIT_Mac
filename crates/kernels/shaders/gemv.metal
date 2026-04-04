#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// GEMV: y = A * x   (matrix-vector multiply for decode path)
//
// Each threadgroup computes one output element y[row].
// Threads split the K reduction dimension using CONTIGUOUS blocks
// (not strided), matching llama.cpp's sequential accumulation order.
//
// Reduction uses Kahan-compensated sequential sum in threadgroup memory
// instead of simd_sum() to eliminate hardware-dependent reduction order.
//
// Grid  : [M, 1, 1]   — one threadgroup per output element
// Group : [32, 1, 1]   — one SIMD lane width
// ──────────────────────────────���──────────────────────────────────────────────

// Sequential Kahan reduction of 32 partial sums in threadgroup memory.
// Only thread 0 executes this; result is returned to thread 0.
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

// ─────────────────────────────────────────────────────────────────────────────
// F16 GEMV
// ─────��─────────────────────────────────────────────────────────────────��─────
kernel void gemv_f16(
    device const half* A     [[buffer(0)]],
    device const half* x_vec [[buffer(1)]],
    device       half* y     [[buffer(2)]],
    constant     uint& M     [[buffer(3)]],
    constant     uint& K     [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;

    // Contiguous block per thread
    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = float(A_row[i]) * float(x_vec[i]) - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        y[row] = half(kahan_reduce_tg(tg, lsize));
    }
}

// Fused GEMV + residual: y[i] = (A * x)[i] + residual[i]
kernel void gemv_add_f16(
    device const half* A        [[buffer(0)]],
    device const half* x_vec    [[buffer(1)]],
    device       half* y        [[buffer(2)]],
    device const half* residual [[buffer(3)]],
    constant     uint& M        [[buffer(4)]],
    constant     uint& K        [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const half* A_row = A + row * K;

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = float(A_row[i]) * float(x_vec[i]) - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        y[row] = half(kahan_reduce_tg(tg, lsize) + float(residual[row]));
    }
}

// ───────────────────────────────���──────────────────────────────��──────────────
// F32-F32 GEMV (dequanted weights, max precision)
// ──────────────────────────��─────────────────────��────────────────────────────

// y_f32 = A_f32 * x_f32
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

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = A_row[i] * x_vec[i] - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize);
}

// y_f16 = A_f32 * x_f32
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

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = A_row[i] * x_vec[i] - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = half(kahan_reduce_tg(tg, lsize));
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
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = A_row[i] * x_vec[i] - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize) + residual[row];
}

// ��──────────────────────────���─────────────────────────────────────────────────
// F16-weight, F32-input GEMV variants
// ──────���──────────────────────��───────────────────────────────────────────────

// y_f16 = A_f16 * x_f32
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

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = float(A_row[i]) * x_vec[i] - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = half(kahan_reduce_tg(tg, lsize));
}

// y_f32 = (A_f16 * x_f16) + residual_f32
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

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = float(A_row[i]) * float(x_vec[i]) - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize) + residual[row];
}

// y_f32 = (A_f16 * x_f32) + residual_f32
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

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = float(A_row[i]) * x_vec[i] - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize) + residual[row];
}

// y_f32 = A_f16 * x_f32  (full f32 output)
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

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = float(A_row[i]) * x_vec[i] - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize);
}

// ═══════���═════════════════════════════════════��═══════════════════════════════
// F32-WEIGHT GEMV VARIANTS
// ═══════════════════���═════════════════════════════════════════════════════════

// y_f16 = A_f32 * x_f16
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

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = A_row[i] * float(x_vec[i]) - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = half(kahan_reduce_tg(tg, lsize));
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
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = A_row[i] * float(x_vec[i]) - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = half(kahan_reduce_tg(tg, lsize) + float(residual[row]));
}

// y_f16 = A_f32 * x_f32
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

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = A_row[i] * x_vec[i] - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = half(kahan_reduce_tg(tg, lsize));
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
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = A_row[i] * float(x_vec[i]) - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize) + residual[row];
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
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize[[threads_per_threadgroup]])
{
    if (row >= M) return;
    device const float* A_row = A + row * K;

    uint block = K / lsize;
    uint start = lid * block;
    uint end   = (lid == lsize - 1) ? K : start + block;

    float acc = 0.0f;
    float comp = 0.0f;
    for (uint i = start; i < end; i++) {
        float prod = A_row[i] * x_vec[i] - comp;
        float t = acc + prod;
        comp = (t - acc) - prod;
        acc = t;
    }

    threadgroup float tg[32];
    tg[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) y[row] = kahan_reduce_tg(tg, lsize) + residual[row];
}
