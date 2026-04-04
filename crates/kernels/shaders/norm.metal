#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
//
// rms_norm(x) = x / sqrt( mean(x²) + eps ) * gamma
//
// No mean subtraction — that's the key difference from LayerNorm.
// Used in Qwen2, LLaMA-2+.
//
// One threadgroup per row of the input (one token embedding vector).
// Threads split the hidden dimension and reduce within the threadgroup.
//
// Grid  : [batch * seq_len, 1, 1]   — one threadgroup per row
// Group : [256, 1, 1]
// SRAM  : 256 * sizeof(float) = 1 KB
// ─────────────────────────────────────────────────────────────────────────────
kernel void rms_norm_f16(
    device const half*  x      [[buffer(0)]],  // input  [rows, hidden]
    device       half*  y      [[buffer(1)]],  // output [rows, hidden]
    device const half*  gamma  [[buffer(2)]],  // scale  [hidden]
    constant     float& eps    [[buffer(3)]],
    constant     uint&  hidden [[buffer(4)]],
    threadgroup  float* sram   [[threadgroup(0)]],  // [threads_per_tg] scratch
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    // Pointer to this row
    device const half* row = x + gid * hidden;
    device       half* out = y + gid * hidden;

    // ── Phase 1: each thread accumulates sum of squares for its slice ──
    float local_sum = 0.0f;
    for (uint i = lid; i < hidden; i += lsize) {
        float v = float(row[i]);
        local_sum += v * v;
    }
    sram[lid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: parallel reduction in sram ───────────────────────────
    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sram[lid] += sram[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 3: compute rms scale and write output ───────────────────
    float rms = rsqrt(sram[0] / float(hidden) + eps);
    for (uint i = lid; i < hidden; i += lsize) {
        out[i] = half(float(row[i]) * rms * float(gamma[i]));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm with f32 input → f16 output
// Used when the residual stream is stored in f32 for precision.
// ─────────────────────────────────────────────────────────────────────────────
kernel void rms_norm_f32in_f16out(
    device const float* x      [[buffer(0)]],
    device       half*  y      [[buffer(1)]],
    device const half*  gamma  [[buffer(2)]],
    constant     float& eps    [[buffer(3)]],
    constant     uint&  hidden [[buffer(4)]],
    threadgroup  float* sram   [[threadgroup(0)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    device const float* row = x + gid * hidden;
    device       half*  out = y + gid * hidden;

    float local_sum = 0.0f;
    for (uint i = lid; i < hidden; i += lsize) {
        float v = row[i];
        local_sum += v * v;
    }
    sram[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) sram[lid] += sram[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(sram[0] / float(hidden) + eps);
    for (uint i = lid; i < hidden; i += lsize) {
        out[i] = half(row[i] * rms * float(gamma[i]));
    }
}

// RMSNorm f32 input → f32 output with f16 gamma
kernel void rms_norm_f32_f32(
    device const float* x      [[buffer(0)]],
    device       float* y      [[buffer(1)]],
    device const half*  gamma  [[buffer(2)]],
    constant     float& eps    [[buffer(3)]],
    constant     uint&  hidden [[buffer(4)]],
    threadgroup  float* sram   [[threadgroup(0)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    device const float* row = x + gid * hidden;
    device       float* out = y + gid * hidden;
    float local_sum = 0.0f;
    for (uint i = lid; i < hidden; i += lsize) {
        float v = row[i];
        local_sum += v * v;
    }
    sram[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) sram[lid] += sram[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(sram[0] / float(hidden) + eps);
    for (uint i = lid; i < hidden; i += lsize) {
        out[i] = row[i] * rms * float(gamma[i]);
    }
}

// RMSNorm f32 input → f32 output with f32 gamma (full precision norm weights)
kernel void rms_norm_f32_f32_f32g(
    device const float* x      [[buffer(0)]],
    device       float* y      [[buffer(1)]],
    device const float* gamma  [[buffer(2)]],
    constant     float& eps    [[buffer(3)]],
    constant     uint&  hidden [[buffer(4)]],
    threadgroup  float* sram   [[threadgroup(0)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    device const float* row = x + gid * hidden;
    device       float* out = y + gid * hidden;
    float local_sum = 0.0f;
    for (uint i = lid; i < hidden; i += lsize) {
        float v = row[i];
        local_sum += v * v;
    }
    sram[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) sram[lid] += sram[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(sram[0] / float(hidden) + eps);
    for (uint i = lid; i < hidden; i += lsize) {
        out[i] = row[i] * rms * gamma[i];
    }
}
