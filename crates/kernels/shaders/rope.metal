#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embeddings (RoPE) — in-place on Q or K
//
// NON-INTERLEAVED pairing (Qwen2/LLaMA standard):
// For each pair index i in [0, head_dim/2):
//   Elements paired are (i, i + head_dim/2)
//   freq    = pos / theta^(2i / head_dim)
//   x[i]          =  x[i]          * cos(freq) - x[i+half] * sin(freq)
//   x[i+half]     =  x[i+half]     * cos(freq) + x[i]      * sin(freq)
//
// Launched with:
//   Grid  : [head_dim/2, n_heads, batch*seq_len]
//   Group : [1, 1, 1]   (one thread per pair)
//
// x layout: [batch*seq_len, n_heads, head_dim]  (contiguous)
// ─────────────────────────────────────────────────────────────────────────────
kernel void rope_inplace_f16(
    device       half*  x         [[buffer(0)]],  // in-place Q or K
    constant     uint&  head_dim  [[buffer(1)]],
    constant     uint&  n_heads   [[buffer(2)]],
    constant     uint&  seq_pos   [[buffer(3)]],  // absolute token position
    constant     float& theta     [[buffer(4)]],  // 1e6 for Qwen2, 1e4 for LLaMA-2
    uint3 gid [[thread_position_in_grid]])        // [pair, head, token]
{
    uint pair    = gid.x;   // index in [0, head_dim/2)
    uint head    = gid.y;   // head index
    uint tok_idx = gid.z;   // token index in the batch

    if (pair >= head_dim / 2) return;

    uint half_dim = head_dim / 2;
    uint head_base = (tok_idx * n_heads + head) * head_dim;
    uint idx0 = head_base + pair;              // first element
    uint idx1 = head_base + pair + half_dim;   // paired element in second half

    float x0 = float(x[idx0]);
    float x1 = float(x[idx1]);

    // Frequency for this pair
    float freq = float(seq_pos) / pow(theta, float(2 * pair) / float(head_dim));
    float c = cos(freq);
    float s = sin(freq);

    x[idx0] = half(x0 * c - x1 * s);
    x[idx1] = half(x1 * c + x0 * s);
}

// f32 variant of RoPE — reads/writes float instead of half.
// Used when Q/K projections are kept in f32 for full precision.
kernel void rope_inplace_f32(
    device       float* x         [[buffer(0)]],
    constant     uint&  head_dim  [[buffer(1)]],
    constant     uint&  n_heads   [[buffer(2)]],
    constant     uint&  seq_pos   [[buffer(3)]],
    constant     float& theta     [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair    = gid.x;
    uint head    = gid.y;
    uint tok_idx = gid.z;

    if (pair >= head_dim / 2) return;

    uint half_dim = head_dim / 2;
    uint head_base = (tok_idx * n_heads + head) * head_dim;
    uint idx0 = head_base + pair;
    uint idx1 = head_base + pair + half_dim;

    float x0 = x[idx0];
    float x1 = x[idx1];

    float freq = float(seq_pos) / pow(theta, float(2 * pair) / float(head_dim));
    float c = cos(freq);
    float s = sin(freq);

    x[idx0] = x0 * c - x1 * s;
    x[idx1] = x1 * c + x0 * s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Batched RoPE — for prefill: applies RoPE to all positions in one call.
//
// Input x layout: [seq_len, n_heads, head_dim]
// Each position seq[s] gets position (start_pos + s).
//
// Grid  : [head_dim/2, n_heads, seq_len]
// Group : [1, 1, 1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void rope_batch_inplace_f16(
    device       half*  x          [[buffer(0)]],  // [seq_len, n_heads, head_dim]
    constant     uint&  head_dim   [[buffer(1)]],
    constant     uint&  n_heads    [[buffer(2)]],
    constant     uint&  start_pos  [[buffer(3)]],  // absolute position of seq[0]
    constant     float& theta      [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])          // [pair, head, seq_pos]
{
    uint pair    = gid.x;
    uint head    = gid.y;
    uint seq_idx = gid.z;

    if (pair >= head_dim / 2) return;

    uint abs_pos = start_pos + seq_idx;
    uint half_dim = head_dim / 2;
    uint head_base = (seq_idx * n_heads + head) * head_dim;
    uint idx0 = head_base + pair;
    uint idx1 = head_base + pair + half_dim;

    float x0 = float(x[idx0]);
    float x1 = float(x[idx1]);

    float freq = float(abs_pos) / pow(theta, float(2 * pair) / float(head_dim));
    float c = cos(freq);
    float s = sin(freq);

    x[idx0] = half(x0 * c - x1 * s);
    x[idx1] = half(x1 * c + x0 * s);
}
