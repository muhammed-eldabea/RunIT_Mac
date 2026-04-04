#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// FUSED QK RoPE: Apply rotary embeddings to BOTH Q and K in ONE dispatch
//
// Replaces 2 separate RoPE dispatches per layer with 1:
//   OLD: rope_f32(Q) → rope_f32(K)
//   NEW: fused_rope_qk_f32(Q, K)
//
// Saves 24 dispatches for 24 layers = 24 fewer pipeline state switches.
//
// Grid:  [(head_dim/2), max(q_heads, kv_heads), 1]
// Group: [min(head_dim/2, 32), 1, 1]
// ─────────────────────────────────────────────────────────────────────────────

kernel void fused_rope_qk_f32(
    device float* Q          [[buffer(0)]],   // [q_heads, head_dim] f32
    device float* K          [[buffer(1)]],   // [kv_heads, head_dim] f32
    constant uint& head_dim  [[buffer(2)]],
    constant uint& q_heads   [[buffer(3)]],
    constant uint& kv_heads  [[buffer(4)]],
    constant uint& seq_pos   [[buffer(5)]],
    constant float& theta    [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    const uint d   = tid.x;   // dimension pair index (0..head_dim/2 - 1)
    const uint hid = tid.y;   // head index

    const uint half_dim = head_dim / 2;
    if (d >= half_dim) return;

    // Compute rotation angle: theta_d = pos * theta^(-2d/dim)
    float freq = 1.0f / pow(theta, float(2 * d) / float(head_dim));
    float angle = float(seq_pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    // Apply RoPE to Q (if this head exists in Q)
    if (hid < q_heads) {
        uint q_base = hid * head_dim;
        float q_r = Q[q_base + d];
        float q_i = Q[q_base + d + half_dim];
        Q[q_base + d]            = q_r * cos_a - q_i * sin_a;
        Q[q_base + d + half_dim] = q_r * sin_a + q_i * cos_a;
    }

    // Apply RoPE to K (if this head exists in K)
    if (hid < kv_heads) {
        uint k_base = hid * head_dim;
        float k_r = K[k_base + d];
        float k_i = K[k_base + d + half_dim];
        K[k_base + d]            = k_r * cos_a - k_i * sin_a;
        K[k_base + d + half_dim] = k_r * sin_a + k_i * cos_a;
    }
}
