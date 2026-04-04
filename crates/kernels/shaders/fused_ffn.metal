#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// FUSED FFN: gate_proj + up_proj + SiLU×mul in ONE dispatch
//
// Replaces 3 separate dispatches per layer with 1:
//   OLD: gemv(gate, x) → gemv(up, x) → silu_mul(gate_out, up_out)
//   NEW: fused_ffn(gate_W, up_W, x) → act_out
//
// Benefits:
//   - 2 fewer dispatches per layer (48 total for 24 layers)
//   - Eliminates 2 intermediate buffers (gate_out, up_out never written to VRAM)
//   - The gate and up dot products stay in REGISTERS → silu applied immediately
//   - Input vector x_norm2 read ONCE (shared for both gate and up projections)
//
// Grid:  [ceil(ff_dim/4), 1, 1]   — multi-row: 4 rows per TG
// Group: [128, 1, 1]              — 4 SIMD groups
// ─────────────────────────────────────────────────────────────────────────────

#define FFN_MR_ROWS 4

// SiLU activation: x / (1 + exp(-x))
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fused FFN with Q8_0 weights (most bandwidth-efficient)
// act_out[row] = silu(dot(W_gate[row], x)) × dot(W_up[row], x)
// ═══════════════════════════════════════════════════════════════════════════════

#define Q8_0_BB 34
#define Q8_0_BE 32

kernel void fused_ffn_q8_0_f32(
    device const uchar* W_gate  [[buffer(0)]],  // [ff_dim, K] Q8_0 packed
    device const uchar* W_up    [[buffer(1)]],  // [ff_dim, K] Q8_0 packed
    device const float* x_vec   [[buffer(2)]],  // [K] f32 input
    device       float* act_out [[buffer(3)]],  // [ff_dim] f32 output
    constant     uint&  ff_dim  [[buffer(4)]],
    constant     uint&  K       [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint row  = tg_id * FFN_MR_ROWS + sg;
    if (row >= ff_dim) return;

    const uint n_blocks = K / Q8_0_BE;
    const uint row_bytes = n_blocks * Q8_0_BB;
    device const uchar* gate_row = W_gate + row * row_bytes;
    device const uchar* up_row   = W_up   + row * row_bytes;

    // Distribute blocks across 32 lanes
    uint bpt = (n_blocks + 31) / 32;
    uint blk_start = lane * bpt;
    uint blk_end   = min(blk_start + bpt, n_blocks);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (uint blk = blk_start; blk < blk_end; blk++) {
        device const uchar* g_blk = gate_row + blk * Q8_0_BB;
        device const uchar* u_blk = up_row   + blk * Q8_0_BB;

        float g_d = float(*reinterpret_cast<device const half*>(g_blk));
        float u_d = float(*reinterpret_cast<device const half*>(u_blk));
        device const char* g_qs = reinterpret_cast<device const char*>(g_blk + 2);
        device const char* u_qs = reinterpret_cast<device const char*>(u_blk + 2);

        uint x_base = blk * Q8_0_BE;

        for (uint i = 0; i < 32; i += 4) {
            float4 x4 = *(device const float4*)(x_vec + x_base + i);

            gate_acc += g_d * (float(g_qs[i])   * x4.x + float(g_qs[i+1]) * x4.y +
                               float(g_qs[i+2]) * x4.z + float(g_qs[i+3]) * x4.w);
            up_acc   += u_d * (float(u_qs[i])   * x4.x + float(u_qs[i+1]) * x4.y +
                               float(u_qs[i+2]) * x4.z + float(u_qs[i+3]) * x4.w);
        }
    }

    // SIMD reduction for both accumulators
    float gate_val = simd_sum(gate_acc);
    float up_val   = simd_sum(up_acc);

    // Apply SiLU(gate) × up — entirely in registers, no VRAM write for intermediates!
    if (lane == 0) {
        act_out[row] = silu(gate_val) * up_val;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fused FFN with F16 weights
// ═══════════════════════════════════════════════════════════════════════════════

kernel void fused_ffn_f16w_f32(
    device const half*  W_gate  [[buffer(0)]],  // [ff_dim, K] f16
    device const half*  W_up    [[buffer(1)]],  // [ff_dim, K] f16
    device const float* x_vec   [[buffer(2)]],  // [K] f32 input
    device       float* act_out [[buffer(3)]],  // [ff_dim] f32 output
    constant     uint&  ff_dim  [[buffer(4)]],
    constant     uint&  K       [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    const uint sg   = lid / 32;
    const uint lane = lid % 32;
    const uint row  = tg_id * FFN_MR_ROWS + sg;
    if (row >= ff_dim) return;

    device const half* g_row = W_gate + row * K;
    device const half* u_row = W_up   + row * K;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    uint K4 = K >> 2;
    for (uint i = lane; i < K4; i += 32) {
        float4 x4 = ((device const float4*)x_vec)[i];
        float4 gw = float4(((device const half4*)g_row)[i]);
        float4 uw = float4(((device const half4*)u_row)[i]);

        gate_acc += dot(gw, x4);
        up_acc   += dot(uw, x4);
    }

    float gate_val = simd_sum(gate_acc);
    float up_val   = simd_sum(up_acc);

    if (lane == 0) {
        act_out[row] = silu(gate_val) * up_val;
    }
}
