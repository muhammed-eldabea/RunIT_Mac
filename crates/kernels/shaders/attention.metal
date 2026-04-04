#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention-2 — tiled, GQA-aware, variable head_dim
//
// head_dim is passed as a runtime constant (buffer 9) so the same kernel
// handles both 64-dim (Qwen2.5-0.5B) and 128-dim (Qwen2.5-7B) models.
//
// MSL requires threadgroup pointer offsets to be compile-time constants, so
// tiles are always allocated at HEAD_DIM_MAX stride.  Only the first head_dim
// elements of each row are used; the rest are harmless zeros.
//
// SRAM layout (fixed, always 26 KB):
//   Q_tile [BLOCK_Q, HEAD_DIM_MAX]  f16   8 KB
//   K_tile [BLOCK_K, HEAD_DIM_MAX]  f16   8 KB
//   V_tile [BLOCK_K, HEAD_DIM_MAX]  f16   8 KB
//   S      [BLOCK_Q, BLOCK_K]       f16   2 KB
//
// Tensor layouts (all row-major, contiguous):
//   Q : [batch, num_heads,    q_len,  head_dim]
//   K : [batch, num_kv_heads, kv_len, head_dim]
//   V : [batch, num_kv_heads, kv_len, head_dim]
//   O : [batch, num_heads,    q_len,  head_dim]
//
// Launch:
//   Grid      : [ceil(q_len/BLOCK_Q), num_heads, batch]
//   Threadgroup: [BLOCK_Q, 1, 1]
//   SRAM      : (BLOCK_Q + 2*BLOCK_K)*HEAD_DIM_MAX*2 + BLOCK_Q*BLOCK_K*2
// ─────────────────────────────────────────────────────────────────────────────

#define BLOCK_Q      32
#define BLOCK_K      32
#define HEAD_DIM_MAX 128   // tile-stride constant; only [0..head_dim) are used

kernel void flash_attention_f16(
    device const half*  Q            [[buffer(0)]],
    device const half*  K            [[buffer(1)]],
    device const half*  V            [[buffer(2)]],
    device       half*  O            [[buffer(3)]],
    constant     uint&  q_len        [[buffer(4)]],
    constant     uint&  kv_len       [[buffer(5)]],
    constant     uint&  num_heads    [[buffer(6)]],
    constant     uint&  num_kv_heads [[buffer(7)]],
    constant     float& scale        [[buffer(8)]],
    constant     uint&  head_dim         [[buffer(9)]],  // actual head dim (64 or 128)
    constant     uint&  kv_head_stride   [[buffer(10)]], // elements between kv heads (= max_seq_len for flat KV cache, = kv_len for materialised)
    threadgroup  half*  sram             [[threadgroup(0)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint3 lid_vec [[thread_position_in_threadgroup]])  // must match tgid type
{
    const uint lid    = lid_vec.x;  // 1-D thread index within threadgroup
    const uint q_block = tgid.x;
    const uint q_head  = tgid.y;
    const uint batch   = tgid.z;

    const uint group_size = num_heads / num_kv_heads;
    const uint kv_head    = q_head / group_size;
    const uint q_row      = q_block * BLOCK_Q + lid;

    // SRAM partitions — offsets use HEAD_DIM_MAX (compile-time constant).
    // MSL requires constant threadgroup pointer offsets.
    threadgroup half* Q_tile = sram;
    threadgroup half* K_tile = sram +  BLOCK_Q              * HEAD_DIM_MAX;
    threadgroup half* V_tile = sram + (BLOCK_Q + BLOCK_K)   * HEAD_DIM_MAX;
    threadgroup half* S      = sram + (BLOCK_Q + 2*BLOCK_K) * HEAD_DIM_MAX;

    // Load Q row for this thread (stride = HEAD_DIM_MAX, use only [0..head_dim))
    // Q layout from GEMM: [batch, q_len, num_heads, head_dim]  (positions outer, heads inner)
    bool q_valid = q_row < q_len;
    uint q_base  = ((batch * q_len + q_row) * num_heads + q_head) * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        Q_tile[lid * HEAD_DIM_MAX + d] = q_valid ? Q[q_base + d] : half(0.0f);
    }

    // Online softmax state + output accumulator in registers
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_reg[HEAD_DIM_MAX];   // sized for max; only [0..head_dim) used
    for (uint d = 0; d < head_dim; d++) O_reg[d] = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_kv_blocks = (kv_len + BLOCK_K - 1) / BLOCK_K;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint kv_start = kv_block * BLOCK_K;

        // Load K tile
        uint k_row   = kv_start + lid;
        bool k_valid = k_row < kv_len;
        uint k_base  = ((batch * num_kv_heads + kv_head) * kv_head_stride + k_row) * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            K_tile[lid * HEAD_DIM_MAX + d] = k_valid ? K[k_base + d] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q_row · K_tile^T * scale (with causal mask)
        for (uint kv = 0; kv < BLOCK_K; kv++) {
            float score = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                score += float(Q_tile[lid * HEAD_DIM_MAX + d])
                       * float(K_tile[kv  * HEAD_DIM_MAX + d]);
            }
            bool kv_valid = (kv_start + kv) < kv_len;
            // Causal mask: each query position can only attend to positions <= its own
            bool causal_ok = (kv_start + kv) <= q_row;
            S[lid * BLOCK_K + kv] = half((kv_valid && causal_ok) ? score * scale : -INFINITY);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: update m and l
        float m_new = m_i;
        for (uint kv = 0; kv < BLOCK_K; kv++) {
            m_new = max(m_new, float(S[lid * BLOCK_K + kv]));
        }
        float l_new = exp(m_i - m_new) * l_i;
        for (uint kv = 0; kv < BLOCK_K; kv++) {
            l_new += exp(float(S[lid * BLOCK_K + kv]) - m_new);
        }

        // Load V tile
        uint v_row   = kv_start + lid;
        bool v_valid = v_row < kv_len;
        uint v_base  = ((batch * num_kv_heads + kv_head) * kv_head_stride + v_row) * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            V_tile[lid * HEAD_DIM_MAX + d] = v_valid ? V[v_base + d] : half(0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted V
        float rescale = exp(m_i - m_new);
        for (uint d = 0; d < head_dim; d++) O_reg[d] *= rescale;

        for (uint kv = 0; kv < BLOCK_K; kv++) {
            float w = exp(float(S[lid * BLOCK_K + kv]) - m_new);
            for (uint d = 0; d < head_dim; d++) {
                O_reg[d] += w * float(V_tile[kv * HEAD_DIM_MAX + d]);
            }
        }

        m_i = m_new;
        l_i = l_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!q_valid) return;

    // O layout matches Q: [batch, q_len, num_heads, head_dim]
    uint o_base = ((batch * q_len + q_row) * num_heads + q_head) * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        O[o_base + d] = half(O_reg[d] / l_i);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode attention — specialised for q_len = 1.
//
// The FlashAttention kernel wastes 31/32 threads when q_len=1 (only thread 0
// has a valid query row). This kernel uses all 32 threads to parallelise
// the dot product across head_dim via SIMD reduction, processing KV
// positions sequentially with online softmax.
//
// Launch:
//   Grid      : [num_heads, batch, 1]    — one threadgroup per head
//   Threadgroup: [32, 1, 1]              — one SIMD group
// ─────────────────────────────────────────────────────────────────────────────
kernel void decode_attention_f16(
    device const half*  Q            [[buffer(0)]],   // [batch, num_heads, 1, head_dim]
    device const half*  K            [[buffer(1)]],   // [batch, num_kv_heads, kv_head_stride, head_dim]
    device const half*  V            [[buffer(2)]],   // same layout as K
    device       half*  O            [[buffer(3)]],   // [batch, num_heads, 1, head_dim]
    constant     uint&  kv_len       [[buffer(4)]],   // number of valid KV positions
    constant     uint&  num_heads    [[buffer(5)]],
    constant     uint&  num_kv_heads [[buffer(6)]],
    constant     float& scale        [[buffer(7)]],
    constant     uint&  head_dim         [[buffer(8)]],
    constant     uint&  kv_head_stride   [[buffer(9)]],
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint3 lid3   [[thread_position_in_threadgroup]],
    uint3 lsize3 [[threads_per_threadgroup]])
{
    const uint lid   = lid3.x;
    const uint lsize = lsize3.x;
    const uint q_head = tgid.x;
    const uint batch  = tgid.y;

    const uint group_size = num_heads / num_kv_heads;
    const uint kv_head    = q_head / group_size;

    // Query base pointer for this head
    const uint q_base = (batch * num_heads + q_head) * head_dim;

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Output accumulator — each thread holds ceil(head_dim/32) elements
    float O_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // max HEAD_DIM_MAX/32 = 4

    // Process KV positions one at a time
    for (uint kv_pos = 0; kv_pos < kv_len; kv_pos++) {
        const uint kv_base = ((batch * num_kv_heads + kv_head) * kv_head_stride + kv_pos) * head_dim;

        // Compute Q · K dot product — each thread handles a slice of head_dim
        float dot = 0.0f;
        for (uint d = lid; d < head_dim; d += lsize) {
            dot += float(Q[q_base + d]) * float(K[kv_base + d]);
        }
        float score = simd_sum(dot) * scale;

        // Online softmax update
        float m_new   = max(m_i, score);
        float rescale = exp(m_i - m_new);
        float exp_new = exp(score - m_new);
        float l_new   = rescale * l_i + exp_new;

        // Rescale existing accumulator and add weighted V
        uint e = 0;
        for (uint d = lid; d < head_dim; d += lsize) {
            O_reg[e] = O_reg[e] * rescale + exp_new * float(V[kv_base + d]);
            e++;
        }

        m_i = m_new;
        l_i = l_new;
    }

    // Write output — each thread writes its elements
    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    const uint o_base = (batch * num_heads + q_head) * head_dim;
    uint e = 0;
    for (uint d = lid; d < head_dim; d += lsize) {
        O[o_base + d] = half(O_reg[e] * inv_l);
        e++;
    }
}
