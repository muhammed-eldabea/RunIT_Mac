#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Q4_K_M dequantization → F16
//
// GGML Q4_K block layout (144 bytes per 256 elements):
//
//   Offset  Size  Field
//   ──────  ────  ─────────────────────────────────────────────────────
//      0      2   d    — super-block scale for sub-block scales (f16)
//      2      2   dmin — super-block scale for sub-block mins   (f16)
//      4     12   scales[12] — 8 × (6-bit scale, 6-bit min) packed
//     16    128   qs[128]    — 256 × 4-bit quantised values
//   ──────
//    144 bytes total
//
// Sub-block scale/min extraction (get_scale_min_k4):
//   j < 4  → sc = scales[j]   & 0x3F, m = scales[j+4] & 0x3F
//   j >= 4 → sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//             m = (scales[j+4] >> 4)    | ((scales[j]   >> 6) << 4)
//
// Dequant formula (8 sub-blocks of 32 elements each):
//   actual_scale = d    * sc
//   actual_min   = dmin * m
//   value = actual_scale * q4_nibble - actual_min
//
// Launch:
//   Grid       : [ceil(n_elements / 256), 1, 1]
//   Threadgroup: [256, 1, 1]
//   Each threadgroup processes exactly one Q4_K block (256 elements).
//   Thread lid processes element lid within the block.
// ─────────────────────────────────────────────────────────────────────────────

#define QK_K       256
#define BLOCK_SIZE 144   // sizeof(block_q4_K)
#define SCALES_OFF   4   // byte offset of scales[] within block
#define QS_OFF      16   // byte offset of qs[]    within block

// Extract 6-bit scale and min for sub-block j (0..7)
// Use Metal native types (uchar) instead of C99 uint8_t for portability.
inline void get_scale_min_k4(uint j, device const uchar* scales,
                               thread uchar& sc, thread uchar& m) {
    if (j < 4) {
        sc = scales[j]   & 0x3Fu;
        m  = scales[j+4] & 0x3Fu;
    } else {
        sc = (scales[j+4] & 0x0Fu) | ((scales[j-4] >> 6u) << 4u);
        m  = (scales[j+4] >> 4u)   | ((scales[j]   >> 6u) << 4u);
    }
}

kernel void dequant_q4k_f16(
    device const uchar* input   [[buffer(0)]],  // Q4_K packed bytes
    device       half*  output  [[buffer(1)]],  // F16 output
    constant     uint&  n_blocks[[buffer(2)]],  // number of Q4_K blocks
    uint tgid [[threadgroup_position_in_grid]],   // block_idx (1-D dispatch)
    uint lid  [[thread_position_in_threadgroup]])  // element within block [0..255]
{
    const uint block_idx = tgid;
    if (block_idx >= n_blocks) return;

    // Pointer to the start of this block
    device const uchar* block = input + block_idx * BLOCK_SIZE;

    // Super-block scales (stored as f16 at bytes 0 and 2)
    const float d    = float(*reinterpret_cast<device const half*>(block + 0));
    const float dmin = float(*reinterpret_cast<device const half*>(block + 2));

    device const uchar* scales = block + SCALES_OFF;  // 12-byte scale/min table
    device const uchar* qs     = block + QS_OFF;      // 128-byte 4-bit quants

    // Determine sub-block for this thread
    const uint sub   = lid / 32u;   // sub-block index 0..7
    const uint local = lid % 32u;   // position within sub-block 0..31

    uchar sc, m;
    get_scale_min_k4(sub, scales, sc, m);

    const float actual_scale = d    * float(sc);
    const float actual_min   = dmin * float(m);

    // Unpack 4-bit value for this element
    // qs stores pairs: qs[k] = low_nibble | (high_nibble << 4)
    // element (sub*32 + local):
    //   byte index  = (sub*32 + local) / 2
    //   nibble      = low if local is even, high if odd
    const uint byte_idx = (sub * 32u + local) / 2u;
    const uchar byte_val = qs[byte_idx];
    const uint q4 = (local & 1u) ? (byte_val >> 4u) : (byte_val & 0x0Fu);

    const float val = actual_scale * float(q4) - actual_min;

    output[block_idx * QK_K + lid] = half(val);
}
