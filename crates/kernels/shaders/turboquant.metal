// turboquant.metal — TurboQuant KV-cache compression kernels
// Phase 5: Randomized Hadamard Transform + Lloyd-Max quantization
//
// Rotation: RHT = diag(signs) × WHT × (1/sqrt(d))
// Keys:     normalize → RHT → Lloyd-Max(b-1 bits) → residual → QJL signs
// Values:   per-group symmetric quantization

#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Seeded hash → ±1 (for on-the-fly QJL matrix generation)
// Uses a simple xorshift hash so no matrix storage is needed.
inline float qjl_entry(uint row, uint col, uint seed) {
    uint x = (row * 2654435761u) ^ (col * 2246822519u) ^ seed;
    x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
    return (x & 1u) ? 1.0f : -1.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. tq_normalize_f16
//    Normalize a vector to unit sphere, store its L2 norm.
//    Grid: [1,1,1]  Threadgroup: [d,1,1]  (d = head_dim)
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_normalize_f16(
    device const half*  x      [[buffer(0)]],   // [d] input
    device       half*  x_hat  [[buffer(1)]],   // [d] normalized output
    device       half*  norm   [[buffer(2)]],   // [1] L2 norm output
    constant     uint&  d      [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]])
{
    float v = (lid < d) ? (float)x[lid] : 0.0f;
    tg[lid] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for sum of squares
    for (uint s = d >> 1; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float n = sqrt(tg[0]) + 1e-8f;
    if (lid == 0) norm[0] = (half)n;
    if (lid < d)  x_hat[lid] = (half)(v / n);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. tq_rht_f16
//    Randomized Hadamard Transform: y = WHT(diag(signs) × x) / sqrt(d)
//    Grid: [1,1,1]  Threadgroup: [d,1,1]  (d must be power of 2)
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_rht_f16(
    device const half*   x      [[buffer(0)]],   // [d] input (unit-norm)
    device       half*   y      [[buffer(1)]],   // [d] rotated output
    device const char* signs  [[buffer(2)]],   // [d] random ±1 signs
    constant     uint&   d      [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]])
{
    // Apply diagonal sign flip
    tg[lid] = (lid < d) ? (float)x[lid] * (float)signs[lid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Walsh-Hadamard Transform (in-place butterfly)
    for (uint s = 1; s < d; s <<= 1) {
        uint mask = s - 1;
        if ((lid & s) == 0) {
            float a = tg[lid];
            float b = tg[lid | s];
            tg[lid]      = a + b;
            tg[lid | s]  = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by 1/sqrt(d)
    if (lid < d) y[lid] = (half)(tg[lid] * rsqrt((float)d));
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. tq_rht_inverse_f16
//    Inverse RHT: x = diag(signs) × WHT(y) / sqrt(d)
//    WHT is its own inverse (up to the 1/d scale factor, so apply 1/sqrt(d) again)
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_rht_inverse_f16(
    device const half*   y      [[buffer(0)]],   // [d] rotated input
    device       half*   x      [[buffer(1)]],   // [d] original-space output
    device const char* signs  [[buffer(2)]],   // [d] same ±1 signs
    constant     uint&   d      [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]])
{
    tg[lid] = (lid < d) ? (float)y[lid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // WHT (self-inverse up to 1/d scale)
    for (uint s = 1; s < d; s <<= 1) {
        if ((lid & s) == 0) {
            float a = tg[lid];
            float b = tg[lid | s];
            tg[lid]      = a + b;
            tg[lid | s]  = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale and un-apply sign flip
    if (lid < d)
        x[lid] = (half)(tg[lid] * rsqrt((float)d) * (float)signs[lid]);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. tq_lloyd_quant_f16
//    Lloyd-Max quantize: binary search in decision boundaries → index.
//    Write indices (uint16) and reconstructed centroid values (f16).
//    Grid: [d,1,1]  Threadgroup: [1,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_lloyd_quant_f16(
    device const half*    y           [[buffer(0)]],  // [d] rotated input
    device       ushort* indices    [[buffer(1)]],  // [d] quantized indices
    device       half*    recon       [[buffer(2)]],  // [d] centroid values
    device const float*   boundaries  [[buffer(3)]],  // [n_centroids-1] decision pts
    device const float*   centroids   [[buffer(4)]],  // [n_centroids] centroid values
    constant     uint&    n_centroids [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    float v = (float)y[gid];
    uint n_bounds = n_centroids - 1;

    // Binary search
    uint lo = 0, hi = n_bounds;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (v > boundaries[mid]) lo = mid + 1;
        else                      hi = mid;
    }
    // lo is the centroid index
    uint idx = min(lo, n_centroids - 1);
    indices[gid] = (ushort)idx;
    recon[gid]   = (half)centroids[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. tq_pack_bits_u8
//    Pack uint16 indices (values < 2^bits) into a byte array.
//    bits must be 1, 2, 3, or 4. d must be divisible by (8/bits).
//    Grid: [d/(8/bits),1,1]  Threadgroup: [1,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_pack_bits_u8(
    device const ushort* indices [[buffer(0)]],   // [d]
    device       uchar*  packed  [[buffer(1)]],   // [d*bits/8]
    constant     uint&     bits    [[buffer(2)]],
    constant     uint&     d       [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint elems_per_byte = 8u / bits;
    uint byte_idx = gid;
    uchar out = 0;
    for (uint i = 0; i < elems_per_byte; i++) {
        uint elem_idx = byte_idx * elems_per_byte + i;
        if (elem_idx < d) {
            uchar v = (uchar)(indices[elem_idx] & ((1u << bits) - 1u));
            out |= (v << (i * bits));
        }
    }
    packed[gid] = out;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. tq_unpack_bits_u8
//    Unpack byte array back to uint16 indices.
//    Grid: [d,1,1]  Threadgroup: [1,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_unpack_bits_u8(
    device const uchar*  packed  [[buffer(0)]],   // [d*bits/8]
    device       ushort* indices [[buffer(1)]],   // [d]
    constant     uint&     bits    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint elems_per_byte = 8u / bits;
    uint byte_idx  = gid / elems_per_byte;
    uint bit_shift = (gid % elems_per_byte) * bits;
    uint mask = (1u << bits) - 1u;
    indices[gid] = (ushort)((packed[byte_idx] >> bit_shift) & mask);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. tq_centroid_lookup_f16
//    Map indices → centroid values (f16).
//    Grid: [d,1,1]  Threadgroup: [1,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_centroid_lookup_f16(
    device const ushort* indices   [[buffer(0)]],
    device       half*     recon     [[buffer(1)]],
    device const float*    centroids [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    recon[gid] = (half)centroids[indices[gid]];
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. tq_residual_norm_f16
//    Compute residual r = a - b and its L2 norm.
//    Grid: [1,1,1]  Threadgroup: [d,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_residual_norm_f16(
    device const half* a      [[buffer(0)]],   // [d] original unit-norm vector
    device const half* b      [[buffer(1)]],   // [d] reconstruction
    device       half* r      [[buffer(2)]],   // [d] residual output
    device       half* r_norm [[buffer(3)]],   // [1] ||r||
    constant     uint& d      [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]])
{
    float diff = (lid < d) ? ((float)a[lid] - (float)b[lid]) : 0.0f;
    if (lid < d) r[lid] = (half)diff;
    tg[lid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = d >> 1; s > 0; s >>= 1) {
        if (lid < s) tg[lid] += tg[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) r_norm[0] = (half)sqrt(tg[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. tq_qjl_signs_f16
//    Compute QJL projection: proj[i] = sum_j( qjl_entry(i,j,seed) * r[j] )
//    then pack sign(proj[i]) as 1 bit per i.
//    Grid: [d/8,1,1]  Threadgroup: [8,1,1]
//    (each threadgroup processes 8 rows → 8 sign bits → 1 packed byte)
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_qjl_signs_f16(
    device const half*    r           [[buffer(0)]],  // [d] residual
    device       uchar* signs_packed[[buffer(1)]],  // [d/8] sign bits
    constant     uint&    d           [[buffer(2)]],
    constant     uint&    qjl_seed    [[buffer(3)]],
    uint2 tgid    [[threadgroup_position_in_grid]],
    uint2 lid_vec [[thread_position_in_threadgroup]],  // must match tgid type
    threadgroup float* tg [[threadgroup(0)]])   // [8 * d]
{
    const uint lid = lid_vec.x;  // 1-D thread index 0..7

    // lid = row index within this byte (0..7)
    uint row = tgid.x * 8u + lid;  // global row index

    // Each thread computes dot product of row `row` of QJL matrix with r
    float dot = 0.0f;
    for (uint j = 0; j < d; j++) {
        dot += qjl_entry(row, j, qjl_seed) * (float)r[j];
    }

    // Pack 8 sign bits into one byte
    uchar bit = (dot >= 0.0f) ? 1u : 0u;
    // Use threadgroup to collect 8 bits
    tg[lid] = (float)bit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        uchar byte = 0;
        for (uint i = 0; i < 8; i++) byte |= ((uchar)tg[i] << i);
        signs_packed[tgid.x] = byte;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. tq_qjl_correction_f16
//     Compute QJL correction: corr[j] = (sqrt(pi/2)/d) * r_norm * sum_i(sign_i * qjl(i,j))
//     Grid: [d,1,1]  Threadgroup: [1,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_qjl_correction_f16(
    device const uchar* signs_packed [[buffer(0)]],  // [d/8]
    device       half*    corr         [[buffer(1)]],  // [d]
    device const half*    r_norm       [[buffer(2)]],  // [1]
    constant     uint&    d            [[buffer(3)]],
    constant     uint&    qjl_seed     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    // Column j of QJL^T: compute sum_i( sign_i * qjl(i, gid) )
    float acc = 0.0f;
    for (uint i = 0; i < d; i++) {
        uchar byte = signs_packed[i >> 3];
        float sign_i = ((byte >> (i & 7u)) & 1u) ? 1.0f : -1.0f;
        acc += sign_i * qjl_entry(i, gid, qjl_seed);
    }
    // Scale: sqrt(pi/2) / d * r_norm
    float scale = 1.2533141f / (float)d * (float)r_norm[0];
    corr[gid] = (half)(scale * acc);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. tq_scale_add_f16
//     out[i] = a[i] * scalar + b[i]  (scale-and-add, used for norm restore)
//     Grid: [d,1,1]  Threadgroup: [1,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_scale_add_f16(
    device const half*  a      [[buffer(0)]],
    device const half*  b      [[buffer(1)]],
    device       half*  out    [[buffer(2)]],
    device const half*  scalar [[buffer(3)]],   // [1]
    constant     uint&  d      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < d)
        out[gid] = (half)((float)a[gid] * (float)scalar[0] + (float)b[gid]);
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. tq_group_quant_val_f16
//     Per-group symmetric quantization for Values.
//     group_size elements → packed uint8 (2 or 4 bits) + scale f16
//     Grid: [n_groups,1,1]  Threadgroup: [group_size,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_group_quant_val_f16(
    device const half*    v          [[buffer(0)]],  // [d]
    device       uchar* packed     [[buffer(1)]],  // [d * bits / 8]
    device       half*    scales     [[buffer(2)]],  // [n_groups]
    constant     uint&    group_size [[buffer(3)]],
    constant     uint&    bits       [[buffer(4)]],  // 2 or 4
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    threadgroup float* tg [[threadgroup(0)]])
{
    uint elem_idx = tgid * group_size + lid;
    float val = (float)v[elem_idx];

    // Find max abs value in group for scale
    tg[lid] = abs(val);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size >> 1; s > 0; s >>= 1) {
        if (lid < s) tg[lid] = max(tg[lid], tg[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float absmax = tg[0] + 1e-8f;
    if (lid == 0) scales[tgid] = (half)absmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Quantize and stage in threadgroup so thread 0 can pack without atomics
    uint levels = (1u << bits) - 1u;
    float norm_val = (val / absmax + 1.0f) * 0.5f;  // 0..1
    uchar q = (uchar)clamp((int)(norm_val * (float)levels + 0.5f), 0, (int)levels);

    tg[lid] = (float)q;          // reuse tg[] for bit-packing stage
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 packs all indices for this group sequentially.
    // No atomics needed because the host zeros packed[] before dispatch and
    // only one thread writes each byte.
    if (lid == 0) {
        uint elems_per_byte = 8u / bits;
        uint base_byte = tgid * (group_size / elems_per_byte);
        for (uint i = 0; i < group_size; i++) {
            uint byte_idx = base_byte + i / elems_per_byte;
            uint bit_off  = (i % elems_per_byte) * bits;
            packed[byte_idx] |= ((uchar)(uint)tg[i] << bit_off);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. tq_group_dequant_val_f16
//     Dequantize values: unpack + scale.
//     Grid: [d,1,1]  Threadgroup: [1,1,1]
// ─────────────────────────────────────────────────────────────────────────────
kernel void tq_group_dequant_val_f16(
    device const uchar* packed     [[buffer(0)]],  // [d * bits / 8]
    device const half*    scales     [[buffer(1)]],  // [n_groups]
    device       half*    out        [[buffer(2)]],  // [d]
    constant     uint&    group_size [[buffer(3)]],
    constant     uint&    bits       [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint elems_per_byte = 8u / bits;
    uint byte_idx  = gid / elems_per_byte;
    uint bit_shift = (gid % elems_per_byte) * bits;
    uint mask = (1u << bits) - 1u;
    uint q = (packed[byte_idx] >> bit_shift) & mask;

    uint levels = (1u << bits) - 1u;
    float scale = (float)scales[gid / group_size];
    // Map [0, levels] → [-scale, +scale]
    float val = ((float)q / (float)levels * 2.0f - 1.0f) * scale;
    out[gid] = (half)val;
}
