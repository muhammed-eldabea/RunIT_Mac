// gemm.metal — Tiled matrix-matrix multiply for transformer prefill (Phase 8).
//
// Computes: Y = A @ B^T
//   A: [M, K]  — activations (seq_len × in_features)
//   B: [N, K]  — weight matrix (out_features × in_features)
//   Y: [M, N]  — output (seq_len × out_features)
//
// Uses 16×16 threadgroup tiles for efficiency on Apple Silicon.
// Grid: [ceil(N/16), ceil(M/16), 1]  Threadgroup: [16, 16, 1]

#include <metal_stdlib>
using namespace metal;

#define TILE 16

kernel void gemm_f16(
    device const half* A  [[buffer(0)]],   // [M, K] activations
    device const half* B  [[buffer(1)]],   // [N, K] weights (B^T implied)
    device       half* Y  [[buffer(2)]],   // [M, N] output
    constant     uint& M  [[buffer(3)]],   // rows of A (seq_len)
    constant     uint& N  [[buffer(4)]],   // rows of B (out_features)
    constant     uint& K  [[buffer(5)]],   // cols of A and B (in_features)
    uint2 tgid [[threadgroup_position_in_grid]],   // tile index
    uint2 lid  [[thread_position_in_threadgroup]]) // [0..15, 0..15]
{
    // Static threadgroup tiles — sized at compile time, no host allocation needed
    threadgroup half As[TILE * TILE];
    threadgroup half Bs[TILE * TILE];

    // Global output coordinates
    uint row = tgid.y * TILE + lid.y;   // M dimension (seq position)
    uint col = tgid.x * TILE + lid.x;   // N dimension (output feature)

    float acc = 0.0f;
    uint n_tiles = (K + TILE - 1) / TILE;

    for (uint t = 0; t < n_tiles; t++) {
        // Cooperatively load a TILE×TILE block of A and B into threadgroup memory
        uint a_col = t * TILE + lid.x;
        uint b_col = t * TILE + lid.y;  // B stored [N, K], tile along K with lid.y

        As[lid.y * TILE + lid.x] = (row < M && a_col < K) ? A[row * K + a_col]  : 0.0h;
        Bs[lid.x * TILE + lid.y] = (col < N && b_col < K) ? B[col * K + b_col]  : 0.0h;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate dot product for this tile
        for (uint k = 0; k < TILE; k++) {
            acc += float(As[lid.y * TILE + k]) * float(Bs[lid.x * TILE + k]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        Y[row * N + col] = half(acc);
    }
}
