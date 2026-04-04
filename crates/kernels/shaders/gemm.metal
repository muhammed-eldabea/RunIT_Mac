// gemm.metal — High-performance GEMM for transformer prefill
//
// Computes: Y = A @ B^T
//   A: [M, K]  — activations (seq_len × in_features)
//   B: [N, K]  — weight matrix (out_features × in_features), row-major
//   Y: [M, N]  — output (seq_len × out_features)
//
// Two implementations:
//   1. gemm_f16_simd — simdgroup_matrix hardware (AMX) for M1+ (primary)
//   2. gemm_f16      — scalar tiled fallback
//
// The simdgroup_matrix version computes 32×32 output tiles using 4 SIMD
// groups (128 threads), each processing an 8×8 sub-tile via hardware
// matrix multiply-accumulate at ~2 TFLOPS throughput.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM with simdgroup_matrix (AMX Hardware Matrix Coprocessor)
//
// Architecture:
//   - 128 threads per threadgroup (4 SIMD groups)
//   - Each SIMD group computes one 8×8 output sub-tile
//   - 4 SIMD groups arranged 2×2 → 16×16 output tile per threadgroup
//   - K dimension iterated in steps of 8 using simdgroup_multiply_accumulate
//   - Weight matrix B loaded with transpose=true (B stored as [N,K])
//   - Cooperative loading into threadgroup memory for coalesced access
//
// Grid:  [ceil(N/16), ceil(M/16), 1]
// Group: [32, 4, 1]  (32 threads × 4 SIMD groups = 128 threads)
// SRAM:  2 × 16 × 8 × sizeof(half) = 512 bytes
// ═══════════════════════════════════════════════════════════════════════════════

#define SIMD_TILE 8
#define TG_TILE  16   // 2 SIMD tiles per dimension

kernel void gemm_f16_simd(
    device const half* A  [[buffer(0)]],   // [M, K]
    device const half* B  [[buffer(1)]],   // [N, K] (row-major, we compute A @ B^T)
    device       half* Y  [[buffer(2)]],   // [M, N]
    constant     uint& M  [[buffer(3)]],
    constant     uint& N  [[buffer(4)]],
    constant     uint& K  [[buffer(5)]],
    threadgroup  half* sram [[threadgroup(0)]],
    uint2  tgid    [[threadgroup_position_in_grid]],
    uint2  lid2    [[thread_position_in_threadgroup]])
{
    // Derive linear thread ID and SIMD group from 2D layout [32, 4]
    const uint lid     = lid2.y * 32 + lid2.x;
    const uint sg_id   = lid2.y;       // SIMD group index (0..3)
    const uint sg_lane = lid2.x;       // lane within SIMD group (0..31)

    // Each SIMD group's sub-tile position within the 16×16 output tile
    const uint sg_row = (sg_id / 2) * SIMD_TILE;  // 0 or 8
    const uint sg_col = (sg_id % 2) * SIMD_TILE;  // 0 or 8

    // Global output coordinates for this SIMD group's 8×8 tile
    const uint g_row = tgid.y * TG_TILE + sg_row;
    const uint g_col = tgid.x * TG_TILE + sg_col;

    // Accumulator — 8×8 output tile, initialized to zero
    simdgroup_matrix<half, 8, 8> acc = simdgroup_matrix<half, 8, 8>(0.0h);

    // Threadgroup staging buffers: A_tile [16, 8] and B_tile [16, 8]
    threadgroup half* A_stage = sram;                    // [TG_TILE × SIMD_TILE]
    threadgroup half* B_stage = sram + TG_TILE * SIMD_TILE; // [TG_TILE × SIMD_TILE]

    // Iterate over K in steps of 8 (SIMD tile width)
    for (uint k = 0; k < K; k += SIMD_TILE) {
        // ── Cooperative load: A_tile [16 rows × 8 cols] ──────────────────
        // 128 threads load 128 elements (16×8)
        if (lid < TG_TILE * SIMD_TILE) {
            uint r = lid / SIMD_TILE;   // row within the 16×8 tile
            uint c = lid % SIMD_TILE;   // col within the 16×8 tile
            uint gr = tgid.y * TG_TILE + r;
            uint gk = k + c;
            A_stage[r * SIMD_TILE + c] = (gr < M && gk < K) ? A[gr * K + gk] : half(0);
        }

        // ── Cooperative load: B_tile [16 rows × 8 cols] ──────────────────
        // B is [N, K] row-major. We need B^T, so load B[n, k] into B_stage[n_local, k_local]
        // Then simdgroup_load with transpose=true gives us the [8, 8] tile of B^T
        if (lid < TG_TILE * SIMD_TILE) {
            uint r = lid / SIMD_TILE;   // row = N-dimension offset
            uint c = lid % SIMD_TILE;   // col = K-dimension offset
            uint gn = tgid.x * TG_TILE + r;
            uint gk = k + c;
            B_stage[r * SIMD_TILE + c] = (gn < N && gk < K) ? B[gn * K + gk] : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── simdgroup_matrix multiply-accumulate ─────────────────────────
        simdgroup_matrix<half, 8, 8> a_tile;
        simdgroup_matrix<half, 8, 8> b_tile;

        // Load this SIMD group's 8×8 sub-tile of A from threadgroup memory
        // A_stage is [16, 8], we want rows [sg_row : sg_row+8]
        simdgroup_load(a_tile, A_stage + sg_row * SIMD_TILE, SIMD_TILE);

        // Load B sub-tile with TRANSPOSE — B_stage is [N_local=16, K_local=8]
        // We want B^T[k, n] from B_stage[n, k], so load with transpose=true
        // Rows from B_stage at [sg_col : sg_col+8] give us the N-dimension slice
        simdgroup_load(b_tile, B_stage + sg_col * SIMD_TILE, SIMD_TILE,
                       ulong2(0, 0), true);

        // Hardware matrix multiply-accumulate: acc += a_tile × b_tile
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Store 8×8 result tile to global memory ───────────────────────────
    // Only store valid elements (handle M/N not multiple of 16)
    if (g_row < M && g_col < N) {
        // Use simdgroup_store for efficient coalesced write
        // But need bounds checking — use per-element store for safety at edges
        if (g_row + SIMD_TILE <= M && g_col + SIMD_TILE <= N) {
            simdgroup_store(acc, Y + g_row * N + g_col, N);
        } else {
            // Edge tile: store element by element (only for boundary threadgroups)
            // Each thread in the SIMD group owns specific elements of the 8×8 tile
            // Use a temporary threadgroup buffer for the store
            threadgroup half temp_tile[SIMD_TILE * SIMD_TILE];
            simdgroup_store(acc, temp_tile, SIMD_TILE);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each thread copies its assigned elements, checking bounds
            for (uint idx = sg_lane; idx < SIMD_TILE * SIMD_TILE; idx += 32) {
                uint tr = idx / SIMD_TILE;
                uint tc = idx % SIMD_TILE;
                if (g_row + tr < M && g_col + tc < N) {
                    Y[(g_row + tr) * N + (g_col + tc)] = temp_tile[tr * SIMD_TILE + tc];
                }
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Scalar tiled GEMM fallback (kept as gemm_f16 for backward compatibility)
// ═══════════════════════════════════════════════════════════════════════════════

#define TILE 16

kernel void gemm_f16(
    device const half* A  [[buffer(0)]],
    device const half* B  [[buffer(1)]],
    device       half* Y  [[buffer(2)]],
    constant     uint& M  [[buffer(3)]],
    constant     uint& N  [[buffer(4)]],
    constant     uint& K  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    threadgroup half As[TILE * TILE];
    threadgroup half Bs[TILE * TILE];

    uint row = tgid.y * TILE + lid.y;
    uint col = tgid.x * TILE + lid.x;

    float acc = 0.0f;
    uint n_tiles = (K + TILE - 1) / TILE;

    for (uint t = 0; t < n_tiles; t++) {
        uint a_col = t * TILE + lid.x;
        uint b_col = t * TILE + lid.y;

        As[lid.y * TILE + lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0h;
        Bs[lid.x * TILE + lid.y] = (col < N && b_col < K) ? B[col * K + b_col] : 0.0h;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k += 4) {
            acc += float(As[lid.y * TILE + k])     * float(Bs[lid.x * TILE + k]);
            acc += float(As[lid.y * TILE + k + 1]) * float(Bs[lid.x * TILE + k + 1]);
            acc += float(As[lid.y * TILE + k + 2]) * float(Bs[lid.x * TILE + k + 2]);
            acc += float(As[lid.y * TILE + k + 3]) * float(Bs[lid.x * TILE + k + 3]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        Y[row * N + col] = half(acc);
    }
}
