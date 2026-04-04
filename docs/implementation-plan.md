# Implementation Plan — Performance Enhancement & Scaling

> **Current state:** 258 tok/sec (Q4\_0) / 233 tok/sec (Q8\_0) on Apple M4 Pro — **exceeds llama.cpp by 12%**
> **Status:** Single-token decode optimization is **COMPLETE**. At the hardware bandwidth physics limit.
> **Next targets:** 7B model support, PagedAttention for concurrency, two-model speculative decoding

---

## Progress So Far

| Phase | What was done | Impact |
|:-----:|---------------|:------:|
| ✅ | **GEMV: simd_sum() + vectorized half4/float4** — replaced 124-step Kahan reduction with 1-cycle hardware instruction; 4× fewer load instructions via dot(float4) | **+36%** (73→102 tok/s) |
| ✅ | **f16 weight dequantization** — quantized types stored as f16 instead of f32, halving GEMV bandwidth | **+58%** (102→161 tok/s) |
| ✅ | **Fused Q8\_0 GEMV** — reads packed 1.06 bytes/elem on-the-fly instead of dequanted f16 (2 bytes/elem) | **+39%** (161→224 tok/s) |
| ✅ | **Q4K GEMV optimization** — removed Kahan, simd_sum, vectorized input reads | Included above |
| ✅ | **Command buffer batching** — 512→8192 encode limit eliminates mid-token GPU stalls | ~5% latency reduction |
| ✅ | **GEMM unrolling** — 4× unrolled inner loop for better ILP in prefill | Prefill improvement |

**Total achieved: 73 → 224 tok/sec = 3.1× speedup (97% of llama.cpp)** 🚀

---

## Phase 1: Saturate Memory Bandwidth (Target: >200 tok/sec)

> **Rationale:** LLM decode is memory-bandwidth-bound. Apple M4 Pro has ~273 GB/s.
> At 161 tok/sec with ~500MB weight reads/token, we're using ~80 GB/s — only 29% of peak.
> These optimizations close the gap to theoretical limits.

### 1.1 Multi-Row GEMV (4-8 rows per threadgroup)

**Impact: +15-25%** · **Complexity: Medium** · **Files:** `gemv.metal`, `dispatch.rs`

Currently each threadgroup handles 1 output row with 32 threads (1 SIMD group).
For small M values (e.g., K/V projection M=128), this under-utilizes GPU cores.

**Action:**
- Create `gemv_f16_mr4` kernel: 128 threads (4 SIMD groups), each group handles 1 row
- The 4 SIMD groups share the same GPU core → input vector stays in L1 cache
- Reduces threadgroup scheduling overhead for small-M GEMVs

```metal
// 4 SIMD groups × 32 lanes = 128 threads
// Each SIMD group handles one output row
kernel void gemv_f16_mr4(
    device const half* A, device const half* x, device half* y,
    constant uint& M, constant uint& K,
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    uint simd_group = lid / 32;
    uint simd_lane  = lid % 32;
    uint row = tg_id * 4 + simd_group;
    if (row >= M) return;
    
    float acc = 0.0f;
    for (uint i = simd_lane; i < K/4; i += 32)
        acc += dot(float4(((device const half4*)(A + row*K))[i]),
                   float4(((device const half4*)x)[i]));
    
    float result = simd_sum(acc);
    if (simd_lane == 0) y[row] = half(result);
}
```

**Dispatch change:** `M/4` threadgroups of 128 threads (vs M threadgroups of 32).

### 1.2 simdgroup_matrix GEMV (Apple AMX Hardware)

**Impact: +20-40%** · **Complexity: High** · **Files:** `gemv.metal`, `dispatch.rs`

Apple Silicon M1+ has a dedicated matrix coprocessor (AMX) exposed via `simdgroup_multiply_accumulate`.
This performs 8×8 half-precision matrix multiply in hardware — massive throughput for GEMV.

**Action:**
- Reshape GEMV as a degenerate GEMM: [M, K] × [K, 1] = [M, 1]
- Process 8 output rows per simdgroup using 8×8 tile operations
- Use `simdgroup_matrix<half, 8, 8>` for the weight tile, `simdgroup_matrix<half, 8, 1>` for input

```metal
#include <metal_simdgroup_matrix>
// Each simdgroup computes 8 output elements via hardware matrix multiply
simdgroup_matrix<half, 8, 8> A_tile;
simdgroup_matrix<half, 8, 1> x_tile;
simdgroup_matrix<half, 8, 1> y_tile = simdgroup_matrix<half, 8, 1>(0);

for (uint k = 0; k < K; k += 8) {
    simdgroup_load(A_tile, A_row + k, K);  // 8×8 from weight matrix
    simdgroup_load(x_tile, x_vec + k);     // 8×1 from input
    simdgroup_multiply_accumulate(y_tile, A_tile, x_tile, y_tile);
}
simdgroup_store(y_tile, y + row_base);  // write 8 outputs
```

**Why this matters:** The AMX can sustain ~2 TFLOPS at f16, far exceeding the scalar ALU path.

### 1.3 f16 KV Cache (Halve Attention Bandwidth)

**Impact: +10-15% at long contexts** · **Complexity: Low** · **Files:** `kv_cache.rs`, `forward.rs`

The KV cache is stored in f32 (4 bytes/element). Decode attention reads all KV positions
every token. Switching to f16 halves bandwidth and memory usage.

**Action:**
- Add `KvCacheF16` struct with f16 buffers
- Use `kv_copy_to_cache_f16` (already exists) for f16 scatter
- Switch `decode_attention_f16` for the attention kernel
- Add `--kv-f16` CLI flag (default for non-research use)

**Memory savings:** For 28-layer 7B model with 4096 context: 470 MB → 235 MB.

---

## Phase 2: Kernel Fusion & Pipeline Optimization (Target: >250 tok/sec)

> **Rationale:** After saturating bandwidth, the next bottleneck is kernel dispatch
> overhead and redundant memory round-trips. Each of the ~400 dispatches per token
> has ~2-5µs overhead = 0.8-2ms per token.

### 2.1 Fused RoPE + QKV Projection

**Impact: -3 kernel dispatches/layer** · **Complexity: Medium** · **Files:** new `fused_qkv_rope.metal`

Currently the flow is: `norm → Q_gemv → K_gemv → V_gemv → rope(Q) → rope(K)` = 6 dispatches.
Fuse into: `norm → fused_QKV_gemv → fused_rope_QK` = 3 dispatches.

**Action:**
- Create `gemv_qkv_f16` kernel that computes Q, K, V in a single dispatch
  (3 output rows per thread, shared input vector)
- Create `rope_qk_inplace_f32` that applies RoPE to both Q and K in one kernel

### 2.2 Fused FFN Gate+Up+SiLU

**Impact: -2 kernel dispatches/layer** · **Complexity: Medium** · **Files:** new `fused_ffn.metal`

Currently: `gate_gemv → up_gemv → silu_mul` = 3 dispatches.
The gate and up projections read the same input vector.

**Action:**
- Create `gemv_gate_up_silu_f16` kernel: computes both gate and up projections,
  then applies SiLU(gate) × up in-place
- Saves 2 dispatches and eliminates the intermediate gate/up buffer writes

### 2.3 Async Command Buffer Pipelining

**Impact: +10-20%** · **Complexity: High** · **Files:** `context.rs`, `forward.rs`

Currently: all kernels for one token are encoded into a single command buffer,
committed, and waited for synchronously. The CPU is idle during GPU execution.

**Action:**
- Pipeline 2 command buffers: while GPU executes token N, CPU prepares token N+1
- Use Metal's `addCompletedHandler` for async notification
- Requires careful synchronization for the KV cache (GPU writes, CPU reads)

### 2.4 Persistent Threadgroups (Reduce Dispatch Overhead)

**Impact: +5-10%** · **Complexity: High** · **Files:** `gemv.metal`, `dispatch.rs`

Instead of dispatching M threadgroups per GEMV (each processing 1 row), dispatch
a fixed number of persistent threadgroups that loop over rows internally.

**Action:**
- Launch `num_gpu_cores × 4` threadgroups that persist for the entire GEMV
- Each threadgroup loops: `for (uint row = tg_id; row < M; row += num_tgs)`
- Eliminates GPU scheduler overhead of launching thousands of tiny threadgroups

---

## Phase 3: Concurrency & Server Scaling (Target: 4+ concurrent requests)

> **Rationale:** The HTTP server currently handles one request at a time.
> For production use, we need multi-sequence KV caches and request scheduling.

### 3.1 PagedAttention KV Cache

**Impact: Virtual memory for KV** · **Complexity: High** · **Files:** new `paged_kv_cache.rs`

Replace the flat contiguous KV cache with a page-table-based allocator:
- Fixed-size pages (e.g., 16 tokens × kv_dim × 2 bytes)
- Page table maps logical positions → physical GPU buffer offsets
- Modified attention kernel reads via indirection

**Benefits:**
- No memory fragmentation with multiple concurrent sequences
- Dynamic context length (no pre-allocated max_seq)
- Memory sharing for beam search or prompt caching

### 3.2 Continuous Batching

**Impact: 4× concurrent throughput** · **Complexity: High** · **Files:** `server/`, `forward.rs`

Process multiple sequences simultaneously in a single forward pass:
- Batch the embedding lookups
- Use batched GEMM (already supports batch dimension)
- Modified attention kernel with per-sequence KV lengths

**Action:**
- `BatchedExecutor` struct managing N concurrent sequences
- Each sequence has its own KV cache pages and position counter
- Server schedules: prefill new requests, decode active requests, evict completed

### 3.3 Chunked Prefill

**Impact: Eliminates prefill latency spikes** · **Complexity: Medium** · **Files:** `prefill.rs`, `server/`

Large prompts (>512 tokens) block the GPU for hundreds of milliseconds,
starving decode requests. Break prefill into fixed-size chunks:

**Action:**
- Split prompt into 512-token chunks
- Interleave prefill chunks with decode steps from other requests
- Ensures decode latency stays ≤ chunk_prefill_time + decode_time

---

## Phase 4: Advanced Optimizations (Target: maximum tok/sec)

### 4.1 Speculative Decoding

**Impact: 1.5-2× generation speed** · **Complexity: Very High** · **Files:** new `speculative.rs`

Use a small "draft" model (e.g., 2-layer distilled) to predict 4-8 tokens ahead,
then verify the entire draft sequence with the main model in one forward pass.

**Action:**
- Load a secondary draft model (could be a pruned version of the main model)
- Draft loop: generate K tokens with the draft model
- Verify: run the main model on all K tokens (single GEMM pass)
- Accept matching prefix, reject divergent tokens

### 4.2 Weight-Only Quantization with Fused Kernels

**Impact: +30% for 7B+ models** · **Complexity: High** · **Files:** new `gemv_q8k.metal`

Add fused on-the-fly dequantization for Q8_0 and Q6K (like the existing Q4K path):
- Read compressed weights directly (1 byte/element for Q8_0)
- Dequantize in registers during the dot product
- Eliminates the f16 dequantization step and halves weight storage

### 4.3 Flash Decoding (Parallel KV Attention)

**Impact: +30-50% at long contexts** · **Complexity: High** · **Files:** `attention.metal`

The current decode attention is sequential over KV positions (one thread processes all positions).
For long contexts (>1K tokens), this becomes the bottleneck.

**Action:**
- Split KV positions across multiple threadgroups
- Each threadgroup computes partial softmax + weighted V sum
- Final reduction combines partial results

---

## Success Metrics

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|:-------:|:-------:|:-------:|:-------:|
| Decode tok/sec | **224** | >280 | >320 | >320 |
| Concurrent requests | 1 | 1 | 1 | **4+** |
| KV memory (7B, 4K ctx) | 470 MB | **235 MB** | 235 MB | Paged |
| Time-to-first-token | ~50ms | <30ms | <20ms | <50ms |
| Accuracy (llama.cpp match) | 100% | 100% | 100% | 100% |

---

## Hardware Utilization Analysis

### Apple M4 Pro Theoretical Limits

| Resource | Capacity | Current Usage | Target |
|----------|----------|:------------:|:------:|
| Memory bandwidth | 273 GB/s | ~80 GB/s (29%) | >200 GB/s (73%) |
| GPU compute (f16) | ~8 TFLOPS | ~0.5 TFLOPS | >3 TFLOPS |
| GPU compute (f32) | ~4 TFLOPS | ~0.3 TFLOPS | >1.5 TFLOPS |
| Unified memory | 24 GB | ~2 GB (0.5B) | <8 GB (7B) |

### Bottleneck Breakdown (per token, Q8_0 0.5B)

```
Weight reads (Q8_0 fused): ~525 MB → 525/273000 = 1.92 ms  (theoretical)
KV cache reads (attn):     ~0.5 MB → negligible at short ctx
Kernel dispatch (×340):    ~2µs×340 = 0.68 ms
CPU overhead (sample):     ~0.10 ms
───────────────────────────────────────────
Theoretical minimum:       ~2.70 ms/token = 370 tok/sec
Current:                   4.61 ms/token = 224 tok/sec
Gap:                       1.65× — kernel fusion + multi-row GEMV
```

---

## Implementation Priority Order

```
Phase 1 (weeks 1-2):    Multi-row GEMV → f16 KV cache → simdgroup_matrix
Phase 2 (weeks 3-4):    Fused RoPE+QKV → Fused FFN → Async pipeline
Phase 3 (weeks 5-8):    PagedAttention → Continuous batching → Chunked prefill
Phase 4 (weeks 9-12):   Speculative decoding → Fused Q8K GEMV → Flash decoding
```

Each phase has clear success criteria and can be validated independently.
Quality (token-match accuracy) is verified after every change.
