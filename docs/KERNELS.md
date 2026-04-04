# Metal Kernel Reference

All shaders live in `crates/kernels/shaders/` and are compiled to a single `.metallib`
at build time by `crates/kernels/build.rs` via `xcrun metal` + `xcrun metallib`.

The Rust dispatch layer in `crates/kernels/src/dispatch.rs` wraps each kernel with a
type-safe function. The `MetalContext` in `crates/kernels/src/context.rs` manages
pipeline states and command buffer encoding.

---

## Shader File Index

| File | Kernels | Purpose |
|------|:-------:|---------|
| `gemv.metal` | 14 | Matrix-vector multiply (all dtype/precision combos) |
| `gemv_q4k.metal` | 6 | Fused Q4\_K on-the-fly dequant GEMV |
| `gemv_q8_0.metal` | 7 | Fused Q8\_0 on-the-fly dequant GEMV (**47% less BW**) |
| `attention.metal` | 3 | FlashAttention-2 prefill + decode attention |
| `norm.metal` | 4 | RMSNorm (f16/f32 input, f16/f32 gamma) |
| `rope.metal` | 4 | Rotary position embeddings (single + batch) |
| `activation.metal` | 10 | SwiGLU, add, mul, argmax, KV scatter, bias, MoE |
| `gemm.metal` | 1 | Tiled 16×16 GEMM for prefill |
| `dequant.metal` | 1 | Q4\_K\_M GPU batch dequantization |
| `turboquant.metal` | 12 | TurboQuant KV-cache compression kernels |
| **Total** | **62** | |

---

## GEMV Kernels — The Decode Hot Path

GEMV (matrix-vector multiply) dominates decode time. Every transformer layer
calls GEMV 7 times (Q, K, V, O, gate, up, down). With 24 layers + lm\_head,
that's **169 GEMVs per token**.

### Architecture

```
Grid:  [M, 1, 1]  — one threadgroup per output row
Group: [32, 1, 1]  — one SIMD group (32 lanes)

Each threadgroup computes: y[row] = dot(A[row, :], x[:])

Inner loop: strided access across K dimension
  thread 0 reads elements 0, 32, 64, ...
  thread 1 reads elements 1, 33, 65, ...
  → perfect coalescing: 32 threads read 32 consecutive elements

Reduction: simd_sum() — single-cycle hardware instruction
  Replaces the old 124-step Kahan sequential sum
```

### Vectorization

For f16 weights: `half4` loads + `dot(float4, float4)` accumulation
```metal
uint K4 = K >> 2;
for (uint i = lid; i < K4; i += 32) {
    acc += dot(float4(((device const half4*)A_row)[i]),
               float4(((device const half4*)x_vec)[i]));
}
```
Each SIMD iteration reads 32×4 = 128 elements = 256 bytes (perfectly coalesced).

### Kernel Variants (gemv.metal — 14 kernels)

| Kernel | Weight | Input | Output | Use Case |
|--------|:------:|:-----:|:------:|----------|
| `gemv_f16` | f16 | f16 | f16 | TQ path |
| `gemv_add_f16` | f16 | f16 | f16+res | TQ residual |
| `gemv_f16w_f32in` | f16 | f32 | f16 | MoE expert |
| `gemv_f16w_f32in_f32out` | f16 | f32 | f32 | **Decode Q/K/V** |
| `gemv_add_f32res_f16` | f16 | f16 | f32+res | f32 residual |
| `gemv_add_f32_f16w` | f16 | f32 | f32+res | **Decode O-proj** |
| `gemv_f32_f32out` | f32 | f32 | f32 | Full precision |
| `gemv_f32` | f32 | f32 | f16 | Precision convert |
| `gemv_add_f32` | f32 | f32 | f32+res | f32 residual |
| `gemv_f32w` | f32 | f16 | f16 | Dequanted weights |
| `gemv_add_f32w` | f32 | f16 | f16+res | Dequanted + res |
| `gemv_f32w_f32in` | f32 | f32 | f16 | Mixed precision |
| `gemv_add_f32res_f32w` | f32 | f16 | f32+res | f32 accum |
| `gemv_add_f32_f32w` | f32 | f32 | f32+res | Full f32 path |

---

## Fused Q8\_0 GEMV — The Bandwidth Breakthrough

**File:** `gemv_q8_0.metal` · **7 kernels**

Reads packed Q8\_0 weights directly (1.0625 bytes/element) and dequantizes
during the dot product. Uses **47% less memory bandwidth** than dequanted f16.

### Q8\_0 Block Layout (34 bytes = 32 elements)

```
bytes 0-1:   d (half)    — scale factor
bytes 2-33:  qs[32] (int8) — quantized values, range [-128, 127]

Reconstruction: value[i] = d × qs[i]
```

### Inner Loop

```metal
float d = float(*reinterpret_cast<device const half*>(block));
device const char* qs = reinterpret_cast<device const char*>(block + 2);

for (uint i = 0; i < 32; i += 4) {
    float4 x4 = *(device const float4*)(x_vec + x_base + i);
    acc += d * (float(qs[i])*x4.x + float(qs[i+1])*x4.y +
                float(qs[i+2])*x4.z + float(qs[i+3])*x4.w);
}
```

### Bandwidth Comparison

| Format | Bytes/elem | 0.5B total/tok | Theoretical tok/sec |
|--------|:----------:|:--------------:|:-------------------:|
| f32 (old) | 4.0 | 1976 MB | 138 |
| f16 (v2) | 2.0 | 988 MB | 276 |
| **Q8\_0 fused (v3)** | **1.06** | **525 MB** | **520** |

### Kernel Variants (7 kernels)

| Kernel | Input | Output | Use Case |
|--------|:-----:|:------:|----------|
| `gemv_q8_0_f32in_f32out` | f32 | f32 | **Decode Q/K/V/FFN** |
| `gemv_q8_0_add_f32_f32in` | f32 | f32+res | **Decode O-proj, FFN-down** |
| `gemv_q8_0_f32in` | f32 | f16 | Greedy lm\_head |
| `gemv_q8_0_f16` | f16 | f16 | TQ path |
| `gemv_q8_0_add_f16` | f16 | f16+res | TQ residual |
| `gemv_q8_0_add_f32res_f16` | f16 | f32+res | MoE accum |

---

## Fused Q4\_K GEMV

**File:** `gemv_q4k.metal` · **6 kernels**

Same concept as Q8\_0 fused but for the more complex Q4\_K block format.
Reads 144-byte blocks (256 elements, 4.5 bits/elem effective) on-the-fly.

### Q4\_K Block Layout (144 bytes = 256 elements)

```
bytes  0-1:   d (half)       — super-block scale
bytes  2-3:   dmin (half)    — super-block minimum
bytes  4-15:  scales[12]     — 8 sub-block scale+min (6-bit packed)
bytes 16-143: qs[128]        — 4-bit nibbles (2 per byte)
```

### Scale Extraction (`get_scale_min_k4`)

```metal
if (j < 4) { sc = scales[j] & 0x3F;  m = scales[j+4] & 0x3F; }
else { sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
       m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4); }
```

---

## Attention Kernels

**File:** `attention.metal` · **3 kernels**

### `decode_attention_f32` — Single-token decode

Optimized for q\_len=1. All 32 SIMD threads cooperate on the Q·K dot product
(instead of FlashAttention where 31/32 threads are idle at q\_len=1).

```
Grid:  [num_heads, batch, 1]
Group: [32, 1, 1]

For each KV position:
  score = simd_sum(Q[d] * K[pos, d]) * scale   // parallel dot
  online_softmax_update(score)                   // streaming
  O += softmax_weight * V[pos, :]                // accumulate
```

### `flash_attention_f16` — Prefill (multi-token)

Tiled FlashAttention-2 with causal mask and GQA. SRAM layout:

```
Q_tile [32, 128]  f16   8 KB
K_tile [32, 128]  f16   8 KB
V_tile [32, 128]  f16   8 KB
S      [32, 32]   f16   2 KB
Total: 26 KB threadgroup memory
```

### `decode_attention_f16` — f16 variant for TQ path

---

## Normalization Kernels

**File:** `norm.metal` · **4 kernels**

RMSNorm: `y = x / sqrt(mean(x²) + eps) * gamma`

| Kernel | Input | Output | Gamma | Use Case |
|--------|:-----:|:------:|:-----:|----------|
| `rms_norm_f16` | f16 | f16 | f16 | TQ/prefill path |
| `rms_norm_f32in_f16out` | f32 | f16 | f16 | Precision convert |
| `rms_norm_f32_f32` | f32 | f32 | f16 | **Decode path** |
| `rms_norm_f32_f32_f32g` | f32 | f32 | f32 | **Decode (f32 norms)** |

Uses parallel tree reduction in threadgroup memory (256 threads).

---

## RoPE Kernels

**File:** `rope.metal` · **4 kernels**

Rotary Position Embeddings with **non-interleaved** dimension pairing:
pairs `(i, i + head_dim/2)` as used by Qwen2, LLaMA-3, Mistral.

| Kernel | Precision | Batch | Use Case |
|--------|:---------:|:-----:|----------|
| `rope_inplace_f16` | f16 | 1 | TQ decode |
| `rope_inplace_f32` | f32 | 1 | **Decode path** |
| `rope_batch_inplace_f16` | f16 | N | Prefill |

---

## Activation Kernels

**File:** `activation.metal` · **10 kernels**

| Kernel | Formula | Use Case |
|--------|---------|----------|
| `silu_mul_f16` | out = silu(gate) × up | TQ SwiGLU |
| `silu_mul_f32` | (f32 variant) | **Decode SwiGLU** |
| `add_f16` | out = a + b | Residual connections |
| `mul_f16` | out = a × b | Element-wise |
| `add_f16_into_f32` | acc\_f32 += src\_f16 | MoE accumulate |
| `argmax_f16` | index of max(f16 vec) | Greedy decoding |
| `scale_accumulate_f16` | acc += src × w | MoE weighted sum |
| `add_bias_broadcast_f16` | x[row,i] += bias[i] | QKV bias |
| `kv_copy_to_cache_f16` | scatter to KV cache | f16 cache write |
| `kv_copy_to_cache_f32` | (f32 variant) | **Decode KV write** |

---

## GEMM Kernel

**File:** `gemm.metal` · **1 kernel**

Tiled matrix-matrix multiply for prefill: `Y = A @ B^T`

```
Tile: 16×16 with threadgroup shared memory
Grid: [ceil(N/16), ceil(M/16), 1]
Group: [16, 16, 1]

Inner loop 4× unrolled for instruction-level parallelism
```

---

## TurboQuant Kernels

**File:** `turboquant.metal` · **12 kernels**

KV-cache compression (3-4 bit) based on the TurboQuant paper (ICLR 2026).

| Kernel | Purpose |
|--------|---------|
| `tq_normalize_f16` | L2-normalize key vectors |
| `tq_rht_f16` | Randomized Hadamard Transform |
| `tq_rht_inverse_f16` | Inverse RHT |
| `tq_lloyd_quant_f16` | Lloyd-Max quantization |
| `tq_pack_bits_u8` | Bit-pack quantized values |
| `tq_unpack_bits_u8` | Unpack to indices |
| `tq_centroid_lookup_f16` | Codebook lookup |
| `tq_residual_norm_f16` | Residual norm computation |
| `tq_qjl_signs_f16` | QJL sign extraction |
| `tq_qjl_correction_f16` | QJL correction vector |
| `tq_scale_add_f16` | Scaled addition |
| `tq_group_quant_val_f16` | Group quantization for values |

---

## Build System

`crates/kernels/build.rs` compiles all `.metal` files into a single `.metallib`:

```bash
xcrun -sdk macosx metal -c shaders/*.metal -o /tmp/kernels.air
xcrun -sdk macosx metallib /tmp/kernels.air -o <out_dir>/kernels.metallib
```

The output path is passed to Rust via `cargo:rustc-env=METALLIB_PATH=...`.
`MetalContext::new()` loads the library at runtime with `device.new_library_with_file()`.

On non-macOS systems (CI Linux), `build.rs` skips compilation; all Metal code
is `#[cfg(target_os = "macos")]` gated.

---

## Adding a New Kernel

1. Create or edit a `.metal` file in `crates/kernels/shaders/`.
2. Add the kernel function name to `KERNEL_NAMES` in `crates/kernels/src/context.rs`.
3. Add a dispatch wrapper in `crates/kernels/src/dispatch.rs`.
4. (Optional) Add a `WeightBuf` variant in `crates/engine/src/forward.rs` if it's a new quant format.
5. Call the dispatch function from the forward pass.

### Quick check: `runit-check-shaders` compiles all shaders without building Rust.
