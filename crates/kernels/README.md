# bare-metal-kernels

85+ optimized Metal GPU compute kernels for LLM inference on Apple Silicon.

Part of the [RunIT Engine](https://github.com/muhammed-eldabea/RunIT_Mac) — a from-scratch LLM inference engine in Rust + Metal for Apple Silicon achieving **233 tok/sec** on M4 Pro.

## Kernel Inventory (16 shader files)

| Shader | Kernels | Description |
|--------|:-------:|-------------|
| `gemv.metal` | 14 | Matrix-vector multiply (simd\_sum + half4 vectorized) |
| `gemv_q8_0.metal` | 7 | Fused Q8\_0 on-the-fly dequant GEMV (1.06 B/elem) |
| `gemv_q4_0.metal` | 7 | Fused Q4\_0 on-the-fly dequant GEMV (0.56 B/elem) |
| `gemv_q4k.metal` | 6 | Fused Q4\_K GEMV |
| `gemv_q8_0_wide.metal` | 3 | 256-thread wide GEMV for large K dimensions |
| `gemv_multirow.metal` | 6 | Multi-row GEMV (4 rows/TG, 128 threads) |
| `fused_qkv.metal` | 3 | Q+K+V+bias projection in 1 dispatch |
| `fused_ffn.metal` | 3 | gate+up+silu in 1 dispatch |
| `fused_rope.metal` | 1 | QK RoPE in 1 dispatch |
| `attention.metal` | 3 | FlashAttention-2 + decode attention |
| `gemm.metal` | 2 | simdgroup\_matrix GEMM + tiled fallback |
| `norm.metal` | 4 | RMSNorm (f16/f32 variants) |
| `rope.metal` | 4 | Rotary position embeddings |
| `activation.metal` | 10 | SwiGLU, add, argmax, KV scatter |
| `dequant.metal` | 1 | Q4K GPU batch dequantization |
| `turboquant.metal` | 12 | TurboQuant KV-cache compression |

## Key Optimizations

- **`simd_sum()`** — single-cycle hardware SIMD reduction (replaces 124-step Kahan)
- **Vectorized `half4`/`float4`** — 4× fewer load instructions, perfect coalescing
- **Fused on-the-fly dequant** — reads packed Q8\_0/Q4\_0 weights directly, no intermediate buffer
- **Kernel fusion** — QKV+bias, gate+up+silu, QK RoPE combined into single dispatches
- **Multi-row dispatch** — 4 rows per threadgroup for better GPU occupancy

## Usage

```rust
use bare_metal_kernels::context::MetalContext;
use bare_metal_kernels::dispatch::gemv_f16;

let ctx = MetalContext::new()?;
gemv_f16(&ctx, &weight_buf, &input_buf, &output_buf, m, k)?;
ctx.flush(); // commit and wait
```

## Requirements

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)

## License

MIT
