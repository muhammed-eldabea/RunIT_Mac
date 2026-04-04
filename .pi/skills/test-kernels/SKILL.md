---
name: test-kernels
description: Run and debug Metal GPU kernel tests. Tests GEMV, attention, RMSNorm, RoPE, SiLU, and dequantization kernels. Use when validating GPU compute correctness.
---

# Test Metal Kernels

You are testing Metal GPU compute kernels for a bare-metal LLM inference engine on Apple Silicon.

## Available Kernels

| Kernel | Shader File | Dispatch Function | Purpose |
|--------|------------|-------------------|---------|
| GEMV f16 | gemv.metal | `dispatch_gemv_f16` | Matrix-vector multiply (f16 weights) |
| GEMV f32 | gemv.metal | `dispatch_gemv_f32` | Matrix-vector multiply (f32 weights) |
| GEMV Q4K | gemv_q4k.metal | `dispatch_gemv_q4k` | Quantized matrix-vector multiply |
| RMSNorm | norm.metal | `dispatch_rms_norm` | Root mean square normalization |
| RoPE | rope.metal | `dispatch_rope` | Rotary position embeddings |
| SiLU | activation.metal | `dispatch_silu_elementwise` | SiLU activation + elementwise multiply |
| Attention | attention.metal | `dispatch_attention` | Flash attention (tiled, GQA-aware) |
| Dequant | dequant.metal | `dispatch_dequantize_q4k` | Q4_K_M GPU dequantization |
| GEMM | gemm.metal | `dispatch_gemm` | Matrix-matrix multiply (prefill) |
| TurboQuant | turboquant.metal | — | KV cache compression |

## Running Tests

### All kernel tests
```bash
cargo test -p bare-metal-kernels 2>&1
```

### GEMV-specific test binary
```bash
cargo run --bin test_gemv 2>&1
```

### Engine integration tests (includes kernel validation)
```bash
cargo test -p bare-metal-engine 2>&1
```

### Run specific test by name
```bash
cargo test -p bare-metal-kernels -- <test_name> 2>&1
```

### Run with verbose output for debugging
```bash
cargo test -p bare-metal-kernels -- --nocapture 2>&1
```

## Debugging Kernel Issues

### Common failure patterns:
1. **NaN/Inf outputs**: Check for uninitialized buffers, division by zero in norm kernels
2. **Wrong values**: Compare against CPU reference — use `bare-metal-reference` crate with Candle
3. **Threadgroup issues**: Metal threadgroups must not exceed device limits. Check `dispatch.rs` grid calculations
4. **Buffer alignment**: Metal buffers must be page-aligned for f16 zero-copy

### Debugging workflow:
1. Run the failing test with `--nocapture` to see Metal debug output
2. Check `crates/kernels/src/dispatch.rs` for the kernel dispatch parameters
3. Check the corresponding `.metal` shader in `crates/kernels/shaders/`
4. Verify buffer sizes and grid dimensions match expected tensor shapes
5. Compare with CPU reference if available

### Metal GPU Capture (if needed):
```bash
export MTL_CAPTURE_ENABLED=1
cargo test -p bare-metal-kernels -- --nocapture 2>&1
```

## After Testing

Report:
1. Total tests run, passed, failed
2. For failures: exact test name, error message, and likely root cause
3. If a kernel produces wrong values, suggest checking the shader math vs dispatch parameters
