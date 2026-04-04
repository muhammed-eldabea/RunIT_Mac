---
name: debug-metal
description: Debug Metal GPU shader issues — NaN values, wrong outputs, threadgroup errors, buffer misalignment. Use when GPU kernels produce incorrect results or crash.
---

# Debug Metal Shaders

You are debugging Metal GPU compute shaders for a bare-metal LLM inference engine on Apple Silicon.

## Architecture

```
Rust (dispatch.rs) → Metal API → GPU Shaders (.metal files)
     ↓                               ↓
  Buffer setup              Threadgroup execution
  Grid calculation          Shared memory tiling
  Encoder commands          Compute kernel math
```

## File Locations

| Component | Path |
|-----------|------|
| Shader sources | `crates/kernels/shaders/*.metal` |
| Rust dispatch | `crates/kernels/src/dispatch.rs` |
| Metal context | `crates/kernels/src/context.rs` |
| Build script | `crates/kernels/build.rs` |
| Forward pass | `crates/engine/src/forward.rs` |

## Common Issues & Debugging

### 1. NaN / Inf in outputs
**Cause**: Uninitialized buffers, division by zero, or overflow in f16.
**Debug steps**:
1. Check if RMSNorm has epsilon > 0 in `norm.metal`
2. Check f16 overflow — values > 65504 become Inf
3. Add temporary buffer reads after each kernel to find where NaN first appears
4. Check `dispatch_rms_norm` in dispatch.rs for correct parameter passing

### 2. Wrong numerical values
**Cause**: Incorrect matrix dimensions, transposition errors, or reduction bugs.
**Debug steps**:
1. Compare output against Candle reference in `bare-metal-reference` crate
2. Check GEMV dimensions: M (output), N (reduction/input) in dispatch.rs
3. Verify Kahan summation accumulator initialization in gemv.metal
4. For Q4K: check dequantization scale/min extraction in gemv_q4k.metal

### 3. Threadgroup / grid errors
**Cause**: Grid dimensions exceed Metal device limits or don't cover full output.
**Debug steps**:
1. Metal max threadgroup size: typically 1024 threads
2. Check `threads_per_threadgroup` and `threadgroups_per_grid` in dispatch.rs
3. Ensure output elements = grid.x * threadgroup.x (no missed elements)
4. For attention: verify sequence length fits tiling scheme

### 4. Buffer alignment issues
**Cause**: Metal requires specific alignment for certain operations.
**Debug steps**:
1. F16 zero-copy needs page-aligned buffers (4096 bytes)
2. Check `MTLResourceOptions` in context.rs buffer creation
3. Q4K block alignment: 256 elements per block (32 groups × 8 values)

### 5. Shader compilation failures
**Debug steps**:
```bash
# Compile a single shader manually to see errors
xcrun metal -c crates/kernels/shaders/<shader>.metal -o /dev/null 2>&1
```
```bash
# Check all shaders
for f in crates/kernels/shaders/*.metal; do
  echo "--- $f ---"
  xcrun metal -c "$f" -o /dev/null 2>&1
done
```

## Validation Strategy

1. **Unit test**: Small tensors with known values → verify exact output
2. **Reference comparison**: Same input through Candle → compare outputs within tolerance
3. **Numerical stability**: Test with extreme values (very large, very small, negative)
4. **Shape sweep**: Test multiple dimensions to catch edge cases in grid calculation

## Reading Shader Code

Metal shaders use C++-like syntax with Metal-specific types:
- `half` / `half4` — f16 values and vectors
- `float` / `float4` — f32 values and vectors
- `device` — GPU global memory
- `threadgroup` — shared memory within a threadgroup
- `thread_position_in_grid` — global thread index
- `thread_position_in_threadgroup` — local thread index
- `threadgroup_position_in_grid` — which threadgroup this is

## After Debugging

Report:
1. Root cause identified
2. Which file(s) need changes (shader vs dispatch vs forward)
3. Proposed fix with explanation
4. How to verify the fix (specific test to run)
