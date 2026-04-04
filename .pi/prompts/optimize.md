---
description: Optimize a specific component of the inference engine for performance
---

# Optimize: $@

## Optimization Protocol

1. **Measure first**: Run `bench-perf` skill to get baseline numbers
2. **Profile**: Identify the bottleneck (compute vs memory bandwidth vs latency)
3. **Read the code**: Understand current implementation before changing
4. **Optimize**: Apply targeted changes
5. **Measure again**: Confirm improvement with same benchmark

## Common Optimization Targets

### Metal Kernels (crates/kernels/shaders/)
- Increase threadgroup utilization
- Use SIMD group operations for reductions
- Coalesce memory accesses
- Use shared memory to reduce global memory traffic
- Vectorize with half4/float4

### Forward Pass (crates/engine/src/forward.rs)
- Minimize CPU↔GPU synchronization points
- Batch kernel dispatches where possible
- Reuse buffers across layers
- Pipeline command buffers (async dispatch)

### Memory (crates/engine/src/)
- Zero-copy weight upload for aligned f16 tensors
- KV cache compression via TurboQuant
- Reduce temporary allocations

## Do NOT optimize
- Code clarity for marginal gains (<5%)
- Already-fast paths (measure first!)
- Debug builds (always benchmark with --release)
