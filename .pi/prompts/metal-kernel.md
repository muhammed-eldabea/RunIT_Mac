---
description: Write or modify a Metal GPU compute shader kernel for the inference engine
---

# Metal Kernel Task: $@

## Context
You are working on Metal GPU compute shaders for a bare-metal LLM inference engine (Apple Silicon).

## File Locations
- Shader sources: `crates/kernels/shaders/*.metal`
- Rust dispatch wrappers: `crates/kernels/src/dispatch.rs`
- Metal context (device, queue): `crates/kernels/src/context.rs`
- Build script (shader compilation): `crates/kernels/build.rs`

## Requirements
1. Read the existing shader that's closest to what you need
2. Read `dispatch.rs` to understand the Rust↔Metal interface pattern
3. Follow the existing conventions:
   - Use `half` for f16, `float` for f32
   - Use threadgroup shared memory for reductions
   - Use Kahan summation for numerical stability in dot products
   - Match the dispatch grid pattern in dispatch.rs
4. After writing the shader, add the corresponding Rust dispatch function
5. Test with `cargo test -p bare-metal-kernels -- --nocapture`

## Metal Shader Template
```metal
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (tid >= N) return;
    // kernel logic here
}
```
