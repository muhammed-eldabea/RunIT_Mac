---
description: Review uncommitted changes in the engine for correctness and quality
---

# Review Current Changes

## Steps

1. Run `git diff` to see all uncommitted changes
2. For each changed file, assess:
   - **Correctness**: Will this produce correct results? Check math in shaders, buffer sizes in dispatch, logic in forward pass
   - **Safety**: Any buffer overflows, uninitialized reads, or race conditions?
   - **Performance**: Any regressions? Unnecessary allocations or synchronization?
   - **Style**: Consistent with the rest of the codebase?

## Critical Review Areas
- Metal shaders: Check threadgroup bounds, memory access patterns, numerical precision
- dispatch.rs: Verify grid dimensions match output sizes, buffer indices match shader expectations
- forward.rs: Check weight tensor names match GGUF convention, layer iteration is correct
- sampler.rs: Verify probability math (softmax, top-p cumsum)

## After Review
- List issues found with severity (critical/warning/nit)
- Suggest specific fixes
- Recommend tests to run before committing
