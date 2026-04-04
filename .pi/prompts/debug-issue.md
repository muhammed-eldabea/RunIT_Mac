---
description: Systematically debug an issue in the inference engine
---

# Debug: $@

## Debugging Protocol

1. **Reproduce**: Run the exact command that triggers the issue
2. **Isolate**: Narrow down to the specific crate/module
   - If output quality issue → check `forward.rs`, `sampler.rs`
   - If crash/panic → check the backtrace, look at `dispatch.rs` buffer setup
   - If wrong numerical values → check Metal shaders in `crates/kernels/shaders/`
   - If server issue → check `crates/engine/src/server/`
3. **Read**: Read the relevant source files before suggesting fixes
4. **Fix**: Make the minimal change needed
5. **Verify**: Run tests to confirm the fix

## Key Files by Subsystem
- Forward pass: `crates/engine/src/forward.rs`
- Sampling: `crates/engine/src/sampler.rs`
- KV Cache: `crates/engine/src/kv_cache.rs`, `tq_kv_cache.rs`
- Kernels: `crates/kernels/src/dispatch.rs` + `crates/kernels/shaders/*.metal`
- Model loading: `crates/engine/src/loader.rs`, `weights.rs`
- Config: `crates/engine/src/config.rs`
- GGUF parsing: `crates/gguf/src/parser.rs`, `types.rs`
- Server: `crates/engine/src/server/handlers.rs`

## Run Tests After Fix
```bash
cargo test 2>&1
cargo run --release --bin generate -- --model <path> --prompt "test" --max-tokens 20 2>&1
```
