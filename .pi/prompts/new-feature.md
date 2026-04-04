---
description: Plan and implement a new feature for the inference engine
---

# New Feature: $@

## Implementation Protocol

1. **Understand**: Read relevant existing code to understand the architecture
2. **Plan**: Identify which crates/files need changes
3. **Implement**: Write the code, following existing patterns
4. **Test**: Add tests and verify with existing test suite
5. **Validate**: Run inference to confirm nothing is broken

## Architecture Overview
```
bare-metal-gguf    → GGUF file parsing (mmap)
bare-metal-kernels → Metal GPU dispatch + shaders
bare-metal-engine  → Model config, loader, forward pass, server
bare-metal-tokenizer → HuggingFace BPE tokenizer
bare-metal-reference → Candle validation backend
```

## Key Design Principles
- Zero Python dependency — everything in Rust + Metal
- Unified memory — CPU and GPU share buffers on Apple Silicon
- Synchronous dispatch — each kernel blocks (pipelining is future work)
- GGUF-native — no format conversion needed
- Qwen2 architecture — hardcoded for now, extensible later

## Checklist
- [ ] Read existing code in the affected area
- [ ] Follow existing code style and patterns
- [ ] Add to the appropriate crate (don't mix concerns)
- [ ] Write a dispatch wrapper in dispatch.rs if adding a kernel
- [ ] Add test coverage
- [ ] Run full test suite: `cargo test 2>&1`
- [ ] Test with actual model: `cargo run --release --bin generate -- ...`
