---
name: build-engine
description: Build the RunIT bare-metal LLM inference engine workspace. Compiles Rust crates and Metal shaders. Use when you need to build, rebuild, or check compilation status.
---

# Build Engine

You are building a Rust + Metal GPU inference engine workspace. The workspace has 6 crates:
- `bare-metal-engine` — model loader, forward pass, HTTP server
- `bare-metal-kernels` — Metal GPU kernel dispatch + shader compilation
- `bare-metal-gguf` — GGUF binary format parser
- `bare-metal-tokenizer` — HuggingFace BPE tokenizer
- `bare-metal-reference` — Candle-based reference backend
- `bare-metal-bench` — Criterion benchmarks

## Build Commands

### Full workspace build (debug)
```bash
cargo build 2>&1
```

### Full workspace build (release, optimized)
```bash
cargo build --release 2>&1
```

### Build specific crate
```bash
cargo build -p bare-metal-engine 2>&1
cargo build -p bare-metal-kernels 2>&1
cargo build -p bare-metal-gguf 2>&1
```

### Check only (faster, no codegen)
```bash
cargo check 2>&1
```

## Metal Shader Compilation

Metal shaders are compiled automatically by `crates/kernels/build.rs`:
1. Each `.metal` file in `crates/kernels/shaders/` → `.air` via `xcrun metal`
2. All `.air` files → `.metallib` via `xcrun metallib`
3. The metallib is embedded at build time

If Metal compilation fails:
- Ensure Xcode 15+ is installed: `xcode-select -p`
- Check Metal compiler: `xcrun metal --version`
- Set `METALLIB_PATH` env var to use a pre-compiled metallib

## Troubleshooting

- **Linker errors**: Check that Metal framework is linked: `cargo build -vv 2>&1 | grep framework`
- **Shader errors**: Look at build.rs output: `cargo build -p bare-metal-kernels -vv 2>&1`
- **Dependency issues**: `cargo tree -p bare-metal-engine`

## After Building

Report:
1. Whether the build succeeded or failed
2. Any warnings (especially in Metal shaders)
3. Which crates were recompiled
4. Suggest `cargo build --release` if the user is about to benchmark or test performance
