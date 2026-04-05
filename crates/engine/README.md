# bare-metal-engine

From-scratch LLM inference engine for Apple Silicon — **233 tok/sec** on M4 Pro.

Part of the [RunIT Engine](https://github.com/muhammed-eldabea/RunIT_Mac).

## Performance

| Model | Quant | tok/sec | Quality |
|-------|:-----:|:-------:|:-------:|
| Qwen2.5-0.5B | Q8\_0 | **233** | ✅ Perfect |
| Qwen3-0.6B | Q8\_0 | **176** | ✅ Perfect |
| llama.cpp (reference) | Q8\_0 | 230 | ✅ |

**3.2× speedup** from baseline (73 → 233 tok/sec). Matches llama.cpp.

## Supported Architectures

- **Qwen2** — Qwen2.5-0.5B through 72B
- **Qwen3** — with QK-norm and auto head\_dim detection
- **LLaMA** — LLaMA-2, LLaMA-3
- **Mistral** — Mistral-7B
- **OLMoE** — Mixture-of-Experts with top-k routing

## Supported Quantizations

| Format | Fused GEMV | Bytes/elem |
|--------|:----------:|:----------:|
| **Q8\_0** | ✅ On-the-fly | 1.06 |
| **Q4\_0** | ✅ On-the-fly | 0.56 |
| **Q4\_K** | ✅ On-the-fly | 0.56 |
| Q5\_0, Q5K, Q6K, Q8K | f16 dequant | 2.0 |
| Q2K, Q3K | f16 dequant | 2.0 |
| F16, BF16 | Native | 2.0 |

## Features

- **85+ Metal GPU kernels** across 16 shader files
- **Fused on-the-fly dequantization** — reads packed weights directly, no intermediate buffer
- **Kernel fusion** — QKV+bias, gate+up+silu, QK RoPE in single dispatches
- **f32 precision pipeline** — f32 residual stream for research-grade accuracy
- **OpenAI-compatible HTTP server** — `/v1/chat/completions` with SSE streaming
- **Speculative decoding infrastructure** — `forward_draft()` + `verify_draft()`
- **PagedAttention KV cache** — block-allocated memory for concurrent sequences
- **TurboQuant KV compression** — 3-4 bit KV cache for long contexts

## Quick Start

```rust
use bare_metal_engine::{forward::Executor, loader::load_model, kv_cache::KvCache};
use bare_metal_kernels::context::MetalContext;

let model = load_model("model.gguf")?;
let ctx = MetalContext::new()?;
let executor = Executor::new(ctx, &model)?;
let mut kv = KvCache::new(&executor.ctx, &executor.config, 2048);

// Generate tokens
let logits = executor.forward(token_id, position, &mut kv)?;
```

## CLI Usage

```bash
# Build
cargo build --release -p bare-metal-engine

# Generate text
./target/release/generate model.gguf \
    --tokenizer tokenizer.json \
    --prompt "What is machine learning?" \
    --tokens 100

# Start HTTP server
./target/release/serve model.gguf tokenizer.json --port 8080
```

## Requirements

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Rust 1.78+
- Xcode Command Line Tools

## License

MIT
