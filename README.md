# Bare-Metal MX LLM

A from-scratch LLM inference engine written in **Rust + Metal**, targeting Apple M-series
chips (specifically M4 Pro 24 GB). Runs [Qwen2.5-Coder-7B-Instruct Q4\_K\_M](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)
natively without Python, PyTorch, or llama.cpp.

---

## Goals

| Goal | Status |
|------|--------|
| Zero Python runtime — pure Rust | Done |
| Metal GPU kernels (MSL) | Done |
| Q4\_K\_M weight dequantisation on GPU | Done (Phase 4) |
| KV cache in unified memory | Done (Phase 3) |
| Greedy / token-by-token decode | Done |
| TurboQuant KV-cache compression (3–4 bit) | Done (Phase 5) |
| Text I/O — tokenizer, temperature/top-p sampling | Done (Phase 6) |
| OpenAI-compatible HTTP server | Done (Phase 7) |
| Prefill batching (GEMM kernel) | Done (Phase 8) |

---

## Quick Start

```bash
# Download model and tokenizer (requires huggingface-cli)
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    tokenizer.json

# CLI generation with text I/O (Phase 6)
cargo run --release -p bare-metal-engine --bin generate -- \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --tokenizer tokenizer.json \
    --prompt "Write a Rust function that computes Fibonacci numbers" \
    --tokens 200 --temperature 0.7

# OpenAI-compatible HTTP server (Phase 7)
cargo run --release -p bare-metal-engine --bin serve -- \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf tokenizer.json \
    --port 8080

# Query the server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

Requires: macOS 14+, Xcode 15+, Rust 1.78+.

---

## Repository Layout

```
Bare-Metal-MX-LLM/
├── crates/
│   ├── gguf/           # Phase 0 — GGUF file parser (mmap, tensors, metadata)
│   ├── tokenizer/      # Phase 6 — HuggingFace tokenizer wrapper
│   ├── kernels/        # Phase 1-8 — Metal shaders + Rust dispatch layer
│   │   ├── shaders/    # .metal source files compiled to .metallib at build time
│   │   │   ├── gemv.metal, rms_norm.metal, rope.metal
│   │   │   ├── flash_attn.metal, silu_mul.metal, add.metal
│   │   │   ├── dequant.metal          # Phase 4: Q4_K_M GPU dequant
│   │   │   ├── turboquant.metal       # Phase 5: TurboQuant KV compression
│   │   │   ├── gemm.metal             # Phase 8: tiled GEMM for prefill
│   │   │   └── rope.metal             # Phase 8: batched RoPE appended
│   │   └── src/        # MetalContext, dispatch functions, error types
│   └── engine/         # Phase 2-8 — model config, forward pass, server
│       └── src/
│           ├── config.rs        # ModelConfig from GGUF metadata
│           ├── tensor.rs        # DType enum, Tensor descriptor
│           ├── loader.rs        # Model struct (mmap + tensor index)
│           ├── kv_cache.rs      # Flat KV cache in unified Metal buffers
│           ├── tq_kv_cache.rs   # Phase 5: TurboQuant hybrid KV cache
│           ├── forward.rs       # Executor: weight upload + transformer forward
│           ├── prefill.rs       # Phase 8: batched prefill via GEMM
│           ├── sampler.rs       # Phase 6: temperature / top-k / top-p sampling
│           ├── chat_template.rs # Phase 6: Qwen2 chat template formatter
│           ├── server/          # Phase 7: OpenAI-compatible HTTP server
│           │   ├── mod.rs       #   AppState + axum router
│           │   ├── types.rs     #   Request / response serde types
│           │   └── handlers.rs  #   Route handler implementations
│           └── bin/
│               ├── generate.rs  # CLI: text I/O + benchmark
│               ├── serve.rs     # HTTP server binary
│               └── inspect.rs   # Model inspection utility
├── docs/
│   ├── ARCHITECTURE.md   # System design and data-flow
│   └── KERNELS.md        # Metal kernel reference
└── README.md
```

---

## Architecture Overview

```
GGUF file (mmap)
      │
      ▼
  bare-metal-gguf   ←── zero-copy tensor views
      │
      ▼
  bare-metal-engine
  ┌───────────────────────────────────┐
  │  Executor::new()                  │
  │  ┌──────────────────────────────┐ │
  │  │  upload_weight()             │ │
  │  │  F16 → zero-copy / memcpy   │ │
  │  │  Q4K → GPU dequant → F16    │ │   ← Phase 4
  │  └──────────────────────────────┘ │
  │                                   │
  │  Executor::forward(token, pos, kv)│
  │  per token:                       │
  │   embed → [RMSNorm → QKV → RoPE  │
  │    → KV-cache update             │
  │    → FlashAttention              │
  │    → FFN (SwiGLU)] × N layers    │
  │   → final norm → lm_head → logits│
  └───────────────────────────────────┘
           │
  KvCache  │  Metal buffers, unified memory
  (caller) │  [num_layers, num_kv_heads, max_seq, head_dim] f16
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

---

## Phase Roadmap

| Phase | Branch | Description |
|-------|--------|-------------|
| 0 | `phase-0/gguf-parser` | GGUF parser — mmap + tensor metadata |
| 1 | `phase-1/metal-kernels` | Core Metal kernels (GEMV, RoPE, RMSNorm, SwiGLU, FlashAttn) |
| 2 | `phase-2/model-loading` | ModelConfig, DType, tensor loader, MetalTensor |
| 3 | `phase-3/transformer-forward` | Transformer forward pass + KV cache |
| 4 | `phase-4/q4k-dequant` | Q4\_K\_M GPU dequantisation |
| 5 | `phase-5/turboquant-kv-cache` | TurboQuant KV-cache compression (3–4 bit, ICLR 2026) |
| 6 | `phase-6-7-8/completion` | Text I/O: HuggingFace tokenizer, temperature/top-p/top-k |
| 7 | `phase-6-7-8/completion` | OpenAI-compatible HTTP server (axum, SSE streaming) |
| 8 | `phase-6-7-8/completion` | Prefill batching — tiled GEMM + batched RoPE |

---

## TurboQuant (Phase 5 Preview)

[TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research / ICLR 2026) compresses the
**KV cache** to 3–4 bits per element with provably near-optimal inner-product fidelity —
no retraining required. Unlike Q4\_K\_M (which quantises model *weights*), TurboQuant
targets the *KV activations* that grow with sequence length.

**How it works:**

1. Apply a randomised Hadamard rotation to each K/V vector → coordinates become
   near-Gaussian, amenable to Lloyd-Max quantisation.
2. Quantise each rotated coordinate to 3 or 4 bits using per-bitwidth codebooks.
3. Optionally correct inner-product bias with a 1-bit JL residual sketch.

**Impact on this engine:**

- KV cache memory: 234 MB (F16, 4096 tokens) → ~48 MB at 3 bit (~4.9×).
- Enables much longer context windows within 24 GB VRAM.
- Metal implementation is feasible: Hadamard transform via SIMD butterfly ops,
  codebook lookup in threadgroup memory.

See [docs/ARCHITECTURE.md#turboquant](docs/ARCHITECTURE.md#phase-5-turboquant-kv-cache-compression) for the planned design.

---

## Development

```bash
# Check all crates (Linux-safe)
cargo check --workspace

# Check Metal-specific code (requires macOS SDK)
cargo check --target aarch64-apple-darwin -p bare-metal-engine -p bare-metal-kernels

# Run tests
cargo test --workspace

# Build release
cargo build --release -p bare-metal-engine
```

CI runs on `macos-latest` (GitHub Actions).

---

## License

MIT
