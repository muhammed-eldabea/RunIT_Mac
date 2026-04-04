<p align="center">
  <img src="docs/assets/logo.svg" alt="RunIT — LLM Inference Engine" width="700"/>
</p>

<p align="center">
  <strong>From-scratch LLM inference engine in Rust + Metal for Apple Silicon</strong>
</p>

<p align="center">
  <a href="#-quickstart"><img src="https://img.shields.io/badge/macOS-14%2B-blue?logo=apple&logoColor=white" alt="macOS 14+"/></a>
  <a href="#-quickstart"><img src="https://img.shields.io/badge/Rust-1.78%2B-orange?logo=rust&logoColor=white" alt="Rust 1.78+"/></a>
  <a href="#-quickstart"><img src="https://img.shields.io/badge/Metal_GPU-Apple_Silicon-blueviolet?logo=apple" alt="Metal GPU"/></a>
  <a href="docs/COMPARISON.md"><img src="https://img.shields.io/badge/llama.cpp-100%25_match-brightgreen" alt="llama.cpp match"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
</p>

<p align="center">
  <em>No Python. No PyTorch. No llama.cpp dependency. Pure Rust + Metal shaders.</em>
</p>

---

## ✨ Highlights

```
┌─────────────────────────────────────────────────────────────────┐
│  🎯 100% token-match with llama.cpp on Qwen2.5-0.5B Q8_0      │
│  ⚡ 161 tok/sec decode on Apple M4 Pro (2.2× speedup)          │
│  🦀 Pure Rust — zero Python runtime                            │
│  🔧 Custom Metal GPU kernels — simd_sum + vectorized loads     │
│  📦 GGUF native — loads any GGUF quantized model               │
│  🌐 OpenAI-compatible HTTP server                              │
│  🔬 f32 precision pipeline for research-grade accuracy         │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Accuracy: Token-Perfect Match with llama.cpp

<p align="center">
  <img src="docs/assets/accuracy-chart.svg" alt="15/15 position accuracy" width="680"/>
</p>

Every test produces **identical output** to llama.cpp (greedy, temp=0):

| Prompt | llama.cpp | RunIT Engine | Match |
|--------|-----------|-------------|:-----:|
| "What is 2+2?" | 2 + 2 equals 4. | 2 + 2 equals 4. | ✅ |
| "Capital of France?" | The capital of France is Paris. | The capital of France is Paris. | ✅ |
| "Capital of Japan?" | The capital of Japan is Tokyo. | The capital of Japan is Tokyo. | ✅ |
| "What is machine learning?" | Machine learning is a subset of AI... (92 tok) | Machine learning is a subset of AI... (92 tok) | ✅ |
| "Translate to French" | Bonjour, comment ça va ? | Bonjour, comment ça va ? | ✅ |
| "Write a poem about the moon" | The moon, a celestial sight... | The moon, a celestial sight... | ✅ |

> 📋 Full per-position logit comparison: [docs/COMPARISON.md](docs/COMPARISON.md)

## ⚡ Performance

<p align="center">
  <img src="docs/assets/perf-chart.svg" alt="Performance comparison — Before &amp; After" width="700"/>
</p>

<p align="center">
  <img src="docs/assets/speedup-chart.svg" alt="Per-prompt decode speed" width="700"/>
</p>

### 🚀 2.2× Speedup: 73 → 161 tok/sec

| Metric | Before (v1) | After (v2) | Improvement |
|--------|:-----------:|:----------:|:-----------:|
| **Decode throughput** | 73 tok/sec | **161 tok/sec** | **2.2×** ⚡ |
| **Avg latency** | 13.79 ms/tok | **6.20 ms/tok** | 2.2× faster |
| **p50 latency** | 13.81 ms | **6.19 ms** | 2.2× faster |
| **p95 latency** | 14.49 ms | **6.85 ms** | 2.1× faster |
| **Model load** | 17 ms | 19 ms | — |
| **GPU upload** | 472 ms | 1087 ms | (f16 dequant) |
| **Output quality** | ✅ Perfect | ✅ Perfect | Maintained |

> Measured on Apple M4 Pro · Qwen2.5-0.5B Q8_0 · greedy decode · 50 tokens

### Key Optimizations Applied

| Optimization | Impact | Files |
|-------------|--------|-------|
| **simd_sum() hardware reduction** | Replaces 124-step Kahan sequential sum | `gemv.metal`, `gemv_q4k.metal` |
| **Vectorized half4/float4 loads** | 4× fewer load instructions, perfect coalescing | `gemv.metal` |
| **Kahan removal from inner loop** | -3 extra ops per element (5× fewer FLOPs) | `gemv.metal`, `gemv_q4k.metal` |
| **f16 weight dequantization** | 2× less bandwidth (2 vs 4 bytes/element) | `forward.rs` |
| **Command buffer batching** | Eliminates mid-token GPU stalls | `context.rs` |
| **GEMM inner-loop unrolling** | Better ILP for prefill path | `gemm.metal` |

> 📋 Full implementation plan for reaching >200 tok/sec: [docs/implementation-plan.md](docs/implementation-plan.md)

---

## 🚀 Quickstart

### Prerequisites

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Rust 1.78+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Xcode Command Line Tools (`xcode-select --install`)

### Download a Model

```bash
# Qwen2.5-0.5B Q8_0 (recommended for testing — 675 MB)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q8_0.gguf --local-dir ~/models/

# Qwen2.5-Coder-7B Q4_K_M (for coding tasks — 4.7 GB)
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf --local-dir ~/models/

# Download tokenizer
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    tokenizer.json --local-dir ~/models/
```

### Build & Run

```bash
# Build
cargo build --release -p bare-metal-engine

# Generate text
./target/release/generate ~/models/qwen2.5-0.5b-instruct-q8_0.gguf \
    --tokenizer ~/models/tokenizer.json \
    --prompt '<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
' \
    --tokens 100

# Start OpenAI-compatible server
./target/release/serve ~/models/qwen2.5-0.5b-instruct-q8_0.gguf \
    ~/models/tokenizer.json --port 8080

# Query the server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

### CLI Options

```
generate <model.gguf> [OPTIONS]

Options:
  --tokenizer <path>      Path to tokenizer.json
  --prompt <text>         Input prompt
  --tokens N              Max tokens to generate (default: 30)
  --temperature T         0.0 = greedy (default: 0.0)
  --top-p P               Nucleus sampling (default: 0.9)
  --top-k K               Top-K sampling (default: 50)
  --rep-penalty F         Repetition penalty (default: 1.1)
  --seed S                RNG seed (default: 42)
  --tq                    Enable TurboQuant KV cache compression
```

---

## 🏗️ Architecture

<p align="center">
  <img src="docs/assets/architecture-chart.svg" alt="Per-prompt decode speed" width="900"/>
</p>

<p align="center">
  <img src="docs/assets/docs/assets/architecture-chart.svg" alt="Arch View — Before &amp; After" width="700"/>
</p>

### Crate Structure

| Crate | Description |
|-------|-------------|
| `bare-metal-gguf` | GGUF parser — mmap, tensor metadata, zero-copy views |
| `bare-metal-tokenizer` | HuggingFace tokenizer wrapper |
| `bare-metal-kernels` | Metal shaders + Rust dispatch layer |
| `bare-metal-engine` | Model config, forward pass, KV cache, server |

### Metal GPU Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `gemv_f16w_f32in_f32out` | `gemv.metal` | GEMV with simd_sum + vectorized half4 loads |
| `gemv_q4k_f16` | `gemv_q4k.metal` | Fused Q4\_K dequant + GEMV (simd_sum) |
| `rms_norm_f32_f32_f32g` | `norm.metal` | Full f32 RMSNorm |
| `rope_inplace_f32` | `rope.metal` | Non-interleaved RoPE (Qwen2/LLaMA) |
| `decode_attention_f32` | `attention.metal` | Single-query attention with online softmax |
| `flash_attention_f16` | `attention.metal` | Tiled FlashAttention-2 for prefill |
| `silu_mul_f32` | `activation.metal` | SwiGLU activation (fused) |
| `dequant_q4k_f16` | `dequant.metal` | Q4\_K\_M GPU dequantization |
| + 20 more | various | f16/f32 variants, bias add, argmax, etc. |

---

## 🔬 Precision Pipeline

RunIT uses an **f32 residual stream** with **f16 weight bandwidth optimization** for maximum speed without quality loss:

```
Token Embedding (f32 lookup)
    │
    ▼
RMSNorm (f32 in → f32 out, f32 gamma)
    │
    ▼
Q/K/V Projections (f16 weights × f32 input → f32 output)
    │                    └── simd_sum + vectorized half4 loads
    ▼
RoPE (f32, non-interleaved pairing)
    │
    ▼
KV Cache (f32, flat layout)
    │
    ▼
Decode Attention (f32 Q·K, f32 softmax, f32 output)
    │
    ▼
FFN: gate/up → SiLU → down (f16 weights, f32 accumulation)
    │
    ▼
lm_head (f16 weights × f32 input → f32 logits)
```

> Weights dequantized from Q8\_0/Q5K/Q6K to f16 (2 bytes/element) instead of f32 (4 bytes).
> Quantized formats have ≤8 significant bits — f16's 10-bit mantissa loses nothing.

---

## 📦 Supported Models & Quantizations

| Architecture | Status | Models Tested |
|-------------|--------|---------------|
| **Qwen2** | ✅ Full support | Qwen2.5-0.5B, Qwen2.5-Coder-7B |
| **LLaMA** | ✅ Full support | LLaMA-2, LLaMA-3 |
| **Mistral** | ✅ Full support | Mistral-7B |
| **OLMoE** | ✅ MoE support | OLMoE-1B-7B |

| Quantization | Status | Notes |
|-------------|--------|-------|
| **Q8\_0** | ✅ Best quality | Dequant to f32, precise |
| **Q4\_K\_M** | ✅ Recommended | Fused GPU dequant, 3.5x less bandwidth |
| **Q6\_K** | ✅ Supported | CPU dequant to f32 |
| **Q5\_K** | ✅ Supported | CPU dequant to f32 |
| **Q5\_0** | ✅ Supported | CPU dequant to f32 |
| **Q4\_0** | ✅ Supported | CPU dequant to f32 |
| **F16** | ✅ Native | Zero-copy or memcpy |
| **BF16** | ✅ Converted | BF16 → f32 at load time |

---

## 📈 Development Roadmap

| Phase | Feature | Status |
|:-----:|---------|:------:|
| 0 | GGUF parser (mmap, tensors, metadata) | ✅ Done |
| 1 | Metal GPU kernels (GEMV, RoPE, RMSNorm, Attention) | ✅ Done |
| 2 | Model loading + config | ✅ Done |
| 3 | Transformer forward pass + KV cache | ✅ Done |
| 4 | Q4\_K\_M GPU dequantization | ✅ Done |
| 5 | TurboQuant KV-cache compression (3-4 bit) | ✅ Done |
| 6 | Text I/O — tokenizer, sampling | ✅ Done |
| 7 | OpenAI-compatible HTTP server | ✅ Done |
| 8 | Prefill batching (GEMM kernel) | ✅ Done |
| 9 | **f32 precision pipeline** | ✅ Done |
| 10 | **RoPE fix + llama.cpp parity** | ✅ **Done** |
| 11 | **GEMV optimization** (simd_sum, vectorize, f16 dequant) | ✅ **Done — 2.2× speedup** |
| 12 | simdgroup\_matrix GEMV + fused kernels | 🔜 Next |
| 13 | PagedAttention KV cache + continuous batching | 📋 Planned |

---

## 🐛 Major Bug Fix: RoPE Dimension Pairing

The engine had a critical bug where the **Rotary Position Embedding** used **interleaved** pairing `(0,1), (2,3)...` instead of the correct **non-interleaved** pairing `(i, i+d/2)` used by Qwen2/LLaMA/Mistral.

**Why it was hidden:** At position 0, RoPE is the identity rotation (`cos(0)=1, sin(0)=0`) regardless of pairing — so single-token tests always passed. The bug only manifested at position ≥ 1, scrambling the positional encoding and corrupting multi-token attention.

```
Before (wrong):  pairs (0,1) (2,3) (4,5) ... → scrambled positions at pos≥1
After (correct): pairs (0,32) (1,33) (2,34) ... → perfect llama.cpp match
```

> 📋 Full investigation: [docs/COMPARISON.md](docs/COMPARISON.md) · [docs/STATUS_REPORT.md](docs/STATUS_REPORT.md)

---

## 📂 Repository Layout

```
RunIT/
├── crates/
│   ├── gguf/           # GGUF file parser (mmap, zero-copy)
│   ├── tokenizer/      # HuggingFace tokenizer wrapper
│   ├── kernels/        # Metal shaders + Rust dispatch
│   │   ├── shaders/    # .metal source → .metallib at build time
│   │   │   ├── gemv.metal          # 14 GEMV variants (simd_sum + half4/float4)
│   │   │   ├── gemv_q4k.metal      # Fused Q4K GEMV (6 variants)
│   │   │   ├── attention.metal     # FlashAttention-2 + decode attention
│   │   │   ├── rope.metal          # RoPE (non-interleaved, f16/f32/batch)
│   │   │   ├── norm.metal          # RMSNorm (4 variants)
│   │   │   ├── activation.metal    # SwiGLU, add, argmax, KV scatter
│   │   │   ├── dequant.metal       # Q4K GPU dequantization
│   │   │   ├── gemm.metal          # Tiled GEMM for prefill
│   │   │   └── turboquant.metal    # TurboQuant KV compression
│   │   └── src/        # MetalContext, dispatch, error types
│   └── engine/         # Model loader, forward pass, server
│       └── src/
│           ├── forward.rs       # Executor + 8 quantization decoders
│           ├── kv_cache.rs      # f32 flat KV cache
│           ├── sampler.rs       # temp/top-k/top-p/rep-penalty
│           ├── server/          # OpenAI-compatible HTTP (axum, SSE)
│           └── bin/
│               ├── generate.rs  # CLI text generation + benchmark
│               └── serve.rs     # HTTP server binary
├── docs/
│   ├── assets/          # Logo, charts, diagrams
│   ├── ARCHITECTURE.md  # System design and data-flow
│   ├── COMPARISON.md    # llama.cpp comparison + test results
│   ├── KERNELS.md       # Metal kernel reference
│   └── STATUS_REPORT.md # Bug fixes + precision improvements
└── README.md
```

---

## 🔧 Development

```bash
# Check all crates
cargo check --workspace

# Run tests
cargo test --workspace

# Build release
cargo build --release -p bare-metal-engine

# Debug: per-position logits during prompt processing
DEBUG_LOGITS=1 ./target/release/generate <model.gguf> \
    --tokenizer <tokenizer.json> --tokens 10 \
    --prompt '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n'

# Debug: per-layer hidden state dump
DEBUG_LAYERS=1 ./target/release/generate <model.gguf> \
    --tokenizer <tokenizer.json> --tokens 1 --prompt 'Hello'
```

---

## 📄 License

MIT

---

<p align="center">
  <strong>Built with 🦀 Rust and ⚡ Metal</strong><br/>
  <em>For Apple Silicon, from the ground up</em>
</p>
