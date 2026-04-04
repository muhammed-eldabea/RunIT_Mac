# llama.cpp vs RunIT Engine — Side-by-Side Comparison

**Model:** Qwen2.5-0.5B Q8_0  
**Hardware:** Apple M4 Pro, macOS 25.4.0  
**Date:** 2026-04-04  
**llama.cpp:** b8640 (homebrew)  
**Fair test:** `llama-completion -no-cnv` (same 15-token prompt, no double-wrapping)

---

## ✅ FIXED — Results Now Match llama.cpp

| # | Prompt | llama.cpp | RunIT Engine | Match? |
| --- | ------ | --------- | ------------ | ------ |
| 1 | "What is 2+2?" | **2 + 2 equals 4.** | **2 + 2 equals 4.** | ✅ YES |
| 2 | "Capital of France?" | **The capital of France is Paris.** | **The capital of France is Paris.** | ✅ YES |
| 3 | "Capital of Japan?" | **The capital of Japan is Tokyo.** | **The capital of Japan is Tokyo.** | ✅ YES |
| 4 | "What is machine learning?" | Machine learning is a subset of AI... (80+ tokens, coherent) | Machine learning is a subset of AI... (92 tokens, coherent, hits EOS) | ✅ YES |
| 5 | "Translate to French" | **Bonjour, comment ça va ?** | **Bonjour, comment ça va ?** | ✅ YES |

### Per-Position Logit Verification

Top-1 token matches at every prompt position (15/15):

| Pos | Token | llama.cpp Top-1 | RunIT Top-1 | Match |
|-----|-------|----------------|-------------|-------|
| 0 | 151644 | 72030 (12.58) | 72030 (13.12) | ✅ |
| 1 | 872 | 16 (15.20) | 16 (15.53) | ✅ |
| 2 | 198 | 18493 (13.61) | 18493 (13.64) | ✅ |
| 3 | 3838 | 374 (20.69) | 374 (20.80) | ✅ |
| 4 | 374 | 279 (17.05) | 279 (17.00) | ✅ |
| 5 | 220 | 16 (19.80) | 16 (19.71) | ✅ |
| 6 | 17 | 15 (17.01) | 15 (16.98) | ✅ |
| 7 | 10 | 17 (20.51) | 17 (20.65) | ✅ |
| 8 | 17 | 30 (18.47) | 30 (18.46) | ✅ |
| 9 | 30 | 715 (16.15) | 715 (16.16) | ✅ |
| 10 | 151645 | 872 (20.02) | 872 (19.98) | ✅ |
| 11 | 198 | 872 (17.05) | 872 (17.60) | ✅ |
| 12 | 151644 | 872 (20.70) | 872 (20.68) | ✅ |
| 13 | 77091 | 198 (21.03) | 198 (21.09) | ✅ |
| 14 | 198 | 17 (24.69) | 17 (24.74) | ✅ |

---

## Root Cause: RoPE Pairing Bug (FIXED)

### The bug

The Rotary Position Embedding (RoPE) kernel used **interleaved** dimension pairing:
```
pairs: (0,1), (2,3), (4,5), ...  ← WRONG for Qwen2
```

Qwen2 (and LLaMA, Mistral, etc.) uses **non-interleaved** pairing:
```
pairs: (i, i+d/2) for i < d/2  ← CORRECT
i.e.: (0,32), (1,33), (2,34), ...  for head_dim=64
```

### Why it wasn't caught earlier

At **position 0**, the RoPE rotation angle is `freq = 0 / theta^(2i/d) = 0`, giving `cos(0)=1, sin(0)=0`. This is the **identity rotation** regardless of pairing. So single-token inference (BOS-only) always produced correct results.

The bug only manifested at position ≥ 1, where the actual rotation is applied with different angles to different dimension pairs. Wrong pairing completely scrambles the positional encoding, causing attention to weight past tokens incorrectly.

### The fix

Changed all three RoPE kernels (`rope_inplace_f16`, `rope_inplace_f32`, `rope_batch_inplace_f16`) to use non-interleaved pairing:

```metal
// Before (interleaved — WRONG):
uint base = (tok_idx * n_heads + head) * head_dim + pair * 2;
x0 = x[base];       // element 2*pair
x1 = x[base + 1];   // element 2*pair + 1

// After (non-interleaved — CORRECT):
uint half_dim = head_dim / 2;
uint head_base = (tok_idx * n_heads + head) * head_dim;
uint idx0 = head_base + pair;              // element pair
uint idx1 = head_base + pair + half_dim;   // element pair + d/2
x0 = x[idx0];
x1 = x[idx1];
```

---

## Previous Investigation (precision) — Confirmed NOT the Issue

All precision upgrades were retained but confirmed to be secondary:

| Change | Effect on logits |
| ------ | ---------------- |
| f32 Q/K/V + f32 KV cache + f32 attention | < 0.01 shift |
| f32 FFN intermediates (gate/up/silu/down) | < 0.01 shift |
| f32 logits from lm_head | < 0.01 shift |
| Kahan compensated summation in GEMV | < 0.01 shift |
| Contiguous block assignment (vs strided) | < 0.01 shift |
| f32 token embedding (vs f16) | < 0.01 shift |
| f32 RMSNorm gamma weights | < 0.01 shift |

The real issue was always the RoPE pairing — a logical bug, not a precision issue.

---

## What's Implemented

### Engine (all working, verified)

1. **f32 attention pipeline** — Q/K/V projections, RoPE, KV cache, decode attention, output projection
2. **f32 FFN pipeline** — gate/up projections, SiLU activation, down projection
3. **f32 logits** — lm_head outputs f32 for precise final token ranking
4. **f32 embeddings** — token embedding dequanted to f32
5. **f32 norm weights** — RMSNorm gamma in f32 precision
6. **Kahan compensated summation** — in all GEMV kernels
7. **Contiguous block accumulation** — matching llama.cpp's sequential element processing
8. **Non-interleaved RoPE** — correct (i, i+d/2) pairing for Qwen2/LLaMA/Mistral
9. **Repetition penalty** (default 1.1) — prevents generation loops
10. **min_tokens EOS suppression** (default 3) — prevents premature stops
11. **f32 sampler** — operates on f32 logits throughout

---

## Performance

| Metric | llama.cpp | RunIT Engine |
| ------ | --------- | ------------ |
| Decode speed | ~230 tok/sec | ~72 tok/sec |
| Prompt speed | ~2500 tok/sec | ~72 tok/sec (no batched prefill) |

Performance gap is expected: llama.cpp has optimized ARM NEON GEMV and batched prefill. RunIT uses Metal GPU with f32 precision paths (2x bandwidth vs f16).

---

## Reproduce These Tests

### Prerequisites

```bash
# Models (download once)
# qwen2.5-0.5b-q8_0.gguf  → ~/models/
# qwen-tokenizer.json      → ~/models/

# llama.cpp (homebrew)
brew install llama.cpp

# RunIT Engine (build from source)
cargo build --release -p bare-metal-engine
```

### llama.cpp commands

```bash
# Test 1: Math
echo "" | llama-completion -m ~/models/qwen2.5-0.5b-q8_0.gguf --temp 0 -n 32 \
  --no-display-prompt -no-cnv -p '<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
' 2>/dev/null

# Test 2: France
echo "" | llama-completion -m ~/models/qwen2.5-0.5b-q8_0.gguf --temp 0 -n 32 \
  --no-display-prompt -no-cnv -p '<|im_start|>user
Capital of France?<|im_end|>
<|im_start|>assistant
' 2>/dev/null

# Test 3: Japan
echo "" | llama-completion -m ~/models/qwen2.5-0.5b-q8_0.gguf --temp 0 -n 32 \
  --no-display-prompt -no-cnv -p '<|im_start|>user
What is the capital of Japan?<|im_end|>
<|im_start|>assistant
' 2>/dev/null

# Test 4: Machine Learning
echo "" | llama-completion -m ~/models/qwen2.5-0.5b-q8_0.gguf --temp 0 -n 100 \
  --no-display-prompt -no-cnv -p '<|im_start|>user
What is machine learning?<|im_end|>
<|im_start|>assistant
' 2>/dev/null

# Test 5: Translation
echo "" | llama-completion -m ~/models/qwen2.5-0.5b-q8_0.gguf --temp 0 -n 32 \
  --no-display-prompt -no-cnv -p '<|im_start|>user
Translate to French: Hello, how are you?<|im_end|>
<|im_start|>assistant
' 2>/dev/null
```

### RunIT Engine commands

```bash
# Test 1-5: same pattern
./target/release/generate ~/models/qwen2.5-0.5b-q8_0.gguf \
  --tokenizer ~/models/qwen-tokenizer.json --tokens 32 --rep-penalty 1.0 \
  --prompt '<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
'
# (replace prompt text for each test)
```

### Debug: per-position logits during prompt

```bash
DEBUG_LOGITS=1 ./target/release/generate ~/models/qwen2.5-0.5b-q8_0.gguf \
  --tokenizer ~/models/qwen-tokenizer.json --tokens 10 --rep-penalty 1.0 \
  --prompt '<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
'
```
