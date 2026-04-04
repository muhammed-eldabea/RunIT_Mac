---
name: validate-output
description: Validate engine output against reference implementations. Compare bare-metal results with Candle or llama.cpp. Use when verifying correctness after changes.
---

# Validate Output

You are validating that the bare-metal inference engine produces correct outputs by comparing against reference implementations.

## Reference Backends

### 1. Candle Reference (built-in)
The `bare-metal-reference` crate wraps Candle (Hugging Face's Rust ML framework) for CPU/Metal validation:
```bash
cargo run -p bare-metal-reference -- --model <path> --prompt "test" 2>&1
```

### 2. llama.cpp (external)
If `llama-cli` is installed:
```bash
llama-cli -m <path-to-model.gguf> -p "What is 2+2?" -n 50 --temp 0 2>&1
```

### 3. Integration test with synthetic model
```bash
cargo test -p bare-metal-engine -- qwen2_load_test --nocapture 2>&1
```

## Validation Workflow

### Step 1: Greedy decode comparison
Use temperature=0 for deterministic output on both engines:
```bash
# Our engine
cargo run --release --bin generate -- \
  --model <path> --prompt "The capital of France is" \
  --max-tokens 20 --temperature 0.0 2>&1

# llama.cpp reference
llama-cli -m <path> -p "The capital of France is" -n 20 --temp 0 2>&1
```
Outputs should be token-identical for greedy decoding.

### Step 2: Logit comparison
For deeper validation, compare raw logits before sampling:
1. Add temporary logging in `forward.rs` to dump top-10 logits after lm_head
2. Compare against Candle reference logits
3. Tolerance: ±0.01 for f16, ±0.001 for f32

### Step 3: Per-layer validation
For tracking down divergence:
1. Dump activations after each transformer layer
2. Compare layer-by-layer against reference
3. First layer that diverges reveals the bug

## Quality Test Suite

Run these prompts and check for issues:

| Prompt | Expected Behavior |
|--------|-------------------|
| "What is 2+2?" | Short, correct answer |
| "Capital of France?" | "Paris" |
| "Write a poem about the ocean" | Multi-sentence, creative, no repetition |
| "What is machine learning?" | Factual, multi-sentence explanation |
| "Translate hello to French" | "Bonjour" |
| "" (empty) | Should not crash, may produce EOS |
| Very long prompt (>1000 tokens) | Should handle without OOM |

## Numerical Tolerance Guide

| Comparison | Acceptable Tolerance |
|------------|---------------------|
| F32 logits | ±0.001 |
| F16 logits | ±0.01 |
| Q8_0 logits | ±0.1 |
| Q4_K_M logits | ±0.5 |
| Token identity (greedy) | Exact match |
| Token identity (sampled) | Not comparable (stochastic) |

## After Validation

Report:
1. Whether outputs match reference within tolerance
2. If diverged: at which layer/operation
3. Quality assessment of generated text
4. Token-per-second comparison between engines
5. Recommended next steps for any issues found
