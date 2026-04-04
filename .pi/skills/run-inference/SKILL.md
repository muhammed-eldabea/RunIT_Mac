---
name: run-inference
description: Run LLM inference with the bare-metal engine. Test text generation with various prompts, sampling parameters, and model variants. Use for quick generation testing.
---

# Run Inference

You are running inference on a bare-metal LLM engine targeting Apple Silicon Metal GPUs.

## Model Location

Models are GGUF files. Common locations:
- `~/.cache/lm-studio/models/` — LM Studio cache
- `~/models/` — manual downloads
- Check for `.gguf` files the user has available

## Generation Command

The binary is `generate` in the engine crate:

```bash
cargo run --release --bin generate -- \
  --model <path-to-model.gguf> \
  --prompt "<prompt text>" \
  --max-tokens <N> \
  --temperature <float> \
  --top-p <float> \
  --top-k <int> \
  --repetition-penalty <float> \
  2>&1
```

### Quick test (defaults)
```bash
cargo run --release --bin generate -- \
  --model <path-to-model.gguf> \
  --prompt "What is machine learning?" \
  --max-tokens 100 \
  2>&1
```

### Creative generation
```bash
cargo run --release --bin generate -- \
  --model <path-to-model.gguf> \
  --prompt "Write a poem about the ocean" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95 \
  --repetition-penalty 1.1 \
  2>&1
```

### Deterministic (greedy)
```bash
cargo run --release --bin generate -- \
  --model <path-to-model.gguf> \
  --prompt "The capital of France is" \
  --max-tokens 50 \
  --temperature 0.0 \
  2>&1
```

## Supported Models

The engine targets Qwen2.5 architecture (Qwen2.5-Coder-7B-Instruct). Quantization variants:
- **Q4_K_M** — 4-bit quantized, ~4GB, fastest
- **Q6_K** — 6-bit quantized, ~5.5GB, better quality
- **Q8_0** — 8-bit quantized, ~7.5GB, best quantized quality
- **F16** — full half precision, ~14GB

## Evaluating Output Quality

When testing generation, assess:
1. **Coherence**: Does the text make grammatical sense?
2. **Relevance**: Does it answer the prompt?
3. **Repetition**: Any loops or repeated phrases? (increase repetition_penalty)
4. **Length**: Does it stop naturally (EOS) or hit max_tokens?
5. **Tokens/sec**: Check the benchmark output for performance

## Chat Template

For chat-style prompts (Qwen2 format), the engine wraps prompts automatically:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

## After Running

Report:
1. The generated text
2. Token count and tokens/sec
3. Whether output was coherent and relevant
4. Any issues (repetition, truncation, garbage tokens)
5. Suggested parameter adjustments if quality is poor
