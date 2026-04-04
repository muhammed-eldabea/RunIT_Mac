---
name: inspect-model
description: Inspect GGUF model files — view architecture, tensor inventory, quantization types, memory footprint, and metadata. Use when loading new models or debugging weight issues.
---

# Inspect GGUF Model

You are inspecting GGUF model files used by the bare-metal LLM inference engine.

## Inspect Binary

```bash
cargo run --release --bin inspect -- --model <path-to-model.gguf> 2>&1
```

This shows:
- Model architecture and hyperparameters
- Tensor count and names
- Quantization types per tensor
- Total memory footprint
- Config extracted from GGUF metadata

## Key Metadata Fields

| Field | Description |
|-------|-------------|
| `general.architecture` | Model family (e.g., "qwen2") |
| `general.name` | Model name |
| `qwen2.block_count` | Number of transformer layers |
| `qwen2.embedding_length` | Hidden dimension |
| `qwen2.attention.head_count` | Number of attention heads |
| `qwen2.attention.head_count_kv` | Number of KV heads (GQA) |
| `qwen2.feed_forward_length` | FFN intermediate size |
| `qwen2.rope.freq_base` | RoPE theta base frequency |
| `qwen2.context_length` | Maximum sequence length |
| `tokenizer.ggml.model` | Tokenizer type |

## Tensor Naming Convention

```
blk.{layer}.attn_q.weight     — Query projection
blk.{layer}.attn_k.weight     — Key projection
blk.{layer}.attn_v.weight     — Value projection
blk.{layer}.attn_output.weight — Output projection
blk.{layer}.ffn_gate.weight   — FFN gate (SwiGLU)
blk.{layer}.ffn_up.weight     — FFN up projection
blk.{layer}.ffn_down.weight   — FFN down projection
blk.{layer}.attn_norm.weight  — Attention layer norm
blk.{layer}.ffn_norm.weight   — FFN layer norm
token_embd.weight              — Token embedding table
output_norm.weight             — Final RMS norm
output.weight                  — Language model head
```

## Quantization Types

| Type | Bits/weight | Block size | Notes |
|------|-------------|------------|-------|
| F32 | 32 | 1 | Full precision |
| F16 | 16 | 1 | Half precision |
| Q8_0 | 8 | 32 | 8-bit quantized |
| Q6_K | 6.5 | 256 | 6-bit K-quant |
| Q4_K_M | 4.8 | 256 | 4-bit K-quant medium |
| Q4_K_S | 4.5 | 256 | 4-bit K-quant small |

## Memory Estimation

For a given model:
- **Weights**: Sum of tensor sizes based on quantization
- **KV Cache**: `2 × num_layers × num_kv_heads × max_seq × head_dim × 2 bytes (f16)`
- **Activations**: `hidden_dim × max_batch × 4 bytes` (temporary, reusable)
- **Total**: Weights + KV Cache + Activations + overhead

Example (Qwen2.5-7B Q4K, 2048 ctx):
- Weights: ~4.0 GB
- KV Cache: ~470 MB
- Working memory: ~100 MB
- **Total: ~4.6 GB unified memory**

## After Inspecting

Report:
1. Model architecture and size
2. Quantization type distribution
3. Estimated memory requirement
4. Whether the model is compatible with the engine (must be Qwen2 architecture)
5. Any unusual or missing tensors
