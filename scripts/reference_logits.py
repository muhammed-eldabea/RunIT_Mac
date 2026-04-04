#!/usr/bin/env python3
"""Get reference per-token logits from llama-cpp-python for comparison."""
import numpy as np
from llama_cpp import Llama

MODEL = "/Users/mohamedeldabaa/models/qwen2.5-0.5b-q8_0.gguf"

# Load with logits_all=True to get per-position logits
llm = Llama(model_path=MODEL, n_ctx=64, n_gpu_layers=0, verbose=False, logits_all=True)

# Same prompt tokens as our engine
prompt_tokens = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

# Process all at once
llm.reset()
llm.eval(prompt_tokens)

# Get logits at last position (predicting next token after full prompt)
n_vocab = llm.n_vocab()
print(f"Vocab size: {n_vocab}")

print("\n--- Per-position top-5 logits ---")
for pos in range(len(prompt_tokens)):
    scores = np.array(llm.scores[pos][:n_vocab])
    top5 = np.argsort(scores)[::-1][:5]
    vals = [f"{t}({scores[t]:.2f})" for t in top5]
    print(f"  pos {pos:2d} (tok {prompt_tokens[pos]:6d}): {' '.join(vals)}")

# Also do token-by-token eval for direct comparison
print("\n--- Token-by-token eval top-5 logits ---")
llm.reset()
for pos in range(len(prompt_tokens)):
    llm.eval([prompt_tokens[pos]])
    scores = np.array(llm.scores[0][:n_vocab])
    top5 = np.argsort(scores)[::-1][:5]
    vals = [f"{t}({scores[t]:.2f})" for t in top5]
    print(f"  pos {pos:2d} (tok {prompt_tokens[pos]:6d}): {' '.join(vals)}")
