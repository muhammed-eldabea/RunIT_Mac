#!/usr/bin/env python3
"""Compare per-position logits between RunIT and llama.cpp reference."""
import numpy as np
from llama_cpp import Llama

MODEL = "/Users/mohamedeldabaa/models/qwen2.5-0.5b-q8_0.gguf"
llm = Llama(model_path=MODEL, n_ctx=64, n_gpu_layers=0, verbose=False, logits_all=True)

tokens = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
llm.reset()
llm.eval(tokens)

# Our engine results (from DEBUG_LOGITS run)
our_top1 = {
    0: (72030, 13.12), 1: (16, 15.53), 2: (18493, 13.64), 3: (374, 20.80),
    4: (279, 17.00), 5: (16, 19.71), 6: (15, 16.98), 7: (17, 20.65),
    8: (30, 18.46), 9: (715, 16.16), 10: (872, 19.98), 11: (872, 17.60),
    12: (872, 20.68), 13: (198, 21.09), 14: (17, 24.74)
}

print(f"{'Pos':>3} {'Ref Top1':>10} {'Ref Logit':>10} {'Our Top1':>10} {'Our Logit':>10} {'Match':>6}")
print("-" * 55)
for pos in range(len(tokens)):
    scores = np.array(llm.scores[pos][:llm.n_vocab()])
    ref_top = int(np.argmax(scores))
    ref_val = scores[ref_top]
    our_tok, our_val = our_top1[pos]
    match = "✅" if ref_top == our_tok else "❌"
    print(f"{pos:3d} {ref_top:10d} {ref_val:10.2f} {our_tok:10d} {our_val:10.2f} {match:>6}")
