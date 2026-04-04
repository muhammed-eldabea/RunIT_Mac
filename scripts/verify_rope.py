#!/usr/bin/env python3
"""Verify which RoPE pairing Qwen2 uses."""
import numpy as np

# Check: does Qwen2 use interleaved (pairs 0,1; 2,3) 
# or non-interleaved (pairs i, i+d/2)?

# Hugging Face Qwen2 uses rotate_half which is NON-INTERLEAVED:
# rotate_half(x) = [-x[d/2:], x[:d/2]]
# apply_rotary: q' = q * cos + rotate_half(q) * sin
# This pairs (i, i+d/2) for i < d/2

# Our kernel uses INTERLEAVED: pairs (2i, 2i+1)

# Let's verify with a simple example: head_dim=4
# Non-interleaved pairs: (0,2) and (1,3)
# Interleaved pairs: (0,1) and (2,3)

hd = 8
theta = 1000000.0
pos = 1

x = np.arange(hd, dtype=np.float32) + 1.0  # [1,2,3,4,5,6,7,8]

# Interleaved (our kernel)
def rope_interleaved(x, pos, theta, hd):
    out = x.copy()
    for pair in range(hd // 2):
        freq = pos / (theta ** (2 * pair / hd))
        c, s = np.cos(freq), np.sin(freq)
        x0, x1 = x[2*pair], x[2*pair+1]
        out[2*pair]   = x0 * c - x1 * s
        out[2*pair+1] = x1 * c + x0 * s
    return out

# Non-interleaved (Qwen2/LLaMA standard)
def rope_non_interleaved(x, pos, theta, hd):
    out = x.copy()
    half = hd // 2
    for i in range(half):
        freq = pos / (theta ** (2 * i / hd))
        c, s = np.cos(freq), np.sin(freq)
        x0, x1 = x[i], x[i + half]
        out[i]        = x0 * c - x1 * s
        out[i + half]  = x1 * c + x0 * s
    return out

print(f"Input: {x}")
print(f"Interleaved (our kernel):    {rope_interleaved(x, pos, theta, hd)}")
print(f"Non-interleaved (Qwen2):     {rope_non_interleaved(x, pos, theta, hd)}")
print(f"\nAt pos=0 (identity):")
print(f"Interleaved:    {rope_interleaved(x, 0, theta, hd)}")
print(f"Non-interleaved: {rope_non_interleaved(x, 0, theta, hd)}")
print(f"\nBoth are identity at pos=0, but diverge at pos=1!")
