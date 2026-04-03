# Metal Kernel Reference

All shaders live in `crates/kernels/shaders/` and are compiled to a single `.metallib`
at build time by `crates/kernels/build.rs` via `xcrun metal` + `xcrun metallib`.

The Rust dispatch layer in `crates/kernels/src/dispatch.rs` wraps each kernel with a
type-safe function signature.

---

## Kernel Index

| Kernel name | File | Purpose |
|-------------|------|---------|
| `gemv_f16` | `gemv.metal` | Matrix-vector multiply (decode path) |
| `rms_norm_f16` | `norm.metal` | RMS normalisation |
| `rope_inplace_f16` | `rope.metal` | Rotary position embeddings (in-place) |
| `silu_mul_f16` | `elementwise.metal` | SwiGLU: silu(gate) × up |
| `add_f16` | `elementwise.metal` | Elementwise add (residuals) |
| `mul_f16` | `elementwise.metal` | Elementwise multiply |
| `flash_attention_f16` | `attention.metal` | Tiled FlashAttention (GQA-aware) |
| `dequant_q4k_f16` | `dequant.metal` | Q4\_K\_M → F16 dequantisation |

---

## gemv\_f16

**Purpose:** General matrix-vector multiply `y = W × x` for the decode path (batch = 1).

**Signature:**
```rust
pub fn gemv_f16(ctx, W: &Buffer, x: &Buffer, y: &Buffer, rows: u32, cols: u32) -> Result<()>
```

**Grid:** `[rows, 1, 1]`, threadgroup `[TG_GEMV=32, 1, 1]`

Each threadgroup computes one output row using a SIMD-lane reduction over `cols` elements.
W is `[rows × cols]` f16, x is `[cols]` f16, y is `[rows]` f16.

---

## rms\_norm\_f16

**Purpose:** RMS normalisation: `y[i] = x[i] / sqrt(mean(x²) + eps) * w[i]`

**Signature:**
```rust
pub fn rms_norm_f16(ctx, x: &Buffer, y: &Buffer, w: &Buffer, eps: f32, n: u32, batch: u32) -> Result<()>
```

**Grid:** `[batch, 1, 1]`, threadgroup `[TG_ELEM=256, 1, 1]`

Threads cooperate to compute the row mean-of-squares via threadgroup reduction, then
scale each element in a second pass.

---

## rope\_inplace\_f16

**Purpose:** Apply rotary position embeddings to Q or K vectors in-place.

**Signature:**
```rust
pub fn rope_inplace_f16(ctx, x: &Buffer, head_dim: u32, n_heads: u32, pos: u32, theta: f32, batch: u32) -> Result<()>
```

**Grid:** `[n_heads × (head_dim/2), batch, 1]`, threadgroup `[1, 1, 1]`

Each thread handles one (cos, sin) rotation pair. Position `pos` is the absolute
sequence position; `theta` is the RoPE base frequency (10 000 for Qwen2.5).

---

## silu\_mul\_f16

**Purpose:** SwiGLU activation: `out[i] = silu(gate[i]) × up[i]`

where `silu(x) = x / (1 + exp(-x))`.

**Signature:**
```rust
pub fn silu_mul_f16(ctx, gate: &Buffer, up: &Buffer, out: &Buffer, n: u32) -> Result<()>
```

**Grid:** `[n, 1, 1]`, threadgroup `[TG_ELEM=256, 1, 1]`

---

## add\_f16

**Purpose:** Elementwise add: `out[i] = a[i] + b[i]` (used for residual connections).

**Signature:**
```rust
pub fn add_f16(ctx, a: &Buffer, b: &Buffer, out: &Buffer, n: u32) -> Result<()>
```

**Grid:** `[n, 1, 1]`, threadgroup `[TG_ELEM=256, 1, 1]`

`a` and `out` may alias (in-place add is safe).

---

## mul\_f16

**Purpose:** Elementwise multiply: `out[i] = a[i] * b[i]`.

**Signature:**
```rust
pub fn mul_f16(ctx, a: &Buffer, b: &Buffer, out: &Buffer, n: u32) -> Result<()>
```

Same grid/threadgroup as `add_f16`.

---

## flash\_attention\_f16

**Purpose:** Tiled FlashAttention with GQA (grouped-query attention) support.

**Signature:**
```rust
pub fn flash_attention_f16(
    ctx,
    q:      &Buffer,   // [batch, q_len,  num_heads,    head_dim]  f16
    k:      &Buffer,   // [batch, kv_len, num_kv_heads, head_dim]  f16  (KV cache)
    v:      &Buffer,   // same as k
    out:    &Buffer,   // [batch, q_len,  num_heads,    head_dim]  f16
    batch:  u32,
    q_len:  u32,       // 1 for decode, >1 for prefill
    kv_len: u32,       // total filled KV cache length
    num_heads:    u32,
    num_kv_heads: u32,
    head_dim:     u32,
) -> Result<()>
```

**Grid:** `[ceil(q_len / BLOCK_Q), num_heads, batch]`, threadgroup `[BLOCK_Q=32, 1, 1]`

Implements the online softmax + tiled accumulation from the FlashAttention paper.
GQA is handled by mapping each Q head to its KV group:
`kv_head = q_head / (num_heads / num_kv_heads)`.

Scale factor: `1 / sqrt(head_dim)` applied inside the kernel.

---

## dequant\_q4k\_f16

**Purpose:** Dequantise a Q4\_K\_M weight tensor from packed GGML format to F16.

**Signature:**
```rust
pub fn dequant_q4k_f16(ctx, input: &Buffer, output: &Buffer, n_blocks: u32) -> Result<()>
```

- `input`:   raw Q4\_K bytes — `n_blocks × 144` bytes
- `output`:  pre-allocated F16 buffer — `n_blocks × 256 × 2` bytes
- Called once at load time; result stored in GPU memory as F16.

**Grid:** `[n_blocks, 1, 1]`, threadgroup `[256, 1, 1]`

Each thread (lid 0..255) processes one element within its block.

**Block layout (144 bytes / 256 elements):**

```
 bytes  0–1:   d     (f16) super-block scale
 bytes  2–3:   dmin  (f16) super-block min
 bytes  4–15:  scales[12]  — 6-bit sc + 6-bit m for each of 8 sub-blocks
 bytes 16–143: qs[128]     — 4-bit nibbles, 2 per byte
```

**Scale extraction** for sub-block index `j` (`get_scale_min_k4`):

```metal
if (j < 4) {
    sc = scales[j]   & 0x3F;
    m  = scales[j+4] & 0x3F;
} else {
    sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
    m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4);
}
```

**Dequant formula:**

```
value = (d * sc) * nibble  -  (dmin * m)
```

where `nibble` is the 4-bit quantised value (0–15).

---

## Build System

`crates/kernels/build.rs` compiles all `.metal` files into a single `.metallib`:

```bash
xcrun -sdk macosx metal -c shaders/*.metal -o /tmp/kernels.air
xcrun -sdk macosx metallib /tmp/kernels.air -o <out_dir>/kernels.metallib
```

The output path is passed to Rust via `cargo:rustc-env=METALLIB_PATH=...`.
`MetalContext::new()` loads the library at runtime with `device.new_library_with_file()`.

On non-macOS systems (CI Linux), `build.rs` emits a `cargo:warning=` and skips
compilation; all Metal code is `#[cfg(target_os = "macos")]` gated.

---

## Adding a New Kernel

1. Add the `.metal` shader to `crates/kernels/shaders/`.
2. Add the kernel function name string to `KERNEL_NAMES` in `crates/kernels/src/context.rs`.
3. Add a dispatch wrapper in `crates/kernels/src/dispatch.rs`.
4. Add a `KernelOp` variant in `crates/kernels/src/lib.rs` (for documentation).
5. Use the dispatch function from `crates/engine/src/forward.rs`.
