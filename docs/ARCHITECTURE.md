# Architecture

This document describes the full system design of the bare-metal MX LLM inference engine.

---

## Crate Structure

```
bare-metal-gguf      — GGUF file parser (no Metal dependency)
bare-metal-kernels   — Metal shaders + Rust dispatch layer  [macOS only]
bare-metal-engine    — Model loader, config, forward pass   [macOS only]
```

All inter-crate dependencies are one-directional: `engine → kernels → (system Metal)` and
`engine → gguf`.

---

## Data Flow

### Load Time

```
GGUF file
  │  mmap()  →  &[u8] view, zero heap copy
  ▼
bare_metal_gguf::GgufFile
  ├── metadata: HashMap<String, GgufValue>  (hyperparams, rope theta, …)
  └── tensors:  HashMap<String, TensorBuffer>
                 name → { ptr: *const u8, size, dtype, shape }

ModelConfig::from_gguf()   — extracts architecture hyperparameters

Executor::new(ctx, model)
  for each weight tensor:
    ├── F16  → zero-copy if page-aligned, else memcpy  →  Metal Buffer (StorageModeShared)
    └── Q4K  → memcpy raw bytes  →  GPU dequant kernel  →  F16 Metal Buffer
```

### Decode (per token)

```
token_id (u32)
  │
  ▼ embedding lookup  (CPU memcpy from tok_emb buffer)
  x  [hidden_size]  f16
  │
  ▼  ╔══ for layer l = 0..N ══════════════════════════════════════╗
  │  ║  RMSNorm(x, w_attn_norm)  →  x_norm                       ║
  │  ║                                                             ║
  │  ║  GEMV(w_q, x_norm) + bias_q  →  q  [num_heads × head_dim] ║
  │  ║  GEMV(w_k, x_norm) + bias_k  →  k  [num_kv_heads × head_dim] ║
  │  ║  GEMV(w_v, x_norm) + bias_v  →  v  [num_kv_heads × head_dim] ║
  │  ║                                                             ║
  │  ║  RoPE(q, pos)  in-place                                    ║
  │  ║  RoPE(k, pos)  in-place                                    ║
  │  ║                                                             ║
  │  ║  KvCache.update(l, k, v)   — write into unified buffer     ║
  │  ║                                                             ║
  │  ║  FlashAttention(q, KvCache.k[l], KvCache.v[l])             ║
  │  ║    q_len=1, kv_len=pos+1                                   ║
  │  ║    GQA-aware: num_heads / num_kv_heads groups              ║
  │  ║  →  attn_out  [num_heads × head_dim]                       ║
  │  ║                                                             ║
  │  ║  GEMV(w_o, attn_out)  →  proj  [hidden_size]               ║
  │  ║  x  +=  proj          (residual)                           ║
  │  ║                                                             ║
  │  ║  RMSNorm(x, w_ffn_norm)  →  x_norm2                        ║
  │  ║  GEMV(w_gate, x_norm2)  →  gate  [intermediate_size]       ║
  │  ║  GEMV(w_up,   x_norm2)  →  up   [intermediate_size]        ║
  │  ║  SwiGLU(gate, up)       →  act  [intermediate_size]        ║
  │  ║  GEMV(w_down, act)      →  ffn_out  [hidden_size]          ║
  │  ║  x  +=  ffn_out         (residual)                         ║
  │  ╚═════════════════════════════════════════════════════════════╝
  │
  ▼ RMSNorm(x, output_norm)  →  x_final
  ▼ GEMV(lm_head, x_final)   →  logits  [vocab_size]  f16
```

---

## KV Cache

**Layout per layer:**

```
k_buf[l]: Metal Buffer (StorageModeShared)
  shape: [num_kv_heads, max_seq_len, head_dim]  f16
  size:  num_kv_heads × max_seq_len × head_dim × 2 bytes

v_buf[l]: same layout
```

For Qwen2.5-7B (28 layers, 8 KV heads, 128 head_dim, 4096 max_seq):
- Per layer: 8 × 4096 × 128 × 2 × 2 = 16.8 MB
- Total: 28 × 16.8 MB ≈ **470 MB**

Buffers are CPU-writable via unified memory (`StorageModeShared`). The GPU reads them
directly in the FlashAttention kernel — no explicit copy needed.

`KvCache::update()` writes the new K/V row for the current position using
`ptr::copy_nonoverlapping` into the buffer's `contents()` pointer. `KvCache::advance()`
increments the `filled` counter after all layers complete.

---

## Q4\_K\_M Dequantisation (Phase 4)

**Block format** (144 bytes = 256 elements):

```
bytes  0–3:   d    (f16) — super-block scale
bytes  2–3:   dmin (f16) — super-block min
bytes  4–15:  scales[12] (uint8) — 8 sub-block scale+min pairs packed in 6 bits each
bytes 16–143: qs[128]    (uint8) — 128 bytes of 4-bit values (2 nibbles per byte)
```

**Scale extraction** (`get_scale_min_k4`, sub-block index `j`):

```
j < 4:  sc = scales[j]   & 0x3F
        m  = scales[j+4] & 0x3F
j ≥ 4:  sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
        m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
```

**Dequant formula** for nibble `q`:

```
value = (d * sc) * q  -  (dmin * m)
```

**GPU strategy:** Dequant once at load time → store F16 → reuse existing F16 GEMV kernel.
Memory cost: 4 GB Q4K model → ~14 GB F16 weights (fits in 24 GB unified memory).

---

## Metal Execution Model

All kernels use synchronous dispatch (`cmd.wait_until_completed()`). The command queue
is single-threaded; each kernel call encodes, commits, and waits before returning.

This keeps the Rust API simple and avoids explicit synchronisation between kernel calls.
Future phases may pipeline multiple command buffers for overlap.

**MetalContext** (in `bare-metal-kernels`):

```rust
pub struct MetalContext {
    pub device: metal::Device,
    pub queue:  metal::CommandQueue,
    pipelines:  HashMap<String, metal::ComputePipelineState>,
}
```

All `ComputePipelineState` objects are compiled once at startup from the embedded
`.metallib`. Lookup is by kernel name string.

---

## Phase 5: TurboQuant KV-Cache Compression

[TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) compresses
KV *activations* — not model weights — to 3–4 bits with provably near-optimal inner
product fidelity. No retraining required. **Implemented in Phase 5.**

### Algorithm (as implemented)

Keys and Values use different algorithms because attention needs inner-product fidelity
for keys (dot with Q) but only reconstruction fidelity for values.

#### Key Compression — Algorithm 2 (inner-product preserving)

```
Encode:
  1. norm  = ||k||₂                         save f16
  2. k̂     = k / norm                       unit sphere
  3. y     = RHT(k̂)                         Randomized Hadamard Transform
                                             = WHT( diag(signs) × k̂ ) / √d
             coordinates now ∼ Beta(-1,1)
  4. q     = Lloyd-Max(y, b-1 bits)          binary search in codebook boundaries
             y_hat = centroid[q]             reconstruction in rotated space
  5. mse_packed = pack(q, b-1 bits)          (b-1)*d/8 bytes
  6. x_hat = RHT⁻¹(y_hat)                   rotate back
  7. r     = k̂ - x_hat                      residual vector
     r_norm = ||r||₂                         save f16
  8. signs = sign( S × r )                   QJL projection (on-the-fly hash ±1)
             signs_packed = bit_pack(signs)  d/8 bytes

Stored per key token: mse_packed + signs_packed + (norm, r_norm) = (b-1)*d/8 + d/8 + 4 bytes

Decode:
  1. q     = unpack(mse_packed, b-1 bits)
     y_hat = centroid[q]
  2. x_hat = RHT⁻¹(y_hat)
  3. corr  = √(π/2)/d × r_norm × S^T @ signs   QJL correction (unbiased)
  4. k̂_tq  = x_hat + corr
  5. k     = k̂_tq × norm                        restore scale
```

**Why unbiased:** The QJL correction satisfies `E[q·k̂_tq] = q·k̂` exactly — attention
scores computed from compressed keys have zero systematic bias.

#### Value Compression — per-group symmetric quantization

```
group_size = 128 elements
scale[g]   = max(|v[g*gs : (g+1)*gs]|)
q[i]       = round( (v[i]/scale + 1) / 2 * (2^bits - 1) )
packed     = bit-pack q  →  bits*d/8 bytes  +  n_groups * 2 bytes (scales f16)
```

Values only need reconstruction fidelity, so MSE quantization without inner-product
correction is sufficient.

### RHT Implementation

The Randomized Hadamard Transform uses:
- A random ±1 sign vector `signs[d]` (seed=42, xorshift64, stored as `i8`)
- Walsh-Hadamard Transform via in-place butterfly in threadgroup memory
- Scale by `1/√d`

This requires O(d log d) ops vs O(d²) for full QR rotation, and no d×d matrix storage.
For d=128: 128×7 = 896 ops vs 16384 for QR.

### QJL Projection

The QJL matrix S is never stored. Entry `S[i,j]` is generated on-the-fly in the Metal
kernel using a deterministic hash: `sign = (xorshift(i*A ^ j*B ^ seed) & 1) ? +1 : -1`.
This saves 128×128×4 = 64 KB per head_dim=128 with no accuracy loss.

### Memory Layout

```
Per layer, per kv_head, per compressed token:
  key_mse:    [(b-1)*d/8]       u8   — packed Lloyd-Max indices
  key_signs:  [d/8]             u8   — QJL sign bits
  key_norms:  [2]               f16  — (||k||, ||r||)
  val_pack:   [b_v*d/8]         u8   — packed group-quant values
  val_scales: [d/group_size]    f16  — per-group scale

Recent tokens (buffer_tokens=128) kept as F16 in GPU buffers.
Older tokens stored compressed on CPU (host memory).
```

### Memory Comparison @ Qwen2.5-7B, 4096 tokens

| Cache type | Memory | vs F16 |
|------------|--------|--------|
| F16 (Phase 3) | ~470 MB | 1× |
| TQ3+V4 (3-bit key, 4-bit val) | ~115 MB | **4.1×** |
| TQ3+V2 (3-bit key, 2-bit val) | ~70 MB | **6.7×** |

### TqKvCache Architecture

```rust
pub struct TqKvCache {
    k_f16:      Vec<metal::Buffer>,  // [layers] recent F16 buffer zone
    v_f16:      Vec<metal::Buffer>,
    compressed: Vec<CompressedStore>, // [layers] CPU-side compressed store
    matrices:   TqMatrices,           // codebook + RHT signs on GPU
    buffer_tokens: usize,             // default 128
    filled: usize,
}
```

`materialise_k_f16(ctx, layer)` dequantises all compressed tokens + copies the F16 buffer
zone into a contiguous `[kv_heads, filled, head_dim]` GPU buffer ready for FlashAttention.

---

---

## Phase 6: Text I/O and Sampler

### Tokenizer (`crates/tokenizer`)

Thin wrapper around the HuggingFace `tokenizers` crate (Rust), loading `tokenizer.json`.

```rust
Tokenizer::from_file(path)       // load BPE vocabulary + merges
tokenizer.encode(text, false)    // → Vec<u32>
tokenizer.decode(&ids, true)     // → String (skip special tokens)
```

### Chat Template (`chat_template.rs`)

Qwen2/Qwen2.5 chat format:
```
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{content}<|im_end|>\n
<|im_start|>assistant\n          ← model continues from here
```

`SpecialTokens` holds the IDs for `<|im_start|>` (151644), `<|im_end|>` (151645),
and `<|endoftext|>` (151643). Generation stops when any stop token is sampled.

### Sampler (`sampler.rs`)

```
logits [vocab_size] f16
  │
  ├── temperature = 0.0 → greedy_argmax() — no RNG needed
  └── temperature > 0.0:
        1. divide by temperature
        2. softmax (numerically stable, subtract max)
        3. top-k filter (zero out below k-th largest)
        4. top-p nucleus filter (zero out past cumulative p)
        5. renormalise → multinomial sample (xorshift64 RNG)
```

`SimpleRng` is a deterministic xorshift64 generator — no `rand` crate dependency.

---

## Phase 7: OpenAI-Compatible HTTP Server

The `serve` binary starts an `axum` HTTP server with three routes:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe — returns `200 OK` |
| GET | `/v1/models` | Lists the loaded model |
| POST | `/v1/chat/completions` | Chat inference (blocking or SSE streaming) |

### AppState

```rust
pub struct AppState {
    model_id:  String,
    tokenizer: Tokenizer,
    executor:  Mutex<Executor>,    // Metal not Send+Sync → serialised via Mutex
}
```

### Request Flow

```
POST /v1/chat/completions  (JSON body: messages, temperature, top_p, max_tokens, stream)
  │
  ├── format_prompt()     → prompt string (Qwen2 chat template)
  ├── stream = false:
  │     spawn_blocking {
  │       tokenizer.encode() → prompt_ids
  │       executor.prefill(prompt_ids) → initial logits
  │       decode loop: forward() → sample() → decode token
  │     }
  │     → ChatCompletionResponse (JSON)
  │
  └── stream = true:
        spawn_blocking { same as above, but tx.blocking_send(Event) per token }
        → Sse<ReceiverStream>  (text/event-stream)
```

The `Mutex<Executor>` ensures one inference at a time. A production server would use
a request queue or multiple loaded models; for single-GPU use the mutex is sufficient.

---

## Phase 8: Prefill Batching

### GEMM Kernel (`gemm.metal`)

Tiled 16×16 matrix multiply:

```
Y[M, N] = A[M, K] @ B[N, K]^T

Grid:  [ceil(N/16), ceil(M/16)]
TG:    [16, 16]
SRAM:  As[16][16] + Bs[16][16]  (two f16 tiles = 1024 bytes)
```

Accumulation in `float32`, result cast to `half`. Replaces the sequential GEMV calls
during prompt processing — all query/key/value projections run in a single kernel launch.

### Batched RoPE (`rope_batch_inplace_f16`)

```
Grid: [head_dim/2, n_heads, seq_len]
// Each thread applies one cosine/sine rotation pair at one sequence position
```

Processes all `seq_len` token positions in a single dispatch, vs `seq_len` separate
single-token `rope_inplace_f16` calls.

### `Executor::prefill()` Data Flow

```
tokens [seq]
  │
  ├── embed all → X [seq, hidden]  (CPU memcpy from tok_emb)
  │
  ╔══ for layer l = 0..N ═══════════════════════════════════════════╗
  ║  RMSNorm(X, norm_l)         → X_norm [seq, h]                  ║
  ║  GEMM(X_norm, W_q)          → Q [seq, q_dim]                   ║
  ║  GEMM(X_norm, W_k)          → K [seq, kv_dim]                  ║
  ║  GEMM(X_norm, W_v)          → V [seq, kv_dim]                  ║
  ║  rope_batch(Q, start=0)  in-place                               ║
  ║  rope_batch(K, start=0)  in-place                               ║
  ║  kv_cache.write_at(l, 0..seq, K, V)  — populate all positions  ║
  ║  FlashAttn(Q, K_cache, V_cache, q_len=seq, kv_len=seq)          ║
  ║  GEMM(attn_out, W_o)        → proj [seq, h]                    ║
  ║  X += proj   (residual)                                         ║
  ║  RMSNorm(X, ffn_norm_l)     → X_norm2 [seq, h]                 ║
  ║  GEMM(X_norm2, W_gate)      → gate [seq, ff_dim]               ║
  ║  GEMM(X_norm2, W_up)        → up   [seq, ff_dim]               ║
  ║  SwiGLU(gate, up)           → act  [seq, ff_dim]               ║
  ║  GEMM(act, W_down)          → ffn_out [seq, h]                 ║
  ║  X += ffn_out  (residual)                                       ║
  ╚══════════════════════════════════════════════════════════════════╝
  │
  ├── extract X[seq-1] → last hidden [h]
  ├── RMSNorm + GEMV(lm_head) → logits [vocab_size]
  └── return logits  (caller samples → first decode token)
```

After `prefill()`, `kv_cache.set_seq_len(seq)` advances the sequence pointer so the
decode loop can call `forward(token, pos=seq, kv)` for subsequent tokens.

---

## Error Handling

All Metal dispatch functions return `Result<(), KernelError>`. The `Executor` methods
propagate as `ForwardError` (via `#[from]`). The CLI `generate.rs` prints the error and
exits non-zero.

No `unwrap()` or `expect()` in production paths.
