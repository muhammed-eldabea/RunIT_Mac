/// Integration test: build a synthetic Qwen2-style GGUF and load it.
///
/// Uses the same tensor dtype layout as a real Qwen2.5-0.5B Q4_K_M file:
///   - F32  : norm weights (attn_norm, ffn_norm, output_norm), QKV biases
///   - Q6_K : token embedding table (tied lm_head)
///   - Q4_K : all attention and FFN weight matrices
///
/// On macOS the test also creates a MetalContext + Executor to exercise the
/// full weight-upload path (F32→F16 conversion, Q6K CPU dequant, Q4K GPU
/// dequant) without requiring a real model file on disk.

use std::io::Write as _;

const GGUF_MAGIC: u32 = 0x46554747; // b"GGUF" as LE u32

// ─────────────────────────────────────────────────────────────────────────────
// Tiny model dimensions (everything a multiple of 256 so quantized
// tensor element counts are valid block-aligned values)
// ─────────────────────────────────────────────────────────────────────────────
const HIDDEN:    usize = 256;
const INTER:     usize = 256;
const LAYERS:    usize = 1;
const N_HEADS:   usize = 4;
const N_KV:      usize = 2;
const HEAD_DIM:  usize = 64; // HIDDEN / N_HEADS
const VOCAB:     usize = 256;

const Q_DIM:  usize = N_HEADS * HEAD_DIM; // 256
const KV_DIM: usize = N_KV    * HEAD_DIM; // 128

// GGUF dtype codes
const DT_F32: u32 = 0;
const DT_F16: u32 = 1;
const DT_Q4K: u32 = 12;
const DT_Q6K: u32 = 14;

// Block geometry
const Q4K_BLOCK_ELEMS: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
const Q6K_BLOCK_ELEMS: usize = 256;
const Q6K_BLOCK_BYTES: usize = 210;

// ─────────────────────────────────────────────────────────────────────────────
// GGUF binary builder helpers
// ─────────────────────────────────────────────────────────────────────────────

fn write_u32(buf: &mut Vec<u8>, v: u32)  { buf.extend_from_slice(&v.to_le_bytes()); }
fn write_u64(buf: &mut Vec<u8>, v: u64)  { buf.extend_from_slice(&v.to_le_bytes()); }
fn write_f32(buf: &mut Vec<u8>, v: f32)  { buf.extend_from_slice(&v.to_le_bytes()); }
fn write_f16_val(buf: &mut Vec<u8>, v: f32) {
    let h = half::f16::from_f32(v);
    buf.extend_from_slice(&h.to_le_bytes());
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    write_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

/// Metadata value types
fn write_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
    write_str(buf, key);
    write_u32(buf, 4); // Uint32 type
    write_u32(buf, val);
}

fn write_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
    write_str(buf, key);
    write_u32(buf, 6); // Float32 type
    write_f32(buf, val);
}

fn write_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
    write_str(buf, key);
    write_u32(buf, 8); // String type
    write_str(buf, val);
}

/// Write a string array (for tokenizer.ggml.tokens)
fn write_kv_str_array(buf: &mut Vec<u8>, key: &str, count: usize) {
    write_str(buf, key);
    write_u32(buf, 9); // Array type
    write_u32(buf, 8); // element type = String
    write_u64(buf, count as u64);
    for i in 0..count {
        let tok = format!("tok{i}");
        write_str(buf, &tok);
    }
}

/// Write a tensor info entry.
fn write_tensor_info(buf: &mut Vec<u8>, name: &str, shape: &[u64], dtype: u32, offset: u64) {
    write_str(buf, name);
    write_u32(buf, shape.len() as u32); // n_dims
    for &d in shape { write_u64(buf, d); }
    write_u32(buf, dtype);
    write_u64(buf, offset);
}

fn align_to(pos: usize, align: usize) -> usize {
    (pos + align - 1) & !(align - 1)
}

/// Build minimal Q4K block data (all-zero quantized values, d=1.0).
fn q4k_blocks(n_elems: usize) -> Vec<u8> {
    assert_eq!(n_elems % Q4K_BLOCK_ELEMS, 0);
    let n_blocks = n_elems / Q4K_BLOCK_ELEMS;
    let mut out = vec![0u8; n_blocks * Q4K_BLOCK_BYTES];
    for b in 0..n_blocks {
        let base = b * Q4K_BLOCK_BYTES;
        // d at offset 0 (f16 = 1.0)
        let one = half::f16::from_f32(1.0).to_le_bytes();
        out[base..base+2].copy_from_slice(&one);
        // dmin at offset 2 (f16 = 0.0 — already zero)
        // scales[12] at offset 4 — all zero
        // qs[128] at offset 16 — all zero
    }
    out
}

/// Build minimal Q6K block data (all-zero quantized values, d=1.0).
fn q6k_blocks(n_elems: usize) -> Vec<u8> {
    assert_eq!(n_elems % Q6K_BLOCK_ELEMS, 0);
    let n_blocks = n_elems / Q6K_BLOCK_ELEMS;
    let mut out = vec![0u8; n_blocks * Q6K_BLOCK_BYTES];
    for b in 0..n_blocks {
        let base = b * Q6K_BLOCK_BYTES;
        // d at offset 208 (f16 = 1.0)
        let one = half::f16::from_f32(1.0).to_le_bytes();
        out[base + 208..base + 210].copy_from_slice(&one);
        // ql[128], qh[64], scales[16] — all zero
    }
    out
}

/// Build minimal F32 data (all zeros).
fn f32_zeros(n_elems: usize) -> Vec<u8> {
    vec![0u8; n_elems * 4]
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic GGUF builder
// ─────────────────────────────────────────────────────────────────────────────

struct TensorSpec {
    name:  String,
    shape: Vec<u64>,
    dtype: u32,
}

impl TensorSpec {
    fn q4k(name: &str, shape: &[usize]) -> Self {
        Self { name: name.into(), shape: shape.iter().map(|&d| d as u64).collect(), dtype: DT_Q4K }
    }
    fn q6k(name: &str, shape: &[usize]) -> Self {
        Self { name: name.into(), shape: shape.iter().map(|&d| d as u64).collect(), dtype: DT_Q6K }
    }
    fn f32(name: &str, shape: &[usize]) -> Self {
        Self { name: name.into(), shape: shape.iter().map(|&d| d as u64).collect(), dtype: DT_F32 }
    }
    fn f16(name: &str, shape: &[usize]) -> Self {
        Self { name: name.into(), shape: shape.iter().map(|&d| d as u64).collect(), dtype: DT_F16 }
    }

    fn n_elems(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    fn data_bytes(&self) -> Vec<u8> {
        let n = self.n_elems();
        match self.dtype {
            DT_Q4K => q4k_blocks(n),
            DT_Q6K => q6k_blocks(n),
            DT_F32 => f32_zeros(n),
            DT_F16 => vec![0u8; n * 2],
            _       => panic!("unknown dtype {}", self.dtype),
        }
    }
}

fn build_qwen2_gguf() -> Vec<u8> {
    // ── Tensor specifications (matching real Qwen2.5 Q4_K_M dtype layout) ──
    let mut tensors: Vec<TensorSpec> = Vec::new();

    // Global
    tensors.push(TensorSpec::q6k("token_embd.weight", &[VOCAB, HIDDEN]));
    tensors.push(TensorSpec::f32("output_norm.weight", &[HIDDEN]));
    // output.weight is absent → tied to token_embd

    // Per layer
    for l in 0..LAYERS {
        // Norms (F32 in real GGUF)
        tensors.push(TensorSpec::f32(&format!("blk.{l}.attn_norm.weight"), &[HIDDEN]));
        tensors.push(TensorSpec::f32(&format!("blk.{l}.ffn_norm.weight"),  &[HIDDEN]));

        // Attention projections (Q4K)
        tensors.push(TensorSpec::q4k(&format!("blk.{l}.attn_q.weight"),      &[Q_DIM,  HIDDEN]));
        tensors.push(TensorSpec::q4k(&format!("blk.{l}.attn_k.weight"),      &[KV_DIM, HIDDEN]));
        tensors.push(TensorSpec::q4k(&format!("blk.{l}.attn_v.weight"),      &[KV_DIM, HIDDEN]));
        tensors.push(TensorSpec::q4k(&format!("blk.{l}.attn_output.weight"), &[HIDDEN, Q_DIM]));

        // QKV biases (F32 in real Qwen2 GGUF)
        tensors.push(TensorSpec::f32(&format!("blk.{l}.attn_q.bias"), &[Q_DIM]));
        tensors.push(TensorSpec::f32(&format!("blk.{l}.attn_k.bias"), &[KV_DIM]));
        tensors.push(TensorSpec::f32(&format!("blk.{l}.attn_v.bias"), &[KV_DIM]));

        // FFN (Q4K)
        tensors.push(TensorSpec::q4k(&format!("blk.{l}.ffn_gate.weight"), &[INTER,  HIDDEN]));
        tensors.push(TensorSpec::q4k(&format!("blk.{l}.ffn_up.weight"),   &[INTER,  HIDDEN]));
        tensors.push(TensorSpec::q4k(&format!("blk.{l}.ffn_down.weight"), &[HIDDEN, INTER]));
    }

    // ── Pre-compute tensor data and offsets ──────────────────────────────────
    let tensor_data: Vec<Vec<u8>> = tensors.iter().map(|t| t.data_bytes()).collect();

    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut cur: u64 = 0;
    for data in &tensor_data {
        offsets.push(cur);
        cur += data.len() as u64;
        // Q-type blocks need 32-byte alignment between tensors
        cur = align_to(cur as usize, 32) as u64;
    }

    // ── Count metadata KV pairs ──────────────────────────────────────────────
    // arch_str, embedding_length, feed_forward_length, block_count,
    // attention.head_count, attention.head_count_kv, tokenizer.ggml.tokens
    // (7 entries)
    let n_metadata: u64 = 7;
    let n_tensors:  u64 = tensors.len() as u64;

    // ── Write header ─────────────────────────────────────────────────────────
    let mut buf: Vec<u8> = Vec::new();
    write_u32(&mut buf, GGUF_MAGIC);
    write_u32(&mut buf, 3);          // version 3
    write_u64(&mut buf, n_tensors);
    write_u64(&mut buf, n_metadata);

    // ── Write metadata ───────────────────────────────────────────────────────
    write_kv_str(&mut buf, "general.architecture", "qwen2");
    write_kv_u32(&mut buf, "qwen2.embedding_length",       HIDDEN  as u32);
    write_kv_u32(&mut buf, "qwen2.feed_forward_length",    INTER   as u32);
    write_kv_u32(&mut buf, "qwen2.block_count",            LAYERS  as u32);
    write_kv_u32(&mut buf, "qwen2.attention.head_count",   N_HEADS as u32);
    write_kv_u32(&mut buf, "qwen2.attention.head_count_kv",N_KV    as u32);
    write_kv_str_array(&mut buf, "tokenizer.ggml.tokens",  VOCAB);

    // ── Write tensor info ────────────────────────────────────────────────────
    for (spec, &off) in tensors.iter().zip(&offsets) {
        write_tensor_info(&mut buf, &spec.name, &spec.shape, spec.dtype, off);
    }

    // ── Align to 32 bytes ─────────────────────────────────────────────────────
    let aligned = align_to(buf.len(), 32);
    buf.resize(aligned, 0u8);

    // ── Write tensor data ────────────────────────────────────────────────────
    for data in &tensor_data {
        buf.extend_from_slice(data);
        let aligned = align_to(buf.len(), 32);
        buf.resize(aligned, 0u8);
    }

    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Write the synthetic GGUF to a temp file and return the path.
fn write_temp_gguf() -> tempfile::NamedTempFile {
    let data = build_qwen2_gguf();
    let mut f = tempfile::NamedTempFile::new().expect("tmp file");
    f.write_all(&data).expect("write gguf");
    f.flush().expect("flush");
    f
}

#[test]
fn test_qwen2_gguf_parses() {
    let tmp = write_temp_gguf();
    let model = bare_metal_engine::loader::load_model_opts(
        tmp.path(),
        bare_metal_engine::loader::LoadOptions { validate_weights: true },
    )
    .expect("model should load without errors");

    assert_eq!(model.config.hidden_size, HIDDEN);
    assert_eq!(model.config.num_hidden_layers, LAYERS);
    assert_eq!(model.config.num_attention_heads, N_HEADS);
    assert_eq!(model.config.num_key_value_heads, N_KV);
    assert_eq!(model.config.vocab_size, VOCAB);
    assert_eq!(model.config.head_dim, HEAD_DIM);

    // All required tensors present
    assert!(model.tensors.contains_key("token_embd.weight"));
    assert!(model.tensors.contains_key("output_norm.weight"));
    assert!(model.tensors.contains_key("blk.0.attn_norm.weight"));
    assert!(model.tensors.contains_key("blk.0.attn_q.bias"));
}

/// On macOS: full executor upload + one forward step.
/// This exercises F32→F16, Q6K CPU dequant, and Q4K GPU dequant.
#[test]
#[cfg(target_os = "macos")]
fn test_qwen2_executor_upload_and_forward() {
    use bare_metal_engine::{
        forward::{Executor, ForwardError},
        kv_cache::KvCache,
    };
    use bare_metal_kernels::context::MetalContext;

    let tmp = write_temp_gguf();
    let model = bare_metal_engine::loader::load_model(tmp.path())
        .expect("model load");

    // Skip gracefully if no Metal GPU (CI without GPU passthrough)
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("SKIP: Metal device not available");
            return;
        }
    };

    let executor = match Executor::new(ctx, &model) {
        Ok(e) => e,
        Err(ForwardError::UnsupportedDtype { tensor, dtype }) => {
            panic!("UnsupportedDtype: tensor '{tensor}' dtype {dtype:?} — add support in upload_weight()");
        }
        Err(e) => panic!("Executor::new() failed: {e}"),
    };

    // One decode step: BOS token at position 0
    let max_seq = 16;
    let mut kv = KvCache::new(&executor.ctx, &model.config, max_seq);
    let bos = model.config.bos_token_id.unwrap_or(1) as u32;

    let logits = executor
        .forward(bos, 0, &mut kv)
        .expect("forward pass should succeed");

    assert_eq!(logits.len(), VOCAB, "logits should have vocab_size entries");
    // Logits should be finite (no NaN/inf from bad dequant)
    let any_nan = logits.iter().any(|l| l.to_f32().is_nan());
    let any_inf = logits.iter().any(|l| l.to_f32().is_infinite());
    assert!(!any_nan, "logits contain NaN");
    assert!(!any_inf, "logits contain Inf");
}
