use std::collections::HashMap;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};
use tracing::info;

use bare_metal_gguf::{parse_gguf_file, GgufFile, GgufMetadataValue};

use crate::config::ModelConfig;
use crate::tensor::{DType, Tensor};
use crate::weights::{validate_weights, ModelWeights};

/// Error types for model loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("GGUF parse error: {0}")]
    Gguf(#[from] bare_metal_gguf::GgufError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported tensor dtype in '{name}': {dtype}")]
    UnsupportedDtype { name: String, dtype: String },

    #[error("missing required metadata key: {0}")]
    MissingMetadata(String),

    #[error("invalid model configuration: {0}")]
    InvalidConfig(String),

    #[error("weight validation failed: {0}")]
    WeightValidation(String),
}

/// A single tensor's location in the mmap — ready to hand to Metal.
///
/// In Phase 2, `ptr` and `size` are passed to `MTLDevice.newBufferWithBytesNoCopy`
/// to create a zero-copy `MTLBuffer` backed directly by the mmap page.
#[derive(Debug, Clone, Copy)]
pub struct TensorBuffer {
    /// Raw pointer into the mmap region (page-aligned by the OS)
    pub ptr: *const u8,
    /// Byte length of this tensor's data
    pub size: usize,
}

// Safety: the mmap backing the pointer lives as long as `Model`, which owns it.
unsafe impl Send for TensorBuffer {}
unsafe impl Sync for TensorBuffer {}

/// A fully loaded model — config, weights index, and mmap'd data region.
pub struct Model {
    /// Architecture configuration parsed from GGUF metadata
    pub config: ModelConfig,
    /// All weight tensors, keyed by name
    pub tensors: HashMap<String, Tensor>,
    /// Raw metadata from the GGUF file
    pub metadata: HashMap<String, GgufMetadataValue>,
    /// Memory-mapped file (kept alive; tensors borrow into this region)
    mmap: Mmap,
    /// Absolute byte offset of the data region within the mmap
    data_offset: u64,
}

impl Model {
    /// Get a zero-copy `TensorBuffer` for a tensor by name.
    ///
    /// The returned pointer points directly into the mmap page —
    /// pass it to `newBufferWithBytesNoCopy` in Phase 2.
    pub fn tensor_buffer(&self, name: &str) -> Option<TensorBuffer> {
        let tensor = self.tensors.get(name)?;
        let abs_offset = (self.data_offset + tensor.buffer_offset) as usize;
        let ptr = unsafe { self.mmap.as_ptr().add(abs_offset) };
        Some(TensorBuffer {
            ptr,
            size: tensor.data_size,
        })
    }

    /// Get raw bytes for a tensor (useful for CPU-side validation).
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.get(name)?;
        let start = (self.data_offset + tensor.buffer_offset) as usize;
        let end = start + tensor.data_size;
        Some(&self.mmap[start..end])
    }

    /// Typed weight accessors.
    pub fn weights(&self) -> ModelWeights<'_> {
        ModelWeights::new(&self.tensors)
    }

    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Human-readable tensor inventory.
    pub fn tensor_summary(&self) -> String {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        let mut out = String::new();
        for name in names {
            let t = &self.tensors[name];
            out.push_str(&format!(
                "  {:60} {:>10} {:?}\n",
                name,
                format!("{:?}", t.shape),
                t.dtype,
            ));
        }
        out
    }
}

/// Load a GGUF model file.
///
/// Steps:
/// 1. Parse GGUF header + metadata + tensor info
/// 2. mmap the entire file
/// 3. Build `ModelConfig` from metadata
/// 4. Build tensor descriptor map
/// 5. Validate weight shapes against config (warnings only — don't fail for quantized models
///    where shapes may be transposed or packed differently)
pub fn load_model(path: &Path) -> Result<Model, LoadError> {
    load_model_opts(path, LoadOptions::default())
}

/// Load with explicit options (e.g. skip validation for inspection).
pub fn load_model_opts(path: &Path, opts: LoadOptions) -> Result<Model, LoadError> {
    // ── Step 1: Parse GGUF header ──────────────────────────────────
    let gguf: GgufFile = parse_gguf_file(path)?;

    info!(
        version = gguf.version,
        tensors = gguf.tensors.len(),
        metadata_keys = gguf.metadata.len(),
        data_offset = gguf.data_offset,
        "Parsed GGUF header"
    );

    // ── Step 2: mmap ───────────────────────────────────────────────
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // ── Step 3: ModelConfig ────────────────────────────────────────
    let config = ModelConfig::from_metadata(&gguf.metadata)?;

    info!(
        arch = config.arch_str,
        layers = config.num_hidden_layers,
        heads = config.num_attention_heads,
        kv_heads = config.num_key_value_heads,
        hidden = config.hidden_size,
        "Model config parsed"
    );

    // ── Step 4: Tensor descriptors ─────────────────────────────────
    let mut tensors = HashMap::with_capacity(gguf.tensors.len());

    for t in &gguf.tensors {
        let dtype = match DType::from_gguf(t.dtype) {
            Some(d) => d,
            None => {
                // Skip tensors with unrecognized dtypes (e.g. tokenizer auxiliary
                // tensors stored as I64, F64, or IQ-family types that the engine
                // doesn't use for inference).
                tracing::debug!(
                    tensor = %t.name,
                    dtype  = %t.dtype,
                    "skipping tensor with unsupported dtype"
                );
                continue;
            }
        };

        let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
        let strides = Tensor::compute_strides(&shape);
        let block_size = dtype.block_size();
        let data_size = t.data_size() as usize;

        tensors.insert(
            t.name.clone(),
            Tensor {
                name: t.name.clone(),
                shape,
                strides,
                dtype,
                block_size,
                buffer_offset: t.data_offset,
                data_size,
            },
        );
    }

    // ── Step 5: Weight validation ──────────────────────────────────
    if opts.validate_weights {
        let weights = ModelWeights::new(&tensors);
        validate_weights(&weights, &config).map_err(|e| {
            LoadError::WeightValidation(e.to_string())
        })?;
        info!("Weight validation passed");
    }

    info!(
        num_tensors = tensors.len(),
        "Model loaded successfully"
    );

    Ok(Model {
        config,
        tensors,
        metadata: gguf.metadata,
        mmap,
        data_offset: gguf.data_offset,
    })
}

/// Options for `load_model_opts`.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Run shape validation against ModelConfig (default: true).
    /// Disable for quick inspection of unknown model formats.
    pub validate_weights: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            validate_weights: true,
        }
    }
}

// ── Legacy compatibility ───────────────────────────────────────────────────────

/// Simple weight map (kept for backward compatibility with Phase 0 tests).
pub struct LoadedModel {
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: HashMap<String, Tensor>,
    _mmap: Mmap,
}

impl LoadedModel {
    pub fn metadata_str(&self, key: &str) -> Result<&str, LoadError> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| LoadError::MissingMetadata(key.to_string()))
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.get(name)?;
        let start = tensor.buffer_offset as usize;
        let end = start + tensor.data_size;
        Some(&self._mmap[start..end])
    }

    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }
}

pub fn load_gguf_model(path: &Path) -> Result<LoadedModel, LoadError> {
    let gguf = parse_gguf_file(path)?;
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let mut tensors = HashMap::with_capacity(gguf.tensors.len());

    for t in &gguf.tensors {
        let dtype = match DType::from_gguf(t.dtype) {
            Some(d) => d,
            None => continue, // skip tokenizer aux tensors with unrecognized dtypes
        };
        let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
        let strides = Tensor::compute_strides(&shape);
        tensors.insert(
            t.name.clone(),
            Tensor {
                name: t.name.clone(),
                shape,
                strides,
                dtype,
                block_size: dtype.block_size(),
                buffer_offset: t.data_offset,
                data_size: t.data_size() as usize,
            },
        );
    }

    Ok(LoadedModel {
        metadata: gguf.metadata,
        tensors,
        _mmap: mmap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn build_test_gguf_bytes() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        buf.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF_MAGIC
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&8u32.to_le_bytes());
        let val = b"qwen2";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        let name = b"test.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let current = buf.len() as u64;
        let aligned = if current % 32 == 0 { current } else { current + (32 - current % 32) };
        buf.resize(aligned as usize, 0u8);
        for i in 0..32u32 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_load_gguf_model_legacy() {
        let data = build_test_gguf_bytes();
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&data).unwrap();
        tmp.flush().unwrap();

        let model = load_gguf_model(tmp.path()).unwrap();
        assert_eq!(model.num_tensors(), 1);

        let t = model.tensors.get("test.weight").unwrap();
        assert_eq!(t.shape, vec![4, 8]);
        assert_eq!(t.dtype, DType::F32);

        let bytes = model.tensor_data("test.weight").unwrap();
        assert_eq!(bytes.len(), 128);
    }
}
