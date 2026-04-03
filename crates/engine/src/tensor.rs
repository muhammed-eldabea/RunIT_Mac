use std::fmt;

/// Data types supported by the inference engine.
/// Maps from GGUF dtypes to engine-internal representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
}

impl DType {
    /// Convert from a GGUF dtype.
    pub fn from_gguf(dtype: bare_metal_gguf::GgufDtype) -> Option<Self> {
        use bare_metal_gguf::GgufDtype;
        match dtype {
            GgufDtype::F32  => Some(Self::F32),
            GgufDtype::F16  => Some(Self::F16),
            GgufDtype::BF16 => Some(Self::BF16),
            GgufDtype::Q4_0 => Some(Self::Q4_0),
            GgufDtype::Q4_1 => Some(Self::Q4_1),
            GgufDtype::Q5_0 => Some(Self::Q5_0),
            GgufDtype::Q5_1 => Some(Self::Q5_1),
            GgufDtype::Q8_0 => Some(Self::Q8_0),
            GgufDtype::Q8_1 => Some(Self::Q8_1),
            GgufDtype::Q2K => Some(Self::Q2K),
            GgufDtype::Q3K => Some(Self::Q3K),
            GgufDtype::Q4K => Some(Self::Q4K),
            GgufDtype::Q5K => Some(Self::Q5K),
            GgufDtype::Q6K => Some(Self::Q6K),
            GgufDtype::Q8K => Some(Self::Q8K),
            GgufDtype::I8 => Some(Self::I8),
            GgufDtype::I16 => Some(Self::I16),
            GgufDtype::I32 => Some(Self::I32),
            _ => None,
        }
    }

    /// Block size for quantized types (number of elements per block).
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::I8 | Self::I16 | Self::I32 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Whether this dtype is a quantized format.
    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::I16 | Self::I32
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A tensor descriptor — metadata about a loaded model weight.
///
/// During Phase 0, this holds shape/dtype info and a byte range into the mmap.
/// In Phase 2, this will gain an `Arc<MetalBuffer>` for GPU access.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Human-readable name (e.g. "blk.0.attn_q.weight")
    pub name: String,
    /// Shape dimensions (outermost first, e.g. [out_features, in_features])
    pub shape: Vec<usize>,
    /// Strides in elements (computed from shape, row-major)
    pub strides: Vec<usize>,
    /// Data type / quantization format
    pub dtype: DType,
    /// Block size for quantized types (copied from dtype for fast access)
    pub block_size: usize,
    /// Byte offset of this tensor's data within the mmap region
    pub buffer_offset: u64,
    /// Size of this tensor's data in bytes
    pub data_size: usize,
}

impl Tensor {
    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Compute row-major strides from shape.
    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides() {
        assert_eq!(Tensor::compute_strides(&[3, 4, 5]), vec![20, 5, 1]);
        assert_eq!(Tensor::compute_strides(&[10, 20]), vec![20, 1]);
        assert_eq!(Tensor::compute_strides(&[5]), vec![1]);
        assert_eq!(Tensor::compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_dtype_from_gguf() {
        assert_eq!(
            DType::from_gguf(bare_metal_gguf::GgufDtype::Q4K),
            Some(DType::Q4K)
        );
        assert_eq!(
            DType::from_gguf(bare_metal_gguf::GgufDtype::F16),
            Some(DType::F16)
        );
    }
}
