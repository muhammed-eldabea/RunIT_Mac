use std::collections::HashMap;
use std::fmt;

/// GGUF magic bytes: b"GGUF" read as little-endian u32
/// bytes [0x47, 0x47, 0x55, 0x46] → LE u32 = 0x46554747
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Supported GGUF format versions
pub const GGUF_VERSION_2: u32 = 2;
pub const GGUF_VERSION_3: u32 = 3;

/// Default alignment for tensor data (bytes)
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Parsed GGUF file — header + metadata + tensor index.
/// Does NOT own the tensor data; that lives in an mmap region.
#[derive(Debug)]
pub struct GgufFile {
    /// Format version (2 or 3)
    pub version: u32,
    /// Key-value metadata (architecture, tokenizer config, etc.)
    pub metadata: HashMap<String, GgufMetadataValue>,
    /// Tensor descriptors in file order
    pub tensors: Vec<GgufTensorInfo>,
    /// Absolute byte offset where the tensor data region begins.
    /// All `GgufTensorInfo::data_offset` values are relative to this.
    pub data_offset: u64,
}

/// Raw file header — first 16 bytes of any GGUF file.
#[derive(Debug, Clone, Copy)]
pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// Describes one tensor's location and layout within the GGUF data region.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g. "blk.0.attn_q.weight")
    pub name: String,
    /// Shape dimensions (outermost first)
    pub shape: Vec<u64>,
    /// Data type / quantization format
    pub dtype: GgufDtype,
    /// Byte offset of this tensor's data, relative to `GgufFile::data_offset`
    pub data_offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements in this tensor.
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }

    /// Size of this tensor's data in bytes.
    pub fn data_size(&self) -> u64 {
        let n = self.num_elements();
        self.dtype.tensor_data_size(n)
    }
}

/// GGUF tensor data types.
/// Values match the GGUF spec's `ggml_type` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4 is unused (was Q4_2)
    // 5 is unused (was Q4_3)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GgufDtype {
    /// Parse a u32 from the GGUF file into a dtype.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Block size for quantized types (number of elements per block).
    /// Non-quantized types return 1.
    pub fn block_size(&self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 256,
            Self::IQ3XXS | Self::IQ3S => 256,
            Self::IQ1S | Self::IQ1M => 256,
            Self::IQ4NL | Self::IQ4XS => 32,
        }
    }

    /// Byte size of one quantization block.
    pub fn type_size(&self) -> u64 {
        match self {
            Self::F32  => 4,
            Self::F16  => 2,
            Self::BF16 => 2,
            Self::F64  => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,   // 2 (scale) + 16 (4-bit × 32)
            Self::Q4_1 => 20,   // 2 (scale) + 2 (min) + 16
            Self::Q5_0 => 22,   // 2 + 4 (high bits) + 16
            Self::Q5_1 => 24,   // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,   // 2 (scale) + 32 (8-bit × 32)
            Self::Q8_1 => 40,   // 4 (scale) + 4 (min) + 32
            Self::Q2K => 256 / 16 * 2 + 256 / 4 + 2 + 2, // 84
            Self::Q3K => 256 / 8 * 3 + 256 / 4 + 12 + 2, // 110
            Self::Q4K => 144,   // 2+2+12+128 (256 elements, ~4.5 bpw)
            Self::Q5K => 176,   // 2+2+12+160
            Self::Q6K => 210,   // 2+256/2+256/4+256/16
            Self::Q8K => 292,   // 4+256+256/16*2
            // IQ types — sizes from ggml source
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ1S => 50,
            Self::IQ1M => 56,
            Self::IQ4NL => 18,
            Self::IQ4XS => 36,
        }
    }

    /// Compute the total byte size for `num_elements` values of this type.
    pub fn tensor_data_size(&self, num_elements: u64) -> u64 {
        let bs = self.block_size();
        let num_blocks = (num_elements + bs - 1) / bs;
        num_blocks * self.type_size()
    }
}

impl fmt::Display for GgufDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3S => "IQ3_S",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ4XS => "IQ4_XS",
            Self::BF16  => "BF16",
        };
        write!(f, "{}", name)
    }
}

/// GGUF metadata value types.
/// Values in the key-value metadata store can be any of these types.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufMetadataValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// A typed metadata value from the GGUF key-value store.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufMetadataValue {
    /// Try to extract as a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as u64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as a slice of values (array).
    pub fn as_array(&self) -> Option<&[GgufMetadataValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}
