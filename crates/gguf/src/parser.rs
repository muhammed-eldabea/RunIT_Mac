use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use crate::error::{GgufError, Result};
use crate::types::*;

/// A streaming GGUF parser that reads from any `Read + Seek` source.
pub struct GgufParser<R> {
    reader: R,
    position: u64,
}

impl<R: Read + Seek> GgufParser<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            position: 0,
        }
    }

    /// Parse the entire GGUF file and return a `GgufFile`.
    pub fn parse(&mut self) -> Result<GgufFile> {
        self.reader.seek(SeekFrom::Start(0))?;
        self.position = 0;

        let header = self.read_header()?;

        // Parse metadata KV pairs
        let mut metadata = HashMap::with_capacity(header.metadata_kv_count as usize);
        for _ in 0..header.metadata_kv_count {
            let (key, value) = self.read_metadata_kv()?;
            metadata.insert(key, value);
        }

        // Read alignment from metadata, default to 32
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as u64)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // Parse tensor info array
        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            tensors.push(self.read_tensor_info()?);
        }

        // Data region starts after header+metadata+tensor_info, aligned
        let data_offset = align_offset(self.position, alignment);

        Ok(GgufFile {
            version: header.version,
            metadata,
            tensors,
            data_offset,
        })
    }

    fn read_header(&mut self) -> Result<GgufHeader> {
        let magic = self.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = self.read_u32()?;
        if version != GGUF_VERSION_2 && version != GGUF_VERSION_3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = self.read_u64()?;
        let metadata_kv_count = self.read_u64()?;

        Ok(GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    fn read_metadata_kv(&mut self) -> Result<(String, GgufMetadataValue)> {
        let key = self.read_string()?;
        let value = self.read_metadata_value()?;
        Ok((key, value))
    }

    fn read_metadata_value(&mut self) -> Result<GgufMetadataValue> {
        let value_type = self.read_u32()?;
        let vtype = GgufMetadataValueType::from_u32(value_type)
            .ok_or(GgufError::UnknownMetadataType(value_type))?;
        self.read_typed_value(&vtype)
    }

    fn read_typed_value(&mut self, vtype: &GgufMetadataValueType) -> Result<GgufMetadataValue> {
        match vtype {
            GgufMetadataValueType::Uint8 => Ok(GgufMetadataValue::Uint8(self.read_u8()?)),
            GgufMetadataValueType::Int8 => Ok(GgufMetadataValue::Int8(self.read_i8()?)),
            GgufMetadataValueType::Uint16 => Ok(GgufMetadataValue::Uint16(self.read_u16()?)),
            GgufMetadataValueType::Int16 => Ok(GgufMetadataValue::Int16(self.read_i16()?)),
            GgufMetadataValueType::Uint32 => Ok(GgufMetadataValue::Uint32(self.read_u32()?)),
            GgufMetadataValueType::Int32 => Ok(GgufMetadataValue::Int32(self.read_i32()?)),
            GgufMetadataValueType::Float32 => Ok(GgufMetadataValue::Float32(self.read_f32()?)),
            GgufMetadataValueType::Bool => Ok(GgufMetadataValue::Bool(self.read_u8()? != 0)),
            GgufMetadataValueType::String => Ok(GgufMetadataValue::String(self.read_string()?)),
            GgufMetadataValueType::Uint64 => Ok(GgufMetadataValue::Uint64(self.read_u64()?)),
            GgufMetadataValueType::Int64 => Ok(GgufMetadataValue::Int64(self.read_i64()?)),
            GgufMetadataValueType::Float64 => Ok(GgufMetadataValue::Float64(self.read_f64()?)),
            GgufMetadataValueType::Array => {
                let elem_type = self.read_u32()?;
                let elem_vtype = GgufMetadataValueType::from_u32(elem_type)
                    .ok_or(GgufError::UnknownMetadataType(elem_type))?;
                let count = self.read_u64()? as usize;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.read_typed_value(&elem_vtype)?);
                }
                Ok(GgufMetadataValue::Array(values))
            }
        }
    }

    fn read_tensor_info(&mut self) -> Result<GgufTensorInfo> {
        let name = self.read_string()?;
        let n_dims = self.read_u32()? as usize;

        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(self.read_u64()?);
        }

        let dtype_raw = self.read_u32()?;
        let dtype =
            GgufDtype::from_u32(dtype_raw).ok_or(GgufError::UnknownDtype(dtype_raw))?;

        let data_offset = self.read_u64()?;

        Ok(GgufTensorInfo {
            name,
            shape,
            dtype,
            data_offset,
        })
    }

    // ── Primitive readers ──────────────────────────────────────────

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.reader.read_exact(&mut buf)?;
        self.position += n as u64;
        Ok(buf)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8(self.position))
    }
}

/// Align `offset` up to the next multiple of `alignment`.
fn align_offset(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

// ── Convenience: parse from file path ──────────────────────────────

/// Parse a GGUF file from a file path.
pub fn parse_gguf_file(path: &std::path::Path) -> Result<GgufFile> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut parser = GgufParser::new(reader);
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid GGUF v3 binary with one metadata key and one tensor.
    fn build_test_gguf() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        // Metadata: one string KV pair
        // Key: "general.architecture"
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        // Value type: String (8)
        buf.extend_from_slice(&8u32.to_le_bytes());
        // Value: "qwen2"
        let val = b"qwen2";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        // Tensor info: one F32 tensor [4, 8]
        let name = b"test.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&8u64.to_le_bytes()); // dim 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // dtype = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // data_offset = 0

        // Pad to 32-byte alignment, then add fake tensor data
        let current_len = buf.len() as u64;
        let aligned = align_offset(current_len, 32);
        buf.resize(aligned as usize, 0u8);

        // Tensor data: 4×8 = 32 floats = 128 bytes
        for i in 0..32u32 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }

        buf
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let data = build_test_gguf();
        let cursor = Cursor::new(data);
        let mut parser = GgufParser::new(cursor);
        let gguf = parser.parse().expect("parse should succeed");

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "test.weight");
        assert_eq!(gguf.tensors[0].shape, vec![4, 8]);
        assert_eq!(gguf.tensors[0].dtype, GgufDtype::F32);
        assert_eq!(gguf.tensors[0].data_offset, 0);
        assert_eq!(gguf.data_offset % 32, 0); // aligned

        let arch = gguf.metadata.get("general.architecture").unwrap();
        assert_eq!(arch.as_str(), Some("qwen2"));
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = build_test_gguf();
        data[0] = 0xFF; // corrupt magic
        let cursor = Cursor::new(data);
        let mut parser = GgufParser::new(cursor);
        let err = parser.parse().unwrap_err();
        assert!(matches!(err, GgufError::InvalidMagic(_)));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = build_test_gguf();
        // Overwrite version (bytes 4-7) with version 99
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        let cursor = Cursor::new(data);
        let mut parser = GgufParser::new(cursor);
        let err = parser.parse().unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion(99)));
    }

    #[test]
    fn test_tensor_data_size() {
        let info = GgufTensorInfo {
            name: "test".into(),
            shape: vec![256, 512],
            dtype: GgufDtype::Q4K,
            data_offset: 0,
        };
        let n = info.num_elements(); // 131072
        let expected_blocks = n / 256; // 512 blocks
        let expected_bytes = expected_blocks * 144; // Q4K block = 144 bytes
        assert_eq!(info.data_size(), expected_bytes);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }
}
