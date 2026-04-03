use thiserror::Error;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid GGUF magic: expected 0x46554747 (\"GGUF\"), got 0x{0:08X}")]
    InvalidMagic(u32),

    #[error("unsupported GGUF version: {0} (expected 2 or 3)")]
    UnsupportedVersion(u32),

    #[error("unknown tensor dtype: {0}")]
    UnknownDtype(u32),

    #[error("unknown metadata value type: {0}")]
    UnknownMetadataType(u32),

    #[error("unexpected end of file at offset {offset} (need {needed} bytes, have {available})")]
    UnexpectedEof {
        offset: u64,
        needed: u64,
        available: u64,
    },

    #[error("invalid UTF-8 string in metadata at offset {0}")]
    InvalidUtf8(u64),

    #[error("tensor data region exceeds file size")]
    DataRegionOverflow,
}

pub type Result<T> = std::result::Result<T, GgufError>;
