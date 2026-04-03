pub mod error;
pub mod parser;
pub mod types;

// Re-export primary types at crate root for convenience
pub use error::{GgufError, Result};
pub use parser::{parse_gguf_file, GgufParser};
pub use types::{GgufDtype, GgufFile, GgufHeader, GgufMetadataValue, GgufTensorInfo};
