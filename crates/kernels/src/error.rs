use thiserror::Error;

#[derive(Error, Debug)]
pub enum KernelError {
    #[error("Metal device not found — is this an Apple Silicon Mac?")]
    NoDevice,

    #[error("failed to load Metal library from '{path}': {reason}")]
    LibraryLoad { path: String, reason: String },

    #[error("pipeline function '{0}' not found in .metallib")]
    PipelineNotFound(String),

    #[error("failed to create compute pipeline for '{name}': {reason}")]
    PipelineCreate { name: String, reason: String },

    #[error("kernel dispatch error: {0}")]
    Dispatch(String),

    #[error("buffer size mismatch: expected {expected} bytes, got {actual}")]
    BufferSize { expected: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, KernelError>;
