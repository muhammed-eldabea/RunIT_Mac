use std::path::Path;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),

    #[error("encoding failed: {0}")]
    Encode(String),

    #[error("decoding failed: {0}")]
    Decode(String),
}

/// Thin wrapper around the HuggingFace `tokenizers` crate.
///
/// Loads a `tokenizer.json` (BPE vocabulary + merges) and provides
/// encode/decode operations for the inference pipeline.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Convert a single token ID to its string representation.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Convert a token string to its ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}
