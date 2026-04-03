use std::collections::HashMap;

use bare_metal_gguf::GgufMetadataValue;

use crate::loader::LoadError;

/// Transformer architecture type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    Llama,
    Mistral,
    OlMoE,
    Other(String),
}

impl Architecture {
    fn from_str(s: &str) -> Self {
        match s {
            "qwen2" => Self::Qwen2,
            "llama" => Self::Llama,
            "mistral" => Self::Mistral,
            "olmoe" => Self::OlMoE,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Activation function used in the FFN.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationFn {
    SiluMul, // SwiGLU: silu(gate) * up — used by Qwen2, LLaMA-2+
    Gelu,
    Relu,
}

/// Full model architecture configuration parsed from GGUF metadata.
///
/// All values are read from the GGUF key-value store under the architecture
/// prefix (e.g. "qwen2.attention.head_count" for Qwen2).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Architecture family
    pub architecture: Architecture,
    /// Raw architecture string from metadata (e.g. "qwen2")
    pub arch_str: String,

    // ── Dimensions ────────────────────────────────────────────────
    /// Transformer hidden dimension (d_model)
    pub hidden_size: usize,
    /// Intermediate FFN dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,

    // ── Attention ─────────────────────────────────────────────────
    /// Total number of query heads
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA: may be < num_attention_heads)
    pub num_key_value_heads: usize,
    /// Head dimension = hidden_size / num_attention_heads
    pub head_dim: usize,

    // ── RoPE ──────────────────────────────────────────────────────
    /// RoPE base frequency (default 10000.0; Qwen2 uses 1000000.0)
    pub rope_theta: f32,
    /// Maximum sequence length from training
    pub max_position_embeddings: usize,

    // ── Vocabulary ────────────────────────────────────────────────
    pub vocab_size: usize,

    // ── Normalization ─────────────────────────────────────────────
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,

    // ── FFN ───────────────────────────────────────────────────────
    pub activation_fn: ActivationFn,

    // ── Special tokens ────────────────────────────────────────────
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,

    // ── Mixture of Experts ───────────────────────────────────────
    /// Total number of experts (0 = dense model, no MoE)
    pub num_experts: usize,
    /// Number of experts selected per token (top-k routing)
    pub num_experts_per_tok: usize,
}

impl ModelConfig {
    /// Parse a `ModelConfig` from GGUF metadata.
    ///
    /// Keys follow the pattern `<arch>.<field>` where `<arch>` is the value
    /// of the `general.architecture` metadata key (e.g. "qwen2").
    pub fn from_metadata(
        metadata: &HashMap<String, GgufMetadataValue>,
    ) -> Result<Self, LoadError> {
        // Architecture
        let arch_str = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LoadError::MissingMetadata("general.architecture".into()))?
            .to_string();

        let architecture = Architecture::from_str(&arch_str);
        let p = &arch_str; // prefix for architecture-specific keys

        // Helper closures
        let req_u32 = |key: &str| -> Result<u32, LoadError> {
            metadata
                .get(key)
                .and_then(|v| match v {
                    GgufMetadataValue::Uint32(n) => Some(*n),
                    GgufMetadataValue::Uint64(n) => Some(*n as u32),
                    GgufMetadataValue::Int32(n) => Some(*n as u32),
                    _ => None,
                })
                .ok_or_else(|| LoadError::MissingMetadata(key.to_string()))
        };

        let req_usize = |key: &str| -> Result<usize, LoadError> {
            req_u32(key).map(|v| v as usize)
        };

        let opt_f32 = |key: &str, default: f32| -> f32 {
            metadata
                .get(key)
                .and_then(|v| match v {
                    GgufMetadataValue::Float32(f) => Some(*f),
                    GgufMetadataValue::Float64(f) => Some(*f as f32),
                    _ => None,
                })
                .unwrap_or(default)
        };

        let opt_usize = |key: &str, default: usize| -> usize {
            metadata
                .get(key)
                .and_then(|v| match v {
                    GgufMetadataValue::Uint32(n) => Some(*n as usize),
                    GgufMetadataValue::Uint64(n) => Some(*n as usize),
                    GgufMetadataValue::Int32(n) => Some(*n as usize),
                    _ => None,
                })
                .unwrap_or(default)
        };

        // ── Core dimensions ───────────────────────────────────────
        let hidden_size =
            req_usize(&format!("{p}.embedding_length"))?;
        let intermediate_size =
            req_usize(&format!("{p}.feed_forward_length"))?;
        let num_hidden_layers =
            req_usize(&format!("{p}.block_count"))?;

        // ── Attention ─────────────────────────────────────────────
        let num_attention_heads =
            req_usize(&format!("{p}.attention.head_count"))?;
        let num_key_value_heads =
            opt_usize(&format!("{p}.attention.head_count_kv"), num_attention_heads);

        if hidden_size % num_attention_heads != 0 {
            return Err(LoadError::InvalidConfig(format!(
                "hidden_size ({hidden_size}) not divisible by num_attention_heads ({num_attention_heads})"
            )));
        }
        let head_dim = hidden_size / num_attention_heads;

        // ── RoPE ──────────────────────────────────────────────────
        // Qwen2 uses 1_000_000.0; LLaMA-2 uses 10_000.0
        let rope_theta = opt_f32(
            &format!("{p}.rope.freq_base"),
            if arch_str == "qwen2" { 1_000_000.0 } else { 10_000.0 },
        );
        let max_position_embeddings =
            opt_usize(&format!("{p}.context_length"), 32768);

        // ── Vocabulary ────────────────────────────────────────────
        // Try explicit count keys first, then fall back to token list length.
        let vocab_size = req_usize("tokenizer.ggml.token_count")
            .or_else(|_| req_usize(&format!("{p}.vocab_size")))
            .or_else(|_| {
                // Real GGUF files often omit the count key; the token list is
                // authoritative.
                metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| match v {
                        GgufMetadataValue::Array(arr) => Some(arr.len()),
                        _ => None,
                    })
                    .ok_or_else(|| LoadError::MissingMetadata(
                        "tokenizer.ggml.tokens (vocab size fallback)".into()
                    ))
            })?;

        // ── Normalisation ─────────────────────────────────────────
        let rms_norm_eps = opt_f32(
            &format!("{p}.attention.layer_norm_rms_epsilon"),
            1e-6,
        );

        // ── FFN activation ────────────────────────────────────────
        // Qwen2 and LLaMA-2+ use SwiGLU (silu_mul)
        let activation_fn = {
            let act_key = format!("{p}.feed_forward_act");
            let act_str = metadata
                .get(&act_key)
                .and_then(|v| v.as_str())
                .unwrap_or("silu");
            match act_str {
                "gelu" | "gelu_new" => ActivationFn::Gelu,
                "relu" => ActivationFn::Relu,
                _ => ActivationFn::SiluMul, // default for qwen2/llama
            }
        };

        // ── Special tokens ────────────────────────────────────────
        let bos_token_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| match v {
                GgufMetadataValue::Uint32(n) => Some(*n),
                GgufMetadataValue::Int32(n) => Some(*n as u32),
                _ => None,
            });
        let eos_token_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| match v {
                GgufMetadataValue::Uint32(n) => Some(*n),
                GgufMetadataValue::Int32(n) => Some(*n as u32),
                _ => None,
            });

        // ── Mixture of Experts ─────────────────────────────────────
        let num_experts = opt_usize(&format!("{p}.expert_count"), 0);
        let num_experts_per_tok = opt_usize(&format!("{p}.expert_used_count"), 0);

        Ok(Self {
            architecture,
            arch_str,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_theta,
            max_position_embeddings,
            vocab_size,
            rms_norm_eps,
            activation_fn,
            bos_token_id,
            eos_token_id,
            num_experts,
            num_experts_per_tok,
        })
    }

    /// GQA group size: how many Q heads share each KV head.
    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether this model uses Grouped Query Attention.
    pub fn is_gqa(&self) -> bool {
        self.num_key_value_heads != self.num_attention_heads
    }

    /// Whether this model uses Mixture of Experts routing.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0 && self.num_experts_per_tok > 0
    }

    /// KV cache size per layer per token (bytes, F16).
    pub fn kv_cache_bytes_per_token(&self) -> usize {
        // K + V, each [num_kv_heads, head_dim] f16
        2 * self.num_key_value_heads * self.head_dim * 2
    }
}

impl std::fmt::Display for ModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Architecture   : {} ({})", self.arch_str, if self.is_gqa() { "GQA" } else { "MHA" })?;
        writeln!(f, "Layers         : {}", self.num_hidden_layers)?;
        writeln!(f, "Hidden size    : {}", self.hidden_size)?;
        writeln!(f, "FFN size       : {}", self.intermediate_size)?;
        writeln!(f, "Attn heads     : {} Q / {} KV  (group={})", self.num_attention_heads, self.num_key_value_heads, self.gqa_group_size())?;
        writeln!(f, "Head dim       : {}", self.head_dim)?;
        writeln!(f, "Vocab size     : {}", self.vocab_size)?;
        writeln!(f, "RoPE theta     : {}", self.rope_theta)?;
        writeln!(f, "Max seq len    : {}", self.max_position_embeddings)?;
        writeln!(f, "RMSNorm eps    : {}", self.rms_norm_eps)?;
        if self.is_moe() {
            writeln!(f, "MoE experts    : {} total, top-{}", self.num_experts, self.num_experts_per_tok)?;
        }
        write!(f,   "KV bytes/token : {} B", self.kv_cache_bytes_per_token())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen2_metadata() -> HashMap<String, GgufMetadataValue> {
        let mut m = HashMap::new();
        m.insert("general.architecture".into(), GgufMetadataValue::String("qwen2".into()));
        m.insert("qwen2.embedding_length".into(), GgufMetadataValue::Uint32(3584));
        m.insert("qwen2.feed_forward_length".into(), GgufMetadataValue::Uint32(18944));
        m.insert("qwen2.block_count".into(), GgufMetadataValue::Uint32(28));
        m.insert("qwen2.attention.head_count".into(), GgufMetadataValue::Uint32(28));
        m.insert("qwen2.attention.head_count_kv".into(), GgufMetadataValue::Uint32(4));
        m.insert("qwen2.attention.layer_norm_rms_epsilon".into(), GgufMetadataValue::Float32(1e-6));
        m.insert("qwen2.context_length".into(), GgufMetadataValue::Uint32(32768));
        m.insert("tokenizer.ggml.token_count".into(), GgufMetadataValue::Uint32(152064));
        m.insert("tokenizer.ggml.bos_token_id".into(), GgufMetadataValue::Uint32(151643));
        m.insert("tokenizer.ggml.eos_token_id".into(), GgufMetadataValue::Uint32(151645));
        m
    }

    #[test]
    fn test_qwen2_config() {
        let m = qwen2_metadata();
        let cfg = ModelConfig::from_metadata(&m).unwrap();

        assert_eq!(cfg.architecture, Architecture::Qwen2);
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.intermediate_size, 18944);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 28);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 128);        // 3584 / 28
        assert_eq!(cfg.gqa_group_size(), 7);  // 28 / 4
        assert!(cfg.is_gqa());
        assert_eq!(cfg.vocab_size, 152064);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert_eq!(cfg.bos_token_id, Some(151643));
        assert_eq!(cfg.eos_token_id, Some(151645));
    }

    #[test]
    fn test_missing_required_field() {
        let mut m = qwen2_metadata();
        m.remove("qwen2.embedding_length");
        let err = ModelConfig::from_metadata(&m).unwrap_err();
        assert!(matches!(err, LoadError::MissingMetadata(_)));
    }
}
