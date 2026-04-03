use std::collections::HashMap;

use crate::config::ModelConfig;
use crate::loader::LoadError;
use crate::tensor::Tensor;

/// Typed accessor layer over the raw tensor map.
///
/// Qwen2 GGUF tensor naming convention:
/// - Token embedding        : `token_embd.weight`
/// - Output norm            : `output_norm.weight`
/// - LM head                : `output.weight`
/// - Per-layer attention Q  : `blk.<n>.attn_q.weight`
/// - Per-layer attention K  : `blk.<n>.attn_k.weight`
/// - Per-layer attention V  : `blk.<n>.attn_v.weight`
/// - Per-layer attention out: `blk.<n>.attn_output.weight`
/// - Per-layer attn Q bias  : `blk.<n>.attn_q.bias`
/// - Per-layer attn K bias  : `blk.<n>.attn_k.bias`
/// - Per-layer attn V bias  : `blk.<n>.attn_v.bias`
/// - Per-layer attn norm    : `blk.<n>.attn_norm.weight`
/// - Per-layer FFN norm     : `blk.<n>.ffn_norm.weight`
/// - Per-layer FFN gate     : `blk.<n>.ffn_gate.weight`
/// - Per-layer FFN up       : `blk.<n>.ffn_up.weight`
/// - Per-layer FFN down     : `blk.<n>.ffn_down.weight`
pub struct ModelWeights<'a> {
    tensors: &'a HashMap<String, Tensor>,
}

impl<'a> ModelWeights<'a> {
    pub fn new(tensors: &'a HashMap<String, Tensor>) -> Self {
        Self { tensors }
    }

    // ── Global weights ────────────────────────────────────────────

    pub fn token_embedding(&self) -> Option<&Tensor> {
        self.tensors.get("token_embd.weight")
    }

    pub fn output_norm(&self) -> Option<&Tensor> {
        self.tensors.get("output_norm.weight")
    }

    /// LM head (unembedding). May be tied to token_embedding.
    pub fn lm_head(&self) -> Option<&Tensor> {
        self.tensors
            .get("output.weight")
            .or_else(|| self.tensors.get("token_embd.weight"))
    }

    // ── Per-layer attention weights ───────────────────────────────

    pub fn attn_q(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.attn_q.weight"))
    }

    pub fn attn_k(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.attn_k.weight"))
    }

    pub fn attn_v(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.attn_v.weight"))
    }

    pub fn attn_output(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.attn_output.weight"))
    }

    // ── Per-layer attention biases (Qwen2 has QKV biases) ─────────

    pub fn attn_q_bias(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.attn_q.bias"))
    }

    pub fn attn_k_bias(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.attn_k.bias"))
    }

    pub fn attn_v_bias(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.attn_v.bias"))
    }

    // ── Per-layer norms ───────────────────────────────────────────

    pub fn attn_norm(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.attn_norm.weight"))
    }

    pub fn ffn_norm(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.ffn_norm.weight"))
    }

    // ── Per-layer FFN weights (SwiGLU: gate + up + down) ─────────

    pub fn ffn_gate(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.ffn_gate.weight"))
    }

    pub fn ffn_up(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.ffn_up.weight"))
    }

    pub fn ffn_down(&self, layer: usize) -> Option<&Tensor> {
        self.tensors.get(&format!("blk.{layer}.ffn_down.weight"))
    }

    // ── Validation ────────────────────────────────────────────────

    /// Validate that every required tensor exists and has the correct shape
    /// for the given `ModelConfig`.
    ///
    /// Returns a list of validation errors (empty = all good).
    pub fn validate(&self, cfg: &ModelConfig) -> Vec<WeightValidationError> {
        let mut errors = Vec::new();

        // Global tensors
        self.check_tensor(
            "token_embd.weight",
            &[cfg.vocab_size, cfg.hidden_size],
            &mut errors,
        );
        self.check_tensor("output_norm.weight", &[cfg.hidden_size], &mut errors);

        for layer in 0..cfg.num_hidden_layers {
            let n = layer;

            // Attention Q: [num_heads * head_dim, hidden_size]
            self.check_tensor(
                &format!("blk.{n}.attn_q.weight"),
                &[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size],
                &mut errors,
            );
            // Attention K: [num_kv_heads * head_dim, hidden_size]
            self.check_tensor(
                &format!("blk.{n}.attn_k.weight"),
                &[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size],
                &mut errors,
            );
            // Attention V: [num_kv_heads * head_dim, hidden_size]
            self.check_tensor(
                &format!("blk.{n}.attn_v.weight"),
                &[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size],
                &mut errors,
            );
            // Attention out: [hidden_size, num_heads * head_dim]
            self.check_tensor(
                &format!("blk.{n}.attn_output.weight"),
                &[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim],
                &mut errors,
            );

            // Norms
            self.check_tensor(
                &format!("blk.{n}.attn_norm.weight"),
                &[cfg.hidden_size],
                &mut errors,
            );
            self.check_tensor(
                &format!("blk.{n}.ffn_norm.weight"),
                &[cfg.hidden_size],
                &mut errors,
            );

            // FFN (SwiGLU)
            self.check_tensor(
                &format!("blk.{n}.ffn_gate.weight"),
                &[cfg.intermediate_size, cfg.hidden_size],
                &mut errors,
            );
            self.check_tensor(
                &format!("blk.{n}.ffn_up.weight"),
                &[cfg.intermediate_size, cfg.hidden_size],
                &mut errors,
            );
            self.check_tensor(
                &format!("blk.{n}.ffn_down.weight"),
                &[cfg.hidden_size, cfg.intermediate_size],
                &mut errors,
            );
        }

        errors
    }

    fn check_tensor(
        &self,
        name: &str,
        expected_shape: &[usize],
        errors: &mut Vec<WeightValidationError>,
    ) {
        match self.tensors.get(name) {
            None => errors.push(WeightValidationError::Missing {
                name: name.to_string(),
            }),
            Some(t) => {
                // GGUF stores 2-D weight shapes in [in, out] (GGML column-major)
                // while our convention is [out, in] (PyTorch row-major).
                // Accept either ordering for 2-D tensors; require exact match
                // for 1-D tensors (norms, biases).
                let reversed: Vec<usize> = expected_shape.iter().rev().cloned().collect();
                let shape_ok = t.shape == expected_shape
                    || (expected_shape.len() == 2 && t.shape == reversed.as_slice());
                if !shape_ok {
                    errors.push(WeightValidationError::ShapeMismatch {
                        name: name.to_string(),
                        expected: expected_shape.to_vec(),
                        actual: t.shape.clone(),
                    });
                }
            }
        }
    }
}

/// A single weight validation failure.
#[derive(Debug, Clone)]
pub enum WeightValidationError {
    Missing {
        name: String,
    },
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

impl std::fmt::Display for WeightValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { name } => write!(f, "missing tensor '{name}'"),
            Self::ShapeMismatch { name, expected, actual } => {
                write!(f, "shape mismatch '{name}': expected {expected:?}, got {actual:?}")
            }
        }
    }
}

/// Convenience: validate weights and return a single `LoadError` if anything fails.
pub fn validate_weights(
    weights: &ModelWeights<'_>,
    cfg: &ModelConfig,
) -> Result<(), LoadError> {
    let errors = weights.validate(cfg);
    if errors.is_empty() {
        return Ok(());
    }
    let msg = errors
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("\n  ");
    Err(LoadError::InvalidConfig(format!(
        "{} weight validation error(s):\n  {msg}",
        errors.len()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Tensor};

    fn make_tensor(name: &str, shape: Vec<usize>) -> (String, Tensor) {
        let strides = Tensor::compute_strides(&shape);
        let n: usize = shape.iter().product::<usize>().max(1);
        (
            name.to_string(),
            Tensor {
                name: name.to_string(),
                shape,
                strides,
                dtype: DType::F32,
                block_size: 1,
                buffer_offset: 0,
                data_size: n * 4,
            },
        )
    }

    fn minimal_cfg() -> ModelConfig {
        ModelConfig {
            architecture: crate::config::Architecture::Qwen2,
            arch_str: "qwen2".into(),
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            vocab_size: 100,
            rms_norm_eps: 1e-6,
            activation_fn: crate::config::ActivationFn::SiluMul,
            bos_token_id: None,
            eos_token_id: None,
            num_experts: 0,
            num_experts_per_tok: 0,
        }
    }

    fn full_tensors(cfg: &ModelConfig) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        let mut add = |name: &str, shape: Vec<usize>| {
            let (k, v) = make_tensor(name, shape);
            m.insert(k, v);
        };

        add("token_embd.weight", vec![cfg.vocab_size, cfg.hidden_size]);
        add("output_norm.weight", vec![cfg.hidden_size]);

        for n in 0..cfg.num_hidden_layers {
            add(&format!("blk.{n}.attn_q.weight"), vec![cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size]);
            add(&format!("blk.{n}.attn_k.weight"), vec![cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size]);
            add(&format!("blk.{n}.attn_v.weight"), vec![cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size]);
            add(&format!("blk.{n}.attn_output.weight"), vec![cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim]);
            add(&format!("blk.{n}.attn_norm.weight"), vec![cfg.hidden_size]);
            add(&format!("blk.{n}.ffn_norm.weight"), vec![cfg.hidden_size]);
            add(&format!("blk.{n}.ffn_gate.weight"), vec![cfg.intermediate_size, cfg.hidden_size]);
            add(&format!("blk.{n}.ffn_up.weight"), vec![cfg.intermediate_size, cfg.hidden_size]);
            add(&format!("blk.{n}.ffn_down.weight"), vec![cfg.hidden_size, cfg.intermediate_size]);
        }
        m
    }

    #[test]
    fn test_validate_passes_with_correct_shapes() {
        let cfg = minimal_cfg();
        let tensors = full_tensors(&cfg);
        let weights = ModelWeights::new(&tensors);
        let errors = weights.validate(&cfg);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn test_validate_detects_missing_tensor() {
        let cfg = minimal_cfg();
        let mut tensors = full_tensors(&cfg);
        tensors.remove("blk.0.attn_q.weight");
        let weights = ModelWeights::new(&tensors);
        let errors = weights.validate(&cfg);
        assert_eq!(errors.len(), 1);
        assert!(matches!(&errors[0], WeightValidationError::Missing { name } if name == "blk.0.attn_q.weight"));
    }

    #[test]
    fn test_validate_detects_shape_mismatch() {
        let cfg = minimal_cfg();
        let mut tensors = full_tensors(&cfg);
        // Replace with wrong shape
        let (k, v) = make_tensor("token_embd.weight", vec![99, 16]); // wrong vocab
        tensors.insert(k, v);
        let weights = ModelWeights::new(&tensors);
        let errors = weights.validate(&cfg);
        assert_eq!(errors.len(), 1);
        assert!(matches!(&errors[0], WeightValidationError::ShapeMismatch { name, .. } if name == "token_embd.weight"));
    }
}
