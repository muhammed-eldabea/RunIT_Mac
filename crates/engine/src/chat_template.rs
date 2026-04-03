//! Qwen2/Qwen2.5 chat template — Phase 6.
//!
//! Format:
//!   <|im_start|>system\n{system}<|im_end|>\n
//!   <|im_start|>user\n{content}<|im_end|>\n
//!   <|im_start|>assistant\n

pub const IM_START: &str = "<|im_start|>";
pub const IM_END:   &str = "<|im_end|>";
pub const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ChatMessage {
    pub role:    String,
    pub content: String,
}

/// Format a list of chat messages into the Qwen2 prompt string.
/// The returned string ends with "<|im_start|>assistant\n" so the model
/// continues from that point.
pub fn format_prompt(messages: &[ChatMessage], system: Option<&str>) -> String {
    let mut out = String::with_capacity(512);
    let sys = system.unwrap_or(DEFAULT_SYSTEM);

    // System turn (always first)
    out.push_str(IM_START);
    out.push_str("system\n");
    out.push_str(sys);
    out.push_str(IM_END);
    out.push('\n');

    for msg in messages {
        out.push_str(IM_START);
        out.push_str(&msg.role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str(IM_END);
        out.push('\n');
    }

    // Prompt the assistant to respond
    out.push_str(IM_START);
    out.push_str("assistant\n");
    out
}

/// Token IDs for special tokens in the Qwen2 vocabulary.
pub struct SpecialTokens {
    pub im_start: u32,  // <|im_start|>
    pub im_end:   u32,  // <|im_end|>
    pub eos:      u32,  // <|endoftext|>
}

impl Default for SpecialTokens {
    /// Qwen2/Qwen2.5 default special token IDs.
    fn default() -> Self {
        Self { im_start: 151644, im_end: 151645, eos: 151643 }
    }
}

impl SpecialTokens {
    /// Look up special token IDs from a loaded tokenizer.
    pub fn from_tokenizer(tok: &bare_metal_tokenizer::Tokenizer) -> Self {
        Self {
            im_start: tok.token_to_id(IM_START).unwrap_or(151644),
            im_end:   tok.token_to_id(IM_END).unwrap_or(151645),
            eos:      tok.token_to_id("<|endoftext|>").unwrap_or(151643),
        }
    }

    /// Return true if this token should stop generation.
    pub fn is_stop(&self, id: u32) -> bool {
        id == self.im_end || id == self.eos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_prompt_single_user() {
        let msgs = vec![ChatMessage { role: "user".into(), content: "Hello".into() }];
        let out = format_prompt(&msgs, None);
        assert!(out.contains("<|im_start|>system\n"));
        assert!(out.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_prompt_multi_turn() {
        let msgs = vec![
            ChatMessage { role: "user".into(), content: "Hi".into() },
            ChatMessage { role: "assistant".into(), content: "Hello!".into() },
            ChatMessage { role: "user".into(), content: "Bye".into() },
        ];
        let out = format_prompt(&msgs, None);
        assert!(out.contains("assistant\nHello!<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }
}
