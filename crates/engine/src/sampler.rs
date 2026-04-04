//! Token sampler with repetition penalty.
//!
//! Strategies:
//!   - Greedy (temperature = 0)
//!   - Temperature scaling + softmax
//!   - Top-k filtering
//!   - Top-p (nucleus) filtering
//!   - Repetition penalty

// ── SimpleRng (xorshift64, no external dep) ─────────────────────────────────

pub struct SimpleRng { state: u64 }

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed ^ 0x9e3779b97f4a7c15 } }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; x
    }
    /// Uniform float in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }
}

// ── SamplerConfig ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0.0 = greedy (argmax), > 0 enables probabilistic sampling
    pub temperature: f32,
    /// Top-k: keep only top-k logits before sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p nucleus: cumulative probability cutoff (1.0 = disabled)
    pub top_p: f32,
    /// RNG seed for reproducible sampling
    pub seed: u64,
    /// Repetition penalty (1.0 = disabled, 1.1-1.3 typical)
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 40, top_p: 0.95, seed: 42, repetition_penalty: 1.1 }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self { Self { temperature: 0.0, repetition_penalty: 1.1, ..Default::default() } }
}

// ── Core sampling ────────────────────────────────────────────────────────────

/// Sample the next token from f32 logits using the given config, context, and RNG.
pub fn sample(logits: &[f32], cfg: &SamplerConfig, context: &[u32], rng: &mut SimpleRng) -> u32 {
    let mut scores: Vec<f32> = logits.to_vec();

    // Step 0: repetition penalty
    if cfg.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut scores, context, cfg.repetition_penalty);
    }

    // Greedy fast path
    if cfg.temperature == 0.0 { return greedy_argmax_f32(&scores); }

    // Step 1: temperature scaling
    for s in scores.iter_mut() { *s /= cfg.temperature; }

    // Step 2: softmax (numerically stable)
    softmax_inplace(&mut scores);

    // Step 3: top-k filter
    if cfg.top_k > 0 && cfg.top_k < scores.len() {
        topk_filter(&mut scores, cfg.top_k);
    }

    // Step 4: top-p (nucleus) filter
    if cfg.top_p < 1.0 {
        topp_filter(&mut scores, cfg.top_p);
    }

    // Step 5: renormalize and sample
    let total: f32 = scores.iter().sum();
    if total <= 0.0 { return greedy_argmax_f32(&scores); }

    let r = rng.next_f32() * total;
    let mut cumsum = 0.0f32;
    for (i, &p) in scores.iter().enumerate() {
        cumsum += p;
        if r <= cumsum { return i as u32; }
    }
    greedy_argmax_f32(&scores)
}

/// Greedy argmax on f32 logits.
pub fn greedy_argmax_f32(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Apply repetition penalty: for tokens that appear in context,
/// divide positive logits by penalty and multiply negative logits by penalty.
fn apply_repetition_penalty(logits: &mut [f32], context: &[u32], penalty: f32) {
    for &tok in context {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    for x in v.iter_mut() { *x /= sum; }
}

fn topk_filter(probs: &mut [f32], k: usize) {
    let mut sorted = probs.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    let mut kept = 0usize;
    for p in probs.iter_mut() {
        if *p >= threshold && kept < k { kept += 1; }
        else { *p = 0.0; }
    }
}

fn topp_filter(probs: &mut [f32], p: f32) {
    let mut sorted_idx: Vec<usize> = (0..probs.len()).collect();
    sorted_idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0f32;
    for &i in &sorted_idx {
        if cumsum >= p { probs[i] = 0.0; }
        else { cumsum += probs[i]; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let logits = vec![0.1f32, 5.0, 2.0];
        assert_eq!(greedy_argmax_f32(&logits), 1);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let cfg = SamplerConfig::greedy();
        let mut rng = SimpleRng::new(42);
        assert_eq!(sample(&logits, &cfg, &[], &mut rng), 9);
    }

    #[test]
    fn repetition_penalty_reduces_repeated() {
        let mut logits = vec![1.0f32, 5.0, 3.0];
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert!((logits[1] - 2.5).abs() < 1e-6); // 5.0 / 2.0
        assert!((logits[0] - 1.0).abs() < 1e-6); // unchanged
    }
}
