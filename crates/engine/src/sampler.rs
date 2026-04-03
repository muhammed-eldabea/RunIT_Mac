//! Token sampler — Phase 6.
//!
//! Strategies:
//!   - Greedy (temperature = 0)
//!   - Temperature scaling + softmax
//!   - Top-k filtering
//!   - Top-p (nucleus) filtering
//!   - Combined temperature + top-k + top-p

use half::f16;

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
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 40, top_p: 0.95, seed: 42 }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self { Self { temperature: 0.0, ..Default::default() } }
}

// ── Core sampling ────────────────────────────────────────────────────────────

/// Sample the next token from logits using the given config and RNG.
pub fn sample(logits: &[f16], cfg: &SamplerConfig, rng: &mut SimpleRng) -> u32 {
    // Greedy fast path
    if cfg.temperature == 0.0 { return greedy_argmax(logits); }

    // Step 1: temperature scaling
    let mut scores: Vec<f32> = logits.iter().map(|x| x.to_f32() / cfg.temperature).collect();

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
    if total <= 0.0 { return greedy_argmax(logits); }

    let r = rng.next_f32() * total;
    let mut cumsum = 0.0f32;
    for (i, &p) in scores.iter().enumerate() {
        cumsum += p;
        if r <= cumsum { return i as u32; }
    }
    greedy_argmax(logits)
}

/// Greedy argmax — returns the index of the highest logit.
pub fn greedy_argmax(logits: &[f16]) -> u32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.to_f32().partial_cmp(&b.to_f32()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    for x in v.iter_mut() { *x /= sum; }
}

fn topk_filter(probs: &mut [f32], k: usize) {
    // Find the k-th largest value
    let mut sorted = probs.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    // Zero out everything below threshold
    let mut kept = 0usize;
    for p in probs.iter_mut() {
        if *p >= threshold && kept < k { kept += 1; }
        else { *p = 0.0; }
    }
}

fn topp_filter(probs: &mut [f32], p: f32) {
    // Sort indices by probability descending
    let mut sorted_idx: Vec<usize> = (0..probs.len()).collect();
    sorted_idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    // Zero out tokens past cumulative probability p
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
        let logits = vec![
            f16::from_f32(0.1), f16::from_f32(5.0), f16::from_f32(2.0),
        ];
        assert_eq!(greedy_argmax(&logits), 1);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits: Vec<f16> = (0..10).map(|i| f16::from_f32(i as f32)).collect();
        let cfg = SamplerConfig::greedy();
        let mut rng = SimpleRng::new(42);
        assert_eq!(sample(&logits, &cfg, &mut rng), 9);
    }

    #[test]
    fn rng_produces_different_values() {
        let mut rng = SimpleRng::new(1234);
        let a = rng.next_f32();
        let b = rng.next_f32();
        assert!((a - b).abs() > 1e-6);
        assert!(a >= 0.0 && a < 1.0);
        assert!(b >= 0.0 && b < 1.0);
    }

    #[test]
    fn topk_keeps_k_entries() {
        let mut probs = vec![0.1, 0.3, 0.05, 0.4, 0.15];
        topk_filter(&mut probs, 2);
        let nonzero: Vec<usize> = probs.iter().enumerate()
            .filter(|(_, &p)| p > 0.0).map(|(i, _)| i).collect();
        assert_eq!(nonzero.len(), 2);
        assert!(nonzero.contains(&1)); // 0.3
        assert!(nonzero.contains(&3)); // 0.4
    }
}
