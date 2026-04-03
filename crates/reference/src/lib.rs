use std::fmt;

use thiserror::Error;

#[derive(Error, Debug)]
pub struct ValidationError {
    pub layer: usize,
    pub max_diff: f32,
    pub tolerance: f32,
    pub num_mismatched: usize,
    pub total_elements: usize,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "layer {} validation failed: max_diff={:.6} > tolerance={:.6} ({}/{} elements differ)",
            self.layer, self.max_diff, self.tolerance, self.num_mismatched, self.total_elements,
        )
    }
}

/// Compare per-layer output from the custom engine against the candle reference.
///
/// Returns `Ok(())` if the maximum absolute difference is within `tolerance`.
/// Typical tolerance: 1e-3 for f16 kernels, 1e-5 for f32.
pub fn validate_layer_output(
    reference: &[f32],
    custom: &[f32],
    layer: usize,
    tolerance: f32,
) -> Result<(), ValidationError> {
    assert_eq!(
        reference.len(),
        custom.len(),
        "reference and custom output lengths must match"
    );

    let mut max_diff: f32 = 0.0;
    let mut num_mismatched: usize = 0;

    for (r, c) in reference.iter().zip(custom.iter()) {
        let diff = (r - c).abs();
        if diff > tolerance {
            num_mismatched += 1;
        }
        max_diff = max_diff.max(diff);
    }

    if max_diff > tolerance {
        return Err(ValidationError {
            layer,
            max_diff,
            tolerance,
            num_mismatched,
            total_elements: reference.len(),
        });
    }

    Ok(())
}

/// Compute summary statistics for a difference vector.
/// Useful for debugging when validation fails.
pub fn diff_stats(reference: &[f32], custom: &[f32]) -> DiffStats {
    assert_eq!(reference.len(), custom.len());

    if reference.is_empty() {
        return DiffStats {
            max_abs: 0.0,
            mean_abs: 0.0,
            rms: 0.0,
            num_elements: 0,
        };
    }

    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;

    for (r, c) in reference.iter().zip(custom.iter()) {
        let diff = (r - c).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff as f64;
        sum_sq += (diff as f64) * (diff as f64);
    }

    let n = reference.len() as f64;
    DiffStats {
        max_abs,
        mean_abs: (sum_abs / n) as f32,
        rms: (sum_sq / n).sqrt() as f32,
        num_elements: reference.len(),
    }
}

#[derive(Debug, Clone)]
pub struct DiffStats {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub rms: f32,
    pub num_elements: usize,
}

impl fmt::Display for DiffStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_abs={:.6}, mean_abs={:.6}, rms={:.6} (n={})",
            self.max_abs, self.mean_abs, self.rms, self.num_elements
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_pass() {
        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let custom = vec![1.0001, 2.0002, 3.0003, 4.0004];
        assert!(validate_layer_output(&reference, &custom, 0, 1e-3).is_ok());
    }

    #[test]
    fn test_validate_fail() {
        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let custom = vec![1.0, 2.0, 3.1, 4.0]; // 0.1 diff
        let err = validate_layer_output(&reference, &custom, 5, 1e-3).unwrap_err();
        assert_eq!(err.layer, 5);
        assert!(err.max_diff > 0.09);
        assert_eq!(err.num_mismatched, 1);
    }

    #[test]
    fn test_diff_stats() {
        let reference = vec![1.0, 2.0, 3.0];
        let custom = vec![1.1, 2.0, 2.9];
        let stats = diff_stats(&reference, &custom);
        assert!((stats.max_abs - 0.1).abs() < 1e-5);
        assert_eq!(stats.num_elements, 3);
    }

    #[test]
    fn test_identical_outputs() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(validate_layer_output(&data, &data, 0, 1e-6).is_ok());
        let stats = diff_stats(&data, &data);
        assert_eq!(stats.max_abs, 0.0);
    }
}
