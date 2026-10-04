//! Small numeric helpers the families share when they turn logits into answers.

/// Softmax of `z`, shifted by its maximum so large logits do not overflow.
pub fn softmax(z: &[f32]) -> Vec<f32> {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

/// Index of the largest value; the first one wins a tie. Zero for an empty slice.
pub fn argmax<T: PartialOrd>(v: &[T]) -> usize {
    v.iter().enumerate().fold(0, |best, (i, x)| if *x > v[best] { i } else { best })
}

/// `x` rounded to `places` decimals the way Python's `round` output prints.
pub fn round_to(x: f64, places: usize) -> f64 {
    format!("{x:.places$}").parse().unwrap_or(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let p = softmax(&[1000.0, 1000.0, -1000.0]);
        assert_eq!(p, [0.5, 0.5, 0.0]);
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn argmax_keeps_the_first_of_equal_maxima() {
        assert_eq!(argmax(&[0.2, 0.5, 0.5, 0.1]), 1);
        assert_eq!(argmax(&[3.0f64]), 0);
        assert_eq!(argmax::<f32>(&[]), 0);
        // NaN never compares greater, so it cannot displace an earlier value.
        assert_eq!(argmax(&[0.1, f32::NAN, 0.3]), 2);
    }

    #[test]
    fn round_to_matches_fixed_point_formatting() {
        assert_eq!(round_to(0.12345, 4), 0.1235);
        assert_eq!(round_to(0.125, 2), 0.12); // binary 0.125 formats half-to-even
        assert_eq!(round_to(1.0, 2), 1.0);
        assert!(round_to(f64::NAN, 2).is_nan());
    }
}
