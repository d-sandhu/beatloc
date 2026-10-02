//! Short-term energy: frame RMS in dBFS.
//!
//! Units: decibels relative to full scale, `10 * log10(mean(x^2))` per frame,
//! floored at -120 dBFS. A full-scale sine wave reads ≈ -3.01 dBFS. Computed
//! on the mono analysis-rate signal on the shared frame grid (window/hop).

/// Frame RMS in dBFS for frames `[i*hop, i*hop + window)`. Returns an empty
/// vector when the signal is shorter than one window.
pub fn frame_rms_dbfs(samples: &[f32], window: usize, hop: usize) -> Vec<f32> {
    if samples.len() < window || hop == 0 || window == 0 {
        return Vec::new();
    }
    let n_frames = (samples.len() - window) / hop + 1;
    let mut out = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * hop;
        let frame = &samples[start..start + window];
        // Accumulate in f64: window sums of f32 squares lose precision for
        // quiet material otherwise.
        let mean_square =
            frame.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / window as f64;
        let dbfs = 10.0 * mean_square.log10();
        out.push(dbfs.max(-120.0) as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scale_sine_is_minus_3db() {
        let sr = 22_050_f32;
        let window = 1024;
        // Exactly 20 cycles per window so windowed RMS is exact: f = k*sr/N.
        let freq = 20.0 * sr / window as f32;
        let samples: Vec<f32> = (0..window * 4)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();
        let curve = frame_rms_dbfs(&samples, window, 441);
        assert!(!curve.is_empty());
        for &db in &curve {
            assert!((db - -3.0103).abs() < 0.01, "expected ≈ -3.01 dBFS, got {db}");
        }
    }

    #[test]
    fn silence_hits_the_floor() {
        let samples = vec![0.0f32; 4096];
        let curve = frame_rms_dbfs(&samples, 1024, 441);
        assert!(curve.iter().all(|&db| db == -120.0));
    }

    #[test]
    fn shorter_than_one_window_is_empty() {
        assert!(frame_rms_dbfs(&[0.5; 100], 1024, 441).is_empty());
    }
}
