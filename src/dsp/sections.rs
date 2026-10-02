//! Section boundaries via feature-contrast novelty (classical, model-free).
//!
//! Method: per frame we take [energy (dBFS), log1p(onset flux)], z-normalize
//! each dimension over the track, and measure the Euclidean distance between
//! the mean feature vector of the `CONTRAST` seconds before and after each
//! frame — a two-sided "checkerboard" novelty score. Boundaries are strict
//! local maxima above mean + 1.5σ, at least `MIN_GAP` seconds apart.
//!
//! This detects *changes* in level/texture (quiet→loud, sparse→dense). It
//! deliberately does NOT label what a section *is* (intro/verse/chorus):
//! boundary detection and section semantics are different problems, and the
//! plan requires keeping them separate. Strengths are relative within the
//! track (peak novelty normalized to [0, 1]) — an uncalibrated diagnostic,
//! not a probability.

/// Half-width of the contrast window, in seconds.
const CONTRAST_SECONDS: f64 = 2.0;
/// Minimum separation between boundaries, in seconds.
const MIN_GAP_SECONDS: f64 = 8.0;
/// Peak threshold: mean + 1.5 × std of the novelty curve.
const PEAK_SIGMAS: f64 = 1.5;
/// Novelty smoothing window, in seconds.
const SMOOTH_SECONDS: f64 = 1.0;

/// One detected section boundary.
#[derive(Debug, Clone, Copy)]
pub struct Transition {
    /// Frame index on the analysis grid.
    pub frame: usize,
    /// Novelty peak height relative to the track's maximum, in [0, 1].
    /// Uncalibrated; meaningful only for ranking within the same track.
    pub strength: f32,
}

/// Detect transitions on the shared frame grid from the energy and
/// onset-strength curves (both are `hop`-spaced, same length).
pub fn detect_transitions(energy_dbfs: &[f32], flux: &[f32], hop_seconds: f64) -> Vec<Transition> {
    let n = energy_dbfs.len().min(flux.len());
    if n < 4 {
        return Vec::new();
    }

    // Feature matrix: [z(dbfs), z(log1p(flux))] per frame.
    let feat_energy = z_normalize(&energy_dbfs[..n]);
    let feat_flux = z_normalize(&flux[..n].iter().map(|&v| v.ln_1p()).collect::<Vec<f32>>());

    // Two-sided contrast novelty.
    let w = (CONTRAST_SECONDS / hop_seconds) as usize; // half-window in frames
    let mut novelty = vec![0.0f64; n];
    for t in 0..n {
        let (a0, a1) = (t.saturating_sub(w), t);
        let (b0, b1) = (t, (t + w).min(n));
        if a1 == a0 || b1 == b0 {
            continue;
        }
        let mean_a = window_mean(&feat_energy, &feat_flux, a0, a1);
        let mean_b = window_mean(&feat_energy, &feat_flux, b0, b1);
        novelty[t] =
            ((mean_a[0] - mean_b[0]).powi(2) + (mean_a[1] - mean_b[1]).powi(2)).sqrt();
    }

    // Smooth, then peak-pick with a minimum gap.
    let s = (SMOOTH_SECONDS / hop_seconds).max(1.0) as usize;
    let novelty = moving_average(&novelty, s);
    let max_val = novelty.iter().copied().fold(0.0f64, f64::max);
    if max_val <= 0.0 {
        return Vec::new();
    }
    let mean = novelty.iter().sum::<f64>() / n as f64;
    let std = (novelty.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let threshold = mean + PEAK_SIGMAS * std;
    let min_gap = (MIN_GAP_SECONDS / hop_seconds) as usize;

    let mut transitions = Vec::new();
    let mut last: Option<usize> = None;
    for t in 1..n.saturating_sub(1) {
        let v = novelty[t];
        if v > threshold && v > novelty[t - 1] && v >= novelty[t + 1] {
            if let Some(prev) = last {
                if t - prev < min_gap {
                    // Too close: keep the stronger of the two.
                    if let Some(prev_t) = transitions.last_mut() {
                        if novelty[t] > novelty[prev] {
                            *prev_t = Transition {
                                frame: t,
                                strength: (v / max_val) as f32,
                            };
                            last = Some(t);
                        }
                        continue;
                    }
                }
            }
            transitions.push(Transition { frame: t, strength: (v / max_val) as f32 });
            last = Some(t);
        }
    }
    transitions
}

fn z_normalize(values: &[f32]) -> Vec<f64> {
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let std = (values.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n).sqrt();
    if std <= 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|&v| (f64::from(v) - mean) / std).collect()
}

fn window_mean(energy: &[f64], flux: &[f64], from: usize, to: usize) -> [f64; 2] {
    let len = (to - from) as f64;
    let e = energy[from..to].iter().sum::<f64>() / len;
    let f = flux[from..to].iter().sum::<f64>() / len;
    [e, f]
}

fn moving_average(values: &[f64], radius: usize) -> Vec<f64> {
    let n = values.len();
    (0..n)
        .map(|i| {
            let from = i.saturating_sub(radius);
            let to = (i + radius + 1).min(n);
            values[from..to].iter().sum::<f64>() / (to - from) as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quiet-then-loud synthetic: a clear transition must be found at the
    /// halfway point, with no other boundaries.
    #[test]
    fn finds_the_obvious_transition() {
        let fps = 50.0;
        let frames = (20.0 * fps) as usize;
        let mut energy = vec![-45.0f32; frames];
        let mut flux = vec![0.05f32; frames];
        for i in frames / 2..frames {
            energy[i] = -12.0; // loud section
            flux[i] = 0.6;
        }
        let transitions = detect_transitions(&energy, &flux, 1.0 / fps);
        assert_eq!(transitions.len(), 1, "transitions: {transitions:?}");
        let t = transitions[0].frame as f64 / fps;
        assert!((t - 10.0).abs() < 1.0, "transition at {t} s, expected ~10 s");
        assert!(transitions[0].strength > 0.9);
    }

    #[test]
    fn constant_signal_has_no_transitions() {
        let energy = vec![-20.0f32; 1000];
        let flux = vec![0.1f32; 1000];
        assert!(detect_transitions(&energy, &flux, 0.02).is_empty());
    }

    #[test]
    fn short_inputs_are_safe() {
        assert!(detect_transitions(&[-20.0], &[0.1], 0.02).is_empty());
        assert!(detect_transitions(&[], &[], 0.02).is_empty());
    }
}
