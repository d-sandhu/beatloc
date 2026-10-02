//! Section boundaries via spectral self-similarity novelty (v2).
//!
//! Method: magnitude STFT frames are compressed into `N_BANDS` log-spaced
//! frequency bands (30 Hz–10 kHz, log1p), each band z-normalized over time.
//! Novelty at frame t is the cosine distance between the mean band vector of
//! the `CONTRAST` seconds before t and after t (a two-sided contrast; the
//! useful core of Foote's checkerboard without the full matrix). Boundaries
//! are strict local maxima above mean + 1.5σ and at least `MIN_GAP` seconds
//! apart. Novelty is only computed where both contrast windows are full, so
//! no boundaries are reported within `CONTRAST` seconds of the file edges
//! (end-of-file fades are not musical transitions).
//!
//! v1 → v2: features went from [energy, flux] scalars to 32 spectral bands,
//! so texture changes at constant loudness (e.g. bass drops out, hats enter)
//! are now visible. Validated on synthetic fixtures and the maintainer's own
//! tracks; an annotated-structure benchmark is blocked on a legally
//! downloadable corpus (RWC audio is a 13.4 GB monolith; Harmonix ships no
//! audio).
//!
//! This detects THAT something changed, never WHAT a section is — no
//! semantic labels. Strengths are relative within the track (peak novelty
//! normalized to [0, 1]) — an uncalibrated ranking aid, not a probability.

/// Number of log-spaced frequency bands.
const N_BANDS: usize = 32;
/// Band edge range in Hz.
const BAND_LO_HZ: f32 = 30.0;
const BAND_HI_HZ: f32 = 10_000.0;
/// Half-width of the contrast window, in seconds.
const CONTRAST_SECONDS: f64 = 2.0;
/// Minimum separation between boundaries, in seconds.
const MIN_GAP_SECONDS: f64 = 8.0;
/// Peak threshold: mean + k × std of the novelty curve.
const PEAK_SIGMAS: f64 = 1.5;
/// Novelty smoothing radius, in seconds.
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

/// Detect transitions from magnitude STFT frames on the shared grid.
pub fn detect_transitions(mags: &[Vec<f32>], hop_seconds: f64, sample_rate: u32) -> Vec<Transition> {
    let n = mags.len();
    if n < 4 {
        return Vec::new();
    }
    let features = band_features(mags, sample_rate);
    let features = z_normalize_bands(&features);

    let w = (CONTRAST_SECONDS / hop_seconds) as usize;
    let mut novelty = vec![0.0f64; n];
    // Novelty is only defined where BOTH contrast windows are full: within
    // CONTRAST seconds of the file edges the windows would be asymmetric, so
    // no boundaries are reported there (an end-of-file fade is not a musical
    // transition, and a section shorter than 2 s is not useful anyway).
    for (t, nv) in novelty.iter_mut().enumerate().take(n.saturating_sub(w)).skip(w) {
        *nv = cosine_distance(&band_mean(&features, t - w, t), &band_mean(&features, t, t + w));
    }

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

    let mut transitions: Vec<Transition> = Vec::new();
    for t in (w + 1)..(n.saturating_sub(w + 1)) {
        let v = novelty[t];
        if v > threshold && v > novelty[t - 1] && v >= novelty[t + 1] {
            if let Some(prev) = transitions.last()
                && t - prev.frame < min_gap
            {
                // Too close: keep the stronger of the two.
                if v > novelty[prev.frame] {
                    *transitions.last_mut().unwrap() =
                        Transition { frame: t, strength: (v / max_val) as f32 };
                }
                continue;
            }
            transitions.push(Transition { frame: t, strength: (v / max_val) as f32 });
        }
    }
    transitions
}

/// Compress magnitude frames into log-spaced band energies (log1p-scaled).
fn band_features(mags: &[Vec<f32>], sample_rate: u32) -> Vec<Vec<f32>> {
    let n_bins = mags[0].len();
    let bin_hz = sample_rate as f32 / (2 * (n_bins - 1)) as f32;
    let ratio = BAND_HI_HZ / BAND_LO_HZ;
    let edge = |i: usize| BAND_LO_HZ * ratio.powf(i as f32 / N_BANDS as f32);
    // Precompute bin → band membership.
    let mut band_of_bin = vec![0usize; n_bins];
    for (bin, band) in band_of_bin.iter_mut().enumerate() {
        let f = bin as f32 * bin_hz;
        *band = (0..N_BANDS).find(|&b| f < edge(b + 1)).unwrap_or(N_BANDS - 1);
    }
    mags.iter()
        .map(|frame| {
            let mut bands = vec![0.0f32; N_BANDS];
            for (bin, &m) in frame.iter().enumerate() {
                bands[band_of_bin[bin]] += m;
            }
            bands.iter_mut().for_each(|v| *v = v.ln_1p());
            bands
        })
        .collect()
}

/// z-normalize each band over time (zero-variance bands become all-zero).
fn z_normalize_bands(features: &[Vec<f32>]) -> Vec<Vec<f64>> {
    let n = features.len() as f64;
    let mut out = vec![vec![0.0f64; N_BANDS]; features.len()];
    for b in 0..N_BANDS {
        let mean = features.iter().map(|f| f64::from(f[b])).sum::<f64>() / n;
        let std = (features.iter().map(|f| (f64::from(f[b]) - mean).powi(2)).sum::<f64>() / n).sqrt();
        if std > 0.0 {
            for (i, f) in features.iter().enumerate() {
                out[i][b] = (f64::from(f[b]) - mean) / std;
            }
        }
    }
    out
}

fn band_mean(features: &[Vec<f64>], from: usize, to: usize) -> Vec<f64> {
    let mut mean = vec![0.0f64; N_BANDS];
    for f in &features[from..to] {
        for (m, &v) in mean.iter_mut().zip(f) {
            *m += v;
        }
    }
    mean.iter_mut().for_each(|v| *v /= (to - from) as f64);
    mean
}

fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    1.0 - dot / (na * nb)
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

    const SR: u32 = 22_050;
    const N_BINS: usize = 513;
    const FPS: f64 = 50.0;

    /// Fake magnitude frames: energy only in bins [lo, hi) of each half.
    fn synthetic_mags(half: &[(usize, usize, f32)], frames: usize) -> Vec<Vec<f32>> {
        let mut mags = vec![vec![0.0f32; N_BINS]; frames];
        for (i, frame) in mags.iter_mut().enumerate() {
            let &(lo, hi, level) = if i < frames / 2 { &half[0] } else { &half[1] };
            for v in &mut frame[lo..hi] {
                *v = level;
            }
        }
        mags
    }

    #[test]
    fn finds_loudness_and_texture_changes() {
        // First half: low-frequency energy; second half: high-frequency.
        let mags = synthetic_mags(&[(2, 40, 1.0), (200, 400, 1.0)], (30.0 * FPS) as usize);
        let transitions = detect_transitions(&mags, 1.0 / FPS, SR);
        assert_eq!(transitions.len(), 1, "transitions: {transitions:?}");
        let t = transitions[0].frame as f64 / FPS;
        assert!((t - 15.0).abs() < 1.5, "transition at {t} s, expected ~15 s");
    }

    #[test]
    fn constant_signal_has_no_transitions() {
        let mags = synthetic_mags(&[(2, 400, 1.0), (2, 400, 1.0)], 1000);
        assert!(detect_transitions(&mags, 0.02, SR).is_empty());
    }

    #[test]
    fn fades_at_file_edges_create_no_edge_boundaries() {
        // A fade-out in the last 0.5 s of a 30 s file must never produce a
        // boundary inside the final CONTRAST window (2 s) — the domain rule
        // — even though its novelty bleed may legitimately fire earlier.
        let frames = (30.0 * FPS) as usize;
        let mut mags = synthetic_mags(&[(2, 200, 1.0), (2, 200, 1.0)], frames);
        for (k, frame) in mags.iter_mut().skip(frames - 25).enumerate() {
            let gain = 1.0 - k as f32 / 25.0; // fade to silence
            for v in frame.iter_mut() {
                *v *= gain;
            }
        }
        let transitions = detect_transitions(&mags, 1.0 / FPS, SR);
        assert!(
            transitions.iter().all(|t| t.frame + 100 <= frames),
            "boundary inside the final contrast window: {transitions:?}"
        );
    }

    #[test]
    fn short_inputs_are_safe() {
        assert!(detect_transitions(&[], 0.02, SR).is_empty());
        assert!(detect_transitions(&[vec![1.0; N_BINS]], 0.02, SR).is_empty());
    }
}
