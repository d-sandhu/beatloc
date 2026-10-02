//! Beat tracking: global tempo + dynamic programming, after Ellis (2007),
//! "Beat Tracking by Dynamic Programming" — implemented from the paper with
//! parameters chosen deliberately (see constants), not translated line-by-line
//! from any reference implementation.
//!
//! Method:
//! 1. Global period τ from the onset envelope's autocorrelation
//!    (mean-subtracted, normalized by r[0]), scored against a log-Gaussian
//!    tempo prior centered at 120 BPM (width 1 octave). Sub-frame refinement
//!    by parabolic interpolation of the ACF peak.
//! 2. DP over frames: `score[i] = e[i] + max_δ { score[i−δ] − λ·log2(δ/τ)² }`
//!    with δ ∈ [0.5τ, 1.5τ], envelope max-normalized to [0, 1], and λ = 100 —
//!    chosen so that at τ ≈ 25 frames, a ±1-frame deviation costs ≈ 0.35,
//!    i.e. strong evidence outweighs timing rigidity but noise does not.
//! 3. Backtrace from the (first) global maximum. Beats cover only the region
//!    with onset evidence — we do NOT extrapolate a beat grid into silence.
//!
//! Known limitations (documented failure cases): a single global period means
//! tempo changes and rubato are tracked poorly *by construction*; half/double
//! tempo errors remain possible despite the prior; the phase can lock to
//! offbeats in sparse textures. This is the classical baseline — the neural
//! engine (V0.3) is the accuracy product.

/// Engine identifier emitted in the timeline's provenance fields.
pub const ENGINE: &str = "dsp-ellis2007";

/// Tempo search range.
pub const MIN_BPM: f64 = 40.0;
pub const MAX_BPM: f64 = 240.0;

/// Log-Gaussian tempo prior: center and width in octaves.
const PRIOR_CENTER_BPM: f64 = 120.0;
const PRIOR_WIDTH_OCTAVES: f64 = 1.0;

/// DP tightness λ (see module docs for the calibration reasoning).
const TIGHTNESS: f64 = 100.0;

/// Result of beat tracking an onset-strength envelope.
pub struct BeatTrack {
    /// Global tempo in BPM.
    pub bpm: f64,
    /// Beat positions as frame indices into the onset envelope.
    pub frames: Vec<usize>,
    /// Normalized autocorrelation at the chosen period, in (0, 1].
    /// Uncalibrated periodicity diagnostic — NOT a calibrated probability.
    pub periodicity: f32,
}

/// Track beats in a spectral-flux envelope sampled at `fps` frames/second.
/// Returns `None` when no periodicity can be established (silence, or too
/// short for even two periods at the slowest detectable tempo).
pub fn track_beats(flux: &[f32], fps: f64) -> Option<BeatTrack> {
    let (period, periodicity) = estimate_period(flux, fps)?;
    let frames = dp_track(flux, period);
    if frames.is_empty() {
        return None;
    }
    Some(BeatTrack { bpm: 60.0 * fps / period, frames, periodicity })
}

/// Global period in frames via prior-weighted autocorrelation.
fn estimate_period(flux: &[f32], fps: f64) -> Option<(f64, f32)> {
    let n = flux.len();
    let min_lag = (fps * 60.0 / MAX_BPM).ceil() as usize; // 13 frames @ 50 fps
    let mut max_lag = (fps * 60.0 / MIN_BPM).floor() as usize; // 75 frames
    max_lag = max_lag.min(n / 2); // require ≥2 periods of evidence
    if max_lag <= min_lag {
        return None;
    }

    let mean = flux.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let centered: Vec<f64> = flux.iter().map(|&v| f64::from(v) - mean).collect();
    let r0: f64 = centered.iter().map(|v| v * v).sum();
    if r0 <= 0.0 {
        return None; // constant envelope (e.g. digital silence)
    }
    let acf = |lag: usize| (0..n - lag).map(|i| centered[i] * centered[i + lag]).sum::<f64>() / r0;

    let mut best_lag = min_lag;
    let mut best_score = f64::NEG_INFINITY;
    for lag in min_lag..=max_lag {
        let bpm = 60.0 * fps / lag as f64;
        let octaves_off = (bpm / PRIOR_CENTER_BPM).log2() / PRIOR_WIDTH_OCTAVES;
        let prior = (-0.5 * octaves_off * octaves_off).exp();
        let score = acf(lag) * prior;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }
    let periodicity = acf(best_lag);
    if periodicity <= 0.0 {
        return None;
    }

    // Sub-frame refinement: parabola through the normalized ACF at the peak.
    let mut period = best_lag as f64;
    if best_lag > min_lag && best_lag < max_lag {
        let (a, b, c) = (acf(best_lag - 1), acf(best_lag), acf(best_lag + 1));
        let denom = a - 2.0 * b + c;
        if denom < 0.0 {
            period += (0.5 * (a - c) / denom).clamp(-1.0, 1.0);
        }
    }
    Some((period, periodicity as f32))
}

/// Ellis DP beat picking on the max-normalized envelope.
fn dp_track(flux: &[f32], period: f64) -> Vec<usize> {
    let n = flux.len();
    let max_val = flux.iter().copied().fold(0.0f32, f32::max);
    if n == 0 || max_val <= 0.0 {
        return Vec::new();
    }
    let e: Vec<f64> = flux.iter().map(|&v| f64::from(v / max_val)).collect();

    let lo = (period * 0.5).round().max(1.0) as usize;
    let hi = (period * 1.5).round().max(lo as f64) as usize;

    let mut score = vec![f64::NEG_INFINITY; n];
    let mut link: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        score[i] = e[i]; // a beat sequence may start at any frame
        for delta in lo..=hi.min(i) {
            let j = i - delta;
            if !score[j].is_finite() {
                continue;
            }
            let penalty = TIGHTNESS * (delta as f64 / period).log2().powi(2);
            let cand = score[j] + e[i] - penalty;
            if cand > score[i] {
                score[i] = cand;
                link[i] = Some(j);
            }
        }
    }

    // Backtrace from the first global maximum (deterministic tie-break).
    let mut end = 0;
    for i in 1..n {
        if score[i] > score[end] {
            end = i;
        }
    }
    if !score[end].is_finite() {
        return Vec::new();
    }
    let mut frames = Vec::new();
    let mut cur = Some(end);
    while let Some(i) = cur {
        frames.push(i);
        cur = link[i];
    }
    frames.reverse();
    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Impulse train on the frame grid: strongest possible periodic evidence.
    fn spike_envelope(fps: f64, bpm: f64, count: usize, start_frame: usize) -> Vec<f32> {
        let period = fps * 60.0 / bpm;
        let last = start_frame + (count as f64 * period) as usize + 10;
        let mut env = vec![0.0f32; last];
        for i in 0..count {
            env[start_frame + (i as f64 * period).round() as usize] = 1.0;
        }
        env
    }

    #[test]
    fn finds_120_bpm() {
        let env = spike_envelope(50.0, 120.0, 16, 50);
        let track = track_beats(&env, 50.0).expect("periodicity found");
        assert!((track.bpm - 120.0).abs() < 1.0, "bpm {}", track.bpm);
        assert_eq!(track.frames.len(), 16);
        // Beats must sit exactly on the spikes (period 25 frames, integer).
        for (i, &f) in track.frames.iter().enumerate() {
            assert_eq!(f, 50 + i * 25);
        }
    }

    #[test]
    fn finds_90_bpm_with_subframe_period() {
        let env = spike_envelope(50.0, 90.0, 16, 50);
        let track = track_beats(&env, 50.0).expect("periodicity found");
        assert!((track.bpm - 90.0).abs() < 1.5, "bpm {}", track.bpm);
        assert_eq!(track.frames.len(), 16);
    }

    #[test]
    fn silence_yields_nothing() {
        assert!(track_beats(&[0.0; 500], 50.0).is_none());
    }

    #[test]
    fn too_short_yields_nothing() {
        assert!(track_beats(&[0.0; 10], 50.0).is_none());
    }
}
