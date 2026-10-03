//! Onset detection: spectral flux envelope + simple peak picking.
//!
//! - Envelope: per frame, sum over bins of the positive difference of
//!   *linear* magnitude spectra, `flux[i] = Σ max(0, |X_i| − |X_{i−1}|)`.
//!   `flux[0] = 0` (no previous frame). Units are arbitrary but comparable
//!   within a track.
//!
//!   Departure from the classic Ellis-style log-magnitude flux, decided
//!   deliberately: in log domain, a near-silent frame sits at the ε floor, so
//!   the faintest first touch of a click at a Hann window's weak edge produces
//!   a huge log jump. Measured effect: a systematic ~17 ms *early* bias on
//!   onsets after quiet passages (exactly where video agents care). Linear
//!   magnitude instead peaks where the absolute energy change is largest,
//!   giving a small, bounded late bias (worst case ~½ window + 1 hop).
//! - Picking: frame `i` is an onset when `flux[i]` is a strict local maximum
//!   (±1 frame) above a LOCAL threshold: mean + 1.5σ computed over a ±5 s
//!   window around `i`, floored at 2% of the track-global mean (a guard
//!   against picking codec hiss in near-silence, where local stats collapse
//!   to ~0). Local rather than global because a track-global threshold goes
//!   silent in the quiet passages of high-dynamic-range material (measured:
//!   a 90 s quiet intro produced zero onsets — the loud climax inflated the
//!   global threshold past anything quiet content could reach). The picking
//!   remains a deliberately simple heuristic, honest about its limits: tuned
//!   for clear attacks, will miss soft onsets and merge flams. NOT evaluated
//!   against annotated onset data — treat strengths as relative.

use super::frame_center_seconds;
use super::stft::stft_magnitudes;
use crate::timeline::Onset;

/// Spectral flux envelope over the shared frame grid.
pub fn onset_strength(samples: &[f32], window: usize, hop: usize) -> Vec<f32> {
    flux_from_mags(&stft_magnitudes(samples, window, hop))
}

/// Spectral flux from precomputed magnitude frames:
/// `flux[i] = Σ max(0, |X_i| − |X_{i−1}|)`, `flux[0] = 0`.
pub fn flux_from_mags(mags: &[Vec<f32>]) -> Vec<f32> {
    let mut flux = Vec::with_capacity(mags.len());
    let mut prev: Option<&Vec<f32>> = None;
    for m in mags {
        flux.push(match prev {
            Some(p) => m.iter().zip(p).map(|(a, b)| (a - b).max(0.0)).sum(),
            None => 0.0,
        });
        prev = Some(m);
    }
    flux
}

/// Pick onset peaks from a flux envelope (see module docs for the rule).
pub fn pick_onsets(flux: &[f32], sample_rate: u32, window: usize, hop: usize) -> Vec<Onset> {
    if flux.len() < 3 {
        return Vec::new();
    }
    let n = flux.len();
    let global_mean = flux.iter().sum::<f32>() / n as f32;
    // Silence guard: below this the content is inaudible relative to the
    // track (and exact digital silence yields flux == 0, which the strict
    // `>` comparison rejects regardless).
    let floor = (0.02 * global_mean).max(1e-9);
    // ±5 s local window, in frames (250 frames at 50 fps).
    let w = (5.0 * f64::from(sample_rate) / hop as f64).max(1.0) as usize;

    // Sliding-window mean/variance (f64 accumulators, truncated at edges).
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut lo = 0usize;
    let mut hi = 0usize; // exclusive
    let mut onsets = Vec::new();
    for i in 1..n - 1 {
        let want_lo = i.saturating_sub(w);
        let want_hi = (i + w + 1).min(n);
        while lo > want_lo {
            lo -= 1;
            let v = flux[lo] as f64;
            sum += v;
            sumsq += v * v;
        }
        while hi < want_hi {
            let v = flux[hi] as f64;
            sum += v;
            sumsq += v * v;
            hi += 1;
        }
        while lo < want_lo {
            let v = flux[lo] as f64;
            sum -= v;
            sumsq -= v * v;
            lo += 1;
        }
        while hi > want_hi {
            hi -= 1;
            let v = flux[hi] as f64;
            sum -= v;
            sumsq -= v * v;
        }
        let count = (hi - lo) as f64;
        let mean = sum / count;
        let std = (sumsq / count - mean * mean).max(0.0).sqrt();
        let threshold = (mean + 1.5 * std) as f32;

        let v = flux[i];
        if v > threshold.max(floor) && v > flux[i - 1] && v >= flux[i + 1] {
            onsets.push(Onset {
                time: frame_center_seconds(i, window, hop, sample_rate),
                strength: v,
                beat_index: None, // attached later, once beats exist
                beat_phase: None,
            });
        }
    }
    onsets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints() {
        // Indirectly guards the window formula: flux of pure silence is zero.
        let flux = onset_strength(&[0.0; 4096], 1024, 441);
        assert!(flux.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn single_click_produces_one_strong_peak() {
        let sr = 22_050usize;
        let mut samples = vec![0.0f32; sr * 2];
        let click_at = sr; // t = 1.0 s
        for i in 0..110 {
            // 5 ms decaying 3 kHz burst
            let env = (-(i as f32) / (sr as f32 * 0.001)).exp();
            samples[click_at + i] =
                0.9 * env * (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / sr as f32).sin();
        }
        let flux = onset_strength(&samples, 1024, 441);
        let onsets = pick_onsets(&flux, sr as u32, 1024, 441);
        assert_eq!(
            onsets.len(),
            1,
            "expected exactly one onset, got {onsets:?}"
        );
        assert!(
            (onsets[0].time - 1.0).abs() < 0.055,
            "onset at {}",
            onsets[0].time
        );
    }

    #[test]
    fn silence_has_no_onsets() {
        let flux = onset_strength(&[0.0; 8192], 1024, 441);
        assert!(pick_onsets(&flux, 22_050, 1024, 441).is_empty());
    }

    #[test]
    fn quiet_onsets_survive_loud_sections() {
        // Regression test for the measured failure on high-dynamic-range
        // audio (a 90 s quiet intro produced ZERO onsets): a track-global
        // threshold gets inflated by the loud part past anything quiet
        // content can reach. The local ±5 s threshold must admit both.
        let sr = 22_050usize;
        let mut samples = vec![0.0f32; sr * 13];
        let mut click = |at_s: f64, amp: f32| {
            let at = (at_s * sr as f64) as usize;
            for i in 0..110 {
                let env = (-(i as f32) / (sr as f32 * 0.001)).exp();
                samples[at + i] +=
                    amp * env * (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / sr as f32).sin();
            }
        };
        // Halves separated so no ±5 s window straddles both.
        for s in [1.0, 2.0, 3.0] {
            click(s, 0.05); // quiet half
        }
        for s in [9.0, 10.0, 11.0] {
            click(s, 1.0); // loud half
        }
        let flux = onset_strength(&samples, 1024, 441);
        let onsets = pick_onsets(&flux, sr as u32, 1024, 441);
        let times: Vec<f64> = onsets.iter().map(|o| o.time).collect();
        for want in [1.0, 2.0, 3.0, 9.0, 10.0, 11.0] {
            assert!(
                times.iter().any(|&t| (t - want).abs() < 0.055),
                "missing onset near {want}s (got {times:?})"
            );
        }
        assert_eq!(onsets.len(), 6, "extra onsets: {times:?}");
    }
}
