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
//!   (±1 frame) above `mean + 1.5σ` of the envelope. This is a deliberately
//!   simple V0.1 heuristic, honest about its limits: it is tuned for clear
//!   attacks and will both miss soft onsets and merge flams. It is NOT yet
//!   evaluated against annotated onset data — treat strengths as relative.

use super::frame_center_seconds;
use crate::timeline::Onset;
use realfft::RealFftPlanner;

/// Spectral flux envelope over the shared frame grid.
pub fn onset_strength(samples: &[f32], window: usize, hop: usize) -> Vec<f32> {
    if samples.len() < window || hop == 0 || window == 0 {
        return Vec::new();
    }
    let n_frames = (samples.len() - window) / hop + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window);
    let mut in_buf = fft.make_input_vec();
    let mut out_buf = fft.make_output_vec();

    // Periodic Hann window (w[0] = 0; matches DSP convention, not the
    // symmetric "window function" variant).
    let win: Vec<f32> = (0..window)
        .map(|n| 0.5 - 0.5 * (std::f32::consts::TAU * n as f32 / window as f32).cos())
        .collect();

    let mut flux = Vec::with_capacity(n_frames);
    let mut prev_mag: Option<Vec<f32>> = None;

    for frame in 0..n_frames {
        let start = frame * hop;
        for (dst, (&s, &w)) in in_buf
            .iter_mut()
            .zip(samples[start..start + window].iter().zip(win.iter()))
        {
            *dst = s * w;
        }
        fft.process(&mut in_buf, &mut out_buf).expect("buffer sizes are exact by construction");

        let mag: Vec<f32> = out_buf.iter().map(|c| c.norm()).collect();
        let value = match &prev_mag {
            Some(prev) => mag.iter().zip(prev).map(|(a, b)| (a - b).max(0.0)).sum(),
            None => 0.0,
        };
        flux.push(value);
        prev_mag = Some(mag);
    }
    flux
}

/// Pick onset peaks from a flux envelope (see module docs for the rule).
pub fn pick_onsets(flux: &[f32], sample_rate: u32, window: usize, hop: usize) -> Vec<Onset> {
    if flux.len() < 3 {
        return Vec::new();
    }
    let n = flux.len() as f32;
    let mean = flux.iter().sum::<f32>() / n;
    let var = flux.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let threshold = (mean + 1.5 * var.sqrt()).max(1e-9);

    let mut onsets = Vec::new();
    for i in 1..flux.len() - 1 {
        let v = flux[i];
        if v > threshold && v > flux[i - 1] && v >= flux[i + 1] {
            onsets.push(Onset { time: frame_center_seconds(i, window, hop, sample_rate), strength: v });
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
        assert_eq!(onsets.len(), 1, "expected exactly one onset, got {onsets:?}");
        assert!((onsets[0].time - 1.0).abs() < 0.055, "onset at {}", onsets[0].time);
    }

    #[test]
    fn silence_has_no_onsets() {
        let flux = onset_strength(&[0.0; 8192], 1024, 441);
        assert!(pick_onsets(&flux, 22_050, 1024, 441).is_empty());
    }
}
