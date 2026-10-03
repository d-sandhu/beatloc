//! Tonal-event detection: narrowband harmonic-stack contrast.
//!
//! Why this exists: spectral-flux onsets detect *energy transients* — drums,
//! percussion, broadband hits. A soft-attack tonal event (a synth hook)
//! overlapping drum energy (hits or their tails) contributes almost no
//! spectral change and is structurally invisible to flux (measured: 0%
//! recall at ±30 ms against a human-validated hook oracle on a real track —
//! the hook's dotted-eighth grid decorrelates from the drums' straight
//! eighths).
//!
//! Method (ported from a reference detector the project owner validated by
//! ear on real music; generalized from its song-tuned f0 range): for a grid
//! of candidate fundamentals, measure how far harmonics 2–5 stick out above
//! their sidebands (±~110–190 Hz, in the power domain). Score per frame =
//! the 2nd-weakest harmonic contrast of the best f0, so one partial masked
//! by a drum doesn't kill it, and broadband noise (which raises all bins
//! together) doesn't score.
//!
//! This detector runs its OWN STFT at a 4 ms hop (the shared grid's 20 ms
//! hop cannot express the method's 12 ms rise / 10–50 ms sustain windows —
//! measured: at 50 fps the port caps at ~0.6 recall). Same 1024-sample
//! window as the shared grid, so frequency resolution is unchanged.
//!
//! This detects TONAL NOTE ONSETS, not "the hook": melodic content of any
//! kind fires (vocals included). `f0_hz` is emitted per event so consumers
//! can cluster by pitch. Scores are uncalibrated — relative strength only.
//! Deliberate design constraint: a tone needs ~4 audible partials to score
//! (the 2nd-weakest-harmonic rule), so nearly-pure sinusoids and very dark
//! tones are rejected — that's the price of drums never false-firing.

use super::frame_center_seconds;
use super::stft::stft_magnitudes;
use crate::timeline::TonalEvent;

/// Candidate fundamentals: uniform 6 Hz grid from 80 to 2000 Hz. The 6 Hz
/// step keeps the highest harmonic within ~half a bin of the true partial
/// (21.5 Hz bins at the analysis rate); validated by the reference.
const F0_MIN_HZ: f32 = 80.0;
const F0_MAX_HZ: f32 = 2000.0;
const F0_STEP_HZ: f32 = 6.0;
/// Harmonics scored (the fundamental itself is excluded — it competes with
/// kick/bass energy and is the least reliable partial).
const HARMONICS: [usize; 4] = [2, 3, 4, 5];
/// Onset threshold and required preceding dip, dB of contrast.
const ONSET_DB: f32 = 14.0;
const RISE_DB: f32 = 6.0;
/// Sustain gate: median score over the 10–50 ms after onset must hold this
/// (a real note sustains; junk spikes and near-silence artifacts don't).
const SUSTAIN_DB: f32 = 8.0;
/// Minimum separation between events; dedup window keeps the
/// better-sustained candidate. (Measured on the oracle track: tails live
/// ~30–50 ms after attack; real beep gaps never < ~125 ms there.)
const MIN_GAP_SECONDS: f64 = 0.070;
const MERGE_GAP_SECONDS: f64 = 0.110;
/// Drop events whose harmonic level is this far below the track's median
/// event level (near-silence contrast explosions are ~50 dB down).
const LEVEL_FLOOR_DB: f32 = -20.0;
/// Fine analysis grid for this detector (see module docs): same window as
/// the shared grid, 4 ms hop.
const TONAL_HOP: usize = 88; // 3.99 ms at 22050 Hz

/// Detector provenance for the `tonal_events.engine` field.
pub const ENGINE: &str = "dsp-harmonic-contrast-v1";

/// Contrast score (dB) for one candidate f0 on one frame's power cumsum.
/// None when the 5th harmonic would leave the spectrum.
fn contrast_score(csum: &[f64], n_bins: isize, f0: f32, bin_hz: f32) -> Option<f32> {
    let band = |k0: isize, k1: isize| -> f64 {
        let a = k0.clamp(0, n_bins) as usize;
        let b = k1.clamp(0, n_bins) as usize;
        if b > a { csum[b] - csum[a] } else { 0.0 }
    };
    let mut contrasts = [0.0f32; 4];
    for (ci, &h) in HARMONICS.iter().enumerate() {
        let k = (h as f32 * f0 / bin_hz).round() as isize;
        if k + 10 >= n_bins {
            return None;
        }
        let peak = band(k - 1, k + 2);
        let side = (band(k - 9, k - 5) + band(k + 6, k + 10)) * (3.0 / 8.0);
        contrasts[ci] = (10.0 * ((peak + 1e-12) / (side + 1e-12)).log10()) as f32;
    }
    // 2nd-weakest harmonic (partial fold, no full sort).
    let (_, score) = contrasts
        .iter()
        .fold((f32::INFINITY, f32::INFINITY), |(a, b), &c| {
            if c < a {
                (c, a)
            } else if c < b {
                (a, c)
            } else {
                (a, b)
            }
        });
    Some(score)
}

/// Power cumsum for one magnitude frame (band sums become subtractions).
fn frame_cumsum(m: &[f32]) -> Vec<f64> {
    let mut csum = vec![0.0f64; m.len() + 1];
    for (b, &v) in m.iter().enumerate() {
        csum[b + 1] = csum[b] + (v as f64) * (v as f64);
    }
    csum
}

/// Per-frame tone score: best-f0 2nd-weakest harmonic contrast (dB).
fn tone_scores(mags: &[Vec<f32>], bin_hz: f32) -> (Vec<f32>, Vec<f32>) {
    let n_frames = mags.len();
    let n_bins = mags[0].len() as isize;
    let n_f0 = ((F0_MAX_HZ - F0_MIN_HZ) / F0_STEP_HZ) as usize + 1;

    let mut scores = vec![f32::NEG_INFINITY; n_frames];
    let mut best_f0 = vec![0.0f32; n_frames];
    for (i, m) in mags.iter().enumerate() {
        let csum = frame_cumsum(m);
        let mut best = f32::NEG_INFINITY;
        let mut best_f = 0.0f32;
        for fi in 0..n_f0 {
            let f0 = F0_MIN_HZ + fi as f32 * F0_STEP_HZ;
            if let Some(score) = contrast_score(&csum, n_bins, f0, bin_hz)
                && score > best
            {
                best = score;
                best_f = f0;
            }
        }
        scores[i] = best;
        best_f0[i] = best_f;
    }
    (scores, best_f0)
}

/// Detect tonal events in mono analysis-rate audio. Times are frame
/// centres on this detector's own 4 ms-hop grid.
pub fn detect_tonal_events(samples: &[f32], sample_rate: u32, window: usize) -> Vec<TonalEvent> {
    let mags = stft_magnitudes(samples, window, TONAL_HOP);
    if mags.len() < 16 {
        return Vec::new();
    }
    let bin_hz = sample_rate as f32 / window as f32;
    let (scores, f0s) = tone_scores(&mags, bin_hz);
    let n = scores.len();
    let frame_t = |i: usize| frame_center_seconds(i, window, TONAL_HOP, sample_rate);
    // In frames: ~12 ms rise reference, 10–50 ms sustain window.
    let rise_back = 3usize; // 12 ms
    let sustain_lo = 3usize; // ~12 ms
    let sustain_hi = 12usize; // ~48 ms

    // Rising edge: score crosses ONSET_DB having been RISE_DB lower ~12 ms
    // earlier (the reference rule; now expressible at 4 ms hop).
    let mut raw: Vec<usize> = Vec::new();
    for i in rise_back..n.saturating_sub(sustain_hi) {
        if scores[i] > ONSET_DB && scores[i - rise_back] < ONSET_DB - RISE_DB {
            let t = frame_t(i);
            if raw
                .last()
                .is_some_and(|&j| t - frame_t(j) < MIN_GAP_SECONDS)
            {
                continue;
            }
            let mut window_scores: Vec<f32> = scores[i + sustain_lo..i + sustain_hi].to_vec();
            window_scores.sort_by(f32::total_cmp);
            let sustain = window_scores[window_scores.len() / 2];
            if sustain >= SUSTAIN_DB {
                raw.push(i);
            }
        }
    }

    // Dedup: within MERGE_GAP keep the better-sustained candidate.
    let mut merged: Vec<usize> = Vec::new();
    let sustain_of = |i: usize| {
        let mut w: Vec<f32> = scores[i + sustain_lo..(i + sustain_hi).min(n)].to_vec();
        w.sort_by(f32::total_cmp);
        w[w.len() / 2]
    };
    for &i in &raw {
        if let Some(&j) = merged.last()
            && frame_t(i) - frame_t(j) < MERGE_GAP_SECONDS
        {
            if sustain_of(i) > sustain_of(j) {
                *merged.last_mut().unwrap() = i;
            }
            continue;
        }
        merged.push(i);
    }

    // Track-relative level gate on harmonic power (h2..h4) ~20 ms after
    // onset (post-attack), the reference's near-silence defense.
    let level_db = |i: usize, f0: f32| -> f32 {
        let m = &mags[(i + 5).min(n - 1)];
        let mut p = 0.0f64;
        for h in [2usize, 3, 4] {
            let k = (h as f32 * f0 / bin_hz).round() as isize;
            for b in (k - 1).max(0)..=(k + 1).min(m.len() as isize - 1) {
                p += (m[b as usize] as f64).powi(2);
            }
        }
        (10.0 * (p + 1e-12).log10()) as f32
    };
    let levels: Vec<f32> = merged.iter().map(|&i| level_db(i, f0s[i])).collect();
    let mut sorted = levels.clone();
    sorted.sort_by(f32::total_cmp);
    let median = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };

    merged
        .iter()
        .zip(&levels)
        .filter(|&(_, &lv)| lv - median >= LEVEL_FLOOR_DB)
        .map(|(&i, &lv)| {
            let read = (i + 3).min(n - 1); // f0 read post-attack
            let f0 = octave_preference(&mags[read], f0s[read], bin_hz);
            TonalEvent {
                time: frame_t(i),
                f0_hz: f0,
                contrast_db: scores[i],
                sustain_db: sustain_of(i),
                level_db_rel: lv - median,
            }
        })
        .collect()
}

/// Subharmonic correction: when a tone's upper partials are attenuated,
/// the f0/2 candidate can win the per-frame argmax (its harmonics 2–5 land
/// on the surviving partials). If the octave-up candidate explains the
/// spectrum nearly as well (within tolerance), prefer it. One octave only.
/// Measured necessity: the BUCKED hook reads 452 Hz instead of 916 Hz in
/// the mellower mid sections without this.
const OCTAVE_TOLERANCE_DB: f32 = 2.0;

fn octave_preference(mags: &[f32], f0: f32, bin_hz: f32) -> f32 {
    let f0_up = f0 * 2.0;
    if f0_up > F0_MAX_HZ {
        return f0;
    }
    let csum = frame_cumsum(mags);
    let n_bins = mags.len() as isize;
    match (
        contrast_score(&csum, n_bins, f0, bin_hz),
        contrast_score(&csum, n_bins, f0_up, bin_hz),
    ) {
        (Some(low), Some(high)) if high >= low - OCTAVE_TOLERANCE_DB => f0_up,
        _ => f0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::onset::{onset_strength, pick_onsets};

    const SR: u32 = 22_050;

    /// Deterministic pseudo-noise (LCG) — reproducible tests, no deps.
    fn noise(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn drum_burst(samples: &mut [f32], at_s: f64, seed: &mut u64) {
        let at = (at_s * SR as f64) as usize;
        for i in 0..(0.03 * SR as f64) as usize {
            // 30 ms decaying noise burst
            let env = (-(i as f32) / (SR as f32 * 0.008)).exp();
            samples[at + i] += 0.8 * env * noise(seed);
        }
    }

    fn beep(samples: &mut [f32], at_s: f64, f0: f32) {
        let at = (at_s * SR as f64) as usize;
        let len = (0.06 * SR as f64) as usize; // 60 ms tone
        for i in 0..len {
            let t = i as f32 / SR as f32;
            let env = 1.0 - (i as f32 / len as f32) * 0.5; // gentle decay
            let mut v = 0.0f32;
            for (h, a) in [
                (1.0, 0.10),
                (2.0, 0.30),
                (3.0, 0.22),
                (4.0, 0.16),
                (5.0, 0.12),
            ] {
                v += a * (2.0 * std::f32::consts::PI * f0 * h * t).sin();
            }
            samples[at + i] += env * v;
        }
    }

    #[test]
    fn beeps_under_drums_are_detected() {
        // Drum eighths at 0.4 s spacing; beeps ON a dotted-eighth grid that
        // decorrelates from the drums (the measured failure mode of flux).
        let mut samples = vec![0.0f32; SR as usize * 4];
        let mut seed = 42u64;
        for k in 0..10 {
            drum_burst(&mut samples, 0.1 + k as f64 * 0.4, &mut seed);
        }
        let beep_times = [0.55, 0.95, 1.55, 2.05, 2.55, 3.05];
        for &t in &beep_times {
            beep(&mut samples, t, 913.0);
        }

        let events = detect_tonal_events(&samples, SR, 1024);
        let found = beep_times
            .iter()
            .filter(|&&b| events.iter().any(|e| (e.time - b).abs() < 0.045))
            .count();
        assert!(
            found >= 5,
            "expected >=5/6 beeps, got {found}: {:?}",
            events.iter().map(|e| e.time).collect::<Vec<_>>()
        );

        // Contrast: flux onsets on the same signal should mostly catch the
        // DRUMS, not the beeps (the documented structural miss).
        let flux = onset_strength(&samples, 1024, 441);
        let flux_onsets = pick_onsets(&flux, SR, 1024, 441);
        let flux_found = beep_times
            .iter()
            .filter(|&&b| flux_onsets.iter().any(|o| (o.time - b).abs() < 0.045))
            .count();
        assert!(
            flux_found < found,
            "flux should under-detect beeps (flux={flux_found}, tonal={found})"
        );
    }

    #[test]
    fn octave_preference_prefers_higher_only_on_ambiguity() {
        // (a) Unambiguous tone at 916 Hz with 4 partials: stays 916.
        // (b) Dense stack (partials every ~458 Hz = ambiguous octave):
        //     the f0/2 candidate scores within tolerance, so the event
        //     must report the octave-up f0.
        let mut samples = vec![0.0f32; SR as usize * 3];
        // (a) 916 Hz with partials 2-5 only: 0.5-1.1s
        let (a0, a1) = ((0.5 * SR as f64) as usize, (1.1 * SR as f64) as usize);
        for (i, s) in samples.iter_mut().enumerate().take(a1).skip(a0) {
            let t = i as f32 / SR as f32;
            let mut v = 0.0f32;
            for (h, a) in [
                (1.0, 0.08),
                (2.0, 0.30),
                (3.0, 0.22),
                (4.0, 0.16),
                (5.0, 0.12),
            ] {
                v += a * (2.0 * std::f32::consts::PI * 916.0 * h * t).sin();
            }
            *s += v;
        }
        // (b) dense stack every 458 Hz from 916 to 4580: 1.5-2.1s
        let (b0, b1) = ((1.5 * SR as f64) as usize, (2.1 * SR as f64) as usize);
        for (i, s) in samples.iter_mut().enumerate().take(b1).skip(b0) {
            let t = i as f32 / SR as f32;
            let mut v = 0.0f32;
            for k in 2..=10 {
                v += 0.14 * (2.0 * std::f32::consts::PI * 458.0 * k as f32 * t).sin();
            }
            *s += v;
        }
        let events = detect_tonal_events(&samples, SR, 1024);
        let a = events
            .iter()
            .find(|e| (e.time - 0.5).abs() < 0.2)
            .expect("tone (a) detected");
        assert!(
            (a.f0_hz - 916.0).abs() < 24.0,
            "unambiguous tone must stay: {}",
            a.f0_hz
        );
        let b = events
            .iter()
            .find(|e| (e.time - 1.5).abs() < 0.2)
            .expect("tone (b) detected");
        assert!(
            (b.f0_hz - 916.0).abs() < 24.0,
            "ambiguous stack must correct to the octave-up: {}",
            b.f0_hz
        );
    }

    #[test]
    fn noise_and_silence_do_not_fire() {
        let silence = vec![0.0f32; SR as usize * 2];
        assert!(detect_tonal_events(&silence, SR, 1024).is_empty());

        let mut seed = 7u64;
        let noise_only: Vec<f32> = (0..SR as usize * 2)
            .map(|_| 0.3 * noise(&mut seed))
            .collect();
        let events = detect_tonal_events(&noise_only, SR, 1024);
        assert!(
            events.len() <= 2,
            "white noise produced {} events",
            events.len()
        );
    }

    #[test]
    fn sustained_tone_fires_once() {
        let mut samples = vec![0.0f32; SR as usize * 2];
        let start = SR as usize / 2;
        for (i, s) in samples.iter_mut().enumerate().take(SR as usize).skip(start) {
            let t = i as f32 / SR as f32;
            let mut v = 0.0f32;
            // Four audible partials — the detector's 2nd-weakest-harmonic
            // rule requires them by design.
            for (h, a) in [
                (1.0, 0.08),
                (2.0, 0.30),
                (3.0, 0.22),
                (4.0, 0.16),
                (5.0, 0.12),
            ] {
                v += a * (2.0 * std::f32::consts::PI * 600.0 * h * t).sin();
            }
            *s = v;
        }
        let events = detect_tonal_events(&samples, SR, 1024);
        assert!(
            events.len() <= 2,
            "one sustained tone -> few events, got {}",
            events.len()
        );
        assert!(!events.is_empty(), "sustained tone should fire once");
        assert!(
            (events[0].time - 0.5).abs() < 0.06,
            "onset near 0.5s: {:?}",
            events[0].time
        );
    }
}
