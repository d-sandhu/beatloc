//! Signal processing primitives. Everything here operates on mono f32 audio
//! at the crate-wide analysis rate and shares one frame grid, so all outputs
//! are directly comparable.

pub mod beat;
pub mod energy;
pub mod onset;
pub mod resample;
pub mod sections;
pub mod stft;

/// Seconds from signal start to the centre of analysis frame `frame`.
///
/// This is the single timestamp convention used for every time value in the
/// output JSON: frames are *not* centre-padded, so frame 0 covers samples
/// `[0, window)` and its timestamp is `window/2` samples in.
pub fn frame_center_seconds(frame: usize, window: usize, hop: usize, sample_rate: u32) -> f64 {
    (frame * hop + window / 2) as f64 / f64::from(sample_rate)
}

/// Local tempo track from beat times, engine-agnostic: one point per beat
/// interval, BPM = 60 / median(inter-beat interval over ±`window` intervals).
/// The median makes this robust to single-beat timing jitter; it will smooth
/// over genuine one-bar tempo blips by design. Empty when there are fewer
/// than 2 beats.
pub fn local_tempo_points(beats: &[f64], window: usize) -> Vec<crate::timeline::LocalTempoPoint> {
    let n = beats.len();
    if n < 2 {
        return Vec::new();
    }
    let ibis: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    ibis.iter()
        .enumerate()
        .map(|(i, _)| {
            let lo = i.saturating_sub(window);
            let hi = (i + window + 1).min(ibis.len());
            let mut w: Vec<f64> = ibis[lo..hi].to_vec();
            w.sort_by(f64::total_cmp);
            let median = if w.len() % 2 == 1 {
                w[w.len() / 2]
            } else {
                f64::midpoint(w[w.len() / 2 - 1], w[w.len() / 2])
            };
            crate::timeline::LocalTempoPoint {
                time: f64::midpoint(beats[i], beats[i + 1]),
                bpm: 60.0 / median,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_tempo_is_flat() {
        let beats: Vec<f64> = (0..32).map(|i| 1.0 + 0.5 * i as f64).collect(); // 120 BPM
        let points = local_tempo_points(&beats, 4);
        assert_eq!(points.len(), 31);
        for p in &points {
            assert!((p.bpm - 120.0).abs() < 0.01, "bpm {}", p.bpm);
        }
        // Timestamps are interval midpoints.
        assert!((points[0].time - 1.25).abs() < 1e-12);
    }

    #[test]
    fn tempo_step_is_tracked() {
        let mut beats: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect(); // 120 BPM
        let start = *beats.last().unwrap();
        beats.extend((1..=16).map(|i| start + i as f64 * (60.0 / 90.0))); // 90 BPM
        let points = local_tempo_points(&beats, 4);
        assert!(points[2].bpm > 115.0, "early: {}", points[2].bpm);
        let last = points.last().unwrap();
        assert!((last.bpm - 90.0).abs() < 2.0, "late: {}", last.bpm);
    }

    #[test]
    fn fewer_than_two_beats_is_empty() {
        assert!(local_tempo_points(&[1.0], 4).is_empty());
        assert!(local_tempo_points(&[], 4).is_empty());
    }
}
