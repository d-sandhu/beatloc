//! Resampling to the analysis rate via rubato's FFT resampler.
//!
//! `process_all` trims the resampler's startup delay, so a sound at time *t*
//! in the input stays at time *t* in the output (verified by unit test with
//! an impulse). This is the property all downstream timestamps rely on.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::{ANALYSIS_SAMPLE_RATE, BeatlocError};

/// Convert mono f32 audio from `from_rate` to the analysis rate.
/// Returns the input unchanged (cloned) when no conversion is needed.
pub fn to_analysis_rate(input: &[f32], from_rate: u32) -> Result<Vec<f32>, BeatlocError> {
    if from_rate == ANALYSIS_SAMPLE_RATE || input.is_empty() {
        return Ok(input.to_vec());
    }

    let mut resampler =
        Fft::<f32>::new(from_rate as usize, ANALYSIS_SAMPLE_RATE as usize, 1024, 1, FixedSync::Input)
            .map_err(|e| BeatlocError::Resample(e.to_string()))?;

    let adapter =
        InterleavedSlice::new(input, 1, input.len()).map_err(|e| BeatlocError::Resample(e.to_string()))?;

    let output = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(|e| BeatlocError::Resample(e.to_string()))?;

    Ok(output.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_when_already_at_analysis_rate() {
        let input = vec![0.1, -0.2, 0.3];
        let out = to_analysis_rate(&input, ANALYSIS_SAMPLE_RATE).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn halves_length_from_44100() {
        let input = vec![0.0f32; 44_100];
        let out = to_analysis_rate(&input, 44_100).unwrap();
        let expected = 22_050.0;
        assert!(
            (out.len() as f64 - expected).abs() / expected < 0.01,
            "expected ~{expected} samples, got {}",
            out.len()
        );
    }

    /// The critical alignment property: an impulse must stay at the same time
    /// after resampling (i.e. filter delay is trimmed).
    #[test]
    fn impulse_stays_time_aligned() {
        let sr_in = 44_100u32;
        let mut input = vec![0.0f32; sr_in as usize * 2];
        input[sr_in as usize] = 1.0; // impulse at exactly t = 1.0 s

        let out = to_analysis_rate(&input, sr_in).unwrap();
        let peak = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .expect("non-empty output");

        let peak_seconds = peak as f64 / f64::from(ANALYSIS_SAMPLE_RATE);
        assert!(
            (peak_seconds - 1.0).abs() < 0.005,
            "impulse expected at 1.0 s, found at {peak_seconds} s"
        );
    }
}
