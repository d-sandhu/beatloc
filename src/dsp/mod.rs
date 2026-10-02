//! Signal processing primitives. Everything here operates on mono f32 audio
//! at the crate-wide analysis rate and shares one frame grid, so all outputs
//! are directly comparable.

pub mod beat;
pub mod energy;
pub mod onset;
pub mod resample;

/// Seconds from signal start to the centre of analysis frame `frame`.
///
/// This is the single timestamp convention used for every time value in the
/// output JSON: frames are *not* centre-padded, so frame 0 covers samples
/// `[0, window)` and its timestamp is `window/2` samples in.
pub fn frame_center_seconds(frame: usize, window: usize, hop: usize, sample_rate: u32) -> f64 {
    (frame * hop + window / 2) as f64 / f64::from(sample_rate)
}
