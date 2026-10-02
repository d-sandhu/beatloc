//! Codec coverage: OGG Vorbis and AAC/M4A decode through the same pipeline.
//!
//! Fixtures are tiny files encoded from OUR OWN synthetic click track
//! (`examples/gen_clicks.rs`), so provenance is clean and CI needs no
//! encoders. The WAV twin is the reference: lossy codecs must produce the
//! same onset count at the same times, within a codec-specific tolerance.
//!
//! Tolerances: Vorbis is gapless-capable (tight); AAC carries untrimmed
//! encoder delay in Symphonia today, so its tolerance is looser and the
//! residual offset is asserted bounded rather than zero (see crate docs).

use beatloc::{AnalysisOptions, Engine, analyze_file};
use std::path::Path;

const REFERENCE: &str = "tests/fixtures/clicks_120bpm.wav";

fn onset_times(path: &str) -> Vec<f64> {
    let t = analyze_file(
        Path::new(path),
        AnalysisOptions { engine: Engine::Dsp, ..Default::default() },
    )
    .unwrap_or_else(|e| panic!("{path}: {e}"));
    t.onsets.iter().map(|o| o.time).collect()
}

/// Max |offset| between each file's onsets and the WAV reference's.
fn max_offset(reference: &[f64], other: &[f64]) -> f64 {
    reference
        .iter()
        .map(|r| {
            other
                .iter()
                .map(|o| (o - r).abs())
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

#[test]
fn ogg_vorbis_matches_wav() {
    let reference = onset_times(REFERENCE);
    let ogg = onset_times("tests/fixtures/clicks_120bpm.ogg");
    assert_eq!(ogg.len(), reference.len(), "onset count differs (ogg)");
    assert!(max_offset(&reference, &ogg) < 0.06, "ogg onsets drifted");
}

#[test]
fn aac_m4a_matches_wav_within_encoder_delay() {
    let reference = onset_times(REFERENCE);
        let m4a = onset_times("tests/fixtures/clicks_120bpm.m4a");
    assert_eq!(m4a.len(), reference.len(), "onset count differs (m4a)");
    // AAC encoder delay is not trimmed by Symphonia: allow it, assert bounded.
    assert!(max_offset(&reference, &m4a) < 0.12, "m4a onsets beyond encoder-delay bound");
}

/// Gapless detection (decode-level): only MP3 carries per-file trim state.
#[test]
fn encoder_delay_trim_flag() {
    use beatloc::decode::decode_file;
    use std::path::Path;

    let wav = decode_file(Path::new(REFERENCE)).unwrap();
    assert_eq!(wav.encoder_delay_trimmed, None, "WAV has no trim concept");

    let tagged = decode_file(Path::new("tests/fixtures/clicks_120bpm.mp3")).unwrap();
    assert_eq!(tagged.encoder_delay_trimmed, Some(true), "tagged MP3 must trim");

    let untagged = decode_file(Path::new("tests/fixtures/clicks_120bpm_untrimmed.mp3")).unwrap();
    assert_eq!(untagged.encoder_delay_trimmed, Some(false), "untagged MP3 must be flagged");
}

/// The measurement behind the warning: how far do onsets actually drift on
/// an MP3 with no Xing/LAME tag? Assert bounded, print the measured value.
#[test]
fn untagged_mp3_shift_is_bounded() {
    let reference = onset_times(REFERENCE);
    let tagged = onset_times("tests/fixtures/clicks_120bpm.mp3");
    let untagged = onset_times("tests/fixtures/clicks_120bpm_untrimmed.mp3");

    // Sanity: the tagged file agrees with WAV tightly.
    assert_eq!(tagged.len(), reference.len());
    assert!(max_offset(&reference, &tagged) < 0.06, "tagged mp3 drifted");

    // The untagged file must decode the same onsets, but shifted by the
    // untrimmed encoder delay — measured here so the README's "~50 ms"
    // claim is backed by a number, not folklore.
    assert_eq!(untagged.len(), reference.len());
    let shift_ms = untagged
        .iter()
        .zip(reference.iter())
        .map(|(u, r)| (u - r) * 1000.0)
        .sum::<f64>()
        / untagged.len() as f64;
    eprintln!("measured untagged-MP3 onset shift: {shift_ms:+.1} ms");
    assert!(shift_ms > 0.0, "expected a positive (late) shift, got {shift_ms:.1} ms");
    assert!(shift_ms < 60.0, "shift {shift_ms:.1} ms exceeds the documented bound");
}
