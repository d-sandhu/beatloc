//! V0.1 exit-condition tests: real decoded audio produces deterministic
//! output with tested timestamp alignment.
//!
//! Strategy: synthesize clicks at known ground-truth times, run the real
//! pipeline (file → decode → downmix → resample → detect), and assert the
//! reported onset times land near the truth with no large systematic offset.
//!
//! Tolerance rationale: detection fires at the first frame whose 46.4 ms
//! window contains the click, so per-onset error is bounded by roughly
//! ±(window/2 + hop) ≈ ±43 ms. We assert 60 ms per onset (headroom for the
//! resampled path) plus a 30 ms cap on the *mean signed* offset, which is
//! what actually catches systematic misalignment (untrimmed filter delay,
//! off-by-one frame math, etc.).

mod common;

use beatloc::timeline::Onset;
use beatloc::{AnalysisOptions, Engine, analyze_file};

const CLICKS: [f64; 4] = [0.5, 1.0, 1.5, 2.0];
const DURATION: f64 = 2.5;

/// Signed offset (detected − truth) of the nearest onset per click.
fn matched_offsets(truth: &[f64], onsets: &[Onset], per_onset_tolerance: f64) -> Vec<f64> {
    truth
        .iter()
        .map(|&t| {
            onsets
                .iter()
                .map(|o| o.time - t)
                .min_by(|a, b| a.abs().total_cmp(&b.abs()))
                .filter(|dt| dt.abs() <= per_onset_tolerance)
                .unwrap_or_else(|| {
                    panic!("no onset within {per_onset_tolerance} s of click at {t} s")
                })
        })
        .collect()
}

fn assert_aligned(onsets: &[Onset]) {
    assert_eq!(
        onsets.len(),
        CLICKS.len(),
        "expected one onset per click, got {onsets:?}"
    );
    let offsets = matched_offsets(&CLICKS, onsets, 0.060);
    let mean = offsets.iter().sum::<f64>() / offsets.len() as f64;
    assert!(
        mean.abs() < 0.030,
        "systematic timing offset {mean:.4} s exceeds 30 ms"
    );
}

#[test]
fn onsets_align_at_native_analysis_rate() {
    let path = common::temp_wav_path("native.wav");
    common::write_wav_i16(
        &path,
        22_050,
        1,
        &common::click_track(22_050, &CLICKS, DURATION),
    );

    let t = analyze_file(
        &path,
        AnalysisOptions {
            engine: Engine::Dsp,
            ..Default::default()
        },
    )
    .expect("analysis failed");

    assert_eq!(t.source.sample_rate, 22_050);
    assert_eq!(t.source.channels, 1);
    assert!(
        t.source.codec.starts_with("pcm"),
        "unexpected codec {}",
        t.source.codec
    );
    assert!((t.source.duration_seconds - DURATION).abs() < 0.01);
    assert_aligned(&t.onsets);

    std::fs::remove_file(&path).ok();
}

#[test]
fn onsets_align_after_resampling_and_downmix() {
    let path = common::temp_wav_path("resampled.wav");
    let mono = common::click_track(44_100, &CLICKS, DURATION);
    let stereo = common::to_interleaved(&mono, 2);
    common::write_wav_i16(&path, 44_100, 2, &stereo);

    let t = analyze_file(
        &path,
        AnalysisOptions {
            engine: Engine::Dsp,
            ..Default::default()
        },
    )
    .expect("analysis failed");

    assert_eq!(t.source.sample_rate, 44_100);
    assert_eq!(t.source.channels, 2);
    assert!((t.source.duration_seconds - DURATION).abs() < 0.01);
    assert_aligned(&t.onsets);

    std::fs::remove_file(&path).ok();
}

#[test]
fn output_is_deterministic() {
    let path = common::temp_wav_path("determinism.wav");
    common::write_wav_i16(
        &path,
        22_050,
        1,
        &common::click_track(22_050, &CLICKS, DURATION),
    );

    let opts = AnalysisOptions {
        include_curves: true,
        engine: Engine::Dsp,
        ..Default::default()
    };
    let a = beatloc::serialize::to_json_string(&analyze_file(&path, opts.clone()).unwrap(), false)
        .unwrap();
    let b = beatloc::serialize::to_json_string(&analyze_file(&path, opts).unwrap(), false).unwrap();
    assert_eq!(a, b);

    std::fs::remove_file(&path).ok();
}

#[test]
fn silence_yields_no_onsets_and_floored_energy() {
    let path = common::temp_wav_path("silence.wav");
    common::write_wav_i16(&path, 22_050, 1, &vec![0.0f32; 22_050]);

    let t = analyze_file(
        &path,
        AnalysisOptions {
            include_curves: true,
            engine: Engine::Dsp,
            ..Default::default()
        },
    )
    .expect("analysis failed");
    assert!(t.onsets.is_empty());
    let curves = t.curves.expect("curves requested");
    assert!(curves.energy.values.iter().all(|&v| v == -120.0));
    assert!(curves.onset_strength.values.iter().all(|&v| v == 0.0));

    std::fs::remove_file(&path).ok();
}

#[test]
fn curves_follow_the_documented_grid() {
    let path = common::temp_wav_path("grid.wav");
    common::write_wav_i16(
        &path,
        22_050,
        1,
        &common::click_track(22_050, &CLICKS, DURATION),
    );

    let t = analyze_file(
        &path,
        AnalysisOptions {
            include_curves: true,
            engine: Engine::Dsp,
            ..Default::default()
        },
    )
    .unwrap();
    let curves = t.curves.unwrap();

    // Grid: hop 441 at 22050 Hz = 0.02 s; first centre at 512/22050 s.
    assert!((curves.energy.hop_seconds - 0.02).abs() < 1e-12);
    assert!((curves.energy.start_seconds - 512.0 / 22_050.0).abs() < 1e-12);
    // Valid frames: floor((len - window) / hop) + 1.
    let expected = (22_050 * 5 / 2 - 1024) / 441 + 1;
    assert_eq!(curves.energy.values.len(), expected);
    assert_eq!(curves.onset_strength.values.len(), expected);

    std::fs::remove_file(&path).ok();
}
