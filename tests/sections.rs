//! Section detection through the full pipeline: a synthetic file with a
//! sparse/quiet first half and a dense/loud second half must yield exactly
//! two sections with the boundary near the true switch.

mod common;

use beatloc::{AnalysisOptions, Engine, analyze_file};

#[test]
fn quiet_to_loud_yields_two_sections() {
    let sr = 22_050;
    // 0–10 s: sparse, quiet clicks. 10–20 s: dense, loud clicks.
    let quiet: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let loud: Vec<f64> = (0..40).map(|i| 10.0 + i as f64 * 0.25).collect();
    let mut samples = common::click_track(sr, &quiet, 20.0);
    for s in &mut samples {
        *s *= 0.1;
    }
    let loud_track = common::click_track(sr, &loud, 20.0);
    for (i, s) in loud_track.iter().enumerate() {
        if i as f64 / f64::from(sr) >= 10.0 {
            samples[i] += 0.8 * s;
        }
    }
    let path = common::temp_wav_path("sections.wav");
    common::write_wav_i16(&path, sr, 1, &samples);

    let t = analyze_file(
        &path,
        AnalysisOptions {
            engine: Engine::Dsp,
            ..Default::default()
        },
    )
    .expect("analysis failed");

    assert_eq!(t.sections.engine, "dsp-novelty-v2");
    assert_eq!(
        t.sections.items.len(),
        2,
        "sections: {:?}",
        t.sections.items
    );
    let boundary = t.sections.items[1].start;
    assert!(
        (boundary - 10.0).abs() < 1.0,
        "boundary at {boundary} s, expected ~10 s"
    );
    assert_eq!(t.sections.items[0].transition_strength, None);
    let strength = t.sections.items[1].transition_strength.unwrap();
    assert!(strength > 0.5, "strength {strength}");
    // Sections tile the file.
    assert_eq!(t.sections.items[0].start, 0.0);
    assert_eq!(t.sections.items[0].end, boundary);
    assert!((t.sections.items[1].end - t.source.duration_seconds).abs() < 1e-9);

    std::fs::remove_file(&path).ok();
}

#[test]
fn uniform_signal_yields_one_section() {
    let path = common::temp_wav_path("sections-flat.wav");
    let samples = common::click_track(22_050, &[0.5, 1.0, 1.5, 2.0], 3.0);
    common::write_wav_i16(&path, 22_050, 1, &samples);

    let t = analyze_file(
        &path,
        AnalysisOptions {
            engine: Engine::Dsp,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(t.sections.items.len(), 1);
    assert_eq!(t.sections.items[0].transition_strength, None);

    std::fs::remove_file(&path).ok();
}
