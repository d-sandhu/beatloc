//! V0.2 exit-condition tests: beat timestamps + tempo from the classical
//! DSP engine, validated end-to-end on synthetic audio with known ground
//! truth, plus the documented failure mode (tempo change).

mod common;

use beatloc::eval::matched_offsets;
use beatloc::{AnalysisOptions, Engine, Timeline, analyze_file};

fn click_times(bpm: f64, count: usize, start: f64) -> Vec<f64> {
    (0..count).map(|i| start + i as f64 * 60.0 / bpm).collect()
}

fn analyze_clicks(name: &str, sample_rate: u32, times: &[f64], duration: f64) -> Timeline {
    let path = common::temp_wav_path(name);
    common::write_wav_i16(
        &path,
        sample_rate,
        1,
        &common::click_track(sample_rate, times, duration),
    );
    let timeline = analyze_file(
        &path,
        AnalysisOptions {
            engine: Engine::Dsp,
            ..Default::default()
        },
    )
    .expect("analysis failed");
    std::fs::remove_file(&path).ok();
    timeline
}

fn beat_times(t: &Timeline) -> Vec<f64> {
    t.beats.items.iter().map(|b| b.time).collect()
}

#[test]
fn tracks_120_bpm_click_train() {
    let truth = click_times(120.0, 16, 1.0);
    let t = analyze_clicks("beats-120.wav", 44_100, &truth, 9.5); // exercises resampling too

    let tempo = t.tempo.as_ref().expect("tempo estimated");
    assert_eq!(tempo.engine, "dsp-ellis2007");
    assert!((tempo.bpm - 120.0).abs() < 1.5, "bpm {}", tempo.bpm);
    assert!(
        tempo.periodicity.unwrap() > 0.5,
        "periodicity {:?}",
        tempo.periodicity
    );

    let beats = beat_times(&t);
    assert_eq!(beats.len(), truth.len(), "beats {beats:?}");
    let offsets = matched_offsets(&truth, &beats, 0.045);
    assert_eq!(
        offsets.len(),
        truth.len(),
        "every ground-truth click matched"
    );
    let mean_abs = offsets.iter().map(|o| o.abs()).sum::<f64>() / offsets.len() as f64;
    assert!(mean_abs < 0.025, "mean |offset| {:.4} s", mean_abs);
    // Indices are a clean 0-based counter.
    assert!(
        t.beats
            .items
            .iter()
            .enumerate()
            .all(|(i, b)| b.index == i as u32)
    );
}

#[test]
fn tracks_90_bpm_click_train() {
    // 90 BPM at 50 fps = 33.33 frames per beat: non-integer period exercises
    // the parabolic ACF refinement and the DP's interval tolerance.
    let truth = click_times(90.0, 16, 1.0);
    let t = analyze_clicks("beats-90.wav", 22_050, &truth, 12.5);

    let tempo = t.tempo.as_ref().expect("tempo estimated");
    assert!((tempo.bpm - 90.0).abs() < 1.5, "bpm {}", tempo.bpm);

    let beats = beat_times(&t);
    assert_eq!(beats.len(), truth.len(), "beats {beats:?}");
    let offsets = matched_offsets(&truth, &beats, 0.045);
    assert_eq!(offsets.len(), truth.len());
}

#[test]
fn silence_yields_no_tempo_and_empty_beats() {
    let path = common::temp_wav_path("beats-silence.wav");
    common::write_wav_i16(&path, 22_050, 1, &vec![0.0f32; 22_050 * 2]);

    let t = analyze_file(
        &path,
        AnalysisOptions {
            engine: Engine::Dsp,
            ..Default::default()
        },
    )
    .expect("analysis failed");
    assert!(t.tempo.is_none(), "no tempo may be invented for silence");
    assert!(t.beats.items.is_empty());

    // And the serialized JSON must omit `tempo` entirely (no sentinel).
    let json = beatloc::serialize::to_json_string(&t, false).unwrap();
    assert!(
        !json.contains("\"tempo\""),
        "tempo key must be absent, got: {json}"
    );

    std::fs::remove_file(&path).ok();
}

/// Documented failure case, kept as a regression net: the tracker assumes one
/// global period, so a tempo change cannot be followed correctly. We assert
/// only that behavior stays defined (terminates, reports *a* tempo, emits
/// beats) — the numbers here are expected to be mediocre by construction.
#[test]
fn tempo_change_is_defined_but_not_solved() {
    let mut truth = click_times(120.0, 8, 1.0);
    let switch = *truth.last().unwrap() + 60.0 / 120.0;
    truth.extend(click_times(90.0, 8, switch).iter().skip(1).copied());

    let t = analyze_clicks("beats-tempo-change.wav", 22_050, &truth, 12.5);
    assert!(t.tempo.is_some());
    assert!(!t.beats.items.is_empty());
}
