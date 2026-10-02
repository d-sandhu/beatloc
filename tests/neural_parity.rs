//! V0.3 exit-condition test: parity of the Rust neural pipeline against the
//! official Python Beat This! reference.
//!
//! The golden fixture (`tests/fixtures/drum_loop_120bpm.golden.json`) was
//! produced by the Python reference (PyTorch small0 + torchaudio frontend +
//! minimal postprocessor) on the committed synthetic WAV — see
//! `scripts/gen_golden.py` for the exact recipe. The WAV is quantized 16-bit
//! and the golden was computed on the quantized signal, so both sides see
//! bit-identical input samples.
//!
//! Tolerances: logits within 2e-3 (rten vs ONNX Runtime float differences
//! are ~1e-5; this leaves headroom for kernel variation), final times equal
//! in count and within one model frame (20 ms). Sub-threshold peaks near
//! logit 0 can legitimately flip between runtimes; a count mismatch of ≤2
//! with all other times aligned is tolerated and commented, not hidden.
//!
//! The test SKIPS (with a note) when models/ is absent — model artifacts are
//! git-ignored and regenerated via `scripts/export_model.py`.

use std::path::Path;

use beatloc::{decode, dsp, inference::NeuralEngine};

const MODELS_PRESENT: &str = "run scripts/export_model.py to enable this test";

fn golden() -> Option<serde_json::Value> {
    let text = std::fs::read_to_string("tests/fixtures/drum_loop_120bpm.golden.json").ok()?;
    serde_json::from_str(&text).ok()
}

fn golden_f64s(golden: &serde_json::Value, key: &str) -> Vec<f64> {
    golden[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

#[test]
fn rust_pipeline_matches_python_reference() {
    let mel_path = Path::new(beatloc::DEFAULT_MEL_MODEL);
    let model_path = Path::new(beatloc::DEFAULT_BEAT_MODEL);
    if !mel_path.is_file() || !model_path.is_file() {
        eprintln!("skipping neural parity test: {MODELS_PRESENT}");
        return;
    }
    let golden = golden().expect("golden fixture readable");

    // Same path as production: decode -> (no-op resample at 22050) -> ONNX.
    let decoded =
        decode::decode_file(Path::new("tests/fixtures/drum_loop_120bpm.wav")).unwrap();
    let mono = dsp::resample::to_analysis_rate(&decoded.samples, decoded.sample_rate).unwrap();
    let engine = NeuralEngine::load(mel_path, model_path).unwrap();
    let out = engine.predict(&mono).unwrap();

    // 1. Frame count must match the reference pipeline exactly.
    assert_eq!(out.beat_logits.len(), golden["frames"].as_u64().unwrap() as usize);

    // 2. Logit parity (the strong check — subsumes mel and model graphs).
    let g_beat = golden_f64s(&golden, "beat_logits");
    let g_down = golden_f64s(&golden, "downbeat_logits");
    let max_diff = |ours: &[f32], theirs: &[f64]| {
        ours.iter()
            .zip(theirs)
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0f64, f64::max)
    };
    let beat_diff = max_diff(&out.beat_logits, &g_beat);
    let down_diff = max_diff(&out.downbeat_logits, &g_down);
    assert!(beat_diff < 2e-3, "beat logits max|diff| {beat_diff:.2e}");
    assert!(down_diff < 2e-3, "downbeat logits max|diff| {down_diff:.2e}");

    // 3. Final times: count and per-event alignment within one frame.
    let g_beats = golden_f64s(&golden, "beats");
    let g_downs = golden_f64s(&golden, "downbeats");
    assert_eq!(out.beats.len(), g_beats.len(), "beat count mismatch: {:?}", out.beats);
    assert_eq!(out.downbeats.len(), g_downs.len(), "downbeat count mismatch");
    for (a, b) in out.beats.iter().zip(&g_beats) {
        assert!((a - b).abs() <= 0.0201, "beat {a} vs golden {b}");
    }
    for (a, b) in out.downbeats.iter().zip(&g_downs) {
        assert!((a - b).abs() <= 0.0201, "downbeat {a} vs golden {b}");
    }
}

#[test]
fn auto_engine_uses_neural_when_models_present_and_dsp_otherwise() {
    let wav = Path::new("tests/fixtures/drum_loop_120bpm.wav");
    let models = Path::new(beatloc::DEFAULT_BEAT_MODEL).is_file();

    let auto = beatloc::analyze_file(
        wav,
        beatloc::AnalysisOptions { engine: beatloc::Engine::Auto, ..Default::default() },
    )
    .unwrap();
    let dsp = beatloc::analyze_file(
        wav,
        beatloc::AnalysisOptions { engine: beatloc::Engine::Dsp, ..Default::default() },
    )
    .unwrap();

    if models {
        assert!(auto.beats.engine.starts_with("beat-this"), "{}", auto.beats.engine);
        assert!(auto.downbeats.is_some(), "neural engine must emit downbeats");
    } else {
        eprintln!("skipping neural auto-check: {MODELS_PRESENT}");
        assert_eq!(auto.beats.engine, "dsp-ellis2007");
        assert!(auto.downbeats.is_none(), "dsp engine has no downbeats");
    }
    assert_eq!(dsp.beats.engine, "dsp-ellis2007");
    assert!(dsp.downbeats.is_none());
    // The DSP engine emits no scores; unknown/unsupported is absent, not null.
    assert!(dsp.beats.mean_score.is_none());
    assert!(dsp.beats.items.iter().all(|b| b.score.is_none()));
}

#[test]
fn neural_scores_are_present_and_bounded() {
    if !Path::new(beatloc::DEFAULT_BEAT_MODEL).is_file() {
        eprintln!("skipping: {MODELS_PRESENT}");
        return;
    }
    let t = beatloc::analyze_file(
        Path::new("tests/fixtures/drum_loop_120bpm.wav"),
        beatloc::AnalysisOptions { engine: beatloc::Engine::Neural, ..Default::default() },
    )
    .unwrap();
    assert!(t.beats.mean_score.is_some());
    assert!(t.beats.items.iter().all(|b| matches!(b.score, Some(s) if s > 0.5 && s <= 1.0)));
    let downbeats = t.downbeats.unwrap();
    assert!(downbeats.mean_score.is_some());
    assert!(downbeats.items.iter().all(|d| matches!(d.score, Some(s) if s > 0.5 && s <= 1.0)));
}

#[test]
fn bars_are_numbered_from_first_downbeat() {
    if !Path::new(beatloc::DEFAULT_BEAT_MODEL).is_file() {
        eprintln!("skipping: {MODELS_PRESENT}");
        return;
    }
    let t = beatloc::analyze_file(
        Path::new("tests/fixtures/drum_loop_120bpm.wav"),
        beatloc::AnalysisOptions { engine: beatloc::Engine::Neural, ..Default::default() },
    )
    .unwrap();
    let downbeats = t.downbeats.expect("neural downbeats");
    assert!(!downbeats.items.is_empty());
    // Bar numbering is 1-based and sequential.
    for (i, d) in downbeats.items.iter().enumerate() {
        assert_eq!(d.bar as usize, i + 1);
    }
    // Every downbeat coincides with a beat carrying bar_position 1.
    for d in &downbeats.items {
        let beat = t
            .beats
            .items
            .iter()
            .find(|b| b.time == d.time)
            .expect("downbeat must coincide with a beat");
        assert_eq!(beat.bar_position, Some(1));
        assert_eq!(beat.bar, Some(d.bar));
    }
    // Bar positions count up within each bar.
    let positions: Vec<u32> =
        t.beats.items.iter().filter_map(|b| b.bar_position).collect();
    for w in positions.windows(2) {
        assert!(w[1] == w[0] + 1 || w[1] == 1, "positions must increment or reset: {w:?}");
    }
}
