//! beatloc — convert audio into a machine-readable musical timeline (JSON).
//!
//! Current scope (V0.2): decode WAV/MP3/FLAC, report duration and source
//! metadata, emit onset timestamps, beat timestamps, and a global tempo
//! estimate from the classical DSP baseline (Ellis 2007), plus optional dense
//! energy/onset-strength curves — all as versioned JSON on one documented
//! time base. Downbeats and bar positions arrive with the neural engine.
//!
//! # Numerical assumptions (V0.1)
//!
//! - All analysis runs on mono f32 at [`ANALYSIS_SAMPLE_RATE`] Hz. Stereo and
//!   multichannel input is downmixed by arithmetic mean (NOT energy-preserving
//!   downmix; the mean is what downstream MIR reference pipelines use).
//! - Frames use a periodic Hann window of [`STFT_WINDOW`] samples with hop
//!   [`STFT_HOP`] samples (50 frames/second at 22050 Hz — the same time base
//!   the Beat This! model uses, which keeps later fusion trivial).
//! - All timestamps are seconds from the start of the *decoded* stream (after
//!   gapless trimming) to the *centre* of an analysis frame:
//!   `t(i) = (i * HOP + WINDOW/2) / 22050`.
//! - Resampling uses rubato's FFT resampler via `process_all`, which trims the
//!   filter startup delay, so resampled output stays time-aligned with input.
//!
//! Known limitations (documented, not hidden):
//! - MP3 files without a Xing/LAME tag carry encoder delay that cannot be
//!   trimmed; timestamps may shift by up to ~50 ms for such files. These
//!   files are detected at decode time and a warning is printed to stderr
//!   (measured shift on the test fixture: +40 ms).
//! - AAC (incl. .m4a): Symphonia does not yet trim encoder delay/padding
//!   for AAC — timestamps may shift by up to ~50 ms. WAV/MP3/FLAC/OGG are
//!   sample-accurate.
//! - Decoding currently buffers the whole file in memory: the interleaved
//!   native-rate decode plus the analysis-rate mono signal (~0.3 GB per hour
//!   at 22050 Hz; a 2-hour 44.1 kHz stereo file peaks around ~2.8 GB).
//!   Fine for songs and ad spots; streaming decode is planned if long mixes
//!   become a real use case.

use std::path::{Path, PathBuf};

pub mod decode;
pub mod dsp;
pub mod eval;
pub mod inference;
pub mod serialize;
pub mod timeline;

pub use timeline::Timeline;

/// Sample rate all analysis runs at. Matches the Beat This! preprocessing
/// contract so V0.1 features and the later neural engine share one time base.
pub const ANALYSIS_SAMPLE_RATE: u32 = 22_050;

/// Analysis window size in samples (≈46.4 ms at 22050 Hz).
pub const STFT_WINDOW: usize = 1024;

/// Analysis hop in samples (= 20 ms, 50 fps at 22050 Hz).
pub const STFT_HOP: usize = 441;

/// Schema identity of the emitted JSON document.
pub const SCHEMA_NAME: &str = "beatloc-timeline";
pub const SCHEMA_VERSION: &str = "0.6.0";

/// Default locations of the ONNX model artifacts (regenerate with
/// `scripts/export_model.py`; see models/manifest.json for provenance).
pub const DEFAULT_BEAT_MODEL: &str = "models/beat_this_small0.onnx";
pub const DEFAULT_MEL_MODEL: &str = "models/mel_spectrogram.onnx";

#[derive(Debug, thiserror::Error)]
pub enum BeatlocError {
    #[error("cannot read input: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported or unrecognized audio: {0}")]
    Unsupported(String),
    #[error("decode failed: {0}")]
    Decode(String),
    #[error("resampling failed: {0}")]
    Resample(String),
    #[error("neural inference failed: {0}")]
    Inference(String),
}

/// Which beat engine to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Engine {
    /// Neural if the model files are present, DSP otherwise (recorded in
    /// each section's `engine` provenance field either way).
    #[default]
    Auto,
    /// Classical DSP baseline (Ellis 2007 DP). No downbeats.
    Dsp,
    /// Beat This! neural engine. Errors if the model files are missing.
    Neural,
}

#[derive(Debug, Clone, Default)]
pub struct AnalysisOptions {
    /// Include dense per-frame curves (energy, onset strength) in the output.
    pub include_curves: bool,
    /// Beat engine selection (default: auto).
    pub engine: Engine,
    /// Beat model ONNX path (default: ./models/beat_this_small0.onnx).
    pub beat_model: Option<PathBuf>,
    /// Mel frontend ONNX path (default: ./models/mel_spectrogram.onnx).
    pub mel_model: Option<PathBuf>,
}

/// Run the full V0.1 pipeline: decode → downmix → resample → features → timeline.
pub fn analyze_file(path: &Path, options: AnalysisOptions) -> Result<Timeline, BeatlocError> {
    let decoded = decode::decode_file(path)?;
    if decoded.encoder_delay_trimmed == Some(false) {
        eprintln!(
            "warning: {}: MP3 has no Xing/LAME gapless tag — encoder delay cannot be \
             trimmed; all timestamps may be shifted by up to ~50 ms",
            path.display()
        );
    }
    let duration_seconds = decoded.duration_seconds();

    let mono = dsp::resample::to_analysis_rate(&decoded.samples, decoded.sample_rate)?;

    // One shared STFT grid feeds both the onset envelope and sections.
    let mags = dsp::stft::stft_magnitudes(&mono, STFT_WINDOW, STFT_HOP);
    let flux = dsp::onset::flux_from_mags(&mags);
    let onsets = dsp::onset::pick_onsets(&flux, ANALYSIS_SAMPLE_RATE, STFT_WINDOW, STFT_HOP);
    let energy = dsp::energy::frame_rms_dbfs(&mono, STFT_WINDOW, STFT_HOP);

    // Sections: spans between consecutive transition boundaries (spectral
    // self-similarity novelty over the shared grid; engine-independent).
    let hop_seconds = STFT_HOP as f64 / ANALYSIS_SAMPLE_RATE as f64;
    let transitions = dsp::sections::detect_transitions(&mags, hop_seconds, ANALYSIS_SAMPLE_RATE);
    let mut section_items = Vec::with_capacity(transitions.len() + 1);
    let mut start = 0.0f64;
    for (i, t) in transitions.iter().enumerate() {
        let boundary =
            dsp::frame_center_seconds(t.frame, STFT_WINDOW, STFT_HOP, ANALYSIS_SAMPLE_RATE);
        // The strength on a section belongs to the boundary that STARTS it.
        section_items.push(timeline::Section {
            index: i as u32,
            start,
            end: boundary,
            transition_strength: if i == 0 { None } else { Some(transitions[i - 1].strength) },
        });
        start = boundary;
    }
    let n_transitions = transitions.len();
    section_items.push(timeline::Section {
        index: n_transitions as u32,
        start,
        end: duration_seconds,
        transition_strength: if n_transitions == 0 {
            None
        } else {
            Some(transitions[n_transitions - 1].strength)
        },
    });
    let sections = timeline::Sections { engine: "dsp-novelty-v2", items: section_items };

    // V0.2/V0.3: beat engine. DSP is the baseline; neural runs when selected
    // and its model files are available.
    let fps = ANALYSIS_SAMPLE_RATE as f64 / STFT_HOP as f64;
    let beat_model = options
        .beat_model
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BEAT_MODEL));
    let mel_model =
        options.mel_model.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_MEL_MODEL));
    let neural_available = beat_model.is_file() && mel_model.is_file();

    let (mut tempo, beats, downbeats) = match options.engine {
        Engine::Neural if !neural_available => {
            return Err(BeatlocError::Inference(format!(
                "neural engine selected but model files are missing (expected {} and {}; \
                 regenerate with scripts/export_model.py)",
                beat_model.display(),
                mel_model.display()
            )));
        }
        Engine::Neural | Engine::Auto if neural_available => {
            let engine = inference::NeuralEngine::load(&mel_model, &beat_model)?;
            let out = engine.predict(&mono)?;
            let name = model_engine_name(&beat_model);
            let (mut items, mut downbeat_items) =
                timeline::build_bars(&out.beats, &out.downbeats);
            for (item, &score) in items.iter_mut().zip(&out.beat_scores) {
                item.score = Some(score);
            }
            for (item, &score) in downbeat_items.iter_mut().zip(&out.downbeat_scores) {
                item.score = Some(score);
            }
            let mean_score = |s: &[f32]| {
                if s.is_empty() { None } else { Some(s.iter().sum::<f32>() / s.len() as f32) }
            };
            let tempo = neural_tempo(&out.beats, &name);
            (
                tempo,
                timeline::Beats {
                    engine: name.clone(),
                    mean_score: mean_score(&out.beat_scores),
                    items,
                },
                Some(timeline::Downbeats {
                    engine: name,
                    mean_score: mean_score(&out.downbeat_scores),
                    items: downbeat_items,
                }),
            )
        }
        _ => {
            // DSP baseline (also the auto fallback when models are absent).
            let (tempo, beats) = match dsp::beat::track_beats(&flux, fps) {
                Some(track) => (
                    Some(timeline::Tempo {
                        engine: dsp::beat::ENGINE.to_string(),
                        bpm: track.bpm,
                        periodicity: Some(track.periodicity),
                        local: None,
                    }),
                    timeline::Beats {
                        engine: dsp::beat::ENGINE.to_string(),
                        mean_score: None, // the DSP engine emits no scores
                        items: track
                            .frames
                            .iter()
                            .enumerate()
                            .map(|(index, &frame)| timeline::Beat {
                                time: dsp::frame_center_seconds(
                                    frame,
                                    STFT_WINDOW,
                                    STFT_HOP,
                                    ANALYSIS_SAMPLE_RATE,
                                ),
                                index: index as u32,
                                bar: None,
                                bar_position: None,
                                score: None,
                            })
                            .collect(),
                    },
                ),
                None => (
                    None,
                    timeline::Beats { engine: dsp::beat::ENGINE.to_string(), mean_score: None, items: Vec::new() },
                ),
            };
            (tempo, beats, None)
        }
    };

    // Local tempo track, derived from the final beat list (engine-agnostic).
    if let Some(t) = tempo.as_mut() {
        let times: Vec<f64> = beats.items.iter().map(|b| b.time).collect();
        let points = dsp::local_tempo_points(&times, 4);
        if !points.is_empty() {
            t.local = Some(points);
        }
    }

    let curves = options.include_curves.then(|| timeline::Curves {
        energy: timeline::Curve {
            start_seconds: dsp::frame_center_seconds(0, STFT_WINDOW, STFT_HOP, ANALYSIS_SAMPLE_RATE),
            hop_seconds,
            units: timeline::Units::Dbfs,
            values: energy,
        },
        onset_strength: timeline::Curve {
            start_seconds: dsp::frame_center_seconds(0, STFT_WINDOW, STFT_HOP, ANALYSIS_SAMPLE_RATE),
            hop_seconds,
            units: timeline::Units::SpectralFlux,
            values: flux,
        },
    });

    Ok(Timeline {
        format: timeline::FormatInfo { name: SCHEMA_NAME, version: SCHEMA_VERSION },
        generator: timeline::GeneratorInfo { name: "beatloc", version: env!("CARGO_PKG_VERSION") },
        source: timeline::SourceInfo {
            duration_seconds,
            sample_rate: decoded.sample_rate,
            channels: decoded.channels,
            codec: decoded.codec,
        },
        analysis: timeline::AnalysisInfo {
            sample_rate: ANALYSIS_SAMPLE_RATE,
            window_size: STFT_WINDOW,
            hop_size: STFT_HOP,
            window: "hann (periodic)",
            timestamps: "seconds from decoded stream start (gapless-trimmed) to frame centre",
        },
        tempo,
        beats,
        downbeats,
        sections,
        onsets,
        curves,
    })
}

/// Collect supported audio files under `dir` (recursive iff `recursive`),
/// sorted by path for deterministic batch processing.
pub fn find_audio_files(dir: &Path, recursive: bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if recursive {
                    stack.push(path);
                }
            } else if matches!(
                path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
                Some("wav" | "mp3" | "flac" | "ogg" | "oga" | "m4a" | "aac")
            ) {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Engine provenance string from a model file name, e.g.
/// "beat_this_small0.onnx" → "beat-this-small0".
fn model_engine_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .replace('_', "-")
}

/// Global tempo from (neural) beat times: BPM = 60 / median inter-beat
/// interval. Deliberately simple; local tempo is a later milestone.
fn neural_tempo(beats: &[f64], engine: &str) -> Option<timeline::Tempo> {
    if beats.len() < 2 {
        return None;
    }
    let mut ibis: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    ibis.sort_by(f64::total_cmp);
    let median = if ibis.len() % 2 == 1 {
        ibis[ibis.len() / 2]
    } else {
        f64::midpoint(ibis[ibis.len() / 2 - 1], ibis[ibis.len() / 2])
    };
    if median <= 0.0 {
        return None;
    }
    Some(timeline::Tempo {
        engine: format!("{engine} (median-ibi)"),
        bpm: 60.0 / median,
        periodicity: None,
        local: None,
    })
}
