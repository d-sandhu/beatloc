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
//!   trimmed; timestamps may shift by up to ~50 ms for such files.
//! - Decoding currently buffers the whole file in memory (~13 MB per hour of
//!   mono audio at the analysis rate). Streaming decode is planned before the
//!   neural engine lands.

use std::path::Path;

pub mod decode;
pub mod dsp;
pub mod eval;
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
pub const SCHEMA_VERSION: &str = "0.2.0";

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
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AnalysisOptions {
    /// Include dense per-frame curves (energy, onset strength) in the output.
    pub include_curves: bool,
}

/// Run the full V0.1 pipeline: decode → downmix → resample → features → timeline.
pub fn analyze_file(path: &Path, options: AnalysisOptions) -> Result<Timeline, BeatlocError> {
    let decoded = decode::decode_file(path)?;
    let duration_seconds = decoded.duration_seconds();

    let mono = dsp::resample::to_analysis_rate(&decoded.samples, decoded.sample_rate)?;

    let flux = dsp::onset::onset_strength(&mono, STFT_WINDOW, STFT_HOP);
    let onsets = dsp::onset::pick_onsets(&flux, ANALYSIS_SAMPLE_RATE, STFT_WINDOW, STFT_HOP);

    // V0.2: classical beat baseline on the same envelope (Ellis 2007).
    let fps = ANALYSIS_SAMPLE_RATE as f64 / STFT_HOP as f64;
    let (tempo, beats) = match dsp::beat::track_beats(&flux, fps) {
        Some(track) => (
            Some(timeline::Tempo {
                engine: dsp::beat::ENGINE,
                bpm: track.bpm,
                periodicity: track.periodicity,
            }),
            timeline::Beats {
                engine: dsp::beat::ENGINE,
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
                    })
                    .collect(),
            },
        ),
        None => (None, timeline::Beats { engine: dsp::beat::ENGINE, items: Vec::new() }),
    };

    let curves = options.include_curves.then(|| timeline::Curves {
        energy: timeline::Curve {
            start_seconds: dsp::frame_center_seconds(0, STFT_WINDOW, STFT_HOP, ANALYSIS_SAMPLE_RATE),
            hop_seconds: STFT_HOP as f64 / ANALYSIS_SAMPLE_RATE as f64,
            units: timeline::Units::Dbfs,
            values: dsp::energy::frame_rms_dbfs(&mono, STFT_WINDOW, STFT_HOP),
        },
        onset_strength: timeline::Curve {
            start_seconds: dsp::frame_center_seconds(0, STFT_WINDOW, STFT_HOP, ANALYSIS_SAMPLE_RATE),
            hop_seconds: STFT_HOP as f64 / ANALYSIS_SAMPLE_RATE as f64,
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
        onsets,
        curves,
    })
}
