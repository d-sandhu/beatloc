//! The canonical timeline: schema types for the JSON document.
//!
//! Schema version 0.1.0. Compatibility policy (from v1.0.0 onward): within a
//! major version, consumers must tolerate new fields appearing; fields are
//! never removed or redefined. Until 1.0.0, anything may change.
//!
//! Conventions:
//! - All times are seconds from the start of the decoded (gapless-trimmed)
//!   stream, as f64.
//! - Frame-derived times refer to frame *centres* (see `dsp::frame_center_seconds`).
//! - Dense per-frame data lives in `Curve`: values sampled every
//!   `hop_seconds` starting at `start_seconds`, so `t(i) = start + i * hop`.
//! - There is deliberately no "confidence" field yet: V0.1 onsets have no
//!   calibrated score, and we do not emit numbers whose meaning is undefined.

use serde::Serialize;

/// One detected onset.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Onset {
    /// Seconds from decoded stream start (frame-centre convention).
    pub time: f64,
    /// Spectral flux at the picked frame (arbitrary, relative units).
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Timeline {
    pub format: FormatInfo,
    pub generator: GeneratorInfo,
    pub source: SourceInfo,
    pub analysis: AnalysisInfo,
    pub onsets: Vec<Onset>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curves: Option<Curves>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FormatInfo {
    /// Always "beatloc-timeline".
    pub name: &'static str,
    /// Schema version, semver. Bumped on any output change.
    pub version: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeneratorInfo {
    pub name: &'static str,
    /// beatloc binary version (env!("CARGO_PKG_VERSION")).
    pub version: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceInfo {
    /// Decoded duration in seconds (post gapless trim), from actual sample count.
    pub duration_seconds: f64,
    /// Native sample rate of the source.
    pub sample_rate: u32,
    /// Channel count of the source before downmixing.
    pub channels: usize,
    /// Decoder's short codec name, e.g. "mp3", "flac", "pcm-s16le".
    pub codec: String,
}

/// The numerical assumptions all timestamps and curves rely on.
/// Serialized so consumers can verify alignment semantics programmatically.
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisInfo {
    /// Rate at which analysis ran (all frame times are in this domain).
    pub sample_rate: u32,
    pub window_size: usize,
    pub hop_size: usize,
    pub window: &'static str,
    /// Human-readable statement of the timestamp convention.
    pub timestamps: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Units {
    /// Frame RMS in decibels relative to full scale (floor -120).
    Dbfs,
    /// Sum of positive log-magnitude differences across bins; arbitrary units.
    SpectralFlux,
}

/// A dense per-frame series: `t(i) = start_seconds + i * hop_seconds`.
#[derive(Debug, Clone, Serialize)]
pub struct Curve {
    pub start_seconds: f64,
    pub hop_seconds: f64,
    pub units: Units,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Curves {
    pub energy: Curve,
    pub onset_strength: Curve,
}
