//! The canonical timeline: schema types for the JSON document.
//!
//! Schema version 0.3.0. Compatibility policy (from v1.0.0 onward): within a
//! major version, consumers must tolerate new fields appearing; fields are
//! never removed or redefined. Until 1.0.0, anything may change.
//!
//! Conventions:
//! - All times are seconds from the start of the decoded (gapless-trimmed)
//!   stream, as f64.
//! - Frame-derived times refer to frame *centres* for the DSP tracks (see
//!   `dsp::frame_center_seconds`); the neural engine reports its own model
//!   grid (frame index / 50 s, i.e. centred STFT frames). Both are "seconds
//!   from stream start"; the per-engine convention is recorded via the
//!   `analysis` block and each section's `engine` provenance field.
//! - Dense per-frame data lives in `Curve`: values sampled every
//!   `hop_seconds` starting at `start_seconds`, so `t(i) = start + i * hop`.
//! - Unknown and unsupported results are represented by ABSENT fields, never
//!   sentinel values. There is deliberately no "confidence" field yet: we do
//!   not emit numbers whose meaning is undefined.
//! - Bars are numbered from the first detected downbeat (bar 1). No
//!   time-signature claim is made: beats-per-bar may vary.

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
    /// Global tempo. Absent when no periodicity could be established —
    /// "unknown" is represented by absence, never by a sentinel value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tempo: Option<Tempo>,
    pub beats: Beats,
    /// Downbeats (bar starts). Present only when the engine supports them
    /// (the DSP baseline does not). Absent = unsupported or none detected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downbeats: Option<Downbeats>,
    pub onsets: Vec<Onset>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curves: Option<Curves>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tempo {
    /// Engine that produced the estimate (provenance).
    pub engine: String,
    /// Global tempo in beats per minute.
    pub bpm: f64,
    /// Normalized autocorrelation at the chosen period, in (0, 1].
    /// Uncalibrated diagnostic of periodicity strength — NOT a probability.
    /// Present only for engines where this diagnostic exists (DSP baseline).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub periodicity: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Beats {
    /// Engine that produced the beat track (provenance).
    pub engine: String,
    /// Beat events in ascending time order. Empty when no reliable periodic
    /// structure was found — beats are never extrapolated into silence.
    pub items: Vec<Beat>,
}

/// One beat.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Beat {
    /// Seconds from decoded stream start (per-engine frame convention,
    /// documented in `analysis` and the engine's provenance).
    pub time: f64,
    /// 0-based position in the detected beat sequence (a plain counter).
    pub index: u32,
    /// 1-based bar number, counted from the first detected downbeat (bar 1).
    /// Absent before the first downbeat and whenever downbeats are unknown.
    /// This is NOT a time-signature claim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bar: Option<u32>,
    /// 1-based position of this beat within its bar (downbeat = 1, the
    /// JAMS/Harmonix convention). Absent under the same conditions as `bar`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bar_position: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Downbeats {
    /// Engine that produced the downbeat track (provenance).
    pub engine: String,
    pub items: Vec<Downbeat>,
}

/// One downbeat: the first beat of a bar.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Downbeat {
    /// Seconds from decoded stream start.
    pub time: f64,
    /// 1-based bar number starting at the first detected downbeat.
    pub bar: u32,
}

/// Assign bar numbers and bar positions to beats, given downbeats that have
/// been snapped to beat times. Bars are 1-based from the first downbeat;
/// beats before it (a possible anacrusis) get `None` — the unknown state.
/// We deliberately do NOT infer a time signature: the number of beats in a
/// bar may vary between bars, and that's representable here.
pub fn build_bars(beats: &[f64], downbeats: &[f64]) -> (Vec<Beat>, Vec<Downbeat>) {
    use std::collections::HashSet;
    let downbeat_set: HashSet<u64> = downbeats.iter().map(|t| t.to_bits()).collect();
    let downbeat_items: Vec<Downbeat> = downbeats
        .iter()
        .enumerate()
        .map(|(i, &t)| Downbeat { time: t, bar: i as u32 + 1 })
        .collect();

    let mut items = Vec::with_capacity(beats.len());
    let mut bar = 0u32;
    let mut position = 0u32;
    for (index, &time) in beats.iter().enumerate() {
        let (bar_field, pos_field) = if downbeat_set.contains(&time.to_bits()) {
            bar += 1;
            position = 1;
            (Some(bar), Some(position))
        } else if bar > 0 {
            position += 1;
            (Some(bar), Some(position))
        } else {
            (None, None)
        };
        items.push(Beat { time, index: index as u32, bar: bar_field, bar_position: pos_field });
    }
    (items, downbeat_items)
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
    /// Sum of positive linear-magnitude differences across bins;
    /// arbitrary units.
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
