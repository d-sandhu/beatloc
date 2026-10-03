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
    /// 0-based index of the beat whose interval contains this onset.
    /// Absent when the onset falls outside the beat span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beat_index: Option<u32>,
    /// Position within that beat interval, [0, 1): 0 = on the beat,
    /// 0.5 = halfway to the next. Continuous, NOT quantized — swung events
    /// keep their true position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beat_phase: Option<f64>,
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
    /// Per-bar overview (energy, onset density) derived from beats +
    /// energy + onsets. Absent when bars are unknown (no downbeats).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bars: Option<Vec<Bar>>,
    /// Section boundaries from feature-contrast novelty. Detects THAT a
    /// transition happens, not WHAT the section is — no semantic labels.
    pub sections: Sections,
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
    /// Local tempo track: one point per beat interval (median IBI over ±4
    /// intervals, timestamped at the interval midpoint). Absent when there
    /// are fewer than 2 beats.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local: Option<Vec<LocalTempoPoint>>,
}

/// One local tempo sample.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct LocalTempoPoint {
    /// Seconds from decoded stream start (midpoint of the beat interval).
    pub time: f64,
    /// Local tempo in BPM (median inter-beat interval over ±4 intervals).
    pub bpm: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Beats {
    /// Engine that produced the beat track (provenance).
    pub engine: String,
    /// Mean of the per-beat `score`s. Present only for engines that emit
    /// scores. Uncalibrated — a relative trust signal, not a probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_score: Option<f32>,
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
    /// Engine score for this beat (sigmoid of the model logit at the picked
    /// frame; neural engine only). UNCALIBRATED: higher = the model asserted
    /// the beat more strongly, NOT a probability of correctness.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Downbeats {
    /// Engine that produced the downbeat track (provenance).
    pub engine: String,
    /// Mean of the per-downbeat scores, when available (uncalibrated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_score: Option<f32>,
    pub items: Vec<Downbeat>,
}

/// One downbeat: the first beat of a bar.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Downbeat {
    /// Seconds from decoded stream start.
    pub time: f64,
    /// 1-based bar number starting at the first detected downbeat.
    pub bar: u32,
    /// Engine score (same semantics as `Beat.score`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Sections {
    /// Detector provenance.
    pub engine: &'static str,
    pub items: Vec<Section>,
}

/// One section: the span between two transitions (or file start/end).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Section {
    pub index: u32,
    /// Seconds from decoded stream start.
    pub start: f64,
    pub end: f64,
    /// Strength of the transition that STARTS this section: novelty peak
    /// height relative to the track's maximum, in [0, 1]. Uncalibrated,
    /// meaningful only for ranking within the same track. Absent on the
    /// first section (no preceding boundary).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transition_strength: Option<f32>,
}

/// One bar with derived aggregates — the bar-level overview consumers
/// otherwise recompute by hand (energy shape finds breaks; onset density
/// finds fills).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Bar {
    /// 1-based bar number, matching `beats.items[].bar`.
    pub bar: u32,
    /// Seconds from decoded stream start: first beat of the bar.
    pub start: f64,
    /// First beat of the NEXT bar. For the final bar: last beat + the
    /// median inter-beat interval (an estimate; documented as such).
    pub end: f64,
    /// Mean of the energy frames (dBFS) within [start, end). Absent only in
    /// the degenerate case of a bar containing no energy frames at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_energy_dbfs: Option<f32>,
    /// Number of onsets within [start, end).
    pub onset_count: u32,
}

/// Build the per-bar overview from beats that carry bar numbers, the energy
/// curve, and onset times. Returns None when no beat has a bar (no
/// downbeats — the DSP engine — or everything is anacrusis).
pub fn bar_overview(
    beats: &[Beat],
    energy: &[f32],
    onset_times: &[f64],
    window: usize,
    hop: usize,
    sample_rate: u32,
) -> Option<Vec<Bar>> {
    let max_bar = beats.iter().filter_map(|b| b.bar).max()?;
    if beats.len() < 2 {
        return None;
    }
    let mut ibis: Vec<f64> = beats.windows(2).map(|w| w[1].time - w[0].time).collect();
    ibis.sort_by(f64::total_cmp);
    let median_ibi = ibis[ibis.len() / 2];

    // Frame whose centre time is >= t: inverse of (i*hop + window/2)/sr.
    let frame_from = |t: f64| -> usize {
        (((t * f64::from(sample_rate) - (window / 2) as f64) / hop as f64).ceil())
            .clamp(0.0, energy.len() as f64) as usize
    };

    let mut bars: Vec<Bar> = Vec::new();
    for n in 1..=max_bar {
        let mut in_bar = beats.iter().filter(|b| b.bar == Some(n));
        let start = in_bar.next()?.time;
        let last_in_bar = in_bar.next_back().map(|b| b.time).unwrap_or(start);
        let end = match beats.iter().find(|b| b.bar == Some(n + 1)) {
            Some(b) => b.time,
            None => last_in_bar + median_ibi, // estimate; see struct docs
        };
        let frames = &energy[frame_from(start)..frame_from(end)];
        let mean_energy_dbfs =
            (!frames.is_empty()).then(|| frames.iter().sum::<f32>() / frames.len() as f32);
        let onset_count = onset_times
            .iter()
            .filter(|&&t| t >= start && t < end)
            .count() as u32;
        bars.push(Bar {
            bar: n,
            start,
            end,
            mean_energy_dbfs,
            onset_count,
        });
    }
    if bars.is_empty() { None } else { Some(bars) }
}

/// Attach `beat_index`/`beat_phase` to onsets: the index of the beat whose
/// interval [beat_i, beat_{i+1}) contains the onset, and the fractional
/// position within that interval in [0, 1). Continuous — no grid
/// quantization, so swung events keep their true position. Onsets outside
/// the beat span keep `None` (unknown = absent).
pub fn assign_beat_phase(onsets: &mut [Onset], beat_times: &[f64]) {
    if beat_times.len() < 2 {
        return;
    }
    for o in onsets.iter_mut() {
        o.beat_index = None;
        o.beat_phase = None;
        if o.time < beat_times[0] || o.time >= *beat_times.last().unwrap() {
            continue;
        }
        // partition_point: first index with beat_time > o.time, minus 1.
        let i = beat_times.partition_point(|&bt| bt <= o.time) - 1;
        let span = beat_times[i + 1] - beat_times[i];
        if span > 0.0 {
            o.beat_index = Some(i as u32);
            o.beat_phase = Some(((o.time - beat_times[i]) / span).clamp(0.0, 1.0));
        }
    }
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
        .map(|(i, &t)| Downbeat {
            time: t,
            bar: i as u32 + 1,
            score: None,
        })
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
        items.push(Beat {
            time,
            index: index as u32,
            bar: bar_field,
            bar_position: pos_field,
            score: None,
        });
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

#[cfg(test)]
mod tests {
    use super::*;

    fn beat(time: f64, index: u32, bar: Option<u32>) -> Beat {
        Beat {
            time,
            index,
            bar,
            bar_position: None,
            score: None,
        }
    }

    fn onset(time: f64) -> Onset {
        Onset {
            time,
            strength: 1.0,
            beat_index: None,
            beat_phase: None,
        }
    }

    #[test]
    fn bar_overview_aggregates_per_bar() {
        // 8 beats at 120 BPM in 4/4 (2 bars), downbeat at beat 0.
        let beats: Vec<Beat> = (0..8)
            .map(|i| beat(i as f64 * 0.5, i, Some(i / 4 + 1)))
            .collect();
        // Energy: quiet (-40) in bar 1's span, loud (-10) in bar 2's.
        // Grid: 50 fps, frame i centred at (i*441+512)/22050.
        let mut energy = vec![-40.0f32; 400];
        for (i, e) in energy.iter_mut().enumerate() {
            let t = (i * 441 + 512) as f64 / 22050.0;
            if t >= 2.0 {
                *e = -10.0;
            }
        }
        let onsets = [0.5, 1.0, 2.0, 2.5, 3.0, 3.25, 3.75]; // 2 in bar 1, 5 in bar 2
        let bars = bar_overview(&beats, &energy, &onsets, 1024, 441, 22050).unwrap();
        assert_eq!(bars.len(), 2);
        assert!((bars[0].start - 0.0).abs() < 1e-9 && (bars[0].end - 2.0).abs() < 1e-9);
        assert_eq!(bars[0].onset_count, 2);
        assert_eq!(bars[1].onset_count, 5);
        let e0 = bars[0].mean_energy_dbfs.unwrap();
        let e1 = bars[1].mean_energy_dbfs.unwrap();
        assert!(
            e1 > e0 + 20.0,
            "bar2 {e1} should be much louder than bar1 {e0}"
        );
        // Final bar end is an estimate: last beat + median IBI.
        assert!((bars[1].end - 4.0).abs() < 1e-9, "got {}", bars[1].end);
    }

    #[test]
    fn bar_overview_absent_without_bars() {
        let beats: Vec<Beat> = (0..8).map(|i| beat(i as f64 * 0.5, i, None)).collect();
        assert!(bar_overview(&beats, &[-20.0; 400], &[], 1024, 441, 22050).is_none());
        assert!(bar_overview(&[], &[-20.0; 400], &[], 1024, 441, 22050).is_none());
    }

    #[test]
    fn beat_phase_is_continuous_and_bracketed() {
        let beats = [1.0, 1.5, 2.0, 2.5];
        let mut onsets = vec![
            onset(0.5),  // before first beat: no phase
            onset(1.0),  // exactly on beat 0
            onset(1.75), // halfway between beats 1 and 2
            onset(2.9),  // after last beat: no phase
        ];
        assign_beat_phase(&mut onsets, &beats);
        assert_eq!(onsets[0].beat_index, None);
        assert_eq!(onsets[1].beat_index, Some(0));
        assert!((onsets[1].beat_phase.unwrap() - 0.0).abs() < 1e-9);
        assert_eq!(onsets[2].beat_index, Some(1));
        assert!((onsets[2].beat_phase.unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(onsets[3].beat_index, None);
        // No beats -> nothing assigned, no panic.
        let mut o = vec![onset(1.0)];
        assign_beat_phase(&mut o, &[]);
        assert_eq!(o[0].beat_index, None);
    }
}
