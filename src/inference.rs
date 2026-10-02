//! Native neural beat/downbeat inference: Beat This! (JKU Linz, MIT) running
//! on ONNX via the pure-Rust `rten` runtime. No Python, no system libraries.
//!
//! The pipeline mirrors the reference implementation exactly:
//! 1. Mel frontend as an ONNX graph (exported from torchaudio by
//!    `scripts/export_model.py`), so frontend parity is by construction.
//! 2. The BeatThis transformer over 1500-frame chunks with a 6-frame border
//!    and keep-first seam merging (`split_piece` / `aggregate_prediction`).
//! 3. The reference "minimal" postprocessor: local-max peak picking
//!    (±3 frames, logit > 0), adjacent-peak merging by fractional mean,
//!    downbeats snapped to the nearest beat.
//!
//! Timestamp convention: frame index / 50 s — the model's own grid (its mel
//! frontend uses centered STFT frames, so frame *i* is centred at sample
//! i·441). Note this differs from the DSP tracks' unpadded frame-centre
//! convention by up to half a window (~23 ms); both are "seconds from decoded
//! stream start" and the per-engine convention is documented in the schema's
//! `analysis` block and each engine's provenance.

use std::path::Path;

use rten::{Model, NodeId, ValueOrView};
use rten_tensor::Tensor;
use rten_tensor::prelude::*;

use crate::BeatlocError;

/// Frames per second of the model's output grid (22050 Hz / hop 441).
pub const MODEL_FPS: f64 = 50.0;
/// Chunk length in frames (30 s) — the model's training length.
const CHUNK_SIZE: usize = 1500;
/// Frames discarded at chunk borders (the model was not trained on edges).
const BORDER: i64 = 6;

/// Beat This! small0 (or final0) loaded from ONNX, plus the mel frontend.
pub struct NeuralEngine {
    mel: Model,
    mel_in: NodeId,
    mel_out: NodeId,
    model: Model,
    model_in: NodeId,
    model_out: [NodeId; 2],
}

/// Raw model output for one piece of audio (22050 Hz mono samples in).
pub struct NeuralOutput {
    /// Frame-level logits, length = frames. Exposed for parity testing.
    pub beat_logits: Vec<f32>,
    pub downbeat_logits: Vec<f32>,
    /// Beat times in seconds (postprocessed).
    pub beats: Vec<f64>,
    /// Per-beat score: sigmoid of the beat logit at the picked frame.
    /// Semantics, precisely: how strongly the model asserted "beat" at that
    /// frame. UNCALIBRATED — not a probability of being right.
    pub beat_scores: Vec<f32>,
    /// Downbeat times in seconds (postprocessed, snapped to beats).
    pub downbeats: Vec<f64>,
    /// Per-downbeat score (sigmoid of the downbeat logit pre-snap).
    pub downbeat_scores: Vec<f32>,
}

impl NeuralEngine {
    pub fn load(mel_path: &Path, model_path: &Path) -> Result<Self, BeatlocError> {
        let mel = Model::load_file(mel_path)
            .map_err(|e| BeatlocError::Inference(format!("{}: {e}", mel_path.display())))?;
        let model = Model::load_file(model_path)
            .map_err(|e| BeatlocError::Inference(format!("{}: {e}", model_path.display())))?;

        let find = |m: &Model, name: &str| {
            m.find_node(name)
                .ok_or_else(|| BeatlocError::Inference(format!("model has no node '{name}'")))
        };
        let mel_in = find(&mel, "waveform")?;
        let mel_out = find(&mel, "spect")?;
        let model_in = find(&model, "spect")?;
        let model_out = [find(&model, "beat")?, find(&model, "downbeat")?];
        Ok(NeuralEngine { mel, mel_in, mel_out, model, model_in, model_out })
    }

    /// Full pipeline: mono 22050 Hz samples → logits → beat/downbeat times.
    pub fn predict(&self, samples: &[f32]) -> Result<NeuralOutput, BeatlocError> {
        let (beat_logits, downbeat_logits) = self.logits(samples)?;
        let (beats, beat_scores) = postprocess(&beat_logits);
        let (mut downbeats, mut downbeat_scores) = postprocess(&downbeat_logits);
        snap_downbeats(&mut downbeats, &mut downbeat_scores, &beats);
        Ok(NeuralOutput {
            beat_logits,
            downbeat_logits,
            beats,
            beat_scores,
            downbeats,
            downbeat_scores,
        })
    }

    /// Samples → aggregated frame-level logits (before postprocessing).
    pub fn logits(&self, samples: &[f32]) -> Result<(Vec<f32>, Vec<f32>), BeatlocError> {
        let spect = self.mel_spectrogram(samples)?;
        let frames = spect.len() / 128;
        let starts = split_starts(frames);

        let mut beat = vec![f32::NEG_INFINITY; frames];
        let mut downbeat = vec![f32::NEG_INFINITY; frames];
        let mut written = vec![false; frames];

        for start in starts {
            let (chunk_beat, chunk_downbeat) = self.run_chunk(&spect, frames, start)?;
            // Keep-first: earlier chunks win in overlaps. The written region
            // of this chunk is [start + BORDER, start + CHUNK_SIZE - BORDER).
            let region_start = (start + BORDER).max(0) as usize;
            let region_end = (start + CHUNK_SIZE as i64 - BORDER).min(frames as i64).max(0) as usize;
            #[allow(clippy::needless_range_loop)]
            for f in region_start..region_end {
                if !written[f] {
                    // Map piece frame f back to this chunk's trimmed prediction.
                    let local = (f as i64 - start - BORDER) as usize;
                    beat[f] = chunk_beat[local];
                    downbeat[f] = chunk_downbeat[local];
                    written[f] = true;
                }
            }
        }
        debug_assert!(written.iter().all(|&w| w), "chunking must cover every frame");
        Ok((beat, downbeat))
    }

    fn mel_spectrogram(&self, samples: &[f32]) -> Result<Vec<f32>, BeatlocError> {
        let input = Tensor::from_data(&[samples.len()], samples.to_vec());
        let outputs = self
            .mel
            .run_n(
                vec![(self.mel_in, ValueOrView::from(input))],
                [self.mel_out],
                None,
            )
            .map_err(|e| BeatlocError::Inference(format!("mel frontend failed: {e}")))?;
        let [output] = outputs;
        output
            .into_tensor::<f32>()
            .map(|t| t.to_vec())
            .ok_or_else(|| BeatlocError::Inference("mel output is not an f32 tensor".into()))
    }

    /// Run the beat model on one chunk; returns border-trimmed predictions.
    fn run_chunk(
        &self,
        spect: &[f32],
        frames: usize,
        start: i64,
    ) -> Result<(Vec<f32>, Vec<f32>), BeatlocError> {
        // Port of beat_this.inference.split_piece: slice + zero-pad borders.
        let chunk_start = start.max(0) as usize;
        let chunk_end = (start + CHUNK_SIZE as i64).min(frames as i64) as usize;
        let left_pad = (-start).max(0) as usize;
        let right_pad = (start + CHUNK_SIZE as i64 - frames as i64)
            .clamp(0, BORDER)
            .max(0) as usize;
        let body = &spect[chunk_start * 128..chunk_end * 128];

        let chunk_frames = left_pad + (chunk_end - chunk_start) + right_pad;
        let mut chunk = vec![0.0f32; chunk_frames * 128];
        chunk[left_pad * 128..(left_pad + (chunk_end - chunk_start)) * 128].copy_from_slice(body);

        let input = Tensor::from_data(&[1, chunk_frames, 128], chunk);
        let outputs = self
            .model
            .run_n(
                vec![(self.model_in, ValueOrView::from(input))],
                self.model_out,
                None,
            )
            .map_err(|e| BeatlocError::Inference(format!("beat model failed: {e}")))?;
        let [beat_v, downbeat_v] = outputs;
        let beat_t = beat_v
            .into_tensor::<f32>()
            .ok_or_else(|| BeatlocError::Inference("beat output is not an f32 tensor".into()))?;
        let downbeat_t = downbeat_v
            .into_tensor::<f32>()
            .ok_or_else(|| BeatlocError::Inference("downbeat output is not an f32 tensor".into()))?;

        // Port of aggregate_prediction's border trim: predictions [BORDER..len-BORDER).
        let beat = beat_t.to_vec();
        let downbeat = downbeat_t.to_vec();
        let trim_end = chunk_frames - BORDER as usize;
        Ok((
            beat[BORDER as usize..trim_end].to_vec(),
            downbeat[BORDER as usize..trim_end].to_vec(),
        ))
    }
}

/// Chunk start positions, ported from `split_piece` (border 6, keep-first).
pub fn split_starts(frames: usize) -> Vec<i64> {
    let step = CHUNK_SIZE as i64 - 2 * BORDER; // 1488
    let mut starts = Vec::new();
    let mut s = -BORDER;
    while s < frames as i64 - BORDER {
        starts.push(s);
        s += step;
    }
    // avoid_short_end: shift the last chunk so it ends flush with the piece.
    if frames as i64 > step {
        if let Some(last) = starts.last_mut() {
            *last = frames as i64 - (CHUNK_SIZE as i64 - BORDER);
        }
    }
    starts
}

/// The reference "minimal" postprocessor for one activation curve, ported
/// exactly: local max within ±3 frames AND logit > 0, adjacent peaks merged
/// by their fractional mean, times = frame / fps.
///
/// Returns (times, scores); a peak's score is sigmoid(its logit), the max
/// over a merged group. Scores are UNCALIBRATED model assertions, not
/// probabilities of correctness.
pub fn postprocess(logits: &[f32]) -> (Vec<f64>, Vec<f32>) {
    let peaks = pick_peaks(logits);
    let times = peaks.iter().map(|p| p.frame / MODEL_FPS).collect();
    let scores = peaks.iter().map(|p| sigmoid(p.logit)).collect();
    (times, scores)
}

/// Snap downbeat times to the nearest beat (reference behavior); when several
/// downbeats collapse onto the same beat, keep the highest score.
pub fn snap_downbeats(
    downbeats: &mut Vec<f64>,
    downbeat_scores: &mut Vec<f32>,
    beats: &[f64],
) {
    if beats.is_empty() {
        return;
    }
    for d in downbeats.iter_mut() {
        *d = beats
            .iter()
            .min_by(|a, b| (**a - *d).abs().total_cmp(&(**b - *d).abs()))
            .copied()
            .unwrap_or(*d);
    }
    let mut pairs: Vec<(f64, f32)> =
        downbeats.iter().copied().zip(downbeat_scores.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    pairs.dedup_by(|next, prev| {
        if next.0 == prev.0 {
            prev.1 = prev.1.max(next.1);
            true
        } else {
            false
        }
    });
    downbeats.clear();
    downbeat_scores.clear();
    downbeats.extend(pairs.iter().map(|p| p.0));
    downbeat_scores.extend(pairs.iter().map(|p| p.1));
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct Peak {
    frame: f64,
    logit: f32,
}

/// Peak picking: frame is a peak iff it equals the max over ±3 frames and its
/// logit is > 0 (p > 0.5). Adjacent peaks (gap ≤ 1) merge to their fractional
/// mean (reference `deduplicate_peaks` semantics); score = max group logit.
fn pick_peaks(logits: &[f32]) -> Vec<Peak> {
    let n = logits.len();
    let mut raw: Vec<Peak> = Vec::new();
    for i in 0..n {
        let lo = i.saturating_sub(3);
        let hi = (i + 3).min(n - 1);
        let window_max = logits[lo..=hi].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if logits[i] > 0.0 && logits[i] == window_max {
            raw.push(Peak { frame: i as f64, logit: logits[i] });
        }
    }
    // Merge groups of adjacent peaks: running mean of frames, max logit.
    let mut merged: Vec<Peak> = Vec::new();
    let Some(first) = raw.first() else { return merged };
    let mut cur = Peak { frame: first.frame, logit: first.logit };
    let mut count = 1.0;
    for p in &raw[1..] {
        if p.frame - cur.frame <= 1.0 {
            count += 1.0;
            cur.frame += (p.frame - cur.frame) / count;
            cur.logit = cur.logit.max(p.logit);
        } else {
            merged.push(cur);
            cur = Peak { frame: p.frame, logit: p.logit };
            count = 1.0;
        }
    }
    merged.push(cur);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_short_piece_is_single_padded_chunk() {
        // 800 frames < chunk: one chunk starting at -BORDER, no shift.
        assert_eq!(split_starts(800), vec![-6]);
    }

    #[test]
    fn split_long_piece_matches_reference_indices() {
        // np.arange(-6, 3000-6, 1488) = [-6, 1482, 2970]; last shifted to
        // 3000 - 1494 = 1506.
        assert_eq!(split_starts(3000), vec![-6, 1482, 1506]);
    }

    #[test]
    fn split_covers_all_frames_without_gaps() {
        // The invariant `logits()` relies on: every frame in [0, frames) is
        // written by exactly the keep-first region of some chunk.
        for frames in [1usize, 12, 800, 1487, 1488, 1489, 3000, 10_001] {
            let starts = split_starts(frames);
            assert!(!starts.is_empty());
            let mut covered = vec![false; frames];
            for &s in &starts {
                let lo = (s + BORDER).max(0) as usize;
                let hi =
                    (s + CHUNK_SIZE as i64 - BORDER).clamp(0, frames as i64) as usize;
                for c in covered.iter_mut().take(hi).skip(lo) {
                    *c = true;
                }
            }
            assert!(covered.iter().all(|&c| c), "frames {frames}: gaps in coverage");
        }
    }

    #[test]
    fn peaks_require_local_max_and_positive_logit() {
        let mut logits = vec![-1.0f32; 21];
        logits[10] = 2.0;
        let (times, scores) = postprocess(&logits);
        assert_eq!(times, vec![10.0 / 50.0]);
        assert!((scores[0] - sigmoid(2.0)).abs() < 1e-6);
        // Sub-threshold peak (logit <= 0) is rejected.
        let mut quiet = vec![-1.0f32; 21];
        quiet[10] = -0.2;
        assert!(postprocess(&quiet).0.is_empty());
    }

    #[test]
    fn plateau_merges_to_fractional_mean() {
        let mut logits = vec![-1.0f32; 21];
        logits[10] = 1.0;
        logits[11] = 1.0; // plateau
        let (times, _) = postprocess(&logits);
        assert_eq!(times, vec![10.5 / 50.0]);
    }

    #[test]
    fn downbeats_snap_to_nearest_beat_and_dedupe() {
        let mut beat_logits = vec![-1.0f32; 51];
        beat_logits[10] = 2.0;
        beat_logits[20] = 2.0;
        let mut down_logits = vec![-1.0f32; 51];
        down_logits[11] = 1.0; // snaps to frame 10 -> 0.2 s
        down_logits[19] = 0.7; // snaps to frame 20 -> 0.4 s
        down_logits[20] = 1.2; // also snaps to 0.4 s; higher score wins
        let (beats, _) = postprocess(&beat_logits);
        let (mut downbeats, mut downbeat_scores) = postprocess(&down_logits);
        snap_downbeats(&mut downbeats, &mut downbeat_scores, &beats);
        assert_eq!(beats, vec![0.2, 0.4]);
        assert_eq!(downbeats, vec![0.2, 0.4]);
        assert_eq!(downbeat_scores.len(), 2);
        assert!((downbeat_scores[1] - sigmoid(1.2)).abs() < 1e-6);
    }
}
