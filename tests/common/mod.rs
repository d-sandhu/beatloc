//! Shared helpers for integration tests: synthetic WAVs with ground truth.
// Not every helper is used by every test crate.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn temp_wav_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("beatloc-test-{}-{name}", std::process::id()))
}

/// Write a canonical 44-byte-header PCM16 WAV. Samples are clamped to [-1, 1].
pub fn write_wav_i16(path: &Path, sample_rate: u32, channels: u16, interleaved: &[f32]) {
    let data_len = (interleaved.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * u32::from(channels) * 2).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&(channels * 2).to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &s in interleaved {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("failed to write test WAV");
}

/// Deterministic click track: 5 ms decaying 3 kHz bursts at each given time.
pub fn click_track(sample_rate: u32, times: &[f64], duration_seconds: f64) -> Vec<f32> {
    let n = (duration_seconds * f64::from(sample_rate)) as usize;
    let mut x = vec![0.0f32; n];
    let burst_len = (0.005 * f64::from(sample_rate)) as usize;
    for &t in times {
        let start = (t * f64::from(sample_rate)) as usize;
        for i in 0..burst_len {
            if start + i >= n {
                break;
            }
            let env = (-(i as f64) / (f64::from(sample_rate) * 0.001)).exp() as f32;
            let phase = 2.0 * std::f64::consts::PI * 3000.0 * i as f64 / f64::from(sample_rate);
            x[start + i] += 0.9 * env * phase.sin() as f32;
        }
    }
    x
}

/// Duplicate a mono signal into `channels` identical interleaved channels.
pub fn to_interleaved(mono: &[f32], channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(mono.len() * channels);
    for &s in mono {
        for _ in 0..channels {
            out.push(s);
        }
    }
    out
}
