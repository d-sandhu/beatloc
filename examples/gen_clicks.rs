//! Generate a click-track WAV for manual testing — no external tools needed.
//!
//! ```sh
//! cargo run --example gen_clicks -- /tmp/clicks.wav            # 120 BPM, 8 s @ 44.1 kHz
//! cargo run --example gen_clicks -- /tmp/clicks.wav 22050 90   # custom rate/BPM
//! ```

use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gen_clicks <output.wav> [sample_rate=44100] [bpm=120]");
        std::process::exit(2);
    };
    let sr: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(44_100);
    let bpm: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(120.0);

    // 1 s quiet intro, then 16 clicks at the given tempo.
    let duration = 1.0 + 16.0 * 60.0 / bpm + 0.5;
    let times: Vec<f64> = (0..16).map(|i| 1.0 + i as f64 * 60.0 / bpm).collect();
    let samples = click_track(sr, &times, duration);
    write_wav_i16(Path::new(&path), sr, &samples);
    eprintln!("wrote {path}: {sr} Hz, {bpm} BPM, {} clicks", times.len());
}

fn click_track(sample_rate: u32, times: &[f64], duration_seconds: f64) -> Vec<f32> {
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

fn write_wav_i16(path: &Path, sample_rate: u32, samples: &[f32]) {
    let data_len = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        bytes.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, bytes).expect("failed to write WAV");
}
