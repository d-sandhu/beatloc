//! Render a timeline's event lanes as a click track over the original
//! audio, for human verification: song in the LEFT channel, ticks in the
//! RIGHT. Dev/QA tool — not shipped.
//!
//! ```sh
//! cargo run --release --example render_clicks -- <audio> <timeline.json> <out.wav>
//! ```
//!
//! Ticks are 5 ms exponentially-decaying bursts (short with no tail — a
//! longer tick reads as "two for one" when checking by ear). Lanes are
//! pitch-separated: downbeat 660 Hz, beat 990 Hz, onset 1560 Hz. Output is
//! 44.1 kHz 16-bit stereo WAV (regardless of source rate — decoded through
//! beatloc's own pipeline, then rendered on a 44.1 kHz grid).
//!
//! The workflow this supports: an agent cannot hear, so verification of any
//! lane is a human listen — this makes that listen a one-command step.

use std::path::Path;

const OUT_RATE: u32 = 44_100;
/// 5 ms burst, ~2 ms decay constant: short, no tail.
const TICK_SECONDS: f64 = 0.005;
const TICK_DECAY_SECONDS: f64 = 0.002;

/// One tick synthesis; lane pitch separates the lanes by ear.
fn add_tick(buf: &mut [f32], sample: usize, freq: f32, amp: f32) {
    let len = (TICK_SECONDS * f64::from(OUT_RATE)) as usize;
    for i in 0..len {
        let t = i as f32 / OUT_RATE as f32;
        let env = (-t / TICK_DECAY_SECONDS as f32).exp();
        let v = amp * env * (2.0 * std::f32::consts::PI * freq * t).sin();
        if let Some(s) = buf.get_mut(sample + i) {
            *s += v;
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(audio), Some(json_path), Some(out_path)) = (args.next(), args.next(), args.next())
    else {
        eprintln!("usage: render_clicks <audio> <timeline.json> <out.wav>");
        std::process::exit(2);
    };

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap_or_else(|e| {
            eprintln!("{json_path}: {e}");
            std::process::exit(1)
        }))
        .unwrap_or_else(|e| {
            eprintln!("{json_path}: invalid JSON: {e}");
            std::process::exit(1)
        });

    // Decode the song through beatloc's own decoder so the timeline and the
    // render share one time base. Mono mix is fine for the reference ear.
    let decoded = beatloc::decode::decode_file(Path::new(&audio)).unwrap_or_else(|e| {
        eprintln!("{audio}: {e}");
        std::process::exit(1)
    });
    let mono = beatloc::dsp::resample::to_analysis_rate(&decoded.samples, decoded.sample_rate)
        .unwrap_or_else(|e| {
            eprintln!("resample: {e}");
            std::process::exit(1)
        });
    // analysis rate is 22050; upsample x2 (linear) to the output rate.
    let mut song: Vec<f32> = Vec::with_capacity(mono.len() * 2);
    for w in mono.windows(2) {
        song.push(w[0]);
        song.push((w[0] + w[1]) * 0.5);
    }
    song.extend_from_slice(&mono[mono.len().saturating_sub(1)..]);

    let n = song.len();
    let mut ticks = vec![0.0f32; n];

    // A lane is either a top-level array ("onsets") or an object with an
    // "items" array ("beats", "downbeats"); both hold {time} entries.
    let lane_times = |key: &str| -> Vec<f64> {
        let node = &json[key];
        let entries = node.as_array().or_else(|| node["items"].as_array());
        entries
            .map(|a| a.iter().filter_map(|e| e["time"].as_f64()).collect())
            .unwrap_or_default()
    };
    let lanes: [(&str, f32, f32); 3] = [
        ("downbeats", 660.0, 0.9),
        ("beats", 990.0, 0.6),
        ("onsets", 1560.0, 0.35),
    ];
    let mut counts = [0usize; 3];
    for (i, (key, freq, amp)) in lanes.iter().enumerate() {
        for t in lane_times(key) {
            let s = (t * f64::from(OUT_RATE)) as usize;
            if s < n {
                add_tick(&mut ticks, s, *freq, *amp);
                counts[i] += 1;
            }
        }
    }

    // Interleave: song left, ticks right.
    let mut interleaved = vec![0.0f32; n * 2];
    for i in 0..n {
        interleaved[2 * i] = song[i];
        interleaved[2 * i + 1] = ticks[i];
    }
    write_wav_i16(Path::new(&out_path), OUT_RATE, 2, &interleaved);
    eprintln!(
        "wrote {out_path} ({:.1}s): {} downbeats, {} beats, {} onsets",
        n as f64 / f64::from(OUT_RATE),
        counts[0],
        counts[1],
        counts[2]
    );
}

/// Minimal 16-bit PCM WAV writer (same layout as the test helpers).
fn write_wav_i16(path: &Path, sample_rate: u32, channels: u16, interleaved: &[f32]) {
    let data_len = (interleaved.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * u32::from(channels) * 2).to_le_bytes());
    bytes.extend_from_slice(&(channels * 2).to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &s in interleaved {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("failed to write WAV");
}
