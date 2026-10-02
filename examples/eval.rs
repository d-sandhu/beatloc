//! Evaluate beat tracking against beat annotations. Dev tool — not shipped.
//!
//! ```sh
//! cargo run --release --example eval -- <audio_dir> <annotation_dir> [--trim SECONDS] [--limit N]
//! ```
//!
//! Pairing: `<stem>.<wav|mp3|flac>` under <audio_dir> (recursive) ↔
//! `<stem>.beats` directly under <annotation_dir>.
//!
//! Annotation format: one beat per line, `seconds[<whitespace> ...]` — extra
//! columns (e.g. bar position) are ignored. This is the madmom/Ballroom
//! convention.
//!
//! Methodology:
//! - mir_eval convention: beats earlier than `--trim` seconds (default 5.0)
//!   are dropped from BOTH sequences before scoring, mir_eval default.
//! - Metrics: F-measure @ ±70 ms; CMLc/CMLt/AMLc/AMLt (mir_eval definitions,
//!   17.5% phase/period tolerances); plus signed-offset statistics — a beatloc
//!   addition, since F-measure alone cannot see systematic early/late bias.
//! - `evaluate` runs the release pipeline; reported wall time is total
//!   analysis time / total audio duration (realtime factor).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use beatloc::eval::{continuity, f_measure, matched_offsets};
use beatloc::{AnalysisOptions, analyze_file};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(audio_dir), Some(ann_dir)) = (args.next(), args.next()) else {
        eprintln!("usage: eval <audio_dir> <annotation_dir> [--trim SECONDS] [--limit N]");
        std::process::exit(2);
    };
    let mut trim = 5.0f64;
    let mut limit = usize::MAX;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--trim" => trim = args.next().and_then(|v| v.parse().ok()).unwrap_or(trim),
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or(limit),
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let annotations = collect_annotations(Path::new(&ann_dir));
    let mut audio_files = Vec::new();
    collect_audio(Path::new(&audio_dir), &mut audio_files);
    audio_files.sort();

    let mut scored = 0usize;
    let mut skipped = 0usize;
    let mut total_audio_seconds = 0.0f64;
    let mut total_wall_seconds = 0.0f64;
    let mut per_track: Vec<(String, f64, f64, f64, f64)> = Vec::new(); // name, f1, cmlt, amlt, mean_off
    let mut all_offsets: Vec<f64> = Vec::new();

    for audio in audio_files.into_iter().take(limit) {
        let stem = audio.file_stem().unwrap().to_string_lossy().to_string();
        let Some(ann_path) = annotations.get(&stem) else {
            skipped += 1;
            continue;
        };
        let mut reference = read_beats_annotation(ann_path);
        let started = Instant::now();
        let timeline = match analyze_file(&audio, AnalysisOptions::default()) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("warn: {stem}: analysis failed: {e}");
                skipped += 1;
                continue;
            }
        };
        total_wall_seconds += started.elapsed().as_secs_f64();
        total_audio_seconds += timeline.source.duration_seconds;

        let mut estimate: Vec<f64> = timeline.beats.items.iter().map(|b| b.time).collect();
        reference.retain(|&t| t >= trim);
        estimate.retain(|&t| t >= trim);

        let fm = f_measure(&reference, &estimate, 0.07);
        let cont = continuity(&reference, &estimate);
        let offsets = matched_offsets(&reference, &estimate, 0.07);
        let mean_off = if offsets.is_empty() {
            f64::NAN
        } else {
            offsets.iter().sum::<f64>() / offsets.len() as f64
        };
        all_offsets.extend(offsets);
        per_track.push((stem, fm.f, cont.cmlt, cont.amlt, mean_off));
        scored += 1;
    }

    if scored == 0 {
        eprintln!("no audio/annotation pairs found");
        std::process::exit(1);
    }

    let mean = |f: fn(&(String, f64, f64, f64, f64)) -> f64| {
        per_track.iter().map(f).sum::<f64>() / scored as f64
    };
    let mean_offset = all_offsets.iter().sum::<f64>() / all_offsets.len().max(1) as f64;
    let offset_std = (all_offsets.iter().map(|o| (o - mean_offset).powi(2)).sum::<f64>()
        / all_offsets.len().max(1) as f64)
        .sqrt();

    println!("=== beatloc eval ===");
    println!("tracks scored: {scored} (skipped {skipped}), trim: first {trim} s dropped");
    println!("F-measure @70ms : {:.4}", mean(|t| t.1));
    println!("CMLt            : {:.4}", mean(|t| t.2));
    println!("AMLt            : {:.4}", mean(|t| t.3));
    println!("signed offset   : mean {:+.1} ms, std {:.1} ms", mean_offset * 1000.0, offset_std * 1000.0);
    println!(
        "speed           : {:.1}x realtime ({:.1} s wall for {:.1} s audio)",
        total_audio_seconds / total_wall_seconds,
        total_wall_seconds,
        total_audio_seconds
    );

    per_track.sort_by(|a, b| a.1.total_cmp(&b.1));
    println!("\nworst 5 tracks by F-measure (documented failure cases):");
    for (name, f1, cmlt, amlt, off) in per_track.iter().take(5) {
        println!(
            "  {name:<44} F1 {f1:.3}  CMLt {cmlt:.3}  AMLt {amlt:.3}  off {:+.1} ms",
            off * 1000.0
        );
    }
}

fn collect_audio(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_audio(&path, out);
        } else if matches!(
            path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
            Some("wav" | "mp3" | "flac")
        ) {
            out.push(path);
        }
    }
}

fn collect_annotations(dir: &Path) -> BTreeMap<String, PathBuf> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("beats"))
        .map(|p| (p.file_stem().unwrap().to_string_lossy().to_string(), p))
        .collect()
}

fn read_beats_annotation(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .expect("annotation readable")
        .lines()
        .filter_map(|line| line.split_whitespace().next()?.parse::<f64>().ok())
        .collect()
}
