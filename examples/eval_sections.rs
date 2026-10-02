//! Evaluate section boundary detection against AIST CHORUS structure
//! annotations (RWC popular music subset). Dev tool — not shipped.
//!
//! ```sh
//! cargo run --release --example eval_sections -- <audio_dir> <chorus_dir> [--limit N]
//! ```
//!
//! Pairing: `RWC_Pxxx.wav` in <audio_dir> ↔ `RM-Pxxx.CHORUS.TXT` in
//! <chorus_dir>.
//!
//! Annotation format: tab-separated `start end "label"` lines in
//! CENTISECONDS (verified: last segment end matches decoded duration to
//! ~10 ms on every track; the harness re-checks this per track and skips
//! loudly on violation). Labels are ignored — our detector is
//! boundary-only, so only boundary times are scored. Reference boundaries
//! are segment starts excluding the first (segment 0 begins the file, not
//! a transition).
//!
//! Metric: boundary F-measure at ±3.0 s (mir_eval structure convention)
//! and ±0.5 s (strict). Human annotators agree with each other at roughly
//! F 0.5–0.7 at ±3 s — that is the practical ceiling, not 1.0.
//!
//! Data provenance: RWC 2.0 audio (CC BY-NC 4.0, Zenodo 17177919) + AIST
//! annotations (research use only). Both live in git-ignored datasets/
//! and are never committed. Cite Goto et al. (ISMIR 2002/2003) and
//! Balke et al. (TISMIR 2026, doi:10.5334/tismir.326).

use std::path::{Path, PathBuf};
use std::time::Instant;

use beatloc::eval::f_measure;
use beatloc::{AnalysisOptions, analyze_file};

/// CHORUS timestamps are centiseconds (verified against metadata durations).
const CHORUS_UNITS_PER_SECOND: f64 = 100.0;
/// Sanity bound: annotated last-segment end vs decoded duration. A
/// violation means the unit assumption broke — fail loudly, never silently
/// score garbage.
const DURATION_CHECK_TOLERANCE_S: f64 = 2.0;

/// Parse a CHORUS file into (interior boundary times in seconds, last
/// segment end in seconds).
fn parse_chorus(path: &Path) -> Result<(Vec<f64>, f64), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut starts = Vec::new();
    let mut last_end = 0.0f64;
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let start: f64 = fields
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("{}:{}: bad start field", path.display(), lineno + 1))?;
        let end: f64 = fields
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("{}:{}: bad end field", path.display(), lineno + 1))?;
        // Third field is the section label — deliberately unused.
        starts.push(start / CHORUS_UNITS_PER_SECOND);
        last_end = end / CHORUS_UNITS_PER_SECOND;
    }
    let boundaries = if starts.len() > 1 {
        starts[1..].to_vec()
    } else {
        Vec::new()
    };
    Ok((boundaries, last_end))
}

/// Map `RM-P001.CHORUS` -> `RWC_P001` audio stem.
fn audio_stem_for(chorus_stem: &str) -> Option<String> {
    let number = chorus_stem.strip_prefix("RM-P")?.strip_suffix(".CHORUS")?;
    Some(format!("RWC_P{number}"))
}

fn find_audio(audio_dir: &Path, stem: &str) -> Option<PathBuf> {
    std::fs::read_dir(audio_dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some(stem))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(audio_dir), Some(chorus_dir)) = (
        args.next().map(PathBuf::from),
        args.next().map(PathBuf::from),
    ) else {
        eprintln!("usage: eval_sections <audio_dir> <chorus_dir> [--limit N]");
        std::process::exit(2);
    };
    let mut limit = usize::MAX;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or(limit),
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let mut chorus_files: Vec<PathBuf> = std::fs::read_dir(&chorus_dir)
        .unwrap_or_else(|e| {
            eprintln!("{}: {e}", chorus_dir.display());
            std::process::exit(2);
        })
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".CHORUS.TXT"))
        })
        .collect();
    chorus_files.sort();
    chorus_files.truncate(limit);

    /// (track, P @ 3 s, R @ 3 s, F @ 3 s, F @ 0.5 s)
    type Row = (String, f64, f64, f64, f64);
    let mut rows: Vec<Row> = Vec::new();
    let mut skipped = 0usize;
    let mut total_wall = 0.0f64;

    for chorus_path in &chorus_files {
        let chorus_stem = chorus_path.file_stem().unwrap().to_str().unwrap();
        let Some(stem) = audio_stem_for(chorus_stem) else {
            continue;
        }; // README.TXT etc.
        let Some(audio_path) = find_audio(&audio_dir, &stem) else {
            eprintln!("{stem}: no audio found, skipping");
            skipped += 1;
            continue;
        };
        let (reference, annotated_end) = match parse_chorus(chorus_path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}, skipping");
                skipped += 1;
                continue;
            }
        };

        let start = Instant::now();
        let timeline = match analyze_file(&audio_path, AnalysisOptions::default()) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{stem}: {e}, skipping");
                skipped += 1;
                continue;
            }
        };
        total_wall += start.elapsed().as_secs_f64();

        // Unit invariant: annotated span must match decoded duration.
        let duration = timeline.source.duration_seconds;
        if (annotated_end - duration).abs() > DURATION_CHECK_TOLERANCE_S {
            eprintln!(
                "{stem}: annotation end {annotated_end:.2}s vs audio {duration:.2}s — \
                 unit assumption broken, skipping"
            );
            skipped += 1;
            continue;
        }

        // Our boundaries = interior section starts (section 0 starts at 0).
        let estimate: Vec<f64> = timeline
            .sections
            .items
            .iter()
            .skip(1)
            .map(|s| s.start)
            .collect();

        let m3 = f_measure(&reference, &estimate, 3.0);
        let f05 = f_measure(&reference, &estimate, 0.5).f;
        rows.push((stem, m3.precision, m3.recall, m3.f, f05));
    }

    if rows.is_empty() {
        eprintln!("no audio/annotation pairs scored");
        std::process::exit(1);
    }

    let n = rows.len() as f64;
    let mean = |f: fn(&Row) -> f64| rows.iter().map(f).sum::<f64>() / n;
    println!("tracks scored: {} (skipped {skipped})", rows.len());
    println!("boundary F-measure @ ±3.0 s: {:.3}", mean(|r| r.3));
    println!(
        "  precision / recall @ ±3 s: {:.3} / {:.3}",
        mean(|r| r.1),
        mean(|r| r.2)
    );
    println!("boundary F-measure @ ±0.5 s: {:.3}", mean(|r| r.4));
    println!("total analysis wall time:  {total_wall:.1} s");
    println!("\nworst 5 by F @ 3 s:");
    rows.sort_by(|a, b| a.3.total_cmp(&b.3));
    for (stem, p, r, f3, f05) in rows.iter().take(5) {
        println!("  {stem}: P={p:.3} R={r:.3} F3={f3:.3} F0.5={f05:.3}");
    }
}
