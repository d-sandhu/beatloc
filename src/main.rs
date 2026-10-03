//! beatloc CLI.
//!
//! Contract with consumers (agents):
//! - stdout carries ONLY the machine-readable result.
//! - stderr carries diagnostics; nothing is printed on success in stdout mode.
//! - Exit codes: 0 = success, 1 = analysis/IO error, 2 = CLI usage error
//!   (clap's default).
//! - `--output` never overwrites an existing file unless `--force` is given.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

use beatloc::{AnalysisOptions, Engine};

#[derive(Parser)]
#[command(
    name = "beatloc",
    version,
    about = "Convert audio into a machine-readable musical timeline (JSON)"
)]
struct Cli {
    /// Audio input file (WAV, MP3, FLAC) or a directory for batch mode
    input: PathBuf,

    /// Recurse into subdirectories (batch mode only)
    #[arg(short, long)]
    recursive: bool,

    /// Batch mode: directory to write one <name>.json per input file.
    /// Required when INPUT is a directory.
    #[arg(long, conflicts_with_all = ["output", "json"])]
    output_dir: Option<PathBuf>,

    /// Write JSON to stdout (default when --output is not given)
    #[arg(long, conflicts_with = "output")]
    json: bool,

    /// Write JSON to this file instead of stdout
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Allow --output to overwrite an existing file
    #[arg(long)]
    force: bool,

    /// Include dense per-frame curves (energy, onset strength)
    #[arg(long)]
    curves: bool,

    /// Pretty-print the JSON output
    #[arg(long)]
    pretty: bool,

    /// Beat engine: "neural" (Beat This!, needs model files), "dsp"
    /// (classical baseline), or "auto" (neural if models are present)
    #[arg(long, value_enum, default_value_t = EngineArg::Auto)]
    engine: EngineArg,

    /// Path to the beat model ONNX (default: auto-detected; see README install notes)
    #[arg(long)]
    model: Option<PathBuf>,

    /// Path to the mel frontend ONNX (default: auto-detected; see README install notes)
    #[arg(long = "mel-model")]
    mel_model: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum EngineArg {
    Auto,
    Dsp,
    Neural,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<()> {
    if cli.input.is_dir() {
        return run_batch(&cli);
    }

    let engine = match cli.engine {
        EngineArg::Auto => Engine::Auto,
        EngineArg::Dsp => Engine::Dsp,
        EngineArg::Neural => Engine::Neural,
    };
    if engine == Engine::Auto && !models_available(&cli) {
        eprintln!(
            "note: neural models not found (looked in $BEATLOC_MODEL_DIR, ~/.local/share/beatloc/models, ./models); \
             using the dsp engine"
        );
    }

    let timeline = beatloc::analyze_file(
        &cli.input,
        AnalysisOptions {
            include_curves: cli.curves,
            engine,
            beat_model: cli.model.clone(),
            mel_model: cli.mel_model.clone(),
        },
    )?;
    let json = beatloc::serialize::to_json_string(&timeline, cli.pretty)?;

    match cli.output {
        Some(path) => {
            if path.exists() && !cli.force {
                anyhow::bail!(
                    "{} already exists (pass --force to overwrite)",
                    path.display()
                );
            }
            std::fs::write(&path, &json)?;
            eprintln!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

/// Batch mode: one JSON file per audio file under the input directory.
/// Progress and per-file errors go to stderr; exit 1 if any file failed.
fn run_batch(cli: &Cli) -> anyhow::Result<()> {
    let Some(out_dir) = &cli.output_dir else {
        anyhow::bail!("batch mode requires --output-dir (one JSON per input file)");
    };
    std::fs::create_dir_all(out_dir)?;

    let engine = match cli.engine {
        EngineArg::Auto => Engine::Auto,
        EngineArg::Dsp => Engine::Dsp,
        EngineArg::Neural => Engine::Neural,
    };
    let files = beatloc::find_audio_files(&cli.input, cli.recursive);
    if files.is_empty() {
        anyhow::bail!("no audio files found under {}", cli.input.display());
    }

    let total = files.len();
    let mut failures = 0usize;
    for (i, file) in files.iter().enumerate() {
        // Output path mirrors the input's path relative to the batch root,
        // keeping the full file name (song.mp3.json): the mapping is
        // injective by construction. (The old stem-only naming silently
        // overwrote results when inputs shared a stem — same song as
        // .wav + .mp3, or same file name in different subdirectories.)
        let rel = file.strip_prefix(&cli.input).unwrap_or(file.as_path());
        let mut name = rel.as_os_str().to_owned();
        name.push(".json");
        let out_path = out_dir.join(name);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let result = (|| -> anyhow::Result<()> {
            let timeline = beatloc::analyze_file(
                file,
                AnalysisOptions {
                    include_curves: cli.curves,
                    engine,
                    beat_model: cli.model.clone(),
                    mel_model: cli.mel_model.clone(),
                },
            )?;
            let json = beatloc::serialize::to_json_string(&timeline, cli.pretty)?;
            std::fs::write(&out_path, json)
                .map_err(|e| anyhow::anyhow!("write {}: {e}", out_path.display()))
        })();
        match result {
            Ok(()) => eprintln!("[{}/{}] ok  {}", i + 1, total, file.display()),
            Err(e) => {
                failures += 1;
                eprintln!("[{}/{}] FAILED {}: {e:#}", i + 1, total, file.display());
            }
        }
    }
    if failures > 0 {
        anyhow::bail!("{failures}/{total} files failed");
    }
    Ok(())
}

fn models_available(cli: &Cli) -> bool {
    beatloc::resolve_model_path(cli.model.as_deref(), beatloc::DEFAULT_BEAT_MODEL_FILE).is_file()
        && beatloc::resolve_model_path(cli.mel_model.as_deref(), beatloc::DEFAULT_MEL_MODEL_FILE)
            .is_file()
}
