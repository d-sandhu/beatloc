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

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "beatloc",
    version,
    about = "Convert audio into a machine-readable musical timeline (JSON)"
)]
struct Cli {
    /// Audio input file (WAV, MP3, FLAC)
    input: PathBuf,

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
    let timeline = beatloc::analyze_file(
        &cli.input,
        beatloc::AnalysisOptions { include_curves: cli.curves },
    )?;
    let json = beatloc::serialize::to_json_string(&timeline, cli.pretty)?;

    match cli.output {
        Some(path) => {
            if path.exists() && !cli.force {
                anyhow::bail!("{} already exists (pass --force to overwrite)", path.display());
            }
            std::fs::write(&path, &json)?;
            eprintln!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}
