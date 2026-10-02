//! CLI contract tests: clean stdout, stderr for diagnostics, exit codes,
//! deliberate overwrite behavior.

mod common;

use std::process::Command;

fn beatloc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_beatloc"))
}

#[test]
fn stdout_is_pure_json_and_stderr_is_silent() {
    let wav = common::temp_wav_path("cli-stdout.wav");
    common::write_wav_i16(&wav, 22_050, 1, &common::click_track(22_050, &[0.5, 1.0], 1.5));

    let out = beatloc().arg(&wav).arg("--json").output().unwrap();

    assert!(out.status.success());
    assert!(out.stderr.is_empty(), "stderr not empty: {}", String::from_utf8_lossy(&out.stderr));
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout is not JSON");
    assert_eq!(json["format"]["name"], "beatloc-timeline");
    assert_eq!(json["format"]["version"], "0.1.0");
    assert_eq!(json["generator"]["name"], "beatloc");

    std::fs::remove_file(&wav).ok();
}

#[test]
fn output_file_never_overwrites_without_force() {
    let wav = common::temp_wav_path("cli-overwrite.wav");
    let json_path = wav.with_extension("json");
    common::write_wav_i16(&wav, 22_050, 1, &common::click_track(22_050, &[0.5], 1.0));

    let first = beatloc().arg(&wav).arg("--output").arg(&json_path).output().unwrap();
    assert!(first.status.success());
    assert!(json_path.exists());

    let second = beatloc().arg(&wav).arg("--output").arg(&json_path).output().unwrap();
    assert_eq!(second.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&second.stderr).contains("already exists"));

    let forced =
        beatloc().arg(&wav).arg("--output").arg(&json_path).arg("--force").output().unwrap();
    assert!(forced.status.success());

    std::fs::remove_file(&wav).ok();
    std::fs::remove_file(&json_path).ok();
}

#[test]
fn missing_input_is_a_clean_error() {
    let out = beatloc().arg("/nonexistent/definitely-not-here.wav").output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(!out.stderr.is_empty());
}
