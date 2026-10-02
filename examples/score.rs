//! Score two beat sequences against each other, print metrics as JSON.
//! Dev tool for differential testing against Python mir_eval.
//!
//! ```sh
//! cargo run --release --example score -- reference.beats estimate.beats
//! ```
//!
//! Input format: one time in seconds per line (extra columns ignored).

use std::path::Path;

use beatloc::eval::{continuity, f_measure};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(reference), Some(estimate)) = (args.next(), args.next()) else {
        eprintln!("usage: score <reference.beats> <estimate.beats>");
        std::process::exit(2);
    };
    let reference = read_times(Path::new(&reference));
    let estimate = read_times(Path::new(&estimate));

    let fm = f_measure(&reference, &estimate, 0.07);
    let c = continuity(&reference, &estimate);
    println!(
        "{{\"f_measure\": {:.17}, \"cmlc\": {:.17}, \"cmlt\": {:.17}, \"amlc\": {:.17}, \"amlt\": {:.17}}}",
        fm.f, c.cmlc, c.cmlt, c.amlc, c.amlt
    );
}

fn read_times(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .expect("readable")
        .lines()
        .filter_map(|line| line.split_whitespace().next()?.parse::<f64>().ok())
        .collect()
}
