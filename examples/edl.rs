//! Example agent-side consumer: turn a beatloc timeline into an edit
//! decision list (EDL). This is the product loop proven end to end — beatloc
//! produces the timeline; *this* is how an agent consumes it.
//!
//! ```sh
//! beatloc track.mp3 --json > timeline.json
//! cargo run --example edl -- timeline.json
//! beatloc track.mp3 --json | cargo run --example edl    # reads stdin
//! ```
//!
//! Rules are deliberately simple — creative judgment belongs to the agent:
//! - `scene_change`: every 2 bars from the first downbeat.
//! - `reveal`: the first downbeat at or after the strongest section
//!   transition (the "drop into the chorus" use case).
//! - `section_start`: every detected section boundary.
//!
//! Output is JSON on stdout, ready to pipe onward.

use std::io::Read;

fn main() {
    // Read the timeline from a file argument or stdin.
    let mut input = String::new();
    match std::env::args().nth(1) {
        Some(path) => {
            input = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                eprintln!("error: cannot read {path}: {e}");
                std::process::exit(1);
            });
        }
        None => {
            std::io::stdin().read_to_string(&mut input).expect("stdin readable");
        }
    }
    let timeline: serde_json::Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: input is not a beatloc timeline: {e}");
            std::process::exit(1);
        }
    };

    let mut events: Vec<serde_json::Value> = Vec::new();

    // Scene changes: every 2 bars from the first downbeat.
    if let Some(downbeats) = timeline["downbeats"]["items"].as_array() {
        for pair in downbeats.chunks(2) {
            let time = pair[0]["time"].as_f64().unwrap();
            let bar = pair[0]["bar"].as_u64().unwrap();
            events.push(serde_json::json!({
                "time": time,
                "type": "scene_change",
                "detail": format!("every 2 bars (bar {bar})"),
            }));
        }
    }

    // Section starts.
    let sections = timeline["sections"]["items"].as_array();
    if let Some(sections) = sections {
        for s in sections.iter().skip(1) {
            events.push(serde_json::json!({
                "time": s["start"].as_f64().unwrap(),
                "type": "section_start",
                "detail": format!(
                    "transition strength {:.2}",
                    s["transition_strength"].as_f64().unwrap_or(0.0)
                ),
            }));
        }
    }

    // Reveal: first downbeat at/after the strongest transition.
    let strongest = sections.and_then(|items| {
        items
            .iter()
            .filter(|s| s["transition_strength"].is_number())
            .max_by(|a, b| {
                a["transition_strength"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["transition_strength"].as_f64().unwrap())
            })
            .and_then(|s| s["start"].as_f64())
    });
    if let (Some(strongest), Some(downbeats)) =
        (strongest, timeline["downbeats"]["items"].as_array())
    {
        if let Some(d) = downbeats.iter().find(|d| d["time"].as_f64().unwrap() >= strongest) {
            events.push(serde_json::json!({
                "time": d["time"].as_f64().unwrap(),
                "type": "reveal",
                "detail": format!(
                    "first downbeat (bar {}) at/after strongest transition at {strongest:.2} s",
                    d["bar"].as_u64().unwrap()
                ),
            }));
        }
    }

    events.sort_by(|a, b| {
        a["time"].as_f64().unwrap_or(0.0).total_cmp(&b["time"].as_f64().unwrap_or(0.0))
    });
    let out = serde_json::json!({
        "source_duration": timeline["source"]["duration_seconds"],
        "timeline_schema": timeline["format"]["version"],
        "events": events,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
