# beatloc

Convert audio into a machine-readable musical timeline (JSON) — built for
video-making agents that need to know *where musical events happen* without
reasoning about raw audio.

The CLI supplies musical information; the agent makes the creative decisions.
The analyzer itself contains no LLM.

**Status: early development (V0.1 — signal landmarks).** See the roadmap
below. Output schema is versioned but unstable until 1.0.0.

## Install / build

```sh
cargo build --release        # produces a single self-contained binary
cargo test                   # unit + integration + CLI contract tests
```

## Usage

```sh
beatloc track.mp3 --json                       # JSON timeline to stdout
beatloc track.mp3 --output timeline.json       # to a file (never overwrites...)
beatloc track.mp3 -o timeline.json --force     # ...unless forced
beatloc track.mp3 --curves --json              # include dense per-frame curves
```

Input: WAV, MP3, FLAC (pure-Rust decoding via
[Symphonia](https://github.com/pdeljanov/Symphonia)).

Contract for machine consumers:

- stdout carries **only** the JSON result; diagnostics go to stderr.
- Exit codes: `0` success, `1` analysis/IO error, `2` usage error.

## Output (schema v0.1.0)

```jsonc
{
  "format":    { "name": "beatloc-timeline", "version": "0.1.0" },
  "generator": { "name": "beatloc", "version": "0.1.0" },
  "source":    { "duration_seconds": 9.5, "sample_rate": 44100, "channels": 2, "codec": "mp3" },
  "analysis":  { "sample_rate": 22050, "window_size": 1024, "hop_size": 441,
                 "window": "hann (periodic)",
                 "timestamps": "seconds from decoded stream start (gapless-trimmed) to frame centre" },
  "onsets":    [ { "time": 0.983, "strength": 12.34 } ],
  "curves":    { "energy":         { "start_seconds": 0.0232, "hop_seconds": 0.02, "units": "dbfs", "values": [] },
                 "onset_strength": { "start_seconds": 0.0232, "hop_seconds": 0.02, "units": "spectral_flux", "values": [] } }
}
```

Conventions:

- All times are seconds (f64) from the start of the decoded stream.
  Gapless metadata (e.g. MP3 encoder delay) is trimmed; MP3s without a
  Xing/LAME tag may be offset by up to ~50 ms.
- Analysis runs on mono (arithmetic-mean downmix) at 22050 Hz, frames of
  1024 samples with hop 441 (50 fps). Frame-derived times are frame centres.
- `energy` is frame RMS in dBFS (floor −120). `onset_strength` is spectral
  flux in arbitrary, track-relative units.
- Dense curves are omitted unless `--curves` is passed.

Current limitations (honest ones): onset detection is a simple spectral-flux
heuristic tuned for clear attacks — it misses soft onsets and is not yet
evaluated against annotated data. No beats, tempo, or downbeats yet.

## Roadmap

| Milestone | Content |
|---|---|
| **V0.1** ✅ | decode, metadata, energy, onsets, versioned JSON, alignment tests |
| V0.2 | beat timestamps, tempo (global/local), accuracy evaluation vs. annotated music |
| V0.3 | downbeats & bar positions via neural inference, parity-tested |
| V0.4 | release hardening: schema 1.0.0, cross-platform builds, benchmarks |
| later | section boundaries & labels (chorus entrances, drops), transitions |

## License

MIT OR Apache-2.0
