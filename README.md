# beatloc

Convert audio into a machine-readable musical timeline (JSON) — built for
video-making agents that need to know *where musical events happen* without
reasoning about raw audio.

The CLI supplies musical information; the agent makes the creative decisions.
The analyzer itself contains no LLM.

**Status: active development (beats, downbeats, bars, tempo, onsets, energy,
sections).** Output schema is versioned but unstable until 1.0.0.

## Install / build

```sh
cargo build --release        # produces a single self-contained binary
cargo test                   # unit + integration + CLI contract tests
```

The neural engine needs its ONNX model files (not committed; see
`models/manifest.json` for provenance). Regenerate them with:

```sh
python3 -m venv .venv && .venv/bin/pip install beat-this onnxruntime onnxscript
.venv/bin/python scripts/export_model.py small0   # or final0 (full accuracy)
```

Without model files, beatloc falls back to the DSP baseline (`--engine dsp`).

## Usage

```sh
beatloc track.mp3 --json                       # JSON timeline to stdout
beatloc track.mp3 --output timeline.json       # to a file (never overwrites...)
beatloc track.mp3 -o timeline.json --force     # ...unless forced
beatloc track.mp3 --curves --json              # include dense per-frame curves
beatloc track.mp3 --engine dsp --json          # classical baseline (no downbeats)
beatloc track.mp3 --engine neural --json       # neural engine (beats + downbeats)
beatloc track.mp3 --model models/beat_this_final0.onnx --json   # full-accuracy model
```

Input: WAV, MP3, FLAC (pure-Rust decoding via
[Symphonia](https://github.com/pdeljanov/Symphonia)).

Contract for machine consumers:

- stdout carries **only** the JSON result; diagnostics go to stderr.
- Exit codes: `0` success, `1` analysis/IO error, `2` usage error.

## Output (schema v0.4.0)

```jsonc
{
  "format":    { "name": "beatloc-timeline", "version": "0.4.0" },
  "generator": { "name": "beatloc", "version": "0.1.0" },
  "source":    { "duration_seconds": 9.5, "sample_rate": 44100, "channels": 2, "codec": "mp3" },
  "analysis":  { "sample_rate": 22050, "window_size": 1024, "hop_size": 441,
                 "window": "hann (periodic)",
                 "timestamps": "seconds from decoded stream start (gapless-trimmed) to frame centre" },
  "tempo":     { "engine": "beat-this-small0 (median-ibi)", "bpm": 120.0 },
  "beats":     { "engine": "beat-this-small0",
                 "items": [ { "time": 1.02, "index": 0, "bar": 1, "bar_position": 1 } ] },
  "downbeats": { "engine": "beat-this-small0", "items": [ { "time": 1.02, "bar": 1 } ] },
  "sections":  { "engine": "dsp-novelty-v1",
                 "items": [ { "index": 0, "start": 0.0, "end": 16.42 },
                            { "index": 1, "start": 16.42, "end": 30.05, "transition_strength": 0.91 } ] },
  "onsets":    [ { "time": 0.983, "strength": 12.34 } ],
  "curves":    { "energy":         { "start_seconds": 0.0232, "hop_seconds": 0.02, "units": "dbfs", "values": [] },
                 "onset_strength": { "start_seconds": 0.0232, "hop_seconds": 0.02, "units": "spectral_flux", "values": [] } }
}
```

- `tempo` is **absent** when no periodicity could be established (e.g.
  silence) — unknown is represented by omission, never a sentinel value.
- `beats.items[].index` is a 0-based counter. `bar` is 1-based from the
  first detected downbeat; `bar_position` counts from 1 within the bar
  (downbeat = 1). Neither is a time-signature claim; both are absent when
  downbeats are unknown (e.g. the DSP engine) or before the first downbeat
  (anacrusis).
- `downbeats` is absent for engines without downbeat support.
- `sections` come from a feature-contrast novelty detector
  (`dsp-novelty-v1`): they mark THAT something changed, not WHAT it is
  (no intro/verse/chorus labels). `transition_strength` ranks boundaries
  within the same track only.
- `periodicity` (DSP tempo only) is an uncalibrated diagnostic, not a
  probability.
- Neural-engine times follow the model's own frame grid (frame index / 50 s);
  DSP-feature times use unpadded frame centres. Conventions are recorded in
  `analysis` and each section's `engine` field.

### Consuming the timeline (agent side)

`examples/edl.rs` shows the intended loop: read the timeline JSON and turn it
into edit decisions — scene changes every 2 bars, a reveal on the first
downbeat after the strongest transition:

```sh
beatloc track.mp3 --json | cargo run --example edl
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

Measured on the Ballroom dataset (698 tracks, mir_eval conventions, first
5 s trimmed):

| Engine | Beat F1 | CMLt | AMLt | Downbeat F1 | Mean offset | Speed | Model size |
|---|---|---|---|---|---|---|---|
| `beat-this-final0` | **0.988** | **0.985** | 0.985 | **0.986** | −0.7 ms | 47× RT | 83 MB |
| `beat-this-small0` (default) | 0.984 | 0.979 | 0.980 | 0.982 | +3.2 ms | 83× RT | 10 MB |
| `dsp-ellis2007` (baseline) | 0.771 | 0.575 | 0.850 | — | +7.1 ms | 3650× RT | none |

small0 is the default: within ~0.4 F1 points of the full model at ⅛ the
size. Pass `--model models/beat_this_final0.onnx` for maximum accuracy.

Honest limitations: the neural engine inherits Beat This!'s training-data
biases (Western 4/4-heavy); it can still lock offbeat on unusual textures,
and it does not report a time signature — only beat positions within
detected bars. Onset detection is a spectral-flux heuristic. The DSP engine
assumes one global tempo and exists as a fallback/baseline, not the product.

Reproduce the numbers:

```sh
scripts/fetch_datasets.sh   # downloads Ballroom audio + annotations (research use only)
cargo run --release --example eval -- datasets/BallroomData datasets/ballroom-annotations --engine neural
```

## Roadmap

| Milestone | Content |
|---|---|
| **V0.1** ✅ | decode, metadata, energy, onsets, versioned JSON, alignment tests |
| **V0.2** ✅ | beats + global tempo (DSP baseline); Rust eval metrics differential-tested vs mir_eval; measured accuracy on Ballroom |
| **V0.3** ✅ | neural engine (Beat This! via rten/ONNX): downbeats, bar positions, parity-tested vs the Python reference |
| **V0.4** 🚧 | sections/transitions v0 ✅; schema 1.0.0, cross-platform verification, benchmarks (release itself deferred by project owner) |
| later | section labels (verse/chorus), richer transitions, streaming for very long files |

## License

MIT OR Apache-2.0
