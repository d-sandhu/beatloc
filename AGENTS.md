# AGENTS.md — beatloc

Rust CLI: audio file → machine-readable musical timeline (JSON) for
video-making agents. Beats, downbeats, bars, tempo (global + local), onsets,
energy, sections. The analyzer contains no LLM; agents make creative
decisions from the JSON.

## First thing when joining this project

If `notes/architecture-memo.md` exists locally (it is git-ignored and never
committed), read it before anything else — it is the private design log with
all decisions, measured results, licenses, and risks. Then `README.md` and
`git log --oneline`.

## Build / test / eval

- `cargo build --release` / `cargo test --release` (release mode keeps the
  neural parity test fast).
- Neural models live in `models/` (git-ignored). If missing:
  `python3 -m venv .venv && .venv/bin/pip install beat-this onnxruntime onnxscript`
  then `.venv/bin/python scripts/export_model.py small0` (and `final0`).
- Accuracy eval: `scripts/fetch_datasets.sh`, then
  `cargo run --release --example eval -- datasets/BallroomData datasets/ballroom-annotations`.
- Metrics differential test vs Python mir_eval (dev-only):
  `.venv/bin/python scripts/differential_check.py`.

## Rules that matter

- stdout carries ONLY machine-readable JSON; diagnostics to stderr; exit
  codes 0 = ok, 1 = analysis/IO error, 2 = usage error. Never regress this.
- Unknown/unsupported results are ABSENT fields, never sentinels or nulls.
- Never emit scores whose semantics aren't documented. Model scores are
  uncalibrated — say so.
- Schema changes bump `SCHEMA_VERSION`; every field documents its units and
  timestamp convention (frame centre vs model grid).
- Never commit `datasets/`, `models/`, `notes/`, `.venv/`, or any
  non-synthetic audio. Test fixtures must be synthesized by our own code.
- Git mutations (commit/push/…) only when the user explicitly asks.
- No plugin frameworks, no custom model training, no features without a
  measurement story. Small, tested, reviewable changes.
