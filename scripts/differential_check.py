#!/usr/bin/env python3
"""Differential test: beatloc's Rust metrics vs Python mir_eval.

Generates randomized beat sequences (tempo drift, jitter, dropouts, wrong
metrical levels), scores them with both implementations, and reports the
maximum absolute divergence. Uses raw mir_eval functions (no 5 s trim).

Dev-only tooling — Python is never a runtime dependency of beatloc itself.

Usage:
    .venv/bin/python scripts/differential_check.py [target/release/examples/score]
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import mir_eval


def make_reference(rng: random.Random) -> list[float]:
    bpm = rng.uniform(60, 200)
    ibi = 60.0 / bpm
    n = rng.randint(20, 120)
    beats = []
    t = rng.uniform(0.0, 2.0)
    for _ in range(n):
        beats.append(t)
        t += ibi * rng.uniform(0.97, 1.03)  # mild tempo drift
    return beats


def corrupt(rng: random.Random, ref: list[float]) -> list[float]:
    style = rng.choice(["tight", "jittery", "dropouts", "double", "half", "offbeat"])
    est = ref
    if style == "double":
        est = sorted(t for pair in zip(ref, ref[1:]) for t in pair) + [
            (a + b) / 2 for a, b in zip(ref, ref[1:])
        ]
        est = sorted(set(round(t, 6) for t in est))
    elif style == "half":
        est = ref[::2]
    elif style == "offbeat":
        ibi = (ref[-1] - ref[0]) / max(len(ref) - 1, 1)
        est = [t + ibi / 2 for t in ref]
    jitter = 0.0 if style == "tight" else rng.uniform(0.005, 0.04)
    out = []
    for t in est:
        if style == "dropouts" and rng.random() < 0.25:
            continue
        out.append(t + rng.gauss(0, jitter))
    return sorted(out)


def main() -> int:
    score_bin = sys.argv[1] if len(sys.argv) > 1 else "target/release/examples/score"
    rng = random.Random(20261002)
    worst = 0.0
    trials = 300
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.txt"
        est_path = Path(tmp) / "est.txt"
        for i in range(trials):
            ref = make_reference(rng)
            est = corrupt(rng, ref)
            ref_path.write_text("\n".join(f"{t:.6f}" for t in ref))
            est_path.write_text("\n".join(f"{t:.6f}" for t in est))

            ours = json.loads(subprocess.run(
                [score_bin, str(ref_path), str(est_path)],
                capture_output=True, text=True, check=True,
            ).stdout)

            ref_trimmed = [t for t in ref]  # raw functions, no 5 s trim
            est_trimmed = [t for t in est]
            theirs_f = mir_eval.beat.f_measure(
                *[__import__("numpy").array(x) for x in (ref_trimmed, est_trimmed)]
            )
            theirs_c = mir_eval.beat.continuity(
                *[__import__("numpy").array(x) for x in (ref_trimmed, est_trimmed)]
            )
            for ours_v, theirs_v, name in [
                (ours["f_measure"], theirs_f, "f_measure"),
                (ours["cmlc"], theirs_c[0], "cmlc"),
                (ours["cmlt"], theirs_c[1], "cmlt"),
                (ours["amlc"], theirs_c[2], "amlc"),
                (ours["amlt"], theirs_c[3], "amlt"),
            ]:
                diff = abs(ours_v - theirs_v)
                worst = max(worst, diff)
                if diff > 1e-9:
                    print(f"TRIAL {i}: {name} ours={ours_v!r} theirs={theirs_v!r}")
    print(f"{trials} trials, max |diff| = {worst:.3e}")
    return 0 if worst <= 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
