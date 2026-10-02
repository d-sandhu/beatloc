#!/usr/bin/env python3
"""Generate the parity-test fixture for the neural engine.

Writes (committed to the repo):
  tests/fixtures/drum_loop_120bpm.wav          — deterministic synthetic audio
  tests/fixtures/drum_loop_120bpm.golden.json  — reference outputs

The golden is produced by the OFFICIAL Python reference: PyTorch small0 model,
beat_this's LogMelSpect frontend, split_predict_aggregate chunking
(1500 frames, 6-frame border, keep_first) and the "minimal" postprocessor.
The Rust test compares logits (max abs diff) and final times (±1 frame).

The audio is synthesized from scratch (no licensing concerns) with a
deterministic drum-loop recipe: kick on beats 1&3, snare-ish on 2&4, hats on
eighths, 120 BPM, 4/4, 8 bars. Dev-only tooling; run via .venv.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch

from beat_this.inference import load_model, split_predict_aggregate
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SR = 22050
BPM = 120.0
BARS = 8
CHECKPOINT = "small0"


def synth() -> np.ndarray:
    duration = BARS * 4 * 60.0 / BPM
    t = np.arange(int(duration * SR)) / SR
    out = np.zeros_like(t)

    def add(start: float, freq: float, tau: float, amp: float) -> None:
        i0 = int(start * SR)
        n = min(len(out) - i0, int(0.4 * SR))
        tt = np.arange(n) / SR
        out[i0 : i0 + n] += amp * np.sin(2 * np.pi * freq * tt) * np.exp(-tt / tau)

    beat = 60.0 / BPM
    for bar in range(BARS):
        for b in range(4):
            t0 = (bar * 4 + b) * beat
            if b in (0, 2):  # kick, accented on the downbeat
                add(t0, 50.0, 0.09, 0.9 if b == 0 else 0.75)
            else:  # snare-ish
                add(t0, 190.0, 0.06, 0.6)
                add(t0, 3300.0, 0.03, 0.15)
            for eighth in (0.0, 0.5):  # hats on eighths
                add(t0 + eighth * beat, 8000.0, 0.02, 0.2)
    return (out * 0.95 / max(1.0, np.abs(out).max())).astype(np.float64)


def write_wav16(path: Path, x: np.ndarray) -> None:
    pcm = np.clip(x, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm16)))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm16)))
        f.write(pcm16)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    wav_path = FIXTURES / "drum_loop_120bpm.wav"
    golden_path = FIXTURES / "drum_loop_120bpm.golden.json"

    wave = synth()
    # Quantize FIRST and run the reference on exactly the samples the Rust
    # side will decode from the WAV — no cross-language float drift.
    pcm16 = (np.clip(wave, -1.0, 1.0) * 32767).astype("<i2")
    wave_q = pcm16.astype(np.float64) / 32767.0
    write_wav16(wav_path, wave_q)
    print(f"wrote {wav_path} ({len(wave_q) / SR:.2f} s)")

    spect = LogMelSpect()(torch.from_numpy(wave_q).float())  # (frames, 128)
    model = load_model(CHECKPOINT)
    pred = split_predict_aggregate(
        spect, chunk_size=1500, border_size=6, overlap_mode="keep_first", model=model
    )
    beats, downbeats = Postprocessor(type="minimal", fps=50)(
        pred["beat"], pred["downbeat"]
    )

    golden = {
        "checkpoint": CHECKPOINT,
        "pipeline": "python reference (torch + LogMelSpect + split_predict_aggregate + minimal)",
        "sample_rate": SR,
        "fps": 50,
        "frames": int(spect.shape[0]),
        "beat_logits": [round(float(v), 6) for v in pred["beat"]],
        "downbeat_logits": [round(float(v), 6) for v in pred["downbeat"]],
        "beats": [round(float(v), 6) for v in beats],
        "downbeats": [round(float(v), 6) for v in downbeats],
    }
    golden_path.write_text(json.dumps(golden))
    print(f"wrote {golden_path}: {len(golden['beats'])} beats, {len(golden['downbeats'])} downbeats")


if __name__ == "__main__":
    main()
