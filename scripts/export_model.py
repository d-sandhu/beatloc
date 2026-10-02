#!/usr/bin/env python3
"""Reproducible ONNX export of Beat This! (JKU Linz, MIT) for beatloc.

Exports two graphs:
  1. `mel_spectrogram.onnx` — the torchaudio log-mel frontend
     (waveform @22050 Hz -> [frames, 128]), so frontend parity is by
     construction rather than by reimplementation.
  2. `beat_this_<name>.onnx` — the BeatThis network
     ([1, frames, 128] -> beat logits [1, frames], downbeat logits [1, frames]).

Notes / gotchas (see repo notes for sources):
  - `PartialFTTransformer.forward` does `b = len(x)`; we always trace and run
    with batch=1, so the baked constant is correct and the time axis stays
    dynamic.
  - The SumHead emits beat = beat_logit + downbeat_logit; dict outputs become
    two ONNX outputs in insertion order (verified by assertion below).
  - Export numerics are validated against the PyTorch module on a synthetic
    signal before anything is written to models/manifest.json.

Dev-only tooling (uses the repo's .venv). Never needed at runtime.

Usage:
    .venv/bin/python scripts/export_model.py [small0|final0]   # default small0
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.onnx

from beat_this.inference import load_model
from beat_this.preprocessing import LogMelSpect

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OPSET = 17


class MelExport(torch.nn.Module):
    """Traceable re-implementation of beat_this's LogMelSpect frontend.

    torchaudio's MelSpectrogram calls torch.stft(return_complex=True), which
    the ONNX exporter rejects. Identical math with exportable ops:
    stft(return_complex=False) -> magnitude -> sqrt(n_fft) normalization ->
    Slaney mel matmul -> log1p(1000x). Validated against LogMelSpect below.
    (T,) waveform -> (1, frames, 128).
    """

    N_FFT = 1024
    HOP = 441

    def __init__(self):
        super().__init__()
        from torchaudio.functional import melscale_fbanks

        self.register_buffer(
            "window", torch.hann_window(self.N_FFT), persistent=False
        )  # periodic (torchaudio default)
        fb = melscale_fbanks(
            n_freqs=self.N_FFT // 2 + 1,
            f_min=30.0,
            f_max=11000.0,
            n_mels=128,
            sample_rate=22050,
            norm=None,  # torchaudio MelScale's default; beat_this does not override it
            mel_scale="slaney",
        )
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, waveform):
        spec = torch.stft(
            waveform,
            n_fft=self.N_FFT,
            hop_length=self.HOP,
            win_length=self.N_FFT,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=False,
        )  # (freq, frames, 2)
        mag = spec.pow(2).sum(-1).sqrt() / self.N_FFT**0.5  # frame_length norm
        mel = mag.T @ self.fb  # (frames, 128)
        return torch.log1p(1000 * mel).unsqueeze(0)


def export_mel(path: Path) -> None:
    module = MelExport().eval()
    dummy = torch.zeros(22050, dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        path,
        input_names=["waveform"],
        output_names=["spect"],
        dynamic_axes={"waveform": {0: "time"}, "spect": {1: "frames"}},
        opset_version=OPSET,
        dynamo=False,
    )


def export_model(checkpoint: str, path: Path) -> torch.nn.Module:
    model = load_model(checkpoint).eval()
    dummy = torch.zeros(1, 1500, 128, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["spect"],
        output_names=["beat", "downbeat"],
        dynamic_axes={
            "spect": {1: "frames"},
            "beat": {1: "frames"},
            "downbeat": {1: "frames"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    return model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "small0"
    MODELS.mkdir(exist_ok=True)

    mel_path = MODELS / "mel_spectrogram.onnx"
    model_path = MODELS / f"beat_this_{checkpoint}.onnx"

    print(f"exporting mel frontend -> {mel_path}")
    export_mel(mel_path)
    print(f"exporting {checkpoint} -> {model_path}")
    model = export_model(checkpoint, model_path)

    # Validate the ONNX graphs against the PyTorch references.
    import onnxruntime as ort

    rng = np.random.default_rng(20261002)
    # Musical-ish probe signal: 220 Hz + harmonics with an amplitude pulse.
    t = np.arange(22050 * 3) / 22050.0
    wave = (
        0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
        + 0.01 * rng.standard_normal(t.size)
    ).astype(np.float32)

    mel_session = ort.InferenceSession(str(mel_path))
    (onnx_mel,) = [o for o in mel_session.run(None, {"waveform": wave})]
    with torch.no_grad():
        ref_mel = LogMelSpect()(torch.from_numpy(wave)).unsqueeze(0).numpy()
        traceable_mel = MelExport()(torch.from_numpy(wave)).numpy()
    rewrite_diff = np.abs(traceable_mel - ref_mel).max()
    mel_diff = np.abs(onnx_mel - ref_mel).max()
    assert onnx_mel.shape == ref_mel.shape, (onnx_mel.shape, ref_mel.shape)
    print(f"mel: shape {onnx_mel.shape}, rewrite max|diff| {rewrite_diff:.3e}, onnx max|diff| {mel_diff:.3e}")
    assert rewrite_diff < 1e-3, "traceable mel rewrite diverged from LogMelSpect"
    assert mel_diff < 1e-3, "mel export diverged"

    model_session = ort.InferenceSession(str(model_path))
    outputs = model_session.run(None, {"spect": onnx_mel})
    names = [o.name for o in model_session.get_outputs()]
    assert names == ["beat", "downbeat"], names
    with torch.no_grad():
        ref = model(torch.from_numpy(onnx_mel))
    beat_diff = np.abs(outputs[0] - ref["beat"].numpy()).max()
    down_diff = np.abs(outputs[1] - ref["downbeat"].numpy()).max()
    print(f"model: outputs {[o.shape for o in outputs]}, max|diff| beat {beat_diff:.3e}, downbeat {down_diff:.3e}")
    assert beat_diff < 1e-3 and down_diff < 1e-3, "model export diverged"

    manifest_path = MODELS / "manifest.json"
    # Merge into any existing manifest so all exported models stay tracked.
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"opset": OPSET, "source": "CPJKU/beat_this (MIT), exported by scripts/export_model.py", "files": {}}
    )
    manifest["opset"] = OPSET
    manifest.setdefault("files", {})[model_path.name] = {
        "sha256": sha256(model_path),
        "bytes": model_path.stat().st_size,
        "checkpoint": checkpoint,
    }
    manifest["files"][mel_path.name] = {
        "sha256": sha256(mel_path),
        "bytes": mel_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
