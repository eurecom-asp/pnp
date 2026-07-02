#!/usr/bin/env python
"""Generate compact spectrogram thumbnails for the static audio demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)
    return sample_rate, audio


def save_spectrogram(wav_path: Path, image_path: Path) -> None:
    sample_rate, audio = read_wav(wav_path)
    _, _, zxx = stft(
        audio,
        fs=sample_rate,
        nperseg=512,
        noverlap=384,
        boundary=None,
    )
    magnitude = np.abs(zxx)
    db = 20.0 * np.log10(np.maximum(magnitude, 1e-6))
    top_db = float(np.max(db))
    db = np.clip(db, top_db - 80.0, top_db)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 1.42), dpi=150)
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(image_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("demo/audio"))
    parser.add_argument("--out-root", type=Path, default=Path("demo/spectrograms"))
    args = parser.parse_args()

    audio_root = args.audio_root
    out_root = args.out_root
    wav_files = sorted(audio_root.rglob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No wav files found under {audio_root}")

    for wav_path in wav_files:
        rel = wav_path.relative_to(audio_root).with_suffix(".png")
        image_path = out_root / rel
        save_spectrogram(wav_path, image_path)
        print(image_path)

    print(f"generated={len(wav_files)}")


if __name__ == "__main__":
    main()
