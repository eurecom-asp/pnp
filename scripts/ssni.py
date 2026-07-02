#!/usr/bin/env python
"""Sample-specific step selection for SSNI-style purification.

The selector is intentionally checkpoint- and dataset-agnostic. A purifier
provides a callable that maps one waveform to one scalar score, and this module
maps that score to a purification step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import torch
import torchaudio
import torchaudio.functional as AF


TARGET_SR = 16000


def _ensure_1xt(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() == 3 and audio.shape[0] == 1 and audio.shape[1] == 1:
        audio = audio.squeeze(0)
    if audio.dim() != 2 or audio.shape[0] != 1:
        raise ValueError(f"Expected mono audio shaped [1, T], got {tuple(audio.shape)}.")
    return audio


def _load_audio(path: Path, device: torch.device) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        audio = AF.resample(audio, sr, TARGET_SR)
    return _ensure_1xt(audio).to(device)


def _reference_items(reference_list: Path, reference_root: Path, max_items: int | None) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    with Path(reference_list).open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rels = parts[1:3] if len(parts) >= 3 and parts[0] in {"0", "1"} else parts[:1]
            for rel in rels:
                if rel in seen:
                    continue
                seen.add(rel)
                paths.append(Path(reference_root) / rel)
                if max_items is not None and len(paths) >= max_items:
                    return paths
    return paths


def load_stats(path: Path) -> dict:
    path = Path(path)
    if path.suffix in {".pt", ".pth"}:
        stats = torch.load(str(path), map_location="cpu")
        if "mean" in stats and "std" in stats:
            return stats
        if "eps_mu" in stats and "eps_std" in stats:
            return {
                "mean": float(stats["eps_mu"]),
                "std": float(stats["eps_std"]),
                "num_items": int(stats.get("max_reference_samples", 0)),
            }
        raise ValueError(f"Unsupported SSNI stats fields in {path}.")
    with path.open("r") as f:
        return json.load(f)


def save_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")


class SSNIStepSelector:
    """Map a sample score to a purification step.

    For two-step SSNI, scores above the sigmoid threshold use the larger step.
    With no reference statistics, the selector falls back to a running mean.
    """

    def __init__(
        self,
        scorer: Callable[[torch.Tensor], float],
        device: torch.device,
        min_step: int = 1,
        max_step: int = 2,
        tau: float = 0.5,
        bias: float = 0.9,
        reference_stats: Path | None = None,
        reference_list: Path | None = None,
        reference_root: Path | None = None,
        reference_size: int | None = 1000,
        save_reference_stats: Path | None = None,
    ):
        if min_step > max_step:
            raise ValueError("--ssni-min-step cannot be larger than --ssni-max-step.")
        self.scorer = scorer
        self.device = device
        self.min_step = int(min_step)
        self.max_step = int(max_step)
        self.tau = float(tau)
        self.bias = float(bias)
        self.center: float | None = None
        self.scale: float = 1.0
        self.count = 0
        self.running_mean = 0.0
        self.step_hist: dict[int, int] = {}

        if reference_stats is not None:
            stats = load_stats(reference_stats)
            self.center = float(stats["mean"])
            self.scale = max(float(stats.get("std", 1.0)), 1e-6)
        elif reference_list is not None and reference_root is not None:
            stats = self.build_reference_stats(reference_list, reference_root, reference_size)
            self.center = float(stats["mean"])
            self.scale = max(float(stats.get("std", 1.0)), 1e-6)
            if save_reference_stats is not None:
                save_stats(save_reference_stats, stats)

    def build_reference_stats(
        self,
        reference_list: Path,
        reference_root: Path,
        reference_size: int | None = 1000,
    ) -> dict:
        paths = _reference_items(reference_list, reference_root, reference_size)
        if not paths:
            raise ValueError("No reference audio found for SSNI statistics.")
        scores = []
        for path in paths:
            if not path.exists():
                continue
            scores.append(self.scorer(_load_audio(path, self.device)))
        if not scores:
            raise ValueError("Reference list did not contain any readable audio.")
        values = torch.tensor(scores, dtype=torch.float64)
        return {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).clamp_min(1e-6).item()),
            "num_items": int(values.numel()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }

    def _fallback_center(self, score: float) -> float:
        center = self.running_mean if self.count > 0 else score
        self.count += 1
        self.running_mean += (score - self.running_mean) / float(self.count)
        return center

    def select(self, audio: torch.Tensor) -> tuple[int, float]:
        score = float(self.scorer(_ensure_1xt(audio)))
        center = self.center if self.center is not None else self._fallback_center(score)
        denom = max(self.tau * self.scale, 1e-6)
        logit = max(min((score - center) / denom, 60.0), -60.0)
        probability = 1.0 / (1.0 + math.exp(-logit))

        if self.min_step == self.max_step:
            step = self.min_step
        elif self.max_step - self.min_step == 1:
            step = self.max_step if probability > self.bias else self.min_step
        else:
            adjusted = min(max((probability - self.bias) / max(1.0 - self.bias, 1e-6), 0.0), 1.0)
            step = self.min_step + int(round(adjusted * (self.max_step - self.min_step)))

        self.step_hist[step] = self.step_hist.get(step, 0) + 1
        return step, score

    def summary(self) -> dict:
        return {
            "min_step": self.min_step,
            "max_step": self.max_step,
            "tau": self.tau,
            "bias": self.bias,
            "center": self.center,
            "scale": self.scale,
            "running_count": self.count,
            "running_mean": self.running_mean,
            "step_hist": dict(sorted(self.step_hist.items())),
        }
