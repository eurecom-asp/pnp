#!/usr/bin/env python
"""Shared ASV evaluation helpers for the open-source release."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.functional as AF


ATTACK_SCALE = float(1 << 15)
TARGET_SR = 16000
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


def Fbank(wav: torch.Tensor) -> torch.Tensor:
    mat = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=TARGET_SR,
        window_type="hamming",
        use_energy=False,
    )
    return mat - torch.mean(mat, dim=0)


def read_trials(path: Path) -> list[tuple[int, str, str]]:
    trials: list[tuple[int, str, str]] = []
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid trial line {line_no}: {line}")
            label = 1 if parts[0] in {"1", "target", "true", "True"} else 0
            trials.append((label, parts[1], parts[2]))
    return trials


def load_audio(path: Path, max_len: int | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    if max_len is not None and wav.shape[-1] > max_len:
        wav = wav[:, :max_len]
    return wav.to(device)


def trial_audio_path(root: Path, rel_path: str) -> Path:
    rel = rel_path.replace("\\", "/")
    return root / rel


def load_asv_model(backend: str, checkpoint: Path, device: str | torch.device):
    backend = backend.lower()
    if backend == "torch":
        try:
            model = torch.jit.load(str(checkpoint), map_location=device)
        except (RuntimeError, ValueError):
            model = torch.load(str(checkpoint), map_location=device)
    elif backend == "ecapa":
        obj = torch.load(str(checkpoint), map_location=device)
        if isinstance(obj, torch.nn.Module):
            model = obj
        else:
            from ecapa_tdnn import ECAPA_TDNN_GLOB_c512

            model = ECAPA_TDNN_GLOB_c512(feat_dim=80, embed_dim=192)
            model.load_state_dict(obj, strict=False)
    elif backend == "campp":
        try:
            model = torch.jit.load(str(checkpoint), map_location=device)
        except RuntimeError:
            from campplus import CAMPPlus

            model = CAMPPlus()
            model.load_state_dict(torch.load(str(checkpoint), map_location=device), strict=False)
    elif backend == "resnet":
        from resnet import ResNet221

        model = ResNet221(feat_dim=80, embed_dim=256, two_emb_layer=False)
        model.load_state_dict(torch.load(str(checkpoint), map_location=device), strict=False)
    elif backend == "samresnet":
        from samresnet import SimAM_ResNet100_ASP

        model = SimAM_ResNet100_ASP(embed_dim=256)
        model.load_state_dict(torch.load(str(checkpoint), map_location=device), strict=False)
    else:
        raise ValueError(f"Unsupported ASV backend: {backend}")

    model.to(device)
    model.eval()
    return model


def compute_embedding(model, wav_norm: torch.Tensor, device: str | torch.device):
    wav_attack_space = wav_norm * ATTACK_SCALE
    feat = Fbank(wav_attack_space.cpu()).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(feat)
    return outputs[-1] if isinstance(outputs, tuple) else outputs


def score_pair(model, enroll_wav: torch.Tensor, test_wav: torch.Tensor, device: str | torch.device) -> float:
    enroll_emb = compute_embedding(model, enroll_wav, device)
    test_emb = compute_embedding(model, test_wav, device)
    return float(similarity(enroll_emb, test_emb).view(()).item())


def compute_eer(scores: list[tuple[float, int]]) -> float:
    return compute_eer_threshold(scores)["eer"]


def compute_eer_threshold(scores: list[tuple[float, int]]) -> dict[str, float]:
    if not scores:
        raise ValueError("No scores were provided.")
    score_values = np.asarray([s for s, _ in scores], dtype=np.float64)
    labels = np.asarray([1 if int(y) == 1 else 0 for _, y in scores], dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("EER needs both target and non-target trials.")

    thresholds = np.concatenate(
        ([np.inf], np.sort(np.unique(score_values))[::-1], [-np.inf])
    )
    best_gap = np.inf
    result = {
        "eer": float("nan"),
        "threshold": float("nan"),
        "far": float("nan"),
        "frr": float("nan"),
    }
    for threshold in thresholds:
        accept = score_values >= threshold
        far = np.mean(accept[labels == 0])
        frr = np.mean(~accept[labels == 1])
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            result = {
                "eer": float(0.5 * (far + frr) * 100.0),
                "threshold": float(threshold),
                "far": float(far * 100.0),
                "frr": float(frr * 100.0),
            }
    return result


def compute_threshold_at_far(scores: list[tuple[float, int]], target_far: float) -> dict[str, float]:
    if not scores:
        raise ValueError("No scores were provided.")
    score_values = np.asarray([s for s, _ in scores], dtype=np.float64)
    labels = np.asarray([1 if int(y) == 1 else 0 for _, y in scores], dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("Threshold estimation needs both target and non-target trials.")

    target_far = float(target_far)
    if target_far > 1.0:
        target_far = target_far / 100.0
    thresholds = np.concatenate(
        ([np.inf], np.sort(np.unique(score_values))[::-1], [-np.inf])
    )
    feasible = []
    fallback = None
    fallback_gap = np.inf
    for threshold in thresholds:
        accept = score_values >= threshold
        far = np.mean(accept[labels == 0])
        frr = np.mean(~accept[labels == 1])
        gap = abs(far - target_far)
        item = {
            "threshold": float(threshold),
            "target_far": float(target_far * 100.0),
            "far": float(far * 100.0),
            "frr": float(frr * 100.0),
        }
        if far <= target_far + 1e-12:
            feasible.append(item)
        if gap < fallback_gap:
            fallback_gap = gap
            fallback = item
    if feasible:
        # Among thresholds that satisfy the requested FAR, keep the one that
        # rejects the fewest target trials.
        return min(feasible, key=lambda item: (item["frr"], -item["far"]))
    return fallback


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
