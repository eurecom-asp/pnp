#!/usr/bin/env python
"""Estimate ASV score thresholds from clean trials."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from common_eval import (
    compute_eer_threshold,
    compute_threshold_at_far,
    load_asv_model,
    load_audio,
    read_trials,
    score_pair,
    trial_audio_path,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate an ASV decision threshold from clean trial scores."
    )
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument(
        "--asv-backend",
        choices=["torch", "ecapa", "campp", "resnet", "samresnet"],
        default="ecapa",
    )
    parser.add_argument("--asv-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=16000 * 20)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help="Optional target FAR. Values >1 are interpreted as percentages.",
    )
    parser.add_argument("--out-json", default=Path("outputs/scores/asv_threshold.json"), type=Path)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_asv_model(args.asv_backend, args.asv_checkpoint, device)
    trials = read_trials(args.trials)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    scores: list[tuple[float, int]] = []
    for label, enroll_rel, test_rel in tqdm(trials, desc=f"threshold:{args.asv_backend}", dynamic_ncols=True):
        enroll = load_audio(trial_audio_path(args.clean_root, enroll_rel), args.max_len, device)
        test = load_audio(trial_audio_path(args.clean_root, test_rel), args.max_len, device)
        scores.append((score_pair(model, enroll, test, device), label))

    eer_result = compute_eer_threshold(scores)
    payload = {
        "num_trials": len(scores),
        "asv_backend": args.asv_backend,
        "threshold_at_eer": eer_result["threshold"],
        "eer": eer_result["eer"],
        "far_at_eer_threshold": eer_result["far"],
        "frr_at_eer_threshold": eer_result["frr"],
        "target_far_result": compute_threshold_at_far(scores, args.target_far) if args.target_far is not None else None,
    }
    write_json(args.out_json, payload)

    print(f"threshold_at_eer={payload['threshold_at_eer']:.6f}")
    print(f"eer={payload['eer']:.4f}")
    print(f"far_at_eer_threshold={payload['far_at_eer_threshold']:.4f}")
    print(f"frr_at_eer_threshold={payload['frr_at_eer_threshold']:.4f}")
    if payload["target_far_result"] is not None:
        target = payload["target_far_result"]
        print(f"threshold_at_target_far={target['threshold']:.6f}")
        print(f"target_far={target['target_far']:.4f}")
        print(f"actual_far={target['far']:.4f}")
        print(f"actual_frr={target['frr']:.4f}")
    print(f"out_json={args.out_json}")


if __name__ == "__main__":
    main()
