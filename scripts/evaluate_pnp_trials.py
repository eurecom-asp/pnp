#!/usr/bin/env python
"""Evaluate PnP purification on clean or locally generated attacked trials."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from common_eval import (
    compute_eer,
    load_asv_model,
    load_audio,
    read_trials,
    score_pair,
    trial_audio_path,
    write_json,
)
from pnp_runtime import load_pnp_model, pnp_forward


def maybe_purify(args, pnp_model, wav: torch.Tensor) -> torch.Tensor:
    if args.purifier == "no_defender":
        return wav
    simple_add = args.purifier == "pnp_gaussian"
    t_step = 2 if args.purifier == "pnp_diff_2" else args.t_step
    with torch.no_grad():
        return pnp_forward(
            pnp_model,
            wav,
            t_index=t_step,
            lam=args.lam,
            simple_add=simple_add,
            clamp=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--attack-root", default=None, type=Path)
    parser.add_argument("--asv-backend", choices=["torch", "ecapa", "campp", "resnet", "samresnet"], default="ecapa")
    parser.add_argument("--asv-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--purifier",
        choices=["no_defender", "pnp_gaussian", "pnp_diff", "pnp_diff_2"],
        default="no_defender",
    )
    parser.add_argument("--pnp-checkpoint", default=None, type=Path)
    parser.add_argument("--t-step", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=16000 * 20)
    parser.add_argument("--out-json", default=Path("outputs/eval_pnp_trials.json"), type=Path)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_asv_model(args.asv_backend, args.asv_checkpoint, device)
    pnp_model = None
    if args.purifier != "no_defender":
        if args.pnp_checkpoint is None:
            raise ValueError("--pnp-checkpoint is required for PnP purifiers.")
        pnp_model = load_pnp_model(args.pnp_checkpoint, device=device)

    records = []
    scores = []
    for label, enroll_rel, test_rel in tqdm(read_trials(args.trials), desc=args.purifier, dynamic_ncols=True):
        enroll = load_audio(trial_audio_path(args.clean_root, enroll_rel), args.max_len, device)
        if args.attack_root is not None and trial_audio_path(args.attack_root, test_rel).exists():
            test_path = trial_audio_path(args.attack_root, test_rel)
        else:
            test_path = trial_audio_path(args.clean_root, test_rel)
        test = load_audio(test_path, args.max_len, device)
        test = maybe_purify(args, pnp_model, test)
        score = score_pair(model, enroll, test, device)
        scores.append((score, label))
        records.append(
            {
                "label": label,
                "enroll": enroll_rel,
                "test": test_rel,
                "source": str(test_path),
                "score": score,
            }
        )

    payload = {
        "eer": compute_eer(scores),
        "num_trials": len(scores),
        "purifier": args.purifier,
        "t_step": 2 if args.purifier == "pnp_diff_2" else args.t_step,
        "lambda": args.lam,
        "scores": records,
    }
    write_json(args.out_json, payload)
    print(f"eer={payload['eer']:.4f}")
    print(f"out_json={args.out_json}")


if __name__ == "__main__":
    main()
