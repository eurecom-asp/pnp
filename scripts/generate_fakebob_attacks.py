#!/usr/bin/env python
"""Generate FAKEBOB-style query attacks for ASV trials."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from common_eval import (
    ATTACK_SCALE,
    compute_embedding,
    load_asv_model,
    load_audio,
    read_trials,
    trial_audio_path,
)
from fakebob_attack import fakebob_sv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FAKEBOB-style black-box attacks from a VoxCeleb-style trial file."
    )
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("data/generated/fakebob_attacks"), type=Path)
    parser.add_argument(
        "--asv-backend",
        choices=["torch", "ecapa", "campp", "resnet", "samresnet"],
        default="ecapa",
    )
    parser.add_argument("--asv-checkpoint", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--target-decision", choices=["accept", "reject"], default="accept")
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=1.0)
    parser.add_argument("--min-lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-draw", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--plateau-length", type=int, default=5)
    parser.add_argument("--plateau-drop", type=float, default=2.0)
    parser.add_argument("--eot-steps", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=16000 * 20)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument(
        "--attack-side",
        choices=["enroll", "test"],
        default="enroll",
        help="Which trial side to attack. The paper protocol attacks wav1/enroll.",
    )
    parser.add_argument("--include-target-trials", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_asv_model(args.asv_backend, args.asv_checkpoint, device)
    trials = read_trials(args.trials)

    condition = f"fakebob_{args.asv_backend}_{args.target_decision}_eps{int(args.epsilon)}"
    save_root = args.output_root / condition
    save_root.mkdir(parents=True, exist_ok=True)
    summary_path = save_root / "summary.csv"

    written = 0
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "enroll",
            "test",
            "attack_side",
            "attacked",
            "reference",
            "saved_path",
            "score",
            "snr",
            "success",
            "queries",
        ])
        for label, enroll_rel, test_rel in tqdm(trials, desc=condition, dynamic_ncols=True):
            if args.max_trials is not None and written >= args.max_trials:
                break
            if label == 1 and not args.include_target_trials:
                continue

            attacked_rel = enroll_rel if args.attack_side == "enroll" else test_rel
            reference_rel = test_rel if args.attack_side == "enroll" else enroll_rel
            attacked = load_audio(trial_audio_path(args.clean_root, attacked_rel), args.max_len, device)
            reference = load_audio(trial_audio_path(args.clean_root, reference_rel), args.max_len, device)
            reference_emb = compute_embedding(model, reference, device)

            adv, score, snr, success, history = fakebob_sv(
                model,
                attacked * ATTACK_SCALE,
                reference_emb,
                threshold=args.threshold,
                target_decision=args.target_decision,
                confidence=args.confidence,
                epsilon=args.epsilon,
                max_iter=args.max_iter,
                max_lr=args.max_lr,
                min_lr=args.min_lr,
                samples_per_draw=args.samples_per_draw,
                sigma=args.sigma,
                momentum=args.momentum,
                plateau_length=args.plateau_length,
                plateau_drop=args.plateau_drop,
                eot_steps=args.eot_steps,
                query_batch_size=args.query_batch_size,
                device=device,
                model_type=args.asv_backend,
            )
            if float(adv.abs().max().item()) > 2.0:
                adv = adv / ATTACK_SCALE

            out_path = save_root / attacked_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(out_path), adv.cpu(), sample_rate=16000)
            history_path = out_path.with_suffix(".fakebob.json")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("w") as hf:
                json.dump(history, hf)
            writer.writerow([
                label,
                enroll_rel,
                test_rel,
                args.attack_side,
                attacked_rel,
                reference_rel,
                str(out_path),
                float(score.item()),
                snr,
                int(success),
                len(history),
            ])
            written += 1

    print(f"saved_root={save_root}")
    print(f"summary={summary_path}")
    print(f"num_saved={written}")


if __name__ == "__main__":
    main()
