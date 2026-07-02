#!/usr/bin/env python
"""Generate white-box waveform attacks from a VoxCeleb-style trial file.

The repository does not ship any generated adversarial samples. Use this script
to create them locally after downloading the data and preparing an ASV model.
"""

from __future__ import annotations

import argparse
import csv
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
from fakebob_attack import mifgsm, pgd_2, pgd_i


def attack_tag(name: str) -> str:
    return {"pgd_linf": "pgd_linf", "pgd_l2": "pgd_l2", "mifgsm": "mifgsm"}[name]


def run_attack(args, model, wav_norm, enroll_emb, label, device):
    wav_attack_space = wav_norm * ATTACK_SCALE
    loss_label = torch.tensor([-1 if label == 0 else 1], device=device)
    if args.attack == "pgd_l2":
        return pgd_2(
            model,
            wav_attack_space.clone(),
            enroll_emb,
            loss_label,
            eps=args.eps,
            alpha=args.alpha,
            iters=args.iterations,
            device=device,
            model_type=args.asv_backend,
        )
    if args.attack == "pgd_linf":
        return pgd_i(
            model,
            wav_attack_space.clone(),
            enroll_emb,
            loss_label,
            eps=args.eps,
            alpha=args.alpha,
            iters=args.iterations,
            device=device,
            model_type=args.asv_backend,
        )
    return mifgsm(
        model,
        wav_attack_space.clone(),
        enroll_emb,
        loss_label,
        eps=args.eps,
        alpha=args.alpha,
        decay=args.decay,
        iters=args.iterations,
        device=device,
        model_type=args.asv_backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("data/generated/attacks"), type=Path)
    parser.add_argument("--asv-backend", choices=["torch", "ecapa", "campp", "resnet", "samresnet"], default="ecapa")
    parser.add_argument("--asv-checkpoint", required=True, type=Path)
    parser.add_argument("--attack", choices=["mifgsm", "pgd_l2", "pgd_linf"], default="mifgsm")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--eps", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=1.0)
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

    condition = f"{attack_tag(args.attack)}_{args.asv_backend}_{args.iterations}_{int(args.eps)}_{int(args.alpha)}"
    save_root = args.output_root / condition
    save_root.mkdir(parents=True, exist_ok=True)
    summary_path = save_root / "summary.csv"

    written = 0
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "enroll", "test", "attack_side", "attacked", "reference", "saved_path", "score", "snr"])
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
            adv, score, snr = run_attack(args, model, attacked, reference_emb, label, device)
            if float(adv.abs().max().item()) > 2.0:
                adv = adv / ATTACK_SCALE

            out_path = save_root / attacked_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(out_path), adv.cpu(), sample_rate=16000)
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
            ])
            written += 1

    print(f"saved_root={save_root}")
    print(f"summary={summary_path}")
    print(f"num_saved={written}")


if __name__ == "__main__":
    main()
