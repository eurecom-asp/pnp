#!/usr/bin/env python
"""Online purification evaluation for generated adversarial trials.

The selected purifier is applied in memory immediately before ASV scoring. The
script reports both adversarial EER and purified EER, so demos do not need a
separate precomputed purified-waveform directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from common_eval import (
    TARGET_SR,
    compute_eer,
    load_asv_model,
    load_audio,
    read_trials,
    score_pair,
    trial_audio_path,
    write_json,
)
from defenders import build_defender


DEFENDER_CHOICES = [
    "identity",
    "noise",
    "pnp_gaussian",
    "pnp_diff",
    "pnp_diff_2",
    "pnp_diff_ssni",
    "pnp_diff_audiopure",
    "pnp_diffwavepnp",
    "audiopure",
    "audiopure_grad",
    "audiopure_ssni",
    "speechtokenizer",
    "dac",
    "academicodec",
]


def _save_audio(path: Path, audio: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() == 3 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    torchaudio.save(str(path), audio.detach().cpu(), TARGET_SR)


def add_defender_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--defender", required=True, choices=DEFENDER_CHOICES)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--pnp-checkpoint", default=None, type=Path)
    parser.add_argument("--diffwave-checkpoint", default=None, type=Path)
    parser.add_argument("--t-step", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--puri-step", type=int, default=1)
    parser.add_argument("--puri-type", default="C", choices=["A", "B", "C", "D"])
    parser.add_argument("--fast-sampling", action="store_true")
    parser.add_argument("--audiopure-config", default=None, type=Path)
    parser.add_argument("--audiopure-checkpoint", default=None, type=Path)
    parser.add_argument("--speechtokenizer-repo", default=None, type=Path)
    parser.add_argument("--speechtokenizer-config", default=None, type=Path)
    parser.add_argument("--speechtokenizer-checkpoint", default=None, type=Path)
    parser.add_argument("--dac-repo", default=None, type=Path)
    parser.add_argument("--dac-checkpoint", default=None, type=Path)
    parser.add_argument("--dac-model-type", default="16khz")
    parser.add_argument("--academicodec-repo", default=None, type=Path)
    parser.add_argument("--academicodec-config", default=None, type=Path)
    parser.add_argument("--academicodec-checkpoint", default=None, type=Path)
    parser.add_argument("--ssni-min-step", type=int, default=1)
    parser.add_argument("--ssni-max-step", type=int, default=2)
    parser.add_argument("--ssni-probe-steps", type=int, default=20)
    parser.add_argument("--ssni-tau", type=float, default=0.5)
    parser.add_argument("--ssni-bias", type=float, default=0.9)
    parser.add_argument("--ssni-score-mode", choices=["trajectory", "delta", "rms"], default="trajectory")
    parser.add_argument("--ssni-reference-stats", default=None, type=Path)
    parser.add_argument("--ssni-reference-list", default=None, type=Path)
    parser.add_argument("--ssni-reference-root", default=None, type=Path)
    parser.add_argument("--ssni-reference-size", type=int, default=1000)
    parser.add_argument("--ssni-save-reference-stats", default=None, type=Path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute adversarial EER and online-purified EER in one run."
    )
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--attack-root", required=True, type=Path)
    parser.add_argument(
        "--asv-backend",
        choices=["torch", "ecapa", "campp", "resnet", "samresnet"],
        default="ecapa",
    )
    parser.add_argument("--asv-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=16000 * 20)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--attack-side",
        choices=["enroll", "test"],
        default="enroll",
        help="Which trial side contains adversarial waveforms. The paper protocol attacks wav1/enroll.",
    )
    parser.add_argument(
        "--attack-target-trials",
        action="store_true",
        help="Load adversarial waveforms for target trials too. By default, target trials use clean audio.",
    )
    parser.add_argument("--allow-missing-attacks", action="store_true")
    parser.add_argument("--score-clean", action="store_true")
    parser.add_argument(
        "--score-purified-clean",
        action="store_true",
        help="Also purify the clean attack-side waveform and report defended clean EER.",
    )
    parser.add_argument("--save-purified-root", default=None, type=Path)
    parser.add_argument("--out-json", default=Path("outputs/online_eval.json"), type=Path)
    add_defender_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    asv_model = load_asv_model(args.asv_backend, args.asv_checkpoint, device)
    defender = build_defender(args)
    effective_t_step = 2 if args.defender == "pnp_diff_2" else args.t_step

    trials = read_trials(args.trials)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    adv_scores: list[tuple[float, int]] = []
    purified_scores: list[tuple[float, int]] = []
    clean_scores: list[tuple[float, int]] = []
    purified_clean_scores: list[tuple[float, int]] = []
    records: list[dict] = []

    for label, enroll_rel, test_rel in tqdm(trials, desc=f"online:{args.defender}", dynamic_ncols=True):
        clean_enroll = load_audio(trial_audio_path(args.clean_root, enroll_rel), args.max_len, device)
        clean_test = load_audio(trial_audio_path(args.clean_root, test_rel), args.max_len, device)

        attacked_rel = enroll_rel if args.attack_side == "enroll" else test_rel
        attack_path = trial_audio_path(args.attack_root, attacked_rel)
        use_clean_side = label == 1 and not args.attack_target_trials
        if use_clean_side:
            attack_path = trial_audio_path(args.clean_root, attacked_rel)
        elif not attack_path.exists():
            if not args.allow_missing_attacks:
                raise FileNotFoundError(
                    f"Missing attack waveform for {attacked_rel}. "
                    "Use --allow-missing-attacks only for debugging."
                )
            attack_path = trial_audio_path(args.clean_root, attacked_rel)

        attacked_side = load_audio(attack_path, args.max_len, device)
        purified_side = defender(attacked_side, attack_path)
        if args.attack_side == "enroll":
            adv_enroll, adv_test = attacked_side, clean_test
            purified_enroll, purified_test = purified_side, clean_test
        else:
            adv_enroll, adv_test = clean_enroll, attacked_side
            purified_enroll, purified_test = clean_enroll, purified_side

        clean_score = None
        if args.score_clean:
            clean_score = score_pair(asv_model, clean_enroll, clean_test, device)
            clean_scores.append((clean_score, label))

        purified_clean_score = None
        if args.score_purified_clean:
            clean_side = clean_enroll if args.attack_side == "enroll" else clean_test
            clean_side_path = trial_audio_path(args.clean_root, attacked_rel)
            purified_clean_side = defender(clean_side, clean_side_path)
            if args.attack_side == "enroll":
                purified_clean_score = score_pair(asv_model, purified_clean_side, clean_test, device)
            else:
                purified_clean_score = score_pair(asv_model, clean_enroll, purified_clean_side, device)
            purified_clean_scores.append((purified_clean_score, label))

        adv_score = score_pair(asv_model, adv_enroll, adv_test, device)
        purified_score = score_pair(asv_model, purified_enroll, purified_test, device)
        adv_scores.append((adv_score, label))
        purified_scores.append((purified_score, label))

        saved_path = None
        if args.save_purified_root is not None:
            saved_path = args.save_purified_root / attacked_rel
            _save_audio(saved_path, purified_side)

        records.append(
            {
                "label": label,
                "enroll": enroll_rel,
                "test": test_rel,
                "attack_side": args.attack_side,
                "attacked": attacked_rel,
                "attack_path": str(attack_path),
                "adv_score": adv_score,
                "purified_score": purified_score,
                "clean_score": clean_score,
                "purified_clean_score": purified_clean_score,
                "purified_path": str(saved_path) if saved_path is not None else None,
            }
        )

    adv_eer = compute_eer(adv_scores)
    purified_eer = compute_eer(purified_scores)
    payload = {
        "num_trials": len(trials),
        "asv_backend": args.asv_backend,
        "defender": args.defender,
        "defender_label": getattr(defender, "name", args.defender),
        "attack_side": args.attack_side,
        "attack_target_trials": args.attack_target_trials,
        "seed": args.seed,
        "adv_eer": adv_eer,
        "purified_eer": purified_eer,
        "eer_delta_adv_minus_purified": adv_eer - purified_eer,
        "t_step": effective_t_step,
        "lambda": args.lam,
        "puri_step": args.puri_step,
        "clean_eer": compute_eer(clean_scores) if args.score_clean else None,
        "purified_clean_eer": compute_eer(purified_clean_scores) if args.score_purified_clean else None,
        "ssni_summary": defender.ssni_summary() if hasattr(defender, "ssni_summary") else None,
        "records": records,
    }
    write_json(args.out_json, payload)

    print(f"adv_eer={adv_eer:.4f}")
    print(f"purified_eer={purified_eer:.4f}")
    if payload["clean_eer"] is not None:
        print(f"clean_eer={payload['clean_eer']:.4f}")
    if payload["purified_clean_eer"] is not None:
        print(f"purified_clean_eer={payload['purified_clean_eer']:.4f}")
    print(f"num_trials={len(trials)}")
    print(f"out_json={args.out_json}")


if __name__ == "__main__":
    main()
