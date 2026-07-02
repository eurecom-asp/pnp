#!/usr/bin/env python
"""Compare multiple online purifiers without saving purified waveforms."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from common_eval import (
    compute_eer,
    compute_embedding,
    load_asv_model,
    load_audio,
    read_trials,
    similarity,
    trial_audio_path,
    write_json,
)
from defenders import build_defender


def score_embeddings(enroll_emb: torch.Tensor, test_emb: torch.Tensor) -> float:
    return float(similarity(enroll_emb, test_emb).view(()).item())


def base_defender_args(args: argparse.Namespace, defender: str) -> SimpleNamespace:
    return SimpleNamespace(
        defender=defender,
        device=args.device,
        noise_std=0.01,
        pnp_checkpoint=None,
        diffwave_checkpoint=None,
        t_step=1,
        lam=args.lam,
        puri_step=1,
        puri_type="C",
        fast_sampling=False,
        audiopure_config=None,
        audiopure_checkpoint=None,
        speechtokenizer_config=None,
        speechtokenizer_checkpoint=None,
        dac_checkpoint=None,
        dac_model_type="16khz",
        academicodec_config=None,
        academicodec_checkpoint=None,
    )


def make_defenders(args: argparse.Namespace) -> list[tuple[str, object]]:
    specs: list[tuple[str, SimpleNamespace]] = []
    specs.append(("No defender", base_defender_args(args, "identity")))
    for std in args.noise_stds:
        defender_args = base_defender_args(args, "noise")
        defender_args.noise_std = std
        specs.append((f"Noise-{std:g}", defender_args))

    if args.pnp_gaussian_checkpoint is not None:
        defender_args = base_defender_args(args, "pnp_gaussian")
        defender_args.pnp_checkpoint = args.pnp_gaussian_checkpoint
        specs.append(("PnP-Gaussian", defender_args))
    if args.pnp_diff_checkpoint is not None:
        defender_args = base_defender_args(args, "pnp_diff")
        defender_args.pnp_checkpoint = args.pnp_diff_checkpoint
        defender_args.t_step = 1
        specs.append(("PnP-Diff", defender_args))

        defender_args = base_defender_args(args, "pnp_diff_2")
        defender_args.pnp_checkpoint = args.pnp_diff_checkpoint
        defender_args.t_step = 2
        specs.append(("PnP-Diff-2", defender_args))

    return [(label, build_defender(defender_args)) for label, defender_args in specs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--attack-root", required=True, type=Path)
    parser.add_argument("--attack-side", choices=["enroll", "test"], default="enroll")
    parser.add_argument("--attack-target-trials", action="store_true")
    parser.add_argument("--asv-backend", choices=["torch", "ecapa", "campp", "resnet", "samresnet"], default="ecapa")
    parser.add_argument("--asv-checkpoint", required=True, type=Path)
    parser.add_argument("--pnp-gaussian-checkpoint", default=None, type=Path)
    parser.add_argument("--pnp-diff-checkpoint", default=None, type=Path)
    parser.add_argument("--noise-stds", nargs="*", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=16000 * 20)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    asv_model = load_asv_model(args.asv_backend, args.asv_checkpoint, device)
    defenders = make_defenders(args)
    trials = read_trials(args.trials)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    waveform_cache: dict[tuple[str, str], torch.Tensor] = {}
    embedding_cache: dict[tuple[str, str], torch.Tensor] = {}
    purified_embedding_cache: dict[tuple[str, str, str], torch.Tensor] = {}

    def load_source(kind: str, rel_path: str) -> torch.Tensor:
        key = (kind, rel_path)
        if key not in waveform_cache:
            root = args.clean_root if kind == "clean" else args.attack_root
            waveform_cache[key] = load_audio(trial_audio_path(root, rel_path), args.max_len, device)
        return waveform_cache[key]

    def embed_source(kind: str, rel_path: str) -> torch.Tensor:
        key = (kind, rel_path)
        if key not in embedding_cache:
            embedding_cache[key] = compute_embedding(asv_model, load_source(kind, rel_path), device)
        return embedding_cache[key]

    def purified_embedding(label: str, defender, kind: str, rel_path: str) -> torch.Tensor:
        if label == "No defender":
            return embed_source(kind, rel_path)
        key = (label, kind, rel_path)
        if key not in purified_embedding_cache:
            with torch.no_grad():
                purified = defender(load_source(kind, rel_path), trial_audio_path(args.clean_root if kind == "clean" else args.attack_root, rel_path))
            purified_embedding_cache[key] = compute_embedding(asv_model, purified, device)
        return purified_embedding_cache[key]

    clean_scores: list[tuple[float, int]] = []
    adv_scores: list[tuple[float, int]] = []
    method_scores = {
        label: {"adv": [], "clean": []}
        for label, _ in defenders
    }

    for label, enroll_rel, test_rel in tqdm(trials, desc="compare-online", dynamic_ncols=True):
        attack_rel = enroll_rel if args.attack_side == "enroll" else test_rel
        other_rel = test_rel if args.attack_side == "enroll" else enroll_rel
        clean_attack_emb = embed_source("clean", attack_rel)
        clean_other_emb = embed_source("clean", other_rel)
        clean_score = score_embeddings(clean_attack_emb, clean_other_emb)
        clean_scores.append((clean_score, label))

        attack_kind = "clean" if label == 1 and not args.attack_target_trials else "attack"
        adv_attack_emb = embed_source(attack_kind, attack_rel)
        adv_score = score_embeddings(adv_attack_emb, clean_other_emb)
        adv_scores.append((adv_score, label))

        for method_label, defender in defenders:
            clean_purified_emb = purified_embedding(method_label, defender, "clean", attack_rel)
            clean_purified_score = score_embeddings(clean_purified_emb, clean_other_emb)
            method_scores[method_label]["clean"].append((clean_purified_score, label))

            adv_purified_emb = purified_embedding(method_label, defender, attack_kind, attack_rel)
            adv_purified_score = score_embeddings(adv_purified_emb, clean_other_emb)
            method_scores[method_label]["adv"].append((adv_purified_score, label))

    clean_eer = compute_eer(clean_scores)
    adv_eer = compute_eer(adv_scores)
    methods = {}
    for method_label, scores in method_scores.items():
        methods[method_label] = {
            "purified_clean_eer": compute_eer(scores["clean"]),
            "purified_adv_eer": compute_eer(scores["adv"]),
        }

    payload = {
        "num_trials": len(trials),
        "attack_side": args.attack_side,
        "attack_target_trials": args.attack_target_trials,
        "seed": args.seed,
        "asv_backend": args.asv_backend,
        "clean_eer": clean_eer,
        "adv_eer": adv_eer,
        "methods": methods,
        "cache_sizes": {
            "waveforms": len(waveform_cache),
            "embeddings": len(embedding_cache),
            "purified_embeddings": len(purified_embedding_cache),
        },
    }
    write_json(args.out_json, payload)

    print(f"clean_eer={clean_eer:.4f}")
    print(f"adv_eer={adv_eer:.4f}")
    for method_label, values in methods.items():
        print(
            f"{method_label}: "
            f"clean={values['purified_clean_eer']:.4f}, "
            f"adv={values['purified_adv_eer']:.4f}"
        )
    print(f"out_json={args.out_json}")


if __name__ == "__main__":
    main()
