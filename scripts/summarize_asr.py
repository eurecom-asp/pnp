#!/usr/bin/env python
"""Summarize attack success rate from online purification JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-json", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--score-field", choices=["adv_score", "purified_score"], default="purified_score")
    parser.add_argument("--target-decision", choices=["accept", "reject"], default="accept")
    parser.add_argument("--label", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    with args.online_json.open("r") as f:
        payload = json.load(f)

    records = [record for record in payload["records"] if int(record["label"]) == args.label]
    if not records:
        raise ValueError(f"No records found with label={args.label}.")

    if args.target_decision == "accept":
        successes = [float(record[args.score_field]) >= args.threshold for record in records]
    else:
        successes = [float(record[args.score_field]) < args.threshold for record in records]

    asr = 100.0 * sum(successes) / len(successes)
    print(f"attack_success_rate={asr:.4f}")
    print(f"num_trials={len(records)}")
    print(f"score_field={args.score_field}")


if __name__ == "__main__":
    main()
