#!/usr/bin/env python
"""Train PnP-Diff with paper-aligned defaults."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pnp"


def main() -> None:
    sys.path.insert(0, str(CORE))
    defaults = [
        "--override",
        "simple_add=false",
        "--override",
        "pnp_margin=0.8",
        "--override",
        "max_puri_step=3",
        "--override",
        "adv_path=data/generated/attacks/pgd_l2_ecapa_50_6400_500",
    ]
    sys.argv[1:1] = defaults
    runpy.run_path(str(CORE / "__main__.py"), run_name="__main__")


if __name__ == "__main__":
    main()
