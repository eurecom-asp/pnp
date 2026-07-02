#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CLEAN_ROOT="${CLEAN_ROOT:-data/VoxCeleb1/dev/wav}"
PNP_CKPT="${PNP_CKPT:-checkpoints/purifier/pnp_diff.pth}"
OUT_DIR="${OUT_DIR:-runs/diffwave_pnp}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PNP_LAMBDA="${PNP_LAMBDA:-0.7}"

export PYTHONPATH="${ROOT}/pnp_asv/diffwave_pnp:${ROOT}/pnp_asv:${ROOT}/pnp_asv/pnp:${PYTHONPATH:-}"
python "${ROOT}/pnp_asv/diffwave_pnp/__main__.py" "${OUT_DIR}" "${CLEAN_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --override "pnp_path=${PNP_CKPT}" \
  --override "pnp_lambda=${PNP_LAMBDA}"
