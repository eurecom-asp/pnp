#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-pnp_diff}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CLEAN_ROOT="${CLEAN_ROOT:-data/VoxCeleb1/dev/wav}"
ASV_CKPT="${ASV_CKPT:-checkpoints/asv/ecapa_tdnn.pth}"
OUT_DIR="${OUT_DIR:-runs/${VARIANT}}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PNP_LAMBDA="${PNP_LAMBDA:-0.7}"
PNP_GAMMA="${PNP_GAMMA:-1e-2}"

case "${VARIANT}" in
  pnp_diff)
    ENTRYPOINT="${ROOT}/pnp_asv/pnp_diff/__main__.py"
    ADV_ROOT="${ADV_ROOT:-data/generated/attacks/pgd_l2_ecapa_50_6400_500}"
    SIMPLE_ADD=false
    MARGIN="${MARGIN:-0.8}"
    MAX_PURI_STEP="${MAX_PURI_STEP:-3}"
    ;;
  pnp_gaussian)
    ENTRYPOINT="${ROOT}/pnp_asv/pnp_gaussian/__main__.py"
    ADV_ROOT="${ADV_ROOT:-data/generated/attacks/pgd_l2_ecapa_20_6400_500}"
    SIMPLE_ADD=true
    MARGIN="${MARGIN:-1.0}"
    MAX_PURI_STEP="${MAX_PURI_STEP:-1}"
    ;;
  *)
    echo "Unknown variant: ${VARIANT}. Use pnp_diff or pnp_gaussian." >&2
    exit 2
    ;;
esac

export PYTHONPATH="${ROOT}/pnp_asv:${ROOT}/pnp_asv/pnp:${ROOT}/scripts:${PYTHONPATH:-}"
python "${ENTRYPOINT}" "${OUT_DIR}" "${CLEAN_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --override "adv_path=${ADV_ROOT}" \
  --override "asv_model=${ASV_CKPT}" \
  --override "simple_add=${SIMPLE_ADD}" \
  --override "pnp_lambda=${PNP_LAMBDA}" \
  --override "pnp_lambda_e=${PNP_GAMMA}" \
  --override "pnp_margin=${MARGIN}" \
  --override "max_puri_step=${MAX_PURI_STEP}"
