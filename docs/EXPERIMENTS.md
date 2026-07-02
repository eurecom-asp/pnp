# Reproducing PnP Experiments

This page gives command templates for the PnP-related experiments reported in
the paper. Run all commands from the repository root. Before launching long
runs, check the host and GPU state:

```bash
hostname
nvidia-smi
```

The release does not include VoxCeleb audio, generated adversarial samples, ASV
checkpoints, purifier checkpoints, or experiment logs. Edit
`configs/paths.yaml` or replace the paths below with local files.

```bash
export DEV_TRIALS=data/trials/vox1_e_dev_train.txt
export TEST_TRIALS=data/trials/vox1_test_4000.txt
export DEV_ROOT=data/VoxCeleb1/dev/wav
export TEST_ROOT=data/VoxCeleb1/test/wav
export ECAPA_CKPT=checkpoints/asv/ecapa_tdnn.pth
export CAMPP_CKPT=checkpoints/asv/campp.pt
export RESNET_CKPT=checkpoints/asv/resnet221.pth
export SAMRESNET_CKPT=checkpoints/asv/simamresnet100.pth
export PNPG_CKPT=checkpoints/purifier/pnp_gaussian.pth
export PNPD_CKPT=checkpoints/purifier/pnp_diff.pth
export DIFFWAVE_PNP_CKPT=checkpoints/purifier/diffwave_pnp.ckpt
export ECAPA_THRESHOLD=0.25
```

## 1. Training Attacks

PnP-Diff is trained with 50-step PGD-L2 adversarial examples generated on the
VoxCeleb1 development split:

```bash
python scripts/generate_whitebox_attacks.py \
  --trials "$DEV_TRIALS" \
  --clean-root "$DEV_ROOT" \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --attack pgd_l2 \
  --iterations 50 \
  --eps 6400 \
  --alpha 500 \
  --output-root data/generated/attacks
```

PnP-Gaussian uses the same protocol with 20-step PGD-L2 examples:

```bash
python scripts/generate_whitebox_attacks.py \
  --trials "$DEV_TRIALS" \
  --clean-root "$DEV_ROOT" \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --attack pgd_l2 \
  --iterations 20 \
  --eps 6400 \
  --alpha 500 \
  --output-root data/generated/attacks
```

## 2. PnP Training

Train the two PnP variants with the paper-aligned defaults:

```bash
CLEAN_ROOT="$DEV_ROOT" \
ADV_ROOT=data/generated/attacks/pgd_l2_ecapa_50_6400_500 \
ASV_CKPT="$ECAPA_CKPT" \
OUT_DIR=runs/pnp_diff \
bash scripts/train_pnp.sh pnp_diff

CLEAN_ROOT="$DEV_ROOT" \
ADV_ROOT=data/generated/attacks/pgd_l2_ecapa_20_6400_500 \
ASV_CKPT="$ECAPA_CKPT" \
OUT_DIR=runs/pnp_gaussian \
bash scripts/train_pnp.sh pnp_gaussian
```

The defaults are `lambda=0.7`; PnP-Diff samples `t in {1,2,3}` during
training; PnP-Gaussian is timestep-free. To reproduce lambda or gamma
ablations, override `PNP_LAMBDA` or `PNP_GAMMA`:

```bash
for LAMBDA in 0.5 0.7 0.9 1.0; do
  CLEAN_ROOT="$DEV_ROOT" \
  ADV_ROOT=data/generated/attacks/pgd_l2_ecapa_50_6400_500 \
  ASV_CKPT="$ECAPA_CKPT" \
  PNP_LAMBDA="$LAMBDA" \
  OUT_DIR="runs/ablation/pnp_diff_lambda_${LAMBDA}" \
  bash scripts/train_pnp.sh pnp_diff
done

for GAMMA in 0 1e-2 1e-1; do
  CLEAN_ROOT="$DEV_ROOT" \
  ADV_ROOT=data/generated/attacks/pgd_l2_ecapa_20_6400_500 \
  ASV_CKPT="$ECAPA_CKPT" \
  PNP_GAMMA="$GAMMA" \
  OUT_DIR="runs/ablation/pnp_gaussian_gamma_${GAMMA}" \
  bash scripts/train_pnp.sh pnp_gaussian
done
```

## 3. Held-Out White-Box Attacks

Generate the held-out attack grid used for PnP robustness evaluation:

```bash
for ITERS in 5 10 20 50 100 200; do
  python scripts/generate_whitebox_attacks.py \
    --trials "$TEST_TRIALS" \
    --clean-root "$TEST_ROOT" \
    --asv-backend ecapa \
    --asv-checkpoint "$ECAPA_CKPT" \
    --attack mifgsm \
    --attack-side enroll \
    --iterations "$ITERS" \
    --eps 30 \
    --alpha 1 \
    --output-root data/generated/attacks

  python scripts/generate_whitebox_attacks.py \
    --trials "$TEST_TRIALS" \
    --clean-root "$TEST_ROOT" \
    --asv-backend ecapa \
    --asv-checkpoint "$ECAPA_CKPT" \
    --attack pgd_linf \
    --attack-side enroll \
    --iterations "$ITERS" \
    --eps 30 \
    --alpha 1 \
    --output-root data/generated/attacks

  python scripts/generate_whitebox_attacks.py \
    --trials "$TEST_TRIALS" \
    --clean-root "$TEST_ROOT" \
    --asv-backend ecapa \
    --asv-checkpoint "$ECAPA_CKPT" \
    --attack pgd_l2 \
    --attack-side enroll \
    --iterations "$ITERS" \
    --eps 6400 \
    --alpha 500 \
    --output-root data/generated/attacks
done
```

## 4. Main PnP Evaluation

Use `online_purification_eval.py` for the main table. It scores the attacked
waveform and the purified waveform in the same run. Add `--score-clean` and
`--score-purified-clean` when the table also reports genuine-input EER.

```bash
python scripts/online_purification_eval.py \
  --trials "$TEST_TRIALS" \
  --clean-root "$TEST_ROOT" \
  --attack-root data/generated/attacks/mifgsm_ecapa_50_30_1 \
  --attack-side enroll \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --defender pnp_diff \
  --pnp-checkpoint "$PNPD_CKPT" \
  --t-step 1 \
  --score-clean \
  --score-purified-clean \
  --out-json outputs/scores/pnp_diff_mifgsm50.json
```

Swap `--defender` and checkpoint arguments for PnP variants:

```text
pnp_gaussian        --pnp-checkpoint "$PNPG_CKPT"
pnp_diff            --pnp-checkpoint "$PNPD_CKPT" --t-step 1
pnp_diff_2          --pnp-checkpoint "$PNPD_CKPT"
pnp_diff_ssni       --pnp-checkpoint "$PNPD_CKPT" --ssni-min-step 1 --ssni-max-step 2 --ssni-probe-steps 20 --ssni-tau 0.5 --ssni-bias 0.9
pnp_diff_audiopure  --pnp-checkpoint "$PNPD_CKPT" --audiopure-config checkpoints/purifier/audiopure_config.json --audiopure-checkpoint checkpoints/purifier/audiopure.pth
pnp_diffwavepnp     --pnp-checkpoint "$PNPD_CKPT" --diffwave-checkpoint "$DIFFWAVE_PNP_CKPT"
```

For the inference-time column reported with the purification results, use
`scripts/pnp_runtime.py`. Keep the same audio length and GPU setting when
comparing PnP-Gaussian, PnP-Diff, and PnP-Diff-2:

```bash
python scripts/pnp_runtime.py \
  --pnp-checkpoint "$PNPD_CKPT" \
  --input-root "$TEST_ROOT" \
  --t-step 1 \
  --max-files 100
```

For step analysis, run PnP-Diff with explicit timesteps:

```bash
for STEP in 1 2 3; do
  python scripts/online_purification_eval.py \
    --trials "$TEST_TRIALS" \
    --clean-root "$TEST_ROOT" \
    --attack-root data/generated/attacks/mifgsm_ecapa_50_30_1 \
    --attack-side enroll \
    --asv-backend ecapa \
    --asv-checkpoint "$ECAPA_CKPT" \
    --defender pnp_diff \
    --pnp-checkpoint "$PNPD_CKPT" \
    --t-step "$STEP" \
    --out-json "outputs/scores/pnp_diff_step${STEP}_mifgsm50.json"
done
```

## 5. PnP Across ASV Architectures

Generate attacks and evaluate PnP-Diff separately for each ASV backend. Replace
the checkpoint paths with the local models listed in `configs/paths.yaml`.

```bash
for BACKEND in ecapa campp resnet samresnet; do
  CKPT_VAR="$(printf '%s_CKPT' "$BACKEND" | tr '[:lower:]' '[:upper:]')"
  CKPT="${!CKPT_VAR}"

  python scripts/generate_whitebox_attacks.py \
    --trials "$TEST_TRIALS" \
    --clean-root "$TEST_ROOT" \
    --asv-backend "$BACKEND" \
    --asv-checkpoint "$CKPT" \
    --attack mifgsm \
    --attack-side enroll \
    --iterations 50 \
    --eps 30 \
    --alpha 1 \
    --output-root data/generated/attacks

  python scripts/online_purification_eval.py \
    --trials "$TEST_TRIALS" \
    --clean-root "$TEST_ROOT" \
    --attack-root "data/generated/attacks/mifgsm_${BACKEND}_50_30_1" \
    --attack-side enroll \
    --asv-backend "$BACKEND" \
    --asv-checkpoint "$CKPT" \
    --defender pnp_diff \
    --pnp-checkpoint "$PNPD_CKPT" \
    --score-clean \
    --score-purified-clean \
    --out-json "outputs/scores/pnp_diff_${BACKEND}_mifgsm50.json"
done
```

## 6. Optional DiffWavePnP Cascade

Train the optional diffusion denoiser that uses PnP-Diff noise in the forward
process:

```bash
CLEAN_ROOT="$DEV_ROOT" \
PNP_CKPT="$PNPD_CKPT" \
OUT_DIR=runs/diffwave_pnp \
bash scripts/train_diffwave_pnp.sh
```

Then evaluate the cascade:

```bash
python scripts/online_purification_eval.py \
  --trials "$TEST_TRIALS" \
  --clean-root "$TEST_ROOT" \
  --attack-root data/generated/attacks/mifgsm_ecapa_50_30_1 \
  --attack-side enroll \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --defender pnp_diffwavepnp \
  --pnp-checkpoint "$PNPD_CKPT" \
  --diffwave-checkpoint "$DIFFWAVE_PNP_CKPT" \
  --out-json outputs/scores/pnp_diffwavepnp_mifgsm50.json
```

## 7. Adaptive White-Box Attacks Against PnP

Use `generate_adaptive_pnp_attacks.py` when the attacker differentiates through
the PnP purifier:

```bash
python scripts/generate_adaptive_pnp_attacks.py \
  --trials "$TEST_TRIALS" \
  --clean-root "$TEST_ROOT" \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --pnp-checkpoint "$PNPD_CKPT" \
  --pnp-variant pnp_diff \
  --attack mifgsm \
  --iterations 50 \
  --eps 30 \
  --alpha 1 \
  --eot-steps 4 \
  --output-root data/generated/adaptive_attacks
```

Evaluate the saved adaptive attacks with the same defender:

```bash
python scripts/online_purification_eval.py \
  --trials "$TEST_TRIALS" \
  --clean-root "$TEST_ROOT" \
  --attack-root data/generated/adaptive_attacks/mifgsm_adaptive_ecapa_pnpd_50_30_1 \
  --attack-side enroll \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --defender pnp_diff \
  --pnp-checkpoint "$PNPD_CKPT" \
  --out-json outputs/scores/pnp_diff_adaptive_mifgsm50.json
```

Set `--pnp-variant pnp_gaussian` or `--pnp-variant pnp_diff_2` for the other
PnP variants.

## 8. FAKEBOB Black-Box Attacks

FAKEBOB requires an ASV decision threshold. If you use the released ECAPA
checkpoint from this project, use the ECAPA threshold reported with the
checkpoint release. The examples below use `ECAPA_THRESHOLD=0.25`, which matches
the default threshold used by the released ECAPA FAKEBOB protocol. For any other
ASV checkpoint, estimate the threshold on clean trials first:

```bash
python scripts/estimate_asv_threshold.py \
  --trials "$TEST_TRIALS" \
  --clean-root "$TEST_ROOT" \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --out-json outputs/scores/ecapa_threshold.json
```

Then generate FAKEBOB samples with the selected threshold:

```bash
python scripts/generate_fakebob_attacks.py \
  --trials "$TEST_TRIALS" \
  --clean-root "$TEST_ROOT" \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --threshold "$ECAPA_THRESHOLD" \
  --target-decision accept \
  --epsilon 160 \
  --max-iter 150 \
  --samples-per-draw 50 \
  --output-root data/generated/fakebob_attacks
```

Evaluate a purifier on the saved FAKEBOB samples:

```bash
python scripts/online_purification_eval.py \
  --trials "$TEST_TRIALS" \
  --clean-root "$TEST_ROOT" \
  --attack-root data/generated/fakebob_attacks/fakebob_ecapa_accept_eps160 \
  --attack-side enroll \
  --asv-backend ecapa \
  --asv-checkpoint "$ECAPA_CKPT" \
  --defender pnp_diff \
  --pnp-checkpoint "$PNPD_CKPT" \
  --out-json outputs/scores/pnp_diff_fakebob.json

python scripts/summarize_asr.py \
  --online-json outputs/scores/pnp_diff_fakebob.json \
  --threshold "$ECAPA_THRESHOLD" \
  --score-field purified_score \
  --target-decision accept \
  --label 0
```

Use `--defender identity` and `--score-field adv_score` for the No defender
FAKEBOB ASR.

## Coverage Notes

The commands above cover PnP training, held-out white-box attacks, PnP-Diff-2,
PnP-Diff-SSNI, PnP-Diff + AudioPure, PnP-Diff + DiffWavePnP, ASV-architecture
evaluation, lambda/gamma ablations, adaptive attacks against PnP, FAKEBOB ASR,
and inference time. Non-PnP baselines can be evaluated through the same
`online_purification_eval.py` interface when users provide the corresponding
external checkpoints and repositories. The release does not ship generated
audio, private checkpoints, or private baseline code.
