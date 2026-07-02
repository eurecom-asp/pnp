# Positive-Incentive Noise Predictor for ASV Purification

[![arXiv](https://img.shields.io/badge/arXiv-2607.00899-b31b1b.svg)](https://arxiv.org/abs/2607.00899)
[![Audio Demo](https://img.shields.io/badge/demo-audio_samples-1c8276.svg)](https://eurecom-asp.github.io/pnp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code release for **Positive-Incentive Noise Predictor (PnP)**, an adversarial
purification framework for automatic speaker verification (ASV). PnP learns
task-beneficial waveform noise as a lightweight front-end purifier and can be
used alone or with an optional DiffWave-style denoising cascade.

This repository provides the reusable code for PnP training, waveform attack
generation, online purification evaluation, and the static audio demo. It does
not include full speech datasets, generated attack sets, experiment logs, or
large checkpoints.

## Installation

```bash
conda create -n pnp-asv python=3.10
conda activate pnp-asv
pip install -r requirements.txt
```

Copy the path template and edit it for your machine:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

## Data Preparation

This release includes the trial lists used by the paper:

```text
data/trials/vox1_e_dev_train.txt   # VoxCeleb1-E trials for dev-set training
data/trials/vox1_test_4000.txt     # 4,000-trial VoxCeleb1-O subset for testing
```

Prepare VoxCeleb1 locally so the relative paths in the trial files resolve
under your clean waveform roots, for example:

```text
data/VoxCeleb1/dev/wav
data/VoxCeleb1/test/wav
```

For VoxCeleb2 data used for ASV training, refer to
[WeSpeaker](https://github.com/wenet-e2e/wespeaker), or use the released
pretrained ASV checkpoints.

Place checkpoints under the following convention, or pass explicit paths through
the command line:

```text
checkpoints/asv/ecapa_tdnn.pth
checkpoints/purifier/pnp_gaussian.pth
checkpoints/purifier/pnp_diff.pth
checkpoints/purifier/diffwave_pnp.ckpt
```

More details are in [docs/DATA.md](docs/DATA.md).

## Quick Start: Online Purification

Generate held-out 50-step MI-FGSM examples against ECAPA-TDNN:

```bash
python scripts/generate_whitebox_attacks.py \
  --trials data/trials/vox1_test_4000.txt \
  --clean-root data/VoxCeleb1/test/wav \
  --asv-backend ecapa \
  --asv-checkpoint checkpoints/asv/ecapa_tdnn.pth \
  --attack mifgsm \
  --attack-side enroll \
  --iterations 50 \
  --eps 30 \
  --alpha 1 \
  --output-root data/generated/attacks
```

Evaluate PnP-Diff online without saving purified waveforms:

```bash
python scripts/online_purification_eval.py \
  --trials data/trials/vox1_test_4000.txt \
  --clean-root data/VoxCeleb1/test/wav \
  --attack-root data/generated/attacks/mifgsm_ecapa_50_30_1 \
  --attack-side enroll \
  --asv-backend ecapa \
  --asv-checkpoint checkpoints/asv/ecapa_tdnn.pth \
  --defender pnp_diff \
  --pnp-checkpoint checkpoints/purifier/pnp_diff.pth \
  --t-step 1 \
  --out-json outputs/scores/pnp_diff_mifgsm50.json
```

Use `--score-clean` and `--score-purified-clean` if you also need clean-input
EER. Other PnP variants are exposed through `--defender pnp_gaussian`,
`pnp_diff_2`, `pnp_diff_ssni`, `pnp_diff_audiopure`, and
`pnp_diffwavepnp`.

Full experiment recipes, including training, ablations, FAKEBOB, adaptive
attacks, and inference-time measurement, are in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Audio Demo

The listening demo is hosted with GitHub Pages and maintained as a separate
static site project. It contains two anonymized audio samples with audio players
and spectrogram thumbnails.

## Repository Notes

- Checkpoints and generated data are intentionally excluded by `.gitignore`.
- Demo audio assets are maintained in the separate static demo project.
- Optional baseline purifiers require users to provide the corresponding
  external checkpoints and repositories. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Citation

If you use this repository, please cite the corresponding paper. The final
BibTeX entry will be added after publication.

## License and Acknowledgements

This repository is released under the MIT license. It includes
DiffWave-derived components, and we thank the authors of
[DiffWave](https://github.com/lmnt-com/diffwave) for their open-source release.
Check each third-party dependency before redistributing pretrained checkpoints
or external assets.
