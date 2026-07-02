# Data Preparation

This repository does not include full speech datasets, model checkpoints, or
pre-generated adversarial/purified waveform directories. It only includes the
trial metadata needed to reproduce the paper protocol:

```text
data/trials/vox1_e_dev_train.txt        # VoxCeleb1-E, for dev-set PnP training
data/trials/vox1_test_4000.txt          # 4,000-trial VoxCeleb1-O subset, for held-out testing
```

Each trial line uses:

```text
label enroll_relative_path test_relative_path
```

Prepare VoxCeleb1 locally so these relative paths resolve under your selected
clean waveform root, for example:

```text
data/VoxCeleb1/dev/wav
data/VoxCeleb1/test/wav
```

For VoxCeleb2 data used for ASV training, refer to
[WeSpeaker](https://github.com/wenet-e2e/wespeaker), or use the released
pretrained ASV checkpoints.

Attack roots should mirror the same relative paths for the attacked side:

```text
clean-root/<speaker>/<segment>/<utterance>.wav
attack-root/<speaker>/<segment>/<utterance>.wav
```

By default, `online_purification_eval.py` follows the paper protocol:
`wav1`/enrollment audio is attacked or purified, while `wav2`/test audio is
loaded from the VoxCeleb1 clean root and used only for ASV scoring. Use
`--attack-side test` only if you intentionally generated adversarial waveforms
for the test side. See the README for example commands for attack generation
and evaluation.

For SSNI-style dynamic step selection, pass a clean reference list and root with
`--ssni-reference-list` and `--ssni-reference-root`, or pass a precomputed
`--ssni-reference-stats` JSON. The script accepts the same trial-list format
above and de-duplicates the referenced clean utterances.
