# Checkpoint Layout

Place downloaded checkpoints under this directory.

Released checkpoints are available on
[Hugging Face](https://huggingface.co/RegulusB/pnp-asv-purification-checkpoints/tree/main/checkpoints).
Download the `checkpoints/` folder and place it at the root of this GitHub
repository.

```text
checkpoints/
  asv/
    ecapa_tdnn.pth
    campp.pt
    resnet221.pth
    simamresnet100.pth
  purifier/
    audiopure_config.json
    audiopure.pth
    speechtokenizer_config.json
    speechtokenizer.ckpt
    dac.pth
    academicodec_config.json
    academicodec.pth
    ssni_reference_stats.json
    pnp_gaussian.pth
    pnp_diff.pth
    diffwave_pnp.ckpt
```

`Noise-0.01` does not require a checkpoint. `PnP-Diff-2` reuses
`purifier/pnp_diff.pth` with purification step `t=2`. `PnP-Diff + AudioPure`
combines `purifier/pnp_diff.pth` with `purifier/audiopure_config.json` and
`purifier/audiopure.pth`. `PnP-Diff + DiffWavePnP` combines
`purifier/pnp_diff.pth` with `purifier/diffwave_pnp.ckpt`. SpeechTokenizer uses
`purifier/speechtokenizer_config.json` and `purifier/speechtokenizer.ckpt`.
DAC can either use `purifier/dac.pth` or download the official model through
the `descript-audio-codec` package. AcademiCodec uses
`purifier/academicodec_config.json` and `purifier/academicodec.pth`.
AudioPure-SSNI and PnP-Diff-SSNI can optionally use
`purifier/ssni_reference_stats.json`; otherwise the scripts can rebuild SSNI
statistics from a user-provided clean reference list.
