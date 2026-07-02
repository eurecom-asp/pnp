#!/usr/bin/env python
"""Configurable purifier wrappers for released experiments.

All checkpoints and external repositories are passed by argument. This file does
not contain private paths or generated data paths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
import torchaudio.functional as AF

from ssni import SSNIStepSelector


ROOT = Path(__file__).resolve().parents[1]
PNP_ROOT = ROOT / "pnp_asv" / "pnp"
DIFFWAVE_PNP_ROOT = ROOT / "pnp_asv" / "diffwave_pnp"
PNP_ASV_ROOT = ROOT / "pnp_asv"
DIFFWAVE_UNCOND_ROOT = ROOT / "scripts" / "DiffWave-unconditional"
TARGET_SR = 16000


def _add_path(path: Path | None) -> None:
    if path is None:
        return
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_1xt(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() == 3 and audio.shape[0] == 1 and audio.shape[1] == 1:
        audio = audio.squeeze(0)
    if audio.dim() != 2 or audio.shape[0] != 1:
        raise ValueError(f"Expected mono audio shaped [1, T], got {tuple(audio.shape)}.")
    return audio


def _match_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    audio = _ensure_1xt(audio)
    if audio.shape[-1] > length:
        return audio[:, :length]
    if audio.shape[-1] < length:
        return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))
    return audio


def _load_audio(path: Path, device: torch.device) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        audio = AF.resample(audio, sr, TARGET_SR)
    return _ensure_1xt(audio).to(device)


def _save_audio(path: Path, audio: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), _ensure_1xt(audio.detach().cpu()), TARGET_SR)


class IdentityDefender:
    name = "No defender"

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        return _ensure_1xt(audio)


class NoiseDefender:
    def __init__(self, std: float = 0.01):
        self.std = float(std)
        self.name = f"Noise-{self.std:g}"

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        audio = _ensure_1xt(audio)
        return torch.clamp(audio + torch.randn_like(audio) * self.std, -1.0, 1.0)


class PnPDefender:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        t_step: int = 1,
        lam: float = 0.7,
        simple_add: bool = False,
        name: str | None = None,
    ):
        from pnp_runtime import load_pnp_model, pnp_forward

        self.model = load_pnp_model(str(checkpoint), device=str(device))
        self.device = device
        self.t_step = int(t_step)
        self.lam = float(lam)
        self.simple_add = bool(simple_add)
        self._pnp_forward = pnp_forward
        self.name = name or ("PnP-Gaussian" if simple_add else f"PnP-Diff-{self.t_step}")

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        return self.apply_step(audio, self.t_step)

    def apply_step(self, audio: torch.Tensor, t_step: int) -> torch.Tensor:
        audio = _ensure_1xt(audio).to(self.device)
        with torch.no_grad():
            out = self._pnp_forward(
                self.model,
                audio,
                t_index=int(t_step),
                lam=self.lam,
                simple_add=self.simple_add,
                clamp=True,
            )
        return _match_length(out, audio.shape[-1])

    def score_trajectory(self, audio: torch.Tensor, probe_steps: int = 20) -> float:
        audio = _ensure_1xt(audio).to(self.device)
        embedding = getattr(getattr(self.model, "diffusion_embedding", None), "embedding", None)
        model_steps = int(embedding.shape[0]) if embedding is not None else 50
        max_steps = min(max(1, int(probe_steps)), model_steps)
        scores = []
        with torch.no_grad():
            for step in range(1, max_steps + 1):
                if self.simple_add:
                    t_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
                else:
                    t_tensor = torch.tensor([step], device=self.device, dtype=torch.float32)
                eps = self.model(audio, t_tensor)
                if eps.dim() == 3:
                    eps = eps.squeeze(1)
                score = eps.flatten(1).norm(p=2, dim=1).mean() / math.sqrt(eps.shape[-1])
                scores.append(score)
        return float(torch.stack(scores).norm(p=2).item())


class PnPDiffWavePnPDefender:
    def __init__(
        self,
        diffwave_checkpoint: Path,
        pnp_checkpoint: Path,
        device: torch.device,
        puri_step: int = 1,
        lam: float = 0.7,
        fast_sampling: bool = False,
    ):
        _add_path(PNP_ASV_ROOT)
        _add_path(PNP_ROOT)
        _add_path(DIFFWAVE_PNP_ROOT)
        from inference_pnpplus import load_model, reverse_only

        self.device = device
        self.puri_step = int(puri_step)
        self.fast_sampling = bool(fast_sampling)
        self.model = load_model(
            model_dir=str(diffwave_checkpoint),
            params={"pnp_path": str(pnp_checkpoint), "pnp_lambda": float(lam)},
            device=device,
        )
        self.model.params.pnp_path = str(pnp_checkpoint)
        self.model.params.pnp_lambda = float(lam)
        self.pnp = PnPDefender(
            pnp_checkpoint,
            device,
            t_step=self.puri_step,
            lam=lam,
            simple_add=False,
            name="PnP-Diff",
        )
        self._reverse_only = reverse_only
        self.name = "PnP-Diff + DiffWavePnP"

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        audio = _ensure_1xt(audio).to(self.device)
        pnp_out = self.pnp(audio, wav_path).to(self.device)
        with torch.no_grad():
            out, _ = self._reverse_only(
                self.model,
                spectrogram=None,
                noisy_audio=pnp_out,
                pstep=self.puri_step,
                device=self.device,
                fast_sampling=self.fast_sampling,
            )
        return _match_length(out, audio.shape[-1])


class AudioPureDefender:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: torch.device,
        puri_step: int = 1,
        puri_type: str = "C",
        differentiable: bool = False,
    ):
        _add_path(DIFFWAVE_UNCOND_ROOT)
        from WaveNet import WaveNet_Speech_Commands as WaveNet
        from util import calc_diffusion_hyperparams, puri_sampling_grad

        with Path(config_path).open("r") as f:
            config = json.load(f)
        diffusion_hyperparams = calc_diffusion_hyperparams(**config["diffusion_config"])
        for key, value in list(diffusion_hyperparams.items()):
            if isinstance(value, torch.Tensor):
                diffusion_hyperparams[key] = value.to(device)

        model = WaveNet(**config["wavenet_config"]).to(device)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state)
        model.eval()

        self.model = model
        self.diffusion_hyperparams = diffusion_hyperparams
        self.device = device
        self.puri_step = int(puri_step)
        self.puri_type = puri_type
        self.differentiable = bool(differentiable)
        self.sampling = puri_sampling_grad
        self.name = "AudioPure"

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        return self.apply_step(audio, self.puri_step)

    def apply_step(self, audio: torch.Tensor, puri_step: int) -> torch.Tensor:
        audio = _ensure_1xt(audio).to(self.device)
        if self.differentiable:
            out = self.sampling(
                self.model,
                audio,
                self.diffusion_hyperparams,
                int(puri_step),
                self.puri_type,
            )
        else:
            with torch.no_grad():
                out = self.sampling(
                    self.model,
                    audio,
                    self.diffusion_hyperparams,
                    int(puri_step),
                    self.puri_type,
                )
        if out.dim() == 3:
            out = out.squeeze(0)
        return _match_length(out, audio.shape[-1])

    def score_trajectory(self, audio: torch.Tensor, probe_steps: int = 20) -> float:
        audio = _ensure_1xt(audio).to(self.device)
        noisy = audio.unsqueeze(0)
        alpha_bar = self.diffusion_hyperparams["Alpha_bar"]
        max_steps = min(max(1, int(probe_steps)), len(alpha_bar))
        scores = []
        with torch.no_grad():
            for step in range(1, max_steps + 1):
                t = step - 1
                sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).to(device=self.device, dtype=audio.dtype)
                # Use a deterministic zero-noise probe to make SSNI scores reproducible.
                x_t = sqrt_alpha_bar * noisy
                diffusion_steps = t * torch.ones(1, 1, device=self.device, dtype=torch.float32)
                eps = self.model((x_t, diffusion_steps))
                score = eps.flatten(1).norm(p=2, dim=1).mean() / math.sqrt(eps.shape[-1])
                scores.append(score)
        return float(torch.stack(scores).norm(p=2).item())


class PnPAudioPureDefender:
    name = "PnP-Diff + AudioPure"

    def __init__(
        self,
        pnp_checkpoint: Path,
        audiopure_config: Path,
        audiopure_checkpoint: Path,
        device: torch.device,
        t_step: int = 1,
        lam: float = 0.7,
        puri_step: int = 1,
    ):
        self.pnp = PnPDefender(
            pnp_checkpoint,
            device,
            t_step=t_step,
            lam=lam,
            simple_add=False,
            name="PnP-Diff",
        )
        self.audiopure = AudioPureDefender(
            audiopure_config,
            audiopure_checkpoint,
            device,
            puri_step=puri_step,
            puri_type="D",
            differentiable=False,
        )

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        pnp_out = self.pnp(audio, wav_path)
        return self.audiopure(pnp_out, wav_path)


class SpeechTokenizerDefender:
    name = "SpeechTokenizer"

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: torch.device,
        repo_path: Path | None = None,
    ):
        _add_path(repo_path)
        try:
            from speechtokenizer import SpeechTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install SpeechTokenizer or pass --speechtokenizer-repo pointing to its cloned repository."
            ) from exc

        self.model = SpeechTokenizer.load_from_checkpoint(str(config_path), str(checkpoint_path)).to(device)
        self.model.eval()
        self.device = device

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        audio = _ensure_1xt(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            codes = self.model.encode(audio)
            out = self.model.decode(codes)[0]
        return _match_length(out, audio.shape[-1])


class DACDefender:
    name = "DAC"

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: Path | None = None,
        model_type: str = "16khz",
        repo_path: Path | None = None,
    ):
        _add_path(repo_path)
        try:
            import dac
        except ImportError as exc:
            raise ImportError(
                "Install descript-audio-codec or pass --dac-repo pointing to its cloned repository."
            ) from exc

        model_path = str(checkpoint_path) if checkpoint_path is not None else dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path).to(device)
        self.model.eval()
        self.device = device

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        audio = _ensure_1xt(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self.model.preprocess(audio, TARGET_SR)
            z, *_ = self.model.encode(x)
            out = self.model.decode(z).squeeze(0)
        return _match_length(out, audio.shape[-1])


class AcademiCodecDefender:
    name = "AcademiCodec"

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: torch.device,
        repo_path: Path | None = None,
    ):
        _add_path(repo_path)
        try:
            from academicodec.models.hificodec.vqvae import VQVAE
        except ImportError as exc:
            raise ImportError(
                "Install AcademiCodec or pass --academicodec-repo pointing to its cloned repository."
            ) from exc

        self.model = VQVAE(str(config_path), str(checkpoint_path), with_encoder=True).to(device)
        self.model.eval()
        self.device = device

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        audio = _ensure_1xt(audio).to(self.device)
        with torch.no_grad():
            wav = audio / (audio.abs().max() + 1e-8) * 0.95
            tokens = self.model.encode(wav)
            out = self.model(tokens).squeeze(0)
        return _match_length(out, audio.shape[-1])


class SSNIDefender:
    def __init__(self, base_defender, selector: SSNIStepSelector, name: str):
        self.base = base_defender
        self.selector = selector
        self.name = name

    def __call__(self, audio: torch.Tensor, wav_path: Path | None = None) -> torch.Tensor:
        step, _ = self.selector.select(audio)
        if hasattr(self.base, "apply_step"):
            return self.base.apply_step(audio, step)
        previous_step = getattr(self.base, "puri_step", None)
        self.base.puri_step = step
        try:
            return self.base(audio, wav_path)
        finally:
            if previous_step is not None:
                self.base.puri_step = previous_step

    def ssni_summary(self) -> dict:
        return self.selector.summary()


def _get_arg(args, name: str, default=None):
    return getattr(args, name, default)


def _build_ssni_selector(args, base_defender, device: torch.device) -> SSNIStepSelector:
    score_mode = _get_arg(args, "ssni_score_mode", "trajectory")
    probe_steps = int(_get_arg(args, "ssni_probe_steps", 20))

    if score_mode == "rms":
        def scorer(audio: torch.Tensor) -> float:
            audio = _ensure_1xt(audio).to(device)
            return float(audio.pow(2).mean().sqrt().item())
    elif score_mode == "delta":
        def scorer(audio: torch.Tensor) -> float:
            audio = _ensure_1xt(audio).to(device)
            scores = []
            with torch.no_grad():
                for step in range(
                    int(_get_arg(args, "ssni_min_step", 1)),
                    int(_get_arg(args, "ssni_max_step", 2)) + 1,
                ):
                    if not hasattr(base_defender, "apply_step"):
                        raise ValueError("SSNI delta scoring requires a defender with apply_step().")
                    out = base_defender.apply_step(audio, step)
                    scores.append((out - audio).flatten(1).norm(p=2, dim=1).mean() / math.sqrt(audio.shape[-1]))
            return float(torch.stack(scores).norm(p=2).item())
    else:
        if not hasattr(base_defender, "score_trajectory"):
            raise ValueError("SSNI trajectory scoring requires a defender with score_trajectory().")

        def scorer(audio: torch.Tensor) -> float:
            return float(base_defender.score_trajectory(audio, probe_steps=probe_steps))

    return SSNIStepSelector(
        scorer=scorer,
        device=device,
        min_step=int(_get_arg(args, "ssni_min_step", 1)),
        max_step=int(_get_arg(args, "ssni_max_step", 2)),
        tau=float(_get_arg(args, "ssni_tau", 0.5)),
        bias=float(_get_arg(args, "ssni_bias", 0.9)),
        reference_stats=_get_arg(args, "ssni_reference_stats", None),
        reference_list=_get_arg(args, "ssni_reference_list", None),
        reference_root=_get_arg(args, "ssni_reference_root", None),
        reference_size=_get_arg(args, "ssni_reference_size", 1000),
        save_reference_stats=_get_arg(args, "ssni_save_reference_stats", None),
    )


def build_defender(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    name = args.defender
    if name == "identity":
        return IdentityDefender()
    if name == "noise":
        return NoiseDefender(std=args.noise_std)
    if name == "pnp_gaussian":
        if args.pnp_checkpoint is None:
            raise ValueError("--pnp-checkpoint is required.")
        return PnPDefender(args.pnp_checkpoint, device, args.t_step, args.lam, simple_add=True, name="PnP-Gaussian")
    if name in {"pnp_diff", "pnp_diff_2"}:
        if args.pnp_checkpoint is None:
            raise ValueError("--pnp-checkpoint is required.")
        t_step = 2 if name == "pnp_diff_2" else args.t_step
        label = "PnP-Diff-2" if t_step == 2 else "PnP-Diff"
        return PnPDefender(args.pnp_checkpoint, device, t_step, args.lam, simple_add=False, name=label)
    if name == "pnp_diff_ssni":
        if args.pnp_checkpoint is None:
            raise ValueError("--pnp-checkpoint is required.")
        base = PnPDefender(args.pnp_checkpoint, device, args.t_step, args.lam, simple_add=False, name="PnP-Diff")
        return SSNIDefender(base, _build_ssni_selector(args, base, device), name="PnP-Diff-SSNI")
    if name == "pnp_diffwavepnp":
        if args.diffwave_checkpoint is None or args.pnp_checkpoint is None:
            raise ValueError("--diffwave-checkpoint and --pnp-checkpoint are required.")
        return PnPDiffWavePnPDefender(
            args.diffwave_checkpoint,
            args.pnp_checkpoint,
            device,
            puri_step=args.puri_step,
            lam=args.lam,
            fast_sampling=args.fast_sampling,
        )
    if name == "pnp_diff_audiopure":
        if args.pnp_checkpoint is None or args.audiopure_config is None or args.audiopure_checkpoint is None:
            raise ValueError("--pnp-checkpoint, --audiopure-config, and --audiopure-checkpoint are required.")
        return PnPAudioPureDefender(
            args.pnp_checkpoint,
            args.audiopure_config,
            args.audiopure_checkpoint,
            device,
            t_step=args.t_step,
            lam=args.lam,
            puri_step=args.puri_step,
        )
    if name in {"audiopure", "audiopure_grad"}:
        if args.audiopure_config is None or args.audiopure_checkpoint is None:
            raise ValueError("--audiopure-config and --audiopure-checkpoint are required.")
        return AudioPureDefender(
            args.audiopure_config,
            args.audiopure_checkpoint,
            device,
            puri_step=args.puri_step,
            puri_type=args.puri_type,
            differentiable=name == "audiopure_grad",
        )
    if name == "audiopure_ssni":
        if args.audiopure_config is None or args.audiopure_checkpoint is None:
            raise ValueError("--audiopure-config and --audiopure-checkpoint are required.")
        base = AudioPureDefender(
            args.audiopure_config,
            args.audiopure_checkpoint,
            device,
            puri_step=args.puri_step,
            puri_type=args.puri_type,
            differentiable=False,
        )
        return SSNIDefender(base, _build_ssni_selector(args, base, device), name="AudioPure+SSNI")
    if name == "speechtokenizer":
        if args.speechtokenizer_config is None or args.speechtokenizer_checkpoint is None:
            raise ValueError("--speechtokenizer-config and --speechtokenizer-checkpoint are required.")
        return SpeechTokenizerDefender(
            args.speechtokenizer_config,
            args.speechtokenizer_checkpoint,
            device,
            repo_path=_get_arg(args, "speechtokenizer_repo", None),
        )
    if name == "dac":
        return DACDefender(
            device,
            checkpoint_path=args.dac_checkpoint,
            model_type=args.dac_model_type,
            repo_path=_get_arg(args, "dac_repo", None),
        )
    if name == "academicodec":
        if args.academicodec_config is None or args.academicodec_checkpoint is None:
            raise ValueError("--academicodec-config and --academicodec-checkpoint are required.")
        return AcademiCodecDefender(
            args.academicodec_config,
            args.academicodec_checkpoint,
            device,
            repo_path=_get_arg(args, "academicodec_repo", None),
        )
    raise ValueError(f"Unsupported defender: {name}")


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".wav", ".flac"}:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defender", required=True, choices=[
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
    ])
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    defender = build_defender(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    files = list(iter_audio_files(args.input_dir))
    if args.max_files is not None:
        files = files[: args.max_files]
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(args.input_dir)
        audio = _load_audio(path, device)
        purified = defender(audio, path)
        _save_audio(args.output_dir / rel.with_suffix(".wav"), purified)
        if index % 50 == 0 or index == len(files):
            print(f"{defender.name}: {index}/{len(files)}", flush=True)
    if hasattr(defender, "ssni_summary"):
        print("ssni_summary=" + json.dumps(defender.ssni_summary(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
