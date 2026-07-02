#!/usr/bin/env python
"""Shared PnP checkpoint loading and forward utilities."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common_eval import TARGET_SR, load_audio


_PNP_RUNTIME = None
_PNP_MODEL_CACHE = {}


def _load_pnp_runtime():
    global _PNP_RUNTIME
    if _PNP_RUNTIME is not None:
        return _PNP_RUNTIME

    pnp_root = Path(__file__).resolve().parents[1] / "pnp_asv" / "pnp"
    pnp_root_str = str(pnp_root)
    if pnp_root_str not in sys.path:
        sys.path.insert(0, pnp_root_str)

    from params_pnp import AttrDict, params as base_params_pnp
    from model_pnp import DiffWave as PnpDiffWave

    _PNP_RUNTIME = {
        "AttrDict": AttrDict,
        "params": base_params_pnp,
        "Model": PnpDiffWave,
    }
    return _PNP_RUNTIME


def load_pnp_model(ckpt_path: str | Path, device: str | torch.device = "cuda"):
    cache_key = (str(ckpt_path), str(device))
    if cache_key in _PNP_MODEL_CACHE:
        return _PNP_MODEL_CACHE[cache_key]

    runtime = _load_pnp_runtime()
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model = runtime["Model"](runtime["AttrDict"](runtime["params"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _PNP_MODEL_CACHE[cache_key] = model
    return model


def pnp_forward(
    pnp_model,
    audio_1xT: torch.Tensor,
    t_index: int,
    params=None,
    lam: float = 0.7,
    simple_add: bool = False,
    noise: torch.Tensor | None = None,
    clamp: bool = True,
) -> torch.Tensor:
    """Apply the PnP forward process used by the paper experiments."""
    if audio_1xT.ndim != 2 or audio_1xT.size(0) != 1:
        raise ValueError(f"audio should be [1, T], got {tuple(audio_1xT.shape)}.")

    runtime = _load_pnp_runtime()
    if params is None:
        params = runtime["params"]

    device = audio_1xT.device
    dtype = audio_1xT.dtype
    training_noise_schedule = np.array(params.noise_schedule)
    alpha = 1.0 - training_noise_schedule
    alpha_bar = np.cumprod(alpha)
    t_idx = int(np.clip(t_index, 1, len(alpha_bar))) - 1

    sqrt_alpha_bar_t = torch.tensor(alpha_bar[t_idx] ** 0.5, device=device, dtype=dtype)
    sqrt_one_minus_alpha_bar_t = torch.tensor((1.0 - alpha_bar[t_idx]) ** 0.5, device=device, dtype=dtype)
    if simple_add:
        t_tensor = torch.zeros(1, device=device, dtype=torch.float32)
    else:
        t_tensor = torch.tensor([t_idx + 1], device=device, dtype=torch.float32)

    # Keep the original paper-code behavior: DiffWave returns [B, 1, T],
    # and normalization is applied before any channel squeeze.
    eps_dir = pnp_model(audio_1xT, t_tensor)
    eps_dir = eps_dir / (eps_dir.norm(p=2, dim=1, keepdim=True) + 1e-8)

    if noise is None:
        noise = torch.randn_like(audio_1xT)
    eps_tilde = lam * eps_dir + (1.0 - lam**2) ** 0.5 * noise

    if simple_add:
        out = audio_1xT + eps_tilde
    else:
        out = sqrt_alpha_bar_t * audio_1xT + sqrt_one_minus_alpha_bar_t * eps_tilde

    if clamp:
        out = torch.clamp(out, -1.0, 1.0)
    return out


def _iter_audio_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".wav", ".flac"}:
            yield path


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure PnP inference real-time factor.")
    parser.add_argument("--pnp-checkpoint", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--t-step", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--simple-add", action="store_true", help="Use PnP-Gaussian additive mode.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=16000 * 20)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_pnp_model(args.pnp_checkpoint, device=device)
    files = list(_iter_audio_files(args.input_root))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise ValueError(f"No wav/flac files found under {args.input_root}.")

    audios = [load_audio(path, args.max_len, device) for path in files]
    for audio in audios[: args.warmup]:
        with torch.no_grad():
            _ = pnp_forward(model, audio, args.t_step, lam=args.lam, simple_add=args.simple_add)
    _sync(device)

    total_audio_seconds = sum(audio.shape[-1] / TARGET_SR for audio in audios) * args.repeat
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.repeat):
            for audio in audios:
                _ = pnp_forward(model, audio, args.t_step, lam=args.lam, simple_add=args.simple_add)
    _sync(device)
    elapsed = time.perf_counter() - start
    rtf = elapsed / total_audio_seconds

    print(f"num_files={len(files)}")
    print(f"audio_seconds={total_audio_seconds:.4f}")
    print(f"elapsed_seconds={elapsed:.4f}")
    print(f"rtf={rtf:.6f}")


if __name__ == "__main__":
    main()
