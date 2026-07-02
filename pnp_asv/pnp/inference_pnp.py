# inference.py — pnp forward-only: predict noise at timestep t and add to input
# Copyright ...
import os
import numpy as np
import torch
import torchaudio
import librosa
import random
from argparse import ArgumentParser
from glob import glob
from os import path
from tqdm import tqdm

from params_pnp import AttrDict, params as base_params

# ====== Your pnp model (provide this in your codebase) ======
# Expect a module defining class pnp; adapt import if different.
from model_pnp import DiffWave as pnp  # Example: if your pnp class is in model

# ---------------- utils ----------------
def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------------- pnp load/apply ----------------
def load_pnp(pnp_dir: str, device):
    """Load pnp weights from pnp_dir/weights.pt (adjust if your filename differs)."""
    ckpt_path = path.join(pnp_dir, "weights.pt")
    if not path.exists(ckpt_path):
        raise FileNotFoundError(f"pnp checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # If your checkpoint stores config, pass it to pnp(...)
    model = pnp(**base_params).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

@torch.no_grad()
def pnp_predict_noise(pnp, audio_1xT: torch.Tensor, t_index: int, params, lam: float = 0.7, simple_add: bool = False):
    """
    Predict task-beneficial directional noise with pnp at timestep t, optionally mix with Gaussian.
    audio_1xT: [1, T] waveform in [-1, 1]
    t_index:   1-based index into training noise schedule (DDPM steps)
    returns: eps_tilde [1, T], sqrt_alpha_bar_t (scalar), sqrt_one_minus_alpha_bar_t (scalar)
    """
    assert audio_1xT.ndim == 2 and audio_1xT.size(0) == 1, "audio should be [1, T]"
    device = audio_1xT.device
    training_noise_schedule = np.array(params.noise_schedule)  # beta_t
    alpha = 1.0 - training_noise_schedule
    alpha_bar = np.cumprod(alpha)  # \bar alpha_t
    t_idx = int(np.clip(t_index, 1, len(alpha_bar))) - 1
    sqrt_alpha_bar_t = torch.tensor(alpha_bar[t_idx] ** 0.5, device=device, dtype=audio_1xT.dtype)
    sqrt_one_minus_alpha_bar_t = torch.tensor((1 - alpha_bar[t_idx]) ** 0.5, device=device, dtype=audio_1xT.dtype)

    # pnp directional noise
    # Expect pnp forward signature: pnp(x, t_tensor) -> [1, T]
    if simple_add:
        t_tensor = torch.zeros(1, device=device, dtype=torch.float32)  # for PnP-Gaussian, t is not used
    else:
        t_tensor = torch.tensor([t_idx + 1], device=device, dtype=torch.float32)  # keep 1-based for clarity
    eps_dir = pnp(audio_1xT, t_tensor)  # shape [1, T]
    # L2 normalize to be a direction
    eps_dir = eps_dir / (eps_dir.norm(p=2, dim=1, keepdim=True) + 1e-8)

    xi = torch.randn_like(audio_1xT) # Gaussian noise
    eps_tilde = lam * eps_dir + (1.0 - lam ** 2) ** 0.5 * xi
    return eps_tilde, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t

@torch.no_grad()
def apply_pnp_forward(audio_1xT: torch.Tensor, t_index: int, pnp, params, lam: float = 0.7, simple_add: bool = False):
    """
    Forward-only pnp application.
    If simple_add=True: PnP-Gaussian: out = audio + eps_tilde.
    Else: PnP-Diff: x_t = sqrt(alpha_bar_t)*x + sqrt(1 - alpha_bar_t)*eps_tilde
    """
    eps_tilde, sqrt_a, sqrt_1ma = pnp_predict_noise(pnp, audio_1xT, t_index, params, lam, simple_add)
    if simple_add:
        out = audio_1xT + eps_tilde
    else:
        out = sqrt_a * audio_1xT + sqrt_1ma * eps_tilde
    # out = torch.clamp(out, -1.0, 1.0)
    return out

# ---------------- main pipeline ----------------
def main(args):
    setup_seed(3407)
    device = torch.device("cuda:0")

    # Load pnp
    pnp = load_pnp(args.pnp_dir, device)

    # Params (for alpha schedule / sample_rate)
    params = AttrDict(base_params)

    # Collect wav files
    wav_dir = args.noisy_wav_path
    wav_exts = ("*.wav", "*.flac")
    wav_files = []
    for ext in wav_exts:
        wav_files += glob(path.join(wav_dir, ext))
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No audio found under: {wav_dir}")

    # Prepare output dir
    ensure_dir(args.output)

    print(f"[pnp] Using t={args.tstep}, lam={args.lam}, "
          f"simple_add={args.simple_add}, device={device}")
    for wav_path in tqdm(wav_files, desc="pnp forward"):
        # Load waveform
        x_np, sr = librosa.load(wav_path, sr=params.sample_rate, mono=True)
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1, T], float32

        # Apply pnp forward
        y = apply_pnp_forward(
            audio_1xT=x,
            t_index=int(args.tstep),
            pnp=pnp,
            params=params,
            lam=float(args.lam),
            simple_add=bool(args.simple_add)
        )

        # Save
        base = path.splitext(path.basename(wav_path))[0]
        out_path = path.join(args.output, f"{base}_pnp_t{args.tstep}.wav")
        torchaudio.save(out_path, y.cpu(), sample_rate=params.sample_rate)

    print(f"Done. Wavs written to: {args.output}")

if __name__ == '__main__':
    parser = ArgumentParser(description='pnp forward: predict noise at t and add to input')
    parser.add_argument('pnp_dir', help='directory containing pnp weights (expects weights.pt)')
    parser.add_argument('noisy_wav_path', help='input wav/flac directory')
    parser.add_argument('--output', '-o', default='output/', help='output directory')
    parser.add_argument('--tstep', type=int, default=100, help='forward timestep t (1..T)')
    parser.add_argument('--lam', type=float, default=0.7, help='mixing coefficient for pnp vs Gaussian')
    parser.add_argument('--simple-add', action='store_true', help='use x + eps instead of DDPM forward form')
    args = parser.parse_args()
    main(args)
