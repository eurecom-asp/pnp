# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import sys
import torch
import librosa
import torchaudio
import random
from argparse import ArgumentParser

from params_pnpplus import AttrDict, params as base_params
from model_pnpplus import DiffWave
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pnp')))
from pnp.inference_pnp import apply_pnp_forward, pnp_predict_noise
from pnp.model_pnp import DiffWave as PnpDiffWave
from pnp.params_pnp import AttrDict as PnpAttrDict, params as base_params_pnp
from os import path
from glob import glob
from tqdm import tqdm
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
setup_seed(3407)
models = {}
pnp_models = {}

def load_model(model_dir=None, args=None, params=None,device=torch.device('cuda:0')) :
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    
    model = DiffWave(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model
  model = models[model_dir]
  model.params.override(params)
      
  return model

def _resolve_pnp_checkpoint(pnp_path):
  if not pnp_path:
    raise ValueError('params.pnp_path must be set to a pnp checkpoint or directory.')
  ckpt_path = path.join(pnp_path, 'weights.pt') if path.isdir(pnp_path) else pnp_path
  if not path.exists(ckpt_path):
    raise FileNotFoundError(f'pnp checkpoint not found: {ckpt_path}')
  return ckpt_path

def load_pnp(pnp_path, device):
  if pnp_path in pnp_models:
    return pnp_models[pnp_path]
  ckpt_path = _resolve_pnp_checkpoint(pnp_path)
  ckpt = torch.load(ckpt_path, map_location=device)
  model = PnpDiffWave(PnpAttrDict(base_params_pnp)).to(device)
  model.load_state_dict(ckpt['model'])
  model.eval()
  pnp_models[pnp_path] = model
  return model

def _pnp_noise(audio_1xT, t_index, pnp_model, params):
  lam = float(getattr(params, 'pnp_lambda', 0.7))
  simple_add = bool(getattr(params, 'simple_add', False))
  max_t = max(1, len(base_params_pnp.noise_schedule) - 1)
  t_index = min(int(t_index), max_t)
  eps_tilde, _, _ = pnp_predict_noise(
    pnp_model,
    audio_1xT,
    int(t_index),
    base_params_pnp,
    lam=lam,
    simple_add=simple_add,
  )
  if eps_tilde.ndim == 3:
    eps_tilde = eps_tilde.squeeze(1)
  return eps_tilde

def reverse_only(model, noisy_audio, spectrogram=None, pstep=None, device=torch.device('cuda:0'),
                 fast_sampling=False):
  """
  Reverse-only purification: use DiffWave to predict noise and remove it, without adding extra noise.
  """
  with torch.no_grad():
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule
    if fast_sampling or pstep is None:
      pstep = len(inference_noise_schedule)
    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)

    if not model.params.unconditional and spectrogram is not None:
      if len(spectrogram.shape) == 2:
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
    audio = noisy_audio.to(device)

    for n in range(pstep - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate


def reverse_only_grad(model, noisy_audio, spectrogram=None, pstep=None, device=torch.device('cuda:0'),
                      fast_sampling=False):
  """
  Differentiable reverse-only purification.
  Same logic as reverse_only, but without torch.no_grad().
  """
  training_noise_schedule = np.array(model.params.noise_schedule)
  inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule
  if fast_sampling or pstep is None:
    pstep = len(inference_noise_schedule)
  talpha = 1 - training_noise_schedule
  talpha_cum = np.cumprod(talpha)

  beta = inference_noise_schedule
  alpha = 1 - beta
  alpha_cum = np.cumprod(alpha)
  T = []
  for s in range(len(inference_noise_schedule)):
    for t in range(len(training_noise_schedule) - 1):
      if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
        T.append(t + twiddle)
        break
  T = np.array(T, dtype=np.float32)

  if not model.params.unconditional and spectrogram is not None:
    if len(spectrogram.shape) == 2:
      spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)
  audio = noisy_audio.to(device)

  for n in range(pstep - 1, -1, -1):
    c1 = 1 / alpha[n]**0.5
    c2 = beta[n] / (1 - alpha_cum[n])**0.5
    audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
    audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate

def predict(model,spectrogram=None,noisy_wav=None,pstep=None, device=torch.device('cuda:0'), fast_sampling=False):

  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule
    if fast_sampling:
      pstep = len(inference_noise_schedule)
    # inference_noise_schedule = inference_noise_schedule[:int(pstep)] if pstep is not None else inference_noise_schedule
    # print("inference noise schedule:",inference_noise_schedule)
    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)
    if not model.params.unconditional:
      if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
      audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
      # audio[:,:noisy_wav.shape[0]] = torch.from_numpy(noisy_wav).to(device)
    else:
      audio = torch.from_numpy(noisy_wav).unsqueeze(0).to(device)
    pnp_model = load_pnp(getattr(model.params, 'pnp_path', None), device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float()[pstep-1].to(device)
    # print("noise_scale:",noise_scale)
    noise_scale_sqrt = noise_scale**0.5
    safe_pstep = min(int(pstep), max(1, len(base_params_pnp.noise_schedule) - 1))
    audio = apply_pnp_forward(
      audio,
      safe_pstep,
      pnp_model,
      base_params_pnp,
      lam=float(getattr(model.params, 'pnp_lambda', 0.7)),
      simple_add=bool(getattr(model.params, 'simple_add', False)),
    )
    for n in range(pstep - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      if n > 0:
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        noise = _pnp_noise(audio, n + 1, pnp_model, model.params)
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
      # break
  return audio, model.params.sample_rate


def main(args):
  specnames = []
  print("spectrums:",args.spectrogram_path)
  print("noisy_wavs:",args.wav_path)
  for path in args.spectrogram_path:
    specnames += glob(f'{path}/*.spec.npy', recursive=True)
    # Lazy load model.
  model_dir = args.model_dir
  
  model = load_model(model_dir=model_dir, args=args)
  with torch.no_grad():
    for spec in tqdm(specnames):
      spectrogram = torch.from_numpy(np.load(spec))
      filename = spec.split("/")[-1].replace(".spec.npy","")
      if "LIBRISPEECH" in args.wav_path:
        pass
      noisy_wav, _ = librosa.load(os.path.join(args.wav_path,filename),sr=16000)
      audio, sr = predict(model,spectrogram,noisy_wav,args.pstep, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params)
      torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('spectrogram_path', nargs='+',
      help='directory containing spectrogram generated by diffwave.preprocess')
  parser.add_argument('noisy_wav_path',
  help='input noisy wav directory')
  parser.add_argument('--output', '-o', default='output/',
      help='output dir name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  parser.add_argument('--pstep', default='10', 
      help='number of steps for purification')
  main(parser.parse_args())
