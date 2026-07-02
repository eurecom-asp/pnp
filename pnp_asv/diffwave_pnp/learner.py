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
import json
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pnp')))

from pnp.inference_pnp import pnp_predict_noise
from pnp.model_pnp import DiffWave as PnpDiffWave
from pnp.params_pnp import AttrDict as PnpAttrDict, params as pnp_params

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import from_path, from_gtzan
from model_pnpplus import DiffWave


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffWavePNPLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None
    self.pnp = None

  def _resolve_pnp_checkpoint(self):
    pnp_path = getattr(self.params, 'pnp_path', None)
    if not pnp_path:
      raise ValueError('params.pnp_path must be set to a pnp checkpoint or directory.')
    ckpt_path = os.path.join(pnp_path, 'weights.pt') if os.path.isdir(pnp_path) else pnp_path
    if not os.path.exists(ckpt_path):
      raise FileNotFoundError(f'pnp checkpoint not found: {ckpt_path}')
    return ckpt_path

  def _load_pnp(self, device):
    ckpt_path = self._resolve_pnp_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PnpDiffWave(PnpAttrDict(pnp_params)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

  def _pnp_noisy_audio(self, audio, t):
    if self.pnp is None or next(self.pnp.parameters()).device != audio.device:
      self.pnp = self._load_pnp(audio.device)
    lam = float(getattr(self.params, 'pnp_lambda', 0.7))
    simple_add = bool(getattr(self.params, 'simple_add', False))
    max_t = max(1, len(pnp_params.noise_schedule) - 1)
    eps_list = []
    sqrt_a_list = []
    sqrt_1ma_list = []
    for i in range(audio.size(0)):
      t_index = min(int(t[i].item()), max_t)
      eps_i, sqrt_a_i, sqrt_1ma_i = pnp_predict_noise(
        self.pnp,
        audio[i:i + 1],
        t_index,
        self.params,
        lam=lam,
        simple_add=simple_add,
      )
      if eps_i.ndim == 3:
        eps_i = eps_i.squeeze(1)
      eps_list.append(eps_i)
      sqrt_a_list.append(sqrt_a_i)
      sqrt_1ma_list.append(sqrt_1ma_i)
    eps = torch.cat(eps_list, dim=0)
    sqrt_a = torch.stack(sqrt_a_list).view(-1, 1)
    sqrt_1ma = torch.stack(sqrt_1ma_list).view(-1, 1)
    if simple_add:
      noisy_audio = audio + eps
    else:
      noisy_audio = sqrt_a * audio + sqrt_1ma * eps
    return eps, noisy_audio

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None, max_epochs=None):
    device = next(self.model.parameters()).device
    while True:
      if max_epochs is not None and (self.step // len(self.dataset)) >= max_epochs:
        return
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        if features is None:
          continue
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss)
          if self.step % len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    spectrogram = features['spectrogram']

    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)
    with self.autocast:
      max_step = int(getattr(self.params, 'max_puri_step', len(self.params.noise_schedule)))
      max_step = max(1, min(max_step, len(self.params.noise_schedule)))
      t = torch.randint(0, max_step, [N], device=audio.device)
      noise, noisy_audio = self._pnp_noisy_audio(audio, t)

      predicted = self.model(noisy_audio, t, spectrogram)
      loss = self.loss_fn(noise, predicted.squeeze(1))

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    if not self.params.unconditional:
      writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWavePNPLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  if replica_id == 0:
    os.makedirs(learner.model_dir, exist_ok=True)
    effective_params_dst = os.path.join(learner.model_dir, 'params.effective.json')
    with open(effective_params_dst, 'w') as fp:
      json.dump(dict(params), fp, indent=2, sort_keys=True)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps, max_epochs=args.max_epochs)


def train(args, params):
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params)
  else:
    dataset = from_path(args.data_dirs, params)
  model = DiffWave(params).cuda()
  _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params, is_distributed=True)
  else:
    dataset = from_path(args.data_dirs, params, is_distributed=True)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id],find_unused_parameters=False)
  _train_impl(replica_id, model, dataset, args, params)
