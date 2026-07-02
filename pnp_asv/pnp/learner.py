# Copyright 2020 LMNT, ...


import numpy as np
import os
import json
import torch
import shutil
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import from_path, from_gtzan
from model_pnp import DiffWave
from params_pnp import AttrDict

import torchaudio.compliance.kaldi as kaldi 
def Fbanks(wavs):
    mats = []
    for wav in wavs:
        mat = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
        )
        # mat = mat - torch.mean(mat, dim=0)/(torch.std(mat,dim=0)+1e-10)
        mat = mat - torch.mean(mat, dim=0)
        mats.append(mat)
    return torch.stack(mats)

def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

# ------------------------ ASV helpers ------------------------

def _load_asv_encoder(asv_model_path, device='cpu'):
  asv = torch.load(asv_model_path) 
  return asv.eval()


# ------------------------ Learner ------------------------
class PnPLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    # noise schedule
    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)  # ᾱ_t
    self.noise_level = torch.tensor(noise_level.astype(np.float32))

    self.train_pnp = params.get('pnp', True)
    self.margin     = params.get('pnp_margin', 0.96)
    self.lambda_mix = params.get('pnp_lambda', 0.7)
    self.lambda_e   = params.get('pnp_lambda_e', 1e-3)
    self.asv_path   = params.get('asv_model', None)
    self.max_puri_step = params.get('max_puri_step', 3)  

    self.loss_fn = nn.L1Loss()

    self.summary_writer = None
    self.asv = None  
    self.grad_norm = torch.tensor(0.0)
    self.simple_add = params.get('simple_add', False) #True: PnP-Gaussian; False: PnP-Diffusion

    margin_tag = f"{float(self.margin):g}"
    if self.simple_add:
      print("Using PnP-Gaussian (simple addition) with λ =", self.lambda_mix, "margin =", self.margin)
      self.model_dir = model_dir+f"/pnpg_m_{margin_tag}"
    else:
      print("Using PnP-Diff (diffusion forward style) with λ =", self.lambda_mix, "margin =", self.margin)
      self.model_dir = model_dir+f"/pnpd_m_{margin_tag}"
    print("save to ",self.model_dir)

  # ----------------- checkpoint -----------------
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
      if os.path.exists(link_name):
        os.remove(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  # ----------------- training loop -----------------
  def train(self, max_steps=None, max_epochs=None):
    device = next(self.model.parameters()).device
    self.asv = torch.load(self.asv_path, map_location=device).eval()
    for epoch in range(max_epochs):
      epoch_str = f'Epoch {epoch + 1}/{max_epochs}'
      iterator = tqdm(self.dataset, desc=epoch_str,dynamic_ncols=True) if self.is_master else self.dataset
      for features in iterator:
        if max_steps is not None and self.step >= max_steps:
          return
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

  def _sample_t(self, N, device,max_puri_step=3):
    return torch.randint(0, max_puri_step, [N], device=device)

  def _alpha_terms(self, t, device):
    # ᾱ_t
    self.noise_level = self.noise_level.to(device)
    noise_scale = self.noise_level[t].unsqueeze(1)         # [N,1]
    noise_scale_sqrt = noise_scale**0.5                    # sqrt(ᾱ_t)
    one_minus = (1.0 - noise_scale).clamp(min=1e-9)        # 1-ᾱ_t
    one_minus_sqrt = one_minus**0.5                        # sqrt(1-ᾱ_t)
    return noise_scale_sqrt, one_minus_sqrt

  def _normalize(self, x, eps=1e-6):
    if x.dim() == 3 and x.size(1) == 1:
      x = x.squeeze(1)
    denom = x.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
    return x / denom

  def _cos_sim(self, a, b):
    a = a.unsqueeze(1) if len(a.size()) == 2 else a
    b = b.unsqueeze(1) if len(b.size()) == 2 else b

    assert a.dim() == 3 and b.dim() == 3
    
    a = Fbanks(a)
    b = Fbanks(b)
    ea = self.asv(a)
    eb = self.asv(b)
    return nn.functional.cosine_similarity(ea, eb, dim=-1)

  def _pnp_train_step(self, features):
    for p in self.model.parameters():
      p.grad = None

    x_bf = features.get('audio')
    x_adv = features.get('adv_audio')
    x_adv = x_adv.squeeze(1) if x_adv.dim()==3 else x_adv
    spectrogram = None
  
    N, T = x_bf.shape
    device = x_bf.device
    t = self._sample_t(N, device,self.max_puri_step if hasattr(self,'max_puri_step') else 3)
    sqrt_alphabar, sqrt_one_minus = self._alpha_terms(t, device)

    with self.autocast:
      if self.simple_add:
        t = torch.zeros_like(t) # for PnP-Gaussian, t is not used
      eps_bf  = self.model(x_bf,  t, spectrogram)  
      eps_adv = self.model(x_adv, t, spectrogram)
      eps_bf  = eps_bf.squeeze(1) if eps_bf.dim()==3 else eps_bf
      eps_adv = eps_adv.squeeze(1) if eps_adv.dim()==3 else eps_adv

      lam = float(self.lambda_mix)
      gauss_bf  = torch.randn_like(x_bf)
      gauss_adv = torch.randn_like(x_adv)

      mix_bf  = lam * self._normalize(eps_bf)  + (1.0 - lam**2)**0.5 * gauss_bf
      mix_adv = lam * self._normalize(eps_adv) + (1.0 - lam**2)**0.5 * gauss_adv
      if self.simple_add: # PnP-Gaussian, w_x=w_n=1
        x_t_bf = x_bf + mix_bf
        x_t_adv = x_adv + mix_adv
        # x_t_bf = torch.clamp(x_t_bf, x_bf.min(), x_bf.max())
        # x_t_adv = torch.clamp(x_t_adv, x_adv.min(), x_adv.max())
      else: # PnP-Diffusion, w_x=sqrt(ᾱ_t), w_n=sqrt(1-ᾱ_t)
        x_t_bf  = sqrt_alphabar * x_bf  + sqrt_one_minus * mix_bf
        x_t_adv = sqrt_alphabar * x_adv + sqrt_one_minus * mix_adv

      s1 = self._cos_sim(x_bf,  x_t_bf)
      s2 = self._cos_sim(x_bf,  x_t_adv)
      m  = float(self.margin)
      loss_asv = torch.relu(m - s1).mean() + torch.relu(m - s2).mean()
                            
      loss_reg = self.lambda_e * (mix_bf.pow(2).mean() + mix_adv.pow(2).mean())
      loss = loss_asv + loss_reg

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def train_step(self, features):
    if self.train_pnp:
      return self._pnp_train_step(features)

    for param in self.model.parameters():
      param.grad = None
    audio = features['audio']
    spectrogram = features['spectrogram']
    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise

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
    wav = features.get('audio', None)
    if wav is not None:
      writer.add_audio('feature/audio', wav[0], step, sample_rate=self.params.sample_rate)
    if not self.params.unconditional and 'spectrogram' in features and features['spectrogram'] is not None:
      writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', float(loss.detach().cpu()), step)
    writer.add_scalar('train/grad_norm', float(self.grad_norm.detach().cpu()), step)
    writer.add_scalar('train/pnp_lambda', float(self.lambda_mix), step)
    writer.add_scalar('train/pnp_margin', float(self.margin), step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)


  learner = PnPLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  if replica_id == 0:
    params_src = os.path.join(os.path.dirname(__file__), 'params_pnp.py')
    params_dst = os.path.join(learner.model_dir, 'params.py')
    if os.path.exists(params_src):
      os.makedirs(learner.model_dir, exist_ok=True)
      shutil.copyfile(params_src, params_dst)
    effective_params_dst = os.path.join(learner.model_dir, 'params.effective.json')
    with open(effective_params_dst, 'w') as fp:
      json.dump(dict(params), fp, indent=2, sort_keys=True)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps, max_epochs=args.max_epochs)


def train(args, params):
  params_src = os.path.join(os.path.dirname(__file__), 'params_pnp.py')
  params_dst = os.path.join(args.model_dir, 'params.py')
  if os.path.exists(params_src):
    os.makedirs(args.model_dir, exist_ok=True)
    shutil.copyfile(params_src, params_dst)
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
  
  
  model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=False)
  
  _train_impl(replica_id, model, dataset, args, params)
