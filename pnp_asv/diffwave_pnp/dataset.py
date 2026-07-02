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
import random
import torch
import torch.nn.functional as F
import torchaudio
import librosa

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class ConditionalDataset(torch.utils.data.Dataset):
  def __init__(self,wav_path, paths):
    super().__init__()
    self.wav_path = wav_path
    self.specnames = []
    for path in paths:
      self.specnames += glob(f'{path}/*.spec.npy', recursive=True)

  def __len__(self):
    return len(self.specnames)

  def __getitem__(self, idx):
    audio_filename = self.specnames[idx]
    spec_filename = f'{audio_filename}'
    spec_path = "/".join(spec_filename.split("/")[:-1])
    spec_path = (spec_filename.split("/")[-1]).split("-")
    
    if "LibriSpeech" in self.wav_path:
      spec_path =  spec_path[1] + "/" + spec_path[2] + "/" + spec_path[3] + "-" + spec_path[4]+ "-" + spec_path[5]
    # print(spec_path)
    # spec_path = spec_path[:7] + "/" + spec_path[8:19] + "/" + spec_path[20:]
    # audio_filename = (self.wav_path + spec_path).replace(".spec.npy", "")
    # noisy_filename = (self.noisy_path + spec_path).replace(".spec.npy", "")
    audio_filename = os.path.join(self.wav_path, spec_path.replace(".spec.npy", ""))
    # print(audio_filename)
    signal, _ = librosa.load(audio_filename, sr=16000)
    spectrogram = np.load(spec_filename)
    # print(signal.shape, spectrogram.shape)
    return {
        'audio': signal,
        'spectrogram': spectrogram.T
    }


class UnconditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for root, dirnames,files in os.walk(paths[0]):
       for file in files:
          if file.endswith('.wav') or file.endswith('.flac'):
            self.filenames.append(os.path.join(root,file))

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = torchaudio.load(audio_filename)
    return {
        'audio': signal[0],
        'spectrogram': None
    }


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      if self.params.unconditional:
          # Filter out records that aren't long enough.
          if len(record['audio']) < self.params.audio_len:
            del record['spectrogram']
            del record['audio']
            continue

          # start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
          # end = start + self.params.audio_len
          # record['audio'] = record['audio'][start:end]
          # record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
          record['audio'] = record['audio'][:self.params.audio_len]
      else:
          # Filter out records that aren't long enough.
          if len(record['spectrogram']) < self.params.crop_mel_frames:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
          end = start + self.params.crop_mel_frames
          record['spectrogram'] = record['spectrogram'][start:end].T

          start *= samples_per_frame
          end *= samples_per_frame
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

    valid_records = [record for record in minibatch if 'audio' in record]
    if len(valid_records) == 0:
        return None
    audio = np.stack([record['audio'] for record in valid_records])
    if self.params.unconditional:
        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': None,
        }
    spectrogram = np.stack([record['spectrogram'] for record in valid_records if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
    }

  # for gtzan
  def collate_gtzan(self, minibatch):
    ldata = []
    mean_audio_len = self.params.audio_len # change to fit in gpu memory
    # audio total generated time = audio_len * sample_rate
    # GTZAN statistics
    # max len audio 675808; min len audio sample 660000; mean len audio sample 662117
    # max audio sample 1; min audio sample -1; mean audio sample -0.0010 (normalized)
    # sample rate of all is 22050
    for data in minibatch:
      if data[0].shape[-1] < mean_audio_len:  # pad
        data_audio = F.pad(data[0], (0, mean_audio_len - data[0].shape[-1]), mode='constant', value=0)
      elif data[0].shape[-1] > mean_audio_len:  # crop
        start = random.randint(0, data[0].shape[-1] - mean_audio_len)
        end = start + mean_audio_len
        data_audio = data[0][:, start:end]
      else:
        data_audio = data[0]
      ldata.append(data_audio)
    if len(ldata) == 0:
      return None
    audio = torch.cat(ldata, dim=0)
    return {
          'audio': audio,
          'spectrogram': None,
    }


def from_path(data_dirs, params, is_distributed=False):
  if params.unconditional:
    dataset = UnconditionalDataset(data_dirs)
  else:#with condition
    dataset = ConditionalDataset(data_dirs)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)


def from_gtzan(params, is_distributed=False):
  dataset = torchaudio.datasets.GTZAN('./data', download=True)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_gtzan,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
