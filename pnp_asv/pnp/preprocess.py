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
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import repeat
import os
from params import params


def transform(filename,indir,outdir, use_spec=False):
  audio, sr = T.load(filename)
  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')
  mel_args = {
      'sample_rate': sr,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': sr / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = TT.MelSpectrogram(**mel_args)

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    filename = filename.replace(indir,"")
    if "LibriSpeech" in indir:
      filename = filename.replace("/","-")
    np.save(f'{outdir+"/"+filename}.spec.npy', spectrogram.numpy()[0])


def main(args):
    dir_path = args.dir
    filenames = []
    for root, dirnames,files in os.walk(dir_path):
       for file in files:
          if file.endswith('.wav') or file.endswith('.flac'):
            filenames.append(os.path.join(root,file))
    with ProcessPoolExecutor() as executor:
      list(tqdm(executor.map(transform, filenames, repeat(args.dir), repeat(args.outdir),repeat(args.spec)), desc='Preprocessing', total=len(filenames)))

if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave-based purification models')
  parser.add_argument('dir',
      help='directory containing .wav/.flac files for training')
  parser.add_argument('outdir',
      help='output directory containing .npy files for training')
  parser.add_argument('--spec',default=False,type=bool,
      help='whether to use spectrograms, default is Mel')
  main(parser.parse_args())
