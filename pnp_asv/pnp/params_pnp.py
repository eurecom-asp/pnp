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


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,  # Not used in PnP

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional=True,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # unconditional sample len
    audio_len=16000*5, # unconditional_synthesis_samples

    
    adv_path="data/generated/attacks/pgd_l2_ecapa_50_6400_500",
    max_puri_step=3,  # max purification steps (default: 3)
    asv_model="checkpoints/asv/ecapa_tdnn.pth",
    pnp=True,  # enable pnp forward-only training (default: True)
    pnp_lambda=0.7,  # mixing factor λ for Pi-noise direction and Gaussian, default: 0.7
    pnp_margin=0.8,  # ASV cosine hinge margin, only used for training, default: 0.8
    pnp_lambda_e=1e-2, # weight for regularization loss, only used for training
    simple_add=False,  # True: PnP-Gaussian; False: PnP-Diff
)
