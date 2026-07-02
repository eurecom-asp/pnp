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

import json
from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train, train_distributed
from params_pnp import params


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def _parse_override_value(raw):
  lowered = raw.lower()
  if lowered == 'true':
    return True
  if lowered == 'false':
    return False
  if lowered in ('none', 'null'):
    return None
  try:
    return int(raw)
  except ValueError:
    pass
  try:
    return float(raw)
  except ValueError:
    pass
  if raw.startswith('[') or raw.startswith('{'):
    try:
      return json.loads(raw)
    except json.JSONDecodeError:
      pass
  return raw


def _apply_overrides(base_params, overrides):
  if not overrides:
    return
  merged = {}
  for item in overrides:
    if '=' not in item:
      raise ValueError(f'Override must be formatted as key=value, got: {item}')
    key, raw = item.split('=', 1)
    merged[key] = _parse_override_value(raw)
  base_params.override(merged)


def main(args):
  _apply_overrides(params, args.override)
  replica_count = device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='Train pnp (forward-only) or standard DiffWave')

  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('data_dirs', nargs='+',
      help='space separated list of directories from which to read .wav files for training')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--max_epochs', default=10, type=int,
      help='maximum number of training epochs')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  parser.add_argument('--override', action='append', default=[],
      help='override params_pnp values with key=value, repeatable')


  main(parser.parse_args())
