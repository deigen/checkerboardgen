import argparse
import datetime
import functools
import numpy as np
import os
import pickle
import shutil
import time
from pathlib import Path
import yaml

import torch

from cbgen import ar
from engine import evaluate
import copy


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--eval_reference_data_path', type=str)
    parser.add_argument('--eval_batch_size', type=int, default=128, help='generation batch size')
    parser.add_argument('--eval_num_samples', type=int, default=1000, help='number of samples to eval')
    parser.add_argument('--eval_image_sizes', type=int, nargs='+', default=None, help='image sizes to eval at')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--cfg', type=float, default=2.0, help='classifier free guidance scale')
    parser.add_argument('--cfg_start_step', type=int, default=5, help='starting step for cfg during evaluation')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature')
    parser.add_argument('--steps_per_scale', type=int, default=8, help='number of AR steps per scale to use during eval')
    parser.add_argument('--keep_samples', action='store_true', help='whether to keep generated samples on disk')
    parser.add_argument('--samples_output_dir', type=str, default=None, help='directory to save samples if keep_samples is set')
    parser.add_argument('--samples_only', action='store_true', help='only generate samples without evaluation')
    return parser

def main(args):
    checkpoint_path = args.checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint-last.pth")
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist")

    checkpoint_dir = os.path.dirname(checkpoint_path)

    train_args_fn = os.path.join(checkpoint_dir, 'args.yaml')
    config_fn = os.path.join(checkpoint_dir, 'config.yaml')

    if not os.path.isfile(train_args_fn):
        raise ValueError(f"Train args {train_args_fn} does not exist")
    if not os.path.isfile(config_fn):
        raise ValueError(f"Config {config_fn} does not exist")

    train_args = yaml.safe_load(open(train_args_fn, 'r'))
    config = yaml.safe_load(open(config_fn, 'r'))

    model = ar.AR(**config['model'], num_classes=train_args.get('num_classes', 1000))
    model.num_blocks_per_scale = args.steps_per_scale

    #if model.scale_ratio < 2:
    #    args.eval_batch_size = min(args.eval_batch_size, 64)

    args.data_path = train_args['data_path']

    if args.eval_reference_data_path is None:
        args.eval_reference_data_path = os.path.join(args.data_path, 'eval_reference')

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_ema' if args.use_ema else 'model'], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.output_dir is None:
        args.output_dir = checkpoint_dir

    evaluate(model, device, None, None, args)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    print("{}".format(args).replace(', ', ',\n'))
    main(args)
