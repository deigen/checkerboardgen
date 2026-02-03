import argparse
import datetime
import numpy as np
import os
import sys
import setproctitle
import subprocess
import shutil
import random
import time
from pathlib import Path
import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScaler
from util.loader import CachedVQDataset, create_vq_cache_v2

from cbgen import ar
from engine import train_one_epoch, evaluate
import copy

args = None

def get_args_parser():
    def listarg(type):
        def parser_fn(x):
            lst = [type(i) for i in x.split(',')]
            if len(lst) == 1:
                return lst[0]
            return lst
        return parser_fn

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--config', help='path to config file (required)', default=None)

    # Generation parameters
    parser.add_argument('--eval_freq', type=int, default=0, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=1, help='save last frequency')

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimizer to use (default: "adamw")')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate (if --lr is not specified): lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=5e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--lr_step_epochs', type=listarg(int), default=[],
                        help='epochs to decrease lr by lr_step_factor (used in step lr schedule)')
    parser.add_argument('--lr_step_factor', type=float, default=0.1,
                        help='factor to decrease lr at each step (used in step lr schedule)')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='number of gradient accumulation steps')
    parser.add_argument('--ema_rate', default=0.9999, type=float)
    parser.add_argument('--ema', action='store_true', help='Enable EMA')

    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Gradient clip')

    # inline eval flags
    parser.add_argument('--eval_batch_size', type=int, default=None, help='generation batch size')
    parser.add_argument('--eval_num_samples', type=int, default=10000, help='number of samples to eval')
    parser.add_argument('--samples_output_dir', type=str, default=None, help='directory to save samples during evaluation')
    parser.add_argument('--samples_only', action='store_true', help='only generate samples without evaluation')
    parser.add_argument('--keep_samples', type=int, default=0, help='number of samples to keep after eval')
    parser.add_argument('--cfg', type=listarg(float), default=2.0, help='classifier-free guidance scale (for evaluate)')
    parser.add_argument('--cfg_start_step', type=int, default=5, help='starting step for cfg during evaluation')
    parser.add_argument('--temperature', default=1.0, type=float, help='sampling temperature (for evaluate)')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--eval_reference_data_path', default=None, type=str,
                        help='metrics reference dataset path')

    # Misc
    parser.add_argument('--output_root', default='./outputs', type=str, help='root path for output')
    parser.add_argument('--output_dir', default=None,
                        help='path where to save checkpoint')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--weights_init', default='',
                        help='weights init from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', '--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', '--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached latents')
    parser.add_argument('--create_cache', action='store_true',
                        help='create cached latents')
    parser.add_argument('--cached_path', default='./data/cached', help='root path to cached latents')

    return parser


log_writer = None


def main():
    global args, log_writer

    parser = get_args_parser()
    args = parser.parse_args()

    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # config
    if args.config is None:
        if args.resume or args.weights_init:
            resume_checkpoint = args.resume if args.resume else args.weights_init
            if os.path.isdir(resume_checkpoint):
                args.config = os.path.join(resume_checkpoint, 'config.yaml')
            else:
                args.config = os.path.join(os.path.dirname(resume_checkpoint), 'config.yaml')
        else:
            raise ValueError("args.config is required")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine which args were explicitly provided on command line
    # by checking if their option strings appear in sys.argv (handles both --arg value and --arg=value)
    explicitly_set = set()
    valid_args = set()
    for action in parser._actions:
        if action.dest == 'help':
            continue
        valid_args.add(action.dest)
        for option in action.option_strings:
            # Check for both "--arg value" and "--arg=value" syntax
            if option in sys.argv or any(arg.startswith(option + '=') for arg in sys.argv):
                explicitly_set.add(action.dest)
                break
    # Set args from the config file args: section if they were not on the command line
    if 'args' in config:
        for k, v in config['args'].items():
            if k not in valid_args:
                raise ValueError(f"Unknown argument '{k}' in config file")
            if k not in explicitly_set:
                setattr(args, k, v)

    if args.output_dir is None:
        if global_rank == 0:
            hexstr = hex(random.getrandbits(64))[2:10]
            run_id = datetime.datetime.now().strftime("%y%m%d-%H%M") + '-' + hexstr
            args.output_dir = os.path.join(args.output_root, run_id)
        else:
            args.output_dir = None

    if args.output_dir is None:
        assert global_rank == 0, "--evaluate only supported on single process"
        checkpoint_path = args.resume if args.resume else args.weights_init
        checkpoint_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path
        assert os.path.isdir(checkpoint_dir), "Could not find checkpoint dir for evaluation"
        args.output_dir = os.path.join(checkpoint_dir, 'eval')

    if args.log_dir is None:
        args.log_dir = args.output_dir

    if global_rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        misc.save_exper_tracking(args, output_dir=args.output_dir)

    if global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    print('output_dir: {}'.format(args.output_dir))
    setproctitle.setproctitle(setproctitle.getproctitle() + f'  [{args.output_dir}]')

    img_size = config['model']['img_size']

    # Handle VQ cache creation/loading
    if args.use_cache or args.create_cache:
        # New cache system (v2)
        split = 'train'
        dsname = os.path.relpath(args.data_path, './data').replace('/', '_').replace('.', '_').strip('_')

        # Build cache filename based on autoencoder config
        autoenc_config = config['model']['autoenc']
        autoenc_name = autoenc_config.get('checkpoint_path', 'unknown').replace('/', '_')  # Extract name from path
        all_scale_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16]
        scales_key = ''.join(map(str, all_scale_sizes))
        cache_key = f'cached_data_{img_size}_{autoenc_name}_{scales_key}_v3'
        cache_filepath = os.path.join(args.cached_path, dsname, split, f'{cache_key}.safetensors')

        data_path = os.path.join(args.data_path, split)

        if args.create_cache:
            # Load autoencoder for cache creation
            print("Loading autoencoder for cache creation...")
            assert config['model'].get('scale_min_size', 1) == 1, \
                'creating cache should use scale_min_size=1 for full pyramid'
            model = ar.AR(
                num_classes=1000,
                **config['model']
            )
            autoenc = model.autoenc
            autoenc.to(device)
            autoenc.eval()
            autoenc.set_scale_sizes(all_scale_sizes)

            # Create cache
            create_vq_cache_v2(
                cache_filepath=cache_filepath,
                data_path=data_path,
                autoenc=autoenc,
                img_size=img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device
            )
            print(f"Data cache created at {cache_filepath}")
            return

        # Load cached dataset
        dataset_train = CachedVQDataset(cache_filepath)

    else:
        # Standard training with on-the-fly preprocessing
        # Use TenCrop augmentation (same as cache creation)
        crop_size = int(img_size * 1.1)

        def random_tencrop_transform(img):
            """Apply TenCrop and randomly select one of the 10 crops."""
            ten_crops = transforms.TenCrop(img_size)(img)
            crop_idx = random.randint(0, len(ten_crops) - 1)
            return ten_crops[crop_idx]

        transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size, flip=False)),
            transforms.Lambda(random_tencrop_transform), # chooses one random ten-crop
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        dataset_dirname, dataset_basename = os.path.split(os.path.abspath(args.data_path))
        if dataset_basename == 'celeba':
            dataset_train = datasets.CelebA(dataset_dirname, split='train', transform=transform_train, download=False)
        elif dataset_basename == 'food101':
            dataset_train = datasets.Food101(args.data_path, split='train', transform=transform_train, download=False)
        else:
            dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    if args.eval_reference_data_path is None:
        args.eval_reference_data_path = os.path.join(args.data_path, 'eval_reference')

    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    # Use custom collate function for cached dataset
    collate_fn = None
    if args.use_cache and hasattr(dataset_train, 'get_collate_fn'):
        collate_fn = dataset_train.get_collate_fn()

    # data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers if not args.use_cache else min(args.num_workers, 1),
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # create model
    num_classes = 0
    if hasattr(dataset_train, 'classes') and dataset_train.classes:
        num_classes = len(dataset_train.classes)
    elif hasattr(dataset_train, 'num_classes'):
        num_classes = dataset_train.num_classes
    model = ar.AR(
        num_classes=num_classes,
        **config['model']
    )

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.lr is None:  # only base_lr is specified
        eff_batch_size = args.batch_size * misc.get_world_size()
        args.lr = args.blr * eff_batch_size / 256

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    print("lr: %.2e" % args.lr)

    # add weight decay to params except biases and norm layers 
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.95))
    else:
        raise NotImplementedError(f'Optimizer {args.optimizer} not implemented')
    print(optimizer)
    loss_scaler = NativeScaler()

    # resume training
    if args.resume or args.weights_init:
        # resume takes precedence over weights_init
        resume_checkpoint = args.resume if args.resume else args.weights_init
        if os.path.isdir(resume_checkpoint):
            resume_checkpoint = os.path.join(resume_checkpoint, "checkpoint-last.pth")
        if not os.path.exists(resume_checkpoint):
            raise ValueError(f"Checkpoint {resume_checkpoint} does not exist")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_params = list(model_without_ddp.parameters())
        print("Loaded init checkpoint %s" % resume_checkpoint)

        if args.ema:
            if args.resume and checkpoint.get('model_ema', None) is not None:
                ema_state_dict = checkpoint['model_ema']
                ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
            else:
                ema_params = copy.deepcopy(model_params)
        if args.resume and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    if not args.ema:
        ema_params = None

    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if misc.is_main_process():
            # save checkpoint
            if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, link_last=True)

            # online evaluation
            if args.eval_freq and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
                torch.cuda.empty_cache()
                if args.ema:
                    train_params = misc.swap_params(model_without_ddp, ema_params)
                evaluate(model_without_ddp, device, epoch, log_writer, args)
                if args.ema:
                    misc.swap_params(model_without_ddp, train_params)
                    del train_params
                torch.cuda.empty_cache()

            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    try:
        main()
    finally:
        if log_writer is not None:
            log_writer.flush()
            log_writer.close()
        # check byte length of output files, delete if this run was too short
        if args is not None and misc.is_main_process() and args.output_dir is not None and os.path.exists(args.output_dir):
            keep = False
            for f in os.listdir(args.output_dir):
                fsize = os.path.getsize(os.path.join(args.output_dir, f))
                if fsize > 100 * 1024:
                    keep = True
                    break
            if not keep:
                print(f"Deleting output dir for incomplete run: {args.output_dir}")
                shutil.rmtree(args.output_dir)
