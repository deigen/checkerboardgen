import math
import sys
import tqdm
import itertools
from typing import Iterable
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import shutil
import numpy as np
import os

import torch_fidelity


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


always_print_grad_norm_thresh = None  # Set to a float value to enable


def train_one_epoch(model,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Gradient accumulation setup
    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
    accum_loss = 0.0
    accum_stats = {}

    for data_iter_step, (images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Handle data format: cached is tuple of lists, non-cached is tensor
        if not args.use_cache:
            # Standard path: images is a tensor (B, C, H, W)
            images = images.to(device, non_blocking=True)
        else:
            # Cached path: images is ((zq_pyr_batch, q_inds_batch)), lists of lists of tensors
            images = [ [x.to(device, non_blocking=True) for x in xs] for xs in images ]

        labels = labels.to(device, non_blocking=True)

        # forward
        with torch.cuda.amp.autocast():
            loss, stats = model(images, labels, cached_load=args.use_cache)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Scale loss for gradient averaging over accumulation steps
        loss = loss / grad_accum_steps

        # Accumulate metrics
        accum_loss += loss_value
        for stat_name, stat_value in stats.items():
            accum_stats[stat_name] = accum_stats.get(stat_name, 0.0) + stat_value

        # Backward pass
        loss_scaler(loss, optimizer)

        # Check if this is an accumulation step (time to update parameters)
        is_accum_step = (data_iter_step + 1) % grad_accum_steps == 0
        if not is_accum_step:
            continue

        # Unscale grads
        loss_scaler.unscale_(optimizer)

        # Compute gradient norm after accumulation
        grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None], norm_type=2.0).item()

        # Check for inf or nan gradients
        # should be handled by the loss scaler, but check here for the grad norm logging
        # otherwise, apply gradient clipping if specified
        if not math.isfinite(grad_norm):
            grad_norm = 0.0
        elif args.clip_grad is not None and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # Normal training step
        loss_scaler.step(optimizer)
        loss_scaler.update()

        force_log_stats = False
        if always_print_grad_norm_thresh is not None and grad_norm > always_print_grad_norm_thresh:
            print(f"Step {data_iter_step}  Grad Norm: {grad_norm}  (rank {misc.get_rank()})")
            force_log_stats = True

        optimizer.zero_grad()

        torch.cuda.synchronize()

        # Update EMA after full accumulation step
        if ema_params is not None:
            update_ema(ema_params, model_params, rate=args.ema_rate)

        # Average accumulated metrics
        avg_loss = accum_loss / grad_accum_steps
        avg_stats = {k: v / grad_accum_steps for k, v in accum_stats.items()}

        # Update metric logger
        metric_logger.update(loss=avg_loss)
        metric_logger.update(grad_norm=grad_norm if math.isfinite(grad_norm) else 0.0)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Reduce metrics across all ranks
        loss_value_reduce = misc.all_reduce_mean(avg_loss)
        grad_norm_reduce = misc.all_reduce_mean(grad_norm)

        # All ranks must participate in all_reduce operations (collective ops)
        stats_reduced = {}
        for stat_name, stat_value in avg_stats.items():
            stats_reduced[stat_name] = misc.all_reduce_mean(stat_value)

        # Only rank 0 writes to tensorboard
        if log_writer is not None and (data_iter_step % print_freq == 0 or force_log_stats):
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('grad_norm', grad_norm_reduce, epoch_1000x)

            # Log per-scale stats
            for stat_name, stat_value_reduce in stats_reduced.items():
                log_writer.add_scalar(f'train/{stat_name}', stat_value_reduce, epoch_1000x)

        # Reset accumulators
        accum_loss = 0.0
        accum_stats = {}

    # zero the gradients after epoch ends
    if not is_accum_step:
        optimizer.zero_grad()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if misc.is_main_process():
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, device, epoch, log_writer, args, balance_labels=True, temperature=None, cfg=None, cfg_start_step=None):
    if args.eval_reference_data_path is None and not args.samples_only:
        raise ValueError("args.eval_reference_data_path must be specified for evaluation.")
    temperature = temperature if temperature is not None else args.temperature
    cfg = cfg if cfg is not None else args.cfg
    cfg_list = [cfg] if not isinstance(cfg, list) else cfg
    cfg_start_step = cfg_start_step if cfg_start_step is not None else args.cfg_start_step
    model.eval()
    samples_dir_root = os.path.join(args.samples_output_dir or args.output_dir, "samples")
    if epoch is not None:
        samples_dir_root += "_ep%03d" % epoch
    if misc.is_main_process() and not os.path.exists(samples_dir_root):
        os.makedirs(samples_dir_root)
    header = 'Test: [{}]'.format(epoch) if epoch is not None else 'Test:'
    assert misc.is_main_process(), "distributed eval generation not yet implemented"
    # sample images and save
    for cfg in cfg_list:
        # Clear samples directory at start of each cfg iteration
        samples_dir = samples_dir_root + "_cfg%.2f" % cfg
        if os.path.exists(samples_dir):
            shutil.rmtree(samples_dir)
        try:
            _evaluate_single_cfg(model, device, epoch, log_writer, args, balance_labels, temperature, cfg, cfg_start_step, samples_dir, header,
                               samples_only=args.samples_only,
                               )
        finally:
            # remove samples dir
            if os.path.exists(samples_dir) and not args.keep_samples:
                shutil.rmtree(samples_dir)

def _evaluate_single_cfg(model, device, epoch, log_writer, args, balance_labels, temperature, cfg, cfg_start_step, samples_dir, header,
                        samples_only=False):
    num_samples = 0
    assert misc.is_main_process(), "distributed eval generation not yet implemented"
    max_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    batch_size = min(max_batch_size, args.eval_num_samples)
    progress = tqdm.tqdm(total=args.eval_num_samples, desc=header + " CFG: %.2f" % cfg)
    #ref_sizes = list(map(int, os.listdir(args.eval_reference_data_path)))
    ref_sizes = [256]
    num_classes = model.class_embeddings.weight.shape[0] - 1  # -1 for unconditional class
    classes_rr = itertools.cycle(list(range(num_classes)))
    image_sizes = set()
    ref_sizes.sort()

    with ThreadPoolExecutor(max_workers=32) as writer_pool:
        def _write_image(img, filename):
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            img.save(filename)

        sample_ind = 0
        while num_samples < args.eval_num_samples:
            # Prepare conditioning if needed
            if balance_labels:
                labels = [next(classes_rr) for _ in range(batch_size)]
            else:
                labels = np.random.randint(0, num_classes, size=(batch_size,)).tolist()

            with torch.no_grad():
                sample_results = model.sample(batch_size=batch_size,
                                              temperature=temperature,
                                              labels=torch.as_tensor(labels).to(device),
                                              cfg=cfg,
                                              cfg_start_step=cfg_start_step,
                                              )
            samples_by_scale = [res[0] for res in sample_results]
            for samples in samples_by_scale:
                imsize = samples.shape[2]
                if imsize not in ref_sizes:
                    continue
                image_sizes.add(imsize)
                samples = (samples + 1) / 2
                samples = samples.clamp(0, 1)
                samples = samples.cpu().permute(0, 2, 3, 1).numpy()
                samples_uint8 = (samples * 255).astype(np.uint8)
                for sample in samples_uint8:
                    img = Image.fromarray(sample)
                    filename = os.path.join(samples_dir, str(imsize), "sample_%05d.png" % sample_ind)
                    writer_pool.submit(_write_image, img, filename)
                    sample_ind += 1
            num_samples += batch_size
            progress.update(batch_size)

    if samples_only:
        return

    # compute FID and IS
    for image_size in sorted(image_sizes):
        ref_path = os.path.join(args.eval_reference_data_path, str(image_size))
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=os.path.join(samples_dir, str(image_size)),
            input2=ref_path,
            input2_cache_name=ref_path.replace('/', '_'),  # cache reference stats
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=True,
            verbose=False,
        )
        # precision and recall swapped since input1 is gen, input2 is ref (reverse of torch_fidelity default)
        metrics_dict['precision'], metrics_dict['recall'] = metrics_dict['recall'], metrics_dict['precision']
        metrics_dict['epoch'] = epoch
        metrics_dict['cfg'] = cfg
        metrics_dict['image_size'] = image_size
        metrics_dict['num_samples'] = args.eval_num_samples
        metrics_dict['temperature'] = temperature
        metrics_dict['num_blocks_per_scale'] = model.num_blocks_per_scale
        metrics_dict['scale_ratio'] = model.scale_ratio
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        precision = metrics_dict.get('precision', 0.0)
        recall = metrics_dict.get('recall', 0.0)
        if log_writer is not None:
            log_writer.add_scalar('fid_%d_cfg%.2f' % (image_size, cfg), fid, epoch)
            log_writer.add_scalar('inception_score_%d_cfg%.2f' % (image_size, cfg), inception_score, epoch)
        print(header + "CFG: %.2f IMSIZE: %3d  FID: %.4f  IS: %.4f  P: %.4f  R: %.4f" % (cfg, image_size, fid, inception_score, precision, recall))
        # write metrics to a file
        if misc.is_main_process():
            with open(os.path.join(args.output_dir, "eval_metrics.txt"), "a") as f:
                if epoch is not None:
                    f.write("epoch %d  cfg %.2f  image size %d\n" % (epoch, cfg, image_size))
                f.write("CFG: %.2f IMSIZE: %3d  FID: %.4f  IS: %.4f  P: %.4f  R: %.4f\n" % (cfg, image_size, fid, inception_score, precision, recall))
                f.write("\n")
                f.write(str(metrics_dict))
                f.write("\n")


