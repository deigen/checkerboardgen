import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
import yaml

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from util.crop import center_crop_arr
from cbgen import ar
import copy

import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='', type=str, help='pretrained model')
    parser.add_argument('--use_ema', action='store_true', help='use EMA weights when loading')

    ## Generation parameters
    # TODO
    #parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    #parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--temperature', default=1.0, type=float, help='sampling temperature')
    #parser.add_argument('--class', default=None, type=int)

    parser.add_argument('--random_classes', action='store_true', help='use random classes for sampling')
    parser.add_argument('--output_dir', default='./samples')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cfg', type=float, default=2.0, help='classifier-free guidance scale')
    parser.add_argument('--cfg_start_step', type=int, default=5, help='starting step for cfg')
    parser.add_argument('--calculate_flops', action='store_true', help='calculate flops for one forward pass')
    parser.add_argument('--steps_per_scale', type=int, default=None, help='number of sampling steps at each scale')

    return parser


class ModelSampler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def forward(self, **kwargs):
        return self.model.sample(**kwargs)


def main(args):
    # fix the seed for reproducibility
    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    train_args_fn = os.path.join(os.path.dirname(args.checkpoint), 'args.yaml')
    if os.path.isfile(train_args_fn):
        train_args = yaml.safe_load(open(train_args_fn, 'r'))
    else:
        train_args = {}
    config_fn = os.path.join(os.path.dirname(args.checkpoint), 'config.yaml')
    if os.path.isfile(config_fn):
        config = yaml.safe_load(open(config_fn, 'r'))
    else:
        config = {}

    assert args.checkpoint, 'Checkpoint is required for sampling'
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    num_classes = checkpoint['model']['class_embeddings.weight'].shape[0] - 1  # -1 for unconditional class

    model = ar.AR(num_classes=num_classes, **config['model'])
    model.load_state_dict(checkpoint['model_ema' if args.use_ema else 'model'], strict=True)
    del checkpoint

    ## debug print trainable param count, group by top-level mod
    #param_counts = {}
    #for name, module in model.named_children():
    #    count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    #    param_counts[name] = count
    #for name, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
    #    print(f'Module {name}: {count/1e6:.2f}M params')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.steps_per_scale is not None:
        model.num_blocks_per_scale = args.steps_per_scale

    if args.random_classes:
        labels = np.random.randint(0, num_classes, size=(16,)).tolist()
    else:
        labels = [207, 360, 387, 974, 88, 979, 417, 279, 37, 30, 281, 985, 937, 350, 436, 483, ]
    print(labels)

    model = ModelSampler(model)  # needed for flops counting

    if args.calculate_flops:
        from util import torch_dispatch_flops
        flops_counter = torch_dispatch_flops.FlopCounterMode(model)
        with flops_counter:
            with torch.no_grad():
                labels = labels[:1]  # bsize 1 for flops calculation
                results = model(
                    batch_size=len(labels),
                    temperature=args.temperature,
                    labels=torch.as_tensor(labels).to(device),
                    cfg=args.cfg,
                    cfg_start_step=args.cfg_start_step,
                )
    else:
        results = model(
            batch_size=len(labels),
            temperature=args.temperature,
            labels=torch.as_tensor(labels).to(device),
            cfg=args.cfg,
            cfg_start_step=args.cfg_start_step,
        )

    # results is a list of length num_scales, each containing (img, selection_ts, value_prob_map, selection_entropy) for all B samples
    # Unpack per-scale data
    per_scale_results = results  # Keep original structure for detailed plotting

    model_stride = model.vae_stride

    samples, selection_order_maps, value_prob_maps, position_entropies = zip(*results)
    samples = vis_pyramid(samples, normalize_range_per_scale=False)
    selection_order_map = vis_pyramid(selection_order_maps, normalize_range_per_scale=True)
    value_prob_map = vis_pyramid(value_prob_maps, normalize_range_per_scale=False)
    position_entropy = vis_pyramid(position_entropies, normalize_range_per_scale=False)

    samples = samples.cpu()
    samples_uint8 = (((samples + 1) / 2).clamp(0, 1) * 255).type(torch.uint8)
    samples_uint8 = samples_uint8.permute(0, 2, 3, 1).cpu().numpy()  # B, H, W, C

    selection_order_map = selection_order_map.cpu().numpy()  # B, H, W
    value_prob_map = value_prob_map.cpu().numpy()  # B, H, W
    position_entropy = position_entropy.cpu().numpy()  # B, H, W

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mont = torchvision.utils.make_grid(
        torch.as_tensor(samples_uint8).permute(0,3,1,2),
        nrow=int(np.sqrt(len(samples))), padding=2, normalize=False
        ).cpu().numpy().transpose(1,2,0)
    Image.fromarray(mont).save(os.path.join(args.output_dir, f'samples_mont.png'))

    for i in range(samples.shape[0]):
        sample = samples_uint8[i]
        order_map = selection_order_map[i]  # TODO jet colormap
        prob_map = value_prob_map[i]
        entropy_map = position_entropy[i] / np.log(10)  # vis/clim with base 10 entropy

        Image.fromarray(sample).save(os.path.join(args.output_dir, f'sample_{i:04d}.png'))

        subi = 0
        nsub = 9
        subh = int(np.sqrt(nsub))
        subw = int(np.ceil(nsub / subh))
        plt.clf()

        subi += 1; plt.subplot(subh, subw, subi)
        plt.imshow(sample)
        plt.title('sampled')
        plt.axis('off')

        subi += 1; plt.subplot(subh, subw, subi)
        plt.imshow(order_map, cmap='jet')
        plt.title('selection order')
        plt.axis('off')

        subi += 1; plt.subplot(subh, subw, subi)
        plt.imshow(np.log10(prob_map), cmap='jet')
        plt.title('log10probs of selected values')
        plt.axis('off')
        #plt.clim([0, 1])
        plt.clim([np.log10(1e-4), 0])
        plt.colorbar()

        subi += 1; plt.subplot(subh, subw, subi)
        plt.imshow(entropy_map, cmap='jet')
        plt.title('position entropy')
        plt.axis('off')
        plt.clim([0, np.log10(model.vq_num_tokens)])
        plt.colorbar()

        # Plot selection order vs entropy per scale
        subi += 1; plt.subplot(subh, subw, subi)
        plot_selection_order_vs_entropy(selection_order_maps, position_entropies, i, model_stride)

        # save with increased size for better quality
        plt.gcf().set_size_inches(3*subw, 3*subh)
        plt.savefig(os.path.join(args.output_dir, f'sample_{i:04d}_maps.png'), bbox_inches='tight', pad_inches=0)



def plot_selection_order_vs_entropy(selection_orders, position_entropies, sample_idx, model_stride):
    """
    Plot selection order vs entropy with one line per scale for a single sample.
    This is designed to be called as a subplot in an existing figure.

    Args:
        selection_orders: List of tensors, one per scale. Each has shape (B, H, W)
        position_entropies: List of tensors, one per scale. Each has shape (B, H, W)
        sample_idx: Index of the sample to plot
    """
    num_scales = len(selection_orders)

    for scale_idx in range(num_scales):
        # Extract data for this sample and scale
        order_tensor = selection_orders[scale_idx]  # (B, H, W)
        entropy_tensor = position_entropies[scale_idx]  # (B, H, W)

        # Convert to numpy if needed
        if isinstance(order_tensor, torch.Tensor):
            order = order_tensor[sample_idx].cpu().numpy()  # H, W
        else:
            order = order_tensor[sample_idx]

        if isinstance(entropy_tensor, torch.Tensor):
            entropy = entropy_tensor[sample_idx].cpu().numpy()  # H, W
        else:
            entropy = entropy_tensor[sample_idx]

        # Flatten
        order_flat = order.flatten()
        entropy_flat = entropy.flatten() / np.log(10)  # Convert to log10

        # Group by unique order values and compute mean/percentiles
        unique_orders = np.unique(order_flat)
        mean_entropies = []
        p25_values = []
        p75_values = []

        for order_val in unique_orders:
            mask = order_flat == order_val
            entropies_at_order = entropy_flat[mask]
            mean_entropies.append(np.mean(entropies_at_order))
            p25_values.append(np.percentile(entropies_at_order, 25))
            p75_values.append(np.percentile(entropies_at_order, 75))

        mean_entropies = np.array(mean_entropies)
        p25_values = np.array(p25_values)
        p75_values = np.array(p75_values)

        # Infer scale size from shape
        label = f'Scale {2**scale_idx * model_stride}'

        # Plot mean line with markers
        line = plt.plot(unique_orders, mean_entropies,
                       label=label, linewidth=1.5, marker='o', markersize=3)[0]

        # Add shaded region for 25th-75th percentile range
        plt.fill_between(unique_orders, p25_values, p75_values,
                        alpha=0.2, color=line.get_color())

    plt.xlabel('Sampling order', fontsize=9)
    plt.ylabel('Entropy (log10)', fontsize=9)
    plt.title('Sample Order vs Entropy', fontsize=10)
    plt.legend(loc='best', fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=8)


def unnormalize_image(img):
    if img.ndim == 4:
        assert img.shape[0] == 1
        img = img[0]
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = (img + 1) / 2
    img = img.clip(0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def vis_pyramid(xs, normalize_range_per_scale=False):
    single_channel = xs[0].ndim == 3
    if single_channel:
        xs = [x.unsqueeze(1) for x in xs]
    shapes = torch.tensor([x.shape for x in xs])
    B, C, _, _ = shapes[0]
    assert torch.all(shapes[:, 0] == B) and torch.all(shapes[:, 1] == C)

    H_max = torch.max(shapes[:, 2])
    W_tot = torch.sum(shapes[:, 3])
    dtype = xs[0].dtype if not normalize_range_per_scale else torch.float32
    vis = torch.zeros(B, C, H_max, W_tot, dtype=dtype, device=xs[0].device)
    w0 = 0
    for i, x in enumerate(xs):
        _, _, H, W = x.shape
        if normalize_range_per_scale:
            xmin = x.amin(dim=(1, 2, 3), keepdim=True)
            xmax = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - xmin) / (xmax - xmin + 1e-5)
        vis[:, :, :H, w0:w0+W] = x
        w0 += W

    if single_channel:
        vis = vis[:, 0]
    return vis

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
