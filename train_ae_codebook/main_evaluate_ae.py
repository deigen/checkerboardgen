"""
Evaluate Autoencoder Reconstruction Quality

This script evaluates the reconstruction quality of a pretrained VQ-autoencoder
using the same metrics (FID and Inception Score) as engine.py:evaluate().

Usage:
    python main_evaluate_ae.py \
        --config configs/autoencoder_config.yaml \
        --checkpoint path/to/autoencoder.ckpt \
        --data_path ./data/imagenet/val \
        --num_samples 10000 \
        --output_dir ./outputs/ae_eval
"""

import argparse
import os
import shutil
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch_fidelity

from util.crop import center_crop_arr
import util.misc as misc
from cbgen.autoencoder import load_model as load_vae


def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluate VQ-Autoencoder Reconstruction')

    # Model parameters
    parser.add_argument('--config', type=str, required=True,
                        help='Path to autoencoder config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to autoencoder checkpoint')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/imagenet/val',
                        help='Path to evaluation dataset')
    parser.add_argument('--eval_reference_data_path', type=str)
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for evaluation')

    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to save evaluation results')
    parser.add_argument('--keep_samples', action='store_true',
                        help='Keep generated samples after evaluation')

    # Device parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use for evaluation')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')

    # Quantization parameters
    parser.add_argument('--use_quantization', type=int, default=1,
                        help='Use quantization in reconstruction (default: with quantization)')

    return parser


@torch.no_grad()
def evaluate_autoencoder(model, data_loader, args):
    """
    Evaluate autoencoder by reconstructing images and computing FID/IS metrics.
    """
    model.eval()

    # Create directories for original and reconstructed images
    samples_dir = os.path.join(args.output_dir, "samples")
    original_dir = os.path.join(samples_dir, "original", str(args.img_size))
    recon_dir = os.path.join(samples_dir, "reconstructed", str(args.img_size))

    for dir_path in [original_dir, recon_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print(f"Evaluating autoencoder reconstruction on {args.num_samples} samples...")
    print(f"Saving samples to: {samples_dir}")

    num_processed = 0
    sample_idx = 0

    with ThreadPoolExecutor(max_workers=8) as writer_pool:
        def _write_image(img, filename):
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            img.save(filename)

        progress = tqdm.tqdm(total=args.num_samples, desc="Processing images")

        for batch_idx, (images, labels) in enumerate(data_loader):
            if num_processed >= args.num_samples:
                break

            # Move to device
            images = images.to(args.device, non_blocking=True)
            batch_size = images.shape[0]

            # Reconstruct images through autoencoder
            recon_images, _ = model(images, requantize=bool(args.use_quantization))

            # Convert from [-1, 1] to [0, 1]
            images_save = (images + 1) / 2
            recon_images_save = (recon_images + 1) / 2

            # Clamp to valid range
            images_save = images_save.clamp(0, 1)
            recon_images_save = recon_images_save.clamp(0, 1)

            # Convert to numpy and save
            images_np = images_save.cpu().permute(0, 2, 3, 1).numpy()
            recon_np = recon_images_save.cpu().permute(0, 2, 3, 1).numpy()

            images_uint8 = (images_np * 255).astype(np.uint8)
            recon_uint8 = (recon_np * 255).astype(np.uint8)

            for i in range(batch_size):
                if num_processed >= args.num_samples:
                    break

                # Save original
                orig_img = Image.fromarray(images_uint8[i])
                orig_filename = os.path.join(original_dir, f"sample_{sample_idx:05d}.png")
                writer_pool.submit(_write_image, orig_img, orig_filename)

                # Save reconstruction
                recon_img = Image.fromarray(recon_uint8[i])
                recon_filename = os.path.join(recon_dir, f"sample_{sample_idx:05d}.png")
                writer_pool.submit(_write_image, recon_img, recon_filename)

                sample_idx += 1
                num_processed += 1

            progress.update(batch_size)

        progress.close()

    print(f"\nProcessed {num_processed} images")
    print("\nComputing FID and Inception Score metrics...")

    ref_dir = args.eval_reference_data_path if args.eval_reference_data_path else original_dir

    # Compute metrics comparing reconstructions to given reference or originals
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=recon_dir,
        input2=ref_dir,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )

    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']

    print(f"\n{'='*60}")
    print(f"Autoencoder Reconstruction Evaluation Results")
    print(f"{'='*60}")
    print(f"Image Size:        {args.img_size}")
    print(f"Number of Samples: {num_processed}")
    print(f"Quantization:      {'Enabled' if args.use_quantization else 'Disabled'}")
    print(f"{'='*60}")
    print(f"FID:               {fid:.4f}")
    print(f"Inception Score:   {inception_score:.4f}")
    print(f"{'='*60}\n")

    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, "ae_eval_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Autoencoder Reconstruction Evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Image Size:        {args.img_size}\n")
        f.write(f"Number of Samples: {num_processed}\n")
        f.write(f"Quantization:      {'Enabled' if args.use_quantization else 'Disabled'}\n")
        f.write("=" * 60 + "\n")
        f.write(f"FID:               {fid:.4f}\n")
        f.write(f"Inception Score:   {inception_score:.4f}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Full metrics:\n")
        f.write(str(metrics_dict))
        f.write("\n")

    print(f"Metrics saved to: {metrics_file}")

    # Clean up samples directory if requested
    if not args.keep_samples:
        print(f"\nRemoving samples directory: {samples_dir}")
        shutil.rmtree(samples_dir)
    else:
        print(f"\nSamples kept in: {samples_dir}")

    return metrics_dict


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    print("Autoencoder Evaluation Script")
    print("=" * 60)
    print(f"Config:      {args.config}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Data:        {args.data_path}")
    print(f"Samples:     {args.num_samples}")
    print(f"Batch Size:  {args.batch_size}")
    print("=" * 60 + "\n")

    # Set output directory
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint) if os.path.isfile(args.checkpoint) else args.checkpoint
        args.output_dir = os.path.join(checkpoint_dir, 'ae_eval')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Load autoencoder
    print("Loading autoencoder...")
    autoencoder = load_vae(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    autoencoder.to(device)
    autoencoder.eval()
    print("Autoencoder loaded successfully")

    # Count parameters
    n_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"Number of parameters: {n_params / 1e6:.2f}M\n")

    # Create dataset
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size, flip=False)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Support different dataset formats
    dataset_dirname, dataset_basename = os.path.split(os.path.abspath(args.data_path))
    if dataset_basename == 'celeba':
        dataset = datasets.CelebA(dataset_dirname, split='test', transform=transform, download=False)
    elif dataset_basename == 'food101':
        dataset = datasets.Food101(args.data_path, split='test', transform=transform, download=False)
    else:
        # Assume ImageFolder format
        dataset = datasets.ImageFolder(args.data_path, transform=transform)

    print(f"Dataset: {dataset}")
    print(f"Total images in dataset: {len(dataset)}\n")

    # Create dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # shuffle for subsampling
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Run evaluation
    metrics = evaluate_autoencoder(autoencoder, data_loader, args)

    print("\nEvaluation complete!")
    return metrics


if __name__ == '__main__':
    main()
