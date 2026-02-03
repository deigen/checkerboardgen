import os
import random
import numpy as np
import json

import torch
import torchvision.datasets as datasets
from safetensors.torch import save_file, load_file


class DatasetWithIndex(torch.utils.data.Dataset):
    """
    Wrapper around a dataset to return the index along with data and label.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        items = self.base_dataset[index]
        return index, *items

def collate_with_index(batch):
    """
    Custom collate function to handle DatasetWithIndex output.

    Args:
        batch: list of tuples (index, data, label)

    Returns:
        indices: tensor of indices
        data_batch: batched data tensor
        labels: tensor of labels
    """
    indices, images, labels = zip(*batch)
    images = torch.stack(images)
    return indices, images, labels

@torch.no_grad()
def create_vq_cache_v2(
    cache_filepath,
    data_path,
    autoenc,
    img_size,
    batch_size=32,
    num_workers=16,
    device='cuda'
):
    """
    Create VQ encoding cache for entire dataset with TenCrop augmentations.

    Args:
        cache_filepath: Path to save the cache file (.safetensors format)
        data_path: Path to ImageFolder dataset (e.g., ./data/imagenet/train)
        autoenc: VQ autoencoder model (already loaded)
        img_size: Image size for crops
        batch_size: Batch size for processing
        num_workers: Number of worker threads for prefetching images from disk
        device: Device to run encoding on

    Cache format (v3):
        - Tensors saved to {cache_filepath}_shard_XXXX.safetensors (sharded safetensors)
        - Metadata saved to {cache_filepath}_metadata.json
        - Format: 2 tensors per entry (1 entry per image)
          - "{index}_q": flattened concatenation of all q_inds scales (includes 10 augs per scale)
          - "{index}_label": label tensor
        - Each q_inds scale has shape (num_augs=10, *spatial_dims)
        - Metadata contains shape information (q_shapes) and split indices
        - All tensors stored in int16 (q, label)
    """
    from tqdm import tqdm
    from util.crop import center_crop_arr
    import torchvision.transforms as transforms

    print(f"Creating VQ cache at: {cache_filepath}")
    os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)

    # Move autoencoder to device and set to eval mode
    autoenc = autoenc.to(device)
    autoenc.eval()

    # Cache dictionary
    cache_dict = {}

    # Create base dataset
    base_dataset = datasets.ImageFolder(data_path)
    total_images = len(base_dataset)
    total_entries = total_images  # 1 entry per image (with 10 augs stored inside)

    print(f"Processing {total_images} images with 10 augmentations each = {total_entries} cache entries")


    with tqdm(total=total_entries, desc="Creating cache") as pbar:
        # Define transforms (same as uncached training path)
        crop_size = int(img_size * 1.1)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_img: center_crop_arr(pil_img, crop_size, flip=False)),
            transforms.TenCrop(img_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # batch dim is crops/flips of each image
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # Create dataset
        dataset = DatasetWithIndex(datasets.ImageFolder(data_path, transform=transform))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, collate_fn=collate_with_index)

        for indices, images, labels in loader:

            assert isinstance(indices[0], int), "Indices should be integers"
            assert isinstance(labels[0], int), "Labels should be integers"
            assert images.dim() == 5, "Images should be a 5D tensor (batch, crops, C, H, W)"

            num_batch, num_augs = images.shape[:2]

            # Run VQ encoding
            images = images.to(device).reshape(-1, 3, img_size, img_size)  # reshape to (B*10, C, H, W)
            with torch.cuda.amp.autocast(enabled=False):
                pyramid = autoenc.quantization_pyramid(images)

            q_inds = pyramid['q_inds']  # list of tensors
            q_inds_cpu = [q.short().cpu() for q in q_inds]
            q_inds_cpu = [q.reshape(num_batch, num_augs, *q.shape[1:]) for q in q_inds_cpu]

            # Store in cache (convert to fp16/int16, move to CPU)
            for b in range(num_batch):
                index = indices[b]
                label = labels[b]
                q_inds_i = [q[b] for q in q_inds_cpu]
                cache_dict[index] = (q_inds_i, label)

            pbar.update(len(indices))

    # Flatten cache dict and write shards (~1GB each)
    print(f"Flattening cache structure and writing shards...")

    # Get num_scales from first item (all items have same num_scales)
    first_key = next(iter(cache_dict))
    num_scales = len(cache_dict[first_key][0])  # q_inds is first element of tuple

    # Extract shapes from first item for metadata
    # q_shapes includes crop dimension: (num_augs, *spatial_dims)
    first_q_inds, _ = cache_dict[first_key]
    q_shapes = [tuple(q.shape) for q in first_q_inds]

    # Compute split indices for flattened tensors
    q_numel_per_scale = [np.prod(shape) for shape in q_shapes]
    q_split_indices = np.cumsum(q_numel_per_scale)[:-1].tolist()

    print(f"  q_shapes (with aug dim): {q_shapes}")
    print(f"  q flattened total elements: {sum(q_numel_per_scale)}")

    # Shard configuration
    # we have 2 tensors per entry (q_flat, label)
    # Use 100k entries per shard -> 200k tensors
    max_entries_per_shard = 100000

    shard_idx = 0
    shard_tensors = {}
    shard_tensor_count = 0
    shard_size_bytes = 0
    shard_info = []  # List of shard metadata
    shard_map = {}  # Maps "{index}" -> shard_idx (using global indices in keys)

    # Sort cache_dict by index to ensure consistent ordering
    sorted_keys = sorted(cache_dict.keys())
    current_shard_entries = 0

    for entry_num, index in enumerate(sorted_keys):
        q_inds, label = cache_dict[index]

        # Use global index for unique naming across shards
        prefix = f"{index}"

        # Record mapping: index -> shard_idx
        shard_map[f"{index}"] = shard_idx

        # Flatten and concatenate all scales into single tensors
        # q_inds includes crop dimension: list of (num_augs, *spatial_dims)
        q_flat = torch.cat([q.flatten() for q in q_inds])      # [total_q_elements]

        # Add flattened tensors to current shard
        entry_size = 0
        entry_tensor_count = 0

        shard_tensors[f"{prefix}_q"] = q_flat
        entry_size += q_flat.numel() * q_flat.element_size()
        entry_tensor_count += 1

        # Store label as tensor
        label_tensor = torch.tensor([label], dtype=torch.int16)
        shard_tensors[f"{prefix}_label"] = label_tensor
        entry_size += label_tensor.numel() * label_tensor.element_size()
        entry_tensor_count += 1

        shard_size_bytes += entry_size
        shard_tensor_count += entry_tensor_count
        current_shard_entries += 1

        # Check if we should write this shard (reached entry count threshold or last entry)
        is_last_entry = (entry_num == len(sorted_keys) - 1)
        should_write_shard = current_shard_entries >= max_entries_per_shard or is_last_entry

        if should_write_shard and len(shard_tensors) > 0:
            # Write shard file
            shard_filename = cache_filepath.replace('.safetensors', f'_shard_{shard_idx:04d}.safetensors')
            if not cache_filepath.endswith('.safetensors'):
                shard_filename = f"{cache_filepath}_shard_{shard_idx:04d}.safetensors"

            print(f"  Writing shard {shard_idx}: {shard_tensor_count} tensors, {shard_size_bytes / (1024**3):.2f} GB, {current_shard_entries} entries")
            save_file(shard_tensors, shard_filename)

            # Record shard info
            shard_info.append({
                'shard_idx': shard_idx,
                'num_entries': current_shard_entries,
                'num_tensors': shard_tensor_count,
                'size_bytes': shard_size_bytes
            })

            # Reset for next shard
            shard_idx += 1
            shard_tensors = {}
            shard_tensor_count = 0
            shard_size_bytes = 0
            current_shard_entries = 0

    # Save metadata separately as JSON
    metadata = {
        'num_images': total_images,
        'num_scales': num_scales,
        'num_classes': len(base_dataset.classes),
        'classes': base_dataset.classes,
        'class_to_idx': base_dataset.class_to_idx,
        'num_shards': shard_idx,
        'max_entries_per_shard': max_entries_per_shard,
        'shard_info': shard_info,
        'shard_map': shard_map,  # Maps "{index}" -> shard_idx
        # Shape information for reconstructing pyramids from flattened tensors
        'q_shapes': q_shapes,            # List of tuples: shape of q at each scale (includes aug dim: num_augs, *spatial)
        'q_split_indices': q_split_indices,    # Split points for torch.split() on flattened q
    }

    metadata_path = cache_filepath.replace('.safetensors', '_metadata.json')
    if not cache_filepath.endswith('.safetensors'):
        metadata_path = cache_filepath + '_metadata.json'

    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print cache size summary
    total_cache_size_mb = sum(info['size_bytes'] for info in shard_info) / (1024 * 1024)
    total_tensors = sum(info['num_tensors'] for info in shard_info)
    total_entries = sum(info['num_entries'] for info in shard_info)
    metadata_size_kb = os.path.getsize(metadata_path) / 1024
    print(f"Cache created successfully!")
    print(f"  Total shards: {shard_idx}")
    print(f"  Total entries: {total_entries} (max {max_entries_per_shard} per shard)")
    print(f"  Total tensors: {total_tensors} ({total_tensors // total_entries} per entry)")
    print(f"  Total size: {total_cache_size_mb:.2f} MB")
    print(f"  Metadata: {metadata_size_kb:.2f} KB")

    return cache_filepath


def cached_vq_collate_fn(batch):
    """
    Custom collate function for CachedVQDataset (v3 format).

    Args:
        batch: list of tuples ((q_inds,), label)

    Returns:
        batch_cache: tuple of (q_inds_batch,) where q_inds_batch is a list of batched tensors
        batch_labels: tensor of labels
    """
    # Separate cache data and labels
    cache_data_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Extract q_inds from each sample (cache_data is (q_inds,) tuple)
    q_inds_list = [cache_data[0] for cache_data in cache_data_list]

    # Batch each scale separately
    num_scales = len(q_inds_list[0])
    q_inds_batch = []

    for scale_i in range(num_scales):
        # Stack all samples for this scale
        q_scale = torch.stack([sample[scale_i] for sample in q_inds_list])
        q_inds_batch.append(q_scale)

    # Convert labels to tensor
    batch_labels = torch.tensor(labels, dtype=torch.long)

    # Return in format: ((q_inds_batch,), labels)
    return (q_inds_batch,), batch_labels


class CachedVQDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-computed VQ encodings from sharded cache.

    Returns cached quantization pyramid data instead of images.
    """
    def __init__(self, cache_filepath):
        """
        Args:
            cache_filepath: Path to the cache file base (.safetensors)
        """
        print(f"Loading VQ cache from: {cache_filepath}")

        # Load metadata
        metadata_path = cache_filepath.replace('.safetensors', '_metadata.json')
        if not cache_filepath.endswith('.safetensors'):
            metadata_path = cache_filepath + '_metadata.json'

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.num_images = metadata['num_images']
        self.num_scales = metadata['num_scales']
        self.num_classes = metadata['num_classes']
        self.classes = metadata['classes']
        self.class_to_idx = metadata['class_to_idx']
        self.num_shards = metadata['num_shards']
        self.shard_info = metadata['shard_info']
        self.shard_map = metadata['shard_map']

        # Load shape information for reconstructing pyramids
        # q_shapes includes aug dimension: (num_augs=10, *spatial_dims) per scale
        self.q_shapes = [tuple(shape) for shape in metadata['q_shapes']]
        self.q_split_indices = metadata['q_split_indices']

        # Load all shard files into memory
        print(f"Loading {self.num_shards} shard files into memory...")
        self.shards = []
        for shard_idx in range(self.num_shards):
            shard_filename = cache_filepath.replace('.safetensors', f'_shard_{shard_idx:04d}.safetensors')
            if not cache_filepath.endswith('.safetensors'):
                shard_filename = f"{cache_filepath}_shard_{shard_idx:04d}.safetensors"

            print(f"  Loading shard {shard_idx}...")
            shard_tensors = load_file(shard_filename)
            self.shards.append(shard_tensors)

        print(f"Cache loaded: {self.num_images} images, {self.num_classes} classes, {self.num_scales} scales")
        print(f"Total cache entries: {self.num_images} (10 augs per image)")
        print(f"Loaded {self.num_shards} shards")

    @staticmethod
    def get_collate_fn():
        """Return the custom collate function for this dataset."""
        return cached_vq_collate_fn

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        """
        Returns:
            cache_data: tuple of (q_inds,) where
                q_inds: list of tensors (code indices at each scale)
            label: class label (int)
        """
        # Randomly choose one of 10 augmentations
        # TODO put num augs in the metadata file
        aug_idx = random.randint(0, 9)

        # Look up which shard contains this index
        map_key = f"{index}"
        shard_idx = self.shard_map[map_key]

        # Get the shard
        shard_tensors = self.shards[shard_idx]

        # Load flattened tensors using global index
        prefix = f"{index}"
        q_flat = shard_tensors[f"{prefix}_q"]
        label = shard_tensors[f"{prefix}_label"].item()

        # Split flattened tensors into per-scale chunks
        if len(self.q_split_indices) > 0:
            q_chunks = torch.tensor_split(q_flat, self.q_split_indices)
        else:
            q_chunks = [q_flat]

        # Reshape each chunk to original scale shape (includes aug dimension)
        # q_shapes[i] = (num_augs=10, *spatial_dims)
        q_inds = []

        for scale_i in range(self.num_scales):
            # Reshape to (num_augs, *spatial_dims)
            q = q_chunks[scale_i].reshape(self.q_shapes[scale_i])

            # Select the chosen augmentation: (spatial_dims,)
            q_selected = q[aug_idx]

            # Convert back to int64 for processing
            q_inds.append(q_selected.long())

        # Return as tuple to match (images, labels) interface
        # New format: only q_inds, no zq_pyr
        cache_data = (q_inds,)
        return cache_data, label

