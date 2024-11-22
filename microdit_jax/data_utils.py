from jax import Array, numpy as jnp, random as jrand
import flax, cv2, jax, pickle, os
import numpy as np
from flax import nnx
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from collections import namedtuple
import flax.traverse_util
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze
from PIL import Image as pillow
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import AutoTokenizer, T5EncoderModel
from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np
import subprocess, torch
from typing import Optional, Union, List
from pathlib import Path


class config:
    vaescale_factor = 0.13025
    batch_size = 128
    img_size = 256
    seed = 33
    patch_size = (4, 4)
    lr = 1e-4
    mask_ratio = 0.75
    epochs = 5
    data_split = 10_000
    cfg_scale = 2.0
    vae_channels = 4
    celebv_id = 'SwayStar123/CelebV-HQ'
    finevid_id = "HuggingFaceFV/finevideo"
    pd12_id = 'Spawning/PD12M'
    vae_id = "madebyollin/sdxl-vae-fp16-fix"
    t5_id = "google-t5/t5-small"
    mini_data_id = 'uoft-cs/cifar10'
    imagenet_id = "ILSVRC/imagenet-1k"
    device_0 = jax.default_device()

randkey = jrand.key(config.seed)


# jax/numpy implementation of topk selection
def jnp_topk(array: Array, k: int):
    topk_tuple = namedtuple('topk', ['values', 'ids'])
    array = jnp.asarray(array)
    flat = array.ravel()

    sort_indices = jnp.argpartition(flat, -k)[-k:]
    argsort = jnp.argsort(-flat[sort_indices]) # get sorting ids

    sort_idx = sort_indices[argsort]
    values = flat[sort_idx]

    idx = jnp.unravel_index(sort_idx, array.shape)

    if len(idx) == 1:
        idx, = idx

    return topk_tuple(values=values, ids=idx)


# save model params in pickle file
def save_paramdict_pickle(model, filename="model.pkl"):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep=".")

    with open(filename, "wb") as f:
        pickle.dump(frozen_state_dict, f)

    return flat_state_dict


def load_paramdict_pickle(model, filename="model.pkl"):
    with open(filename, "rb") as modelfile:
        params = pickle.load(modelfile)

    params = flax.traverse_util.unflatten_dict(params, sep=".")
    params = unfreeze(params)
    params = from_state_dict(params)
    
    nnx.update(model, params)

    return params


def apply_mask(x: Array, mask, patch_size):
    # basically turns the masked values to 0s
    bs, c, h, w = x.shape
    numpatch_h = h // patch_size[0]
    numpatch_w = w // patch_size[1]

    mask = jnp.reshape(mask, shape=(bs, numpatch_h, numpatch_w))

    mask = jnp.expand_dims(mask, axis=1)
    mask = jnp.tile(mask, reps=(1, 1, patch_size[0], patch_size[1]))
    mask = jnp.reshape(mask, shape=(bs, 1, h, w))

    x_masked = x * mask

    return x_masked


def random_mask(bs, height, width, patch_size, mask_ratio):
    num_patches = (height // patch_size[0]) * (width // patch_size[1])
    num_patches_to_mask = int(num_patches * mask_ratio)

    rand_array = jrand.normal(randkey, shape=(bs, num_patches))
    indices = jnp.argsort(rand_array, axis=1)

    mask = jnp.ones(shape=(bs, num_patches))

    batch_mask_array = jnp.expand_dims(jnp.arange(bs), axis=1)
    # mask[batch_mask_array, indices[:, :num_patches_to_mask]] = 0
    new_mask = mask.at[batch_mask_array, indices[:, :num_patches_to_mask]].set(0)
    mask = new_mask
    mask = jnp.reshape(mask, shape=(bs, num_patches))

    return mask


def nearest_divisor(scaled_num_heads, embed_dim):
    # Find all divisors of embed_dim
    divisors = [k for k in range(1, embed_dim + 1) if embed_dim % k == 0]

    # Find the nearest divisor
    nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))

    return nearest


def remove_masked_patches(patches: Array, mask: Array):
    # Convert and invert mask
    mask = jnp.logical_not(mask)
    bs, num_patches, embed_dim = patches.shape

    # Method 1: Using take with nonzero
    # Reshape mask to 2D (combining batch and patches)
    mask_flat = mask.reshape(-1)
    indices = jnp.nonzero(mask_flat, size=mask.shape[1])[0]

    patches_flat = patches.reshape(-1, embed_dim)

    unmasked_patches = jnp.take(patches_flat, indices, axis=0)

    return unmasked_patches.reshape(bs, -1, embed_dim)


def add_masked_patches(patches: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    # Ensure mask is a boolean tensor
    mask = mask.astype(jnp.bool_)

    bs, num_patches, embed_dim = mask.shape[0], mask.shape[1], patches.shape[-1]

    full_patches = jnp.zeros((bs, num_patches, embed_dim), dtype=patches.dtype)

    full_patches = full_patches.at[mask].set(patches.reshape(-1, embed_dim))

    return full_patches


def display_samples(sample_batch):
    batch = np.array(sample_batch[-1])
    # Set up the grid
    batch_size = batch.shape[0]

    grid_size = int(np.ceil(np.sqrt(batch_size)))  # Square grid

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Plot each image
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            img = (batch[i] * 255).astype(np.uint8)
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.show()


# image grid
def save_image_grid(batch, file_path: str, grid_size=None):
    batch = np.array(batch[-1])
    # Determine grid size
    batch_size = batch.shape[0]
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(batch_size)))  # Square grid
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_size

    # Set up the grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Plot each image
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            img = (batch[i] * 255).astype(np.uint8)  # Scale image to 0-255
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)

    return file_path


def vae_encode(vae: AutoencoderKL, img_array: Array):
    tensor_img = torch.from_numpy(np.array(img_array))
    latent = vae.encode(tensor_img).latent_dist.sample().mul_(config.vaescale_factor)
    np_latent = jnp.asarray(latent.numpy())
    
    return np_latent

def rsync_checkpoint(
    source_path: Union[str, Path],
    remote_host: str,
    remote_path: str,
    ssh_key_path: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
    ssh_port: int = 22,
    compress: bool = True,
    dry_run: bool = False
) -> bool:
    """
    Syncs training checkpoints to a remote machine using rsync over SSH.
    
    Args:
        source_path (Union[str, Path]): Local path to sync (file or directory)
        remote_host (str): Remote host (e.g., 'username@hostname' or 'hostname')
        remote_path (str): Destination path on remote host
        ssh_key_path (Optional[str]): Path to SSH private key file
        exclude_patterns (Optional[List[str]]): Patterns to exclude from sync
        ssh_port (int): SSH port number (default: 22)
        compress (bool): Whether to compress data during transfer
        dry_run (bool): If True, show what would be transferred without actual transfer
    
    Returns:
        bool: True if sync was successful, False otherwise
    """
    try:
        # Convert source path to string if it's a Path object
        source_path = str(source_path)
        
        # Build the rsync command
        rsync_cmd = ["rsync"]
        
        # Add rsync options
        rsync_options = ["-avP"]  # archive mode and verbose
        
        if compress:
            rsync_options.append("-z")  # compression
        
        if dry_run:
            rsync_options.append("--dry-run")
        
        # Add SSH options
        ssh_options = f"ssh -p {ssh_port}"
        if ssh_key_path:
            ssh_key_path = os.path.expanduser(ssh_key_path)
            ssh_options += f" -i {ssh_key_path}"
        
        rsync_options.extend(["-e", ssh_options])
        
        # Add exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                rsync_options.extend(["--exclude", pattern])
        
        # Combine all parts of the command
        rsync_cmd.extend(rsync_options)
        rsync_cmd.extend([source_path, f"{remote_host}:{remote_path}"])
        
        # Setup logging
        print(f"Starting rsync with command: {' '.join(rsync_cmd)}")
        
        # Execute rsync
        result = subprocess.run(
            rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if successful
        if result.returncode == 0:
            print("Rsync completed successfully")
            if result.stdout:
                print(f"Rsync stdout: {result.stdout}")
            return True
        else:
            print(f"Rsync failed with return code {result.returncode}")
            print(f"Error message: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error during rsync: {str(e)}")
        return False

# T5 text encoder / VAE
# t5_tokenizer = AutoTokenizer.from_pretrained(config.t5_id)
# t5_model = T5EncoderModel.from_pretrained(config.t5_id)
# vae = AutoencoderKL.from_pretrained(config.vae_id)

# def text_t5_encode(text_input: str, tokenizer=t5_tokenizer, model=t5_model):
#     input_ids = tokenizer(text_input, return_tensors='np').input_ids  # Batch size 1
#     outputs = model(input_ids=input_ids)
#     last_hidden_states = outputs.last_hidden_state

#     return last_hidden_states


###########################################################
## data loading
image_data = load_dataset(config.mini_data_id, streaming=True, split='train', trust_remote_code=True).take(config.split)
vae = AutoencoderKL.from_pretrained(config.vae_id)

def load_image(url):
    image = pillow.open(urlopen(url=url))
    img_array = np.array(image)
    resized = cv2.resize(img_array, dsize=(config.img_size, config.img_size))

    return resized / 255.0


class Text2ImageDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.dataset = load_dataset(
            config.pd12_id, streaming=True,
            split='train', trust_remote_code=True
        ).take(config.split)

    def __len__(self):
        return config.data_split

    def __iter__(self):
        for sample in self.dataset:
            image = sample["img_url"] # type: ignore
            image = load_image(image)
            img_latents = vae_encode(vae, image)
            caption_encoded = text_t5_encode(sample["caption"]) # type: ignore

            image = jnp.array(img_latents)
            text_encoding = jnp.array(caption_encoded)

            yield image, text_encoding


class ImageClassData(IterableDataset):
    def __init__(self, dataset=image_data):
        super().__init__()
        self.dataset = load_dataset(
            config.imagenet_id, streaming=True, split="train", trust_remote_code=True
        ).take(config.split)

    def __len__(self):
        return config.data_split

    def __iter__(self):
        for sample in self.dataset:
            image = sample["img"]  # type: ignore
            image = load_image(image)
            
            img_latents = vae_encode(vae, image)

            image = jnp.array(img_latents)
            label = jnp.array(sample["label"])

            yield image, label


def jax_collate(batch):
    images, labels = zip(*batch)
    batch = (jnp.array(images), jnp.array(labels))
    batch = jax.tree_util.tree_map(jnp.array, batch)

    return batch


dataset = ImageClassData() # imagenet dataset

train_loader = DataLoader(dataset, batch_size=4, collate_fn=jax_collate)