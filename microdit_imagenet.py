"""
Training a MicroDIT model, ~300M params, 24-depth, 12 heads config for on Imagenet
"""

import jax, optax
from jax import Array, numpy as jnp, random as jrand
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
jax.config.update("jax_default_matmul_precision", "bfloat16")

import numpy as np
from flax import nnx
import flax.traverse_util
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze
from einops import rearrange

import os, wandb, time, pickle, gc, click
import math, flax, torch, streaming
from tqdm.auto import tqdm
from typing import List, Any
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from streaming.base.format.mds.encodings import Encoding, _encodings
from streaming import StreamingDataset

import warnings
warnings.filterwarnings("ignore")


class config:
    batch_size = 128
    img_size = 32
    seed = 222
    patch_size = (2, 2)
    lr = 2e-4
    mask_ratio = 0.75
    epochs = 30
    data_split = 20_000 #imagenet split to train on
    cfg_scale = 3.0
    vaescale_factor = 0.13025


# XLA/JAX flags
JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"

# keys/env seeds
rkey = jrand.key(config.seed)
randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)

# mesh / sharding configs
num_devices = jax.device_count()
devices = jax.devices()
num_processes = jax.process_count()
rank = jax.process_index()


print(f"found {num_devices} JAX device(s)")
for device in devices:
    print(f"{device} / ")

mesh_devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(mesh_devices, axis_names=("data",))
data_sharding = NS(mesh, PS("data"))
rep_sharding = NS(mesh, PS())

# sd VAE for decoding latents
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
print("loaded vae")


def apply_mask(x: jnp.ndarray, mask: jnp.ndarray, patch_size: tuple[int, int]) -> Array:
    """
    Applies a mask to a tensor. Turns the masked values to 0s.

    Args:
        x (jnp.ndarray): Tensor of shape (bs, c, h, w)
        mask (jnp.ndarray): Tensor of shape (bs, num_patches)
        patch_size (tuple[int, int]): Size of each patch.

    Returns:
        jnp.ndarray: Tensor of shape (bs, c, h, w) with the masked values turned to 0s.
    """
    bs, c, h, w = x.shape
    num_patches_h = h // patch_size[0]
    num_patches_w = w // patch_size[1]

    # Ensure that height and width are divisible by patch_size
    assert (
        h % patch_size[0] == 0 and w % patch_size[1] == 0
    ), "Height and width must be divisible by patch_size. Height: {}, Width: {}, Patch size: {}".format(
        h, w, patch_size
    )

    # Reshape mask to (bs, num_patches_h, num_patches_w)
    mask = mask.reshape(bs, num_patches_h, num_patches_w)

    # Expand the mask to cover each patch
    # (bs, num_patches_h, num_patches_w) -> (bs, 1, h, w)
    mask = jnp.expand_dims(mask, axis=1)  # Add channel dimension

    # Repeat for patch_size
    mask = jnp.repeat(mask, patch_size[0], axis=2)
    mask = jnp.repeat(mask, patch_size[1], axis=3)

    # Reshape to (bs, 1, h, w) - not strictly necessary after the repeats if dims are correct
    # mask = mask.reshape(bs, 1, h, w)

    # Apply the mask to the input tensor
    x = x * mask

    return x


def random_mask(
    bs: int, height: int, width: int, patch_size: tuple[int, int], mask_ratio: float
) -> Array:
    """
    Generates a random mask for patched images. Randomly selects patches to mask.

    Args:
        bs (int): Batch size.
        height (int): Height of the image.
        width (int): Width of the image.
        patch_size (tuple of int): Size of the patches.
        mask_ratio (float): Ratio of patches to mask. Ranges from 0 to 1. mask_ratio * 100 = percentage of 1s in the mask

    Returns:
        mask (Array): A tensor of shape (bs, num_patches) with values in {0, 1}.
    """
    num_patches = (height // patch_size[0]) * (width // patch_size[1])
    num_patches_to_mask = int(num_patches * mask_ratio)

    # Create a tensor of random values
    rand_tensor = random.uniform(randkey, (bs, num_patches))

    # Sort the random tensor and get the indices
    indices = jnp.argsort(rand_tensor, axis=1)

    # Create a mask tensor initialized with ones
    mask = jnp.ones((bs, num_patches), dtype=jnp.int32)

    # Set the first num_patches_to_mask indices to 0 for each batch
    row_indices = jnp.arange(bs)[:, None]
    col_indices = indices[:, :num_patches_to_mask]

    # Use jax.numpy.where to conditionally set elements to 0
    mask = mask.at[row_indices, col_indices].set(0)

    return mask

def remove_masked_patches(patches: Array, mask: Array) -> Array:
    """
    Removes the masked patches from the patches tensor while preserving batch dimensions.
    Returned tensor will have shape (bs, number_of_unmasked_patches, embed_dim).
    """
    mask = mask.astype(bool)
    mask = jnp.logical_not(mask)

    # Get batch size and embed dimension
    bs, num_patches, embed_dim = patches.shape

    # Use boolean indexing to select unmasked patches for each batch
    unmasked_patches_list = []
    for i in range(bs):
        unmasked_patches_list.append(patches[i][mask[i]])

    # Find the maximum number of unmasked patches across all batches
    max_unmasked_patches = max(map(len, unmasked_patches_list))

    # Pad the unmasked patches with zeros to have a consistent shape
    padded_unmasked_patches = []
    for unmasked in unmasked_patches_list:
        pad_width = max_unmasked_patches - unmasked.shape[0]
        padding = [(0, pad_width), (0, 0)]  # Pad along the num_patches dimension
        padded_unmasked_patches.append(jnp.pad(unmasked, padding))

    # Stack the padded unmasked patches
    unmasked_patches = jnp.stack(padded_unmasked_patches)

    return unmasked_patches


def add_masked_patches(patches: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Adds the masked patches to the patches tensor.
    Returned tensor will have shape (bs, num_patches, embed_dim).
    The missing patches will be filled with 0s.
    """
   # Ensure mask is a boolean array
    mask = mask.astype(bool)

    # Get the total number of patches and embed dimension
    bs, num_patches, embed_dim = mask.shape[0], mask.shape[1], patches.shape[-1]

    # Create a tensor of zeros with the same shape and dtype as the patches tensor
    full_patches = jnp.zeros((bs, num_patches, embed_dim), dtype=patches.dtype)

    # Iterate over each batch and place unmasked patches back in their original positions
    for i in range(bs):
        full_patches = full_patches.at[i, mask[i]].set(patches[i])

    return full_patches


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


_encodings["uint8"] = uint8
remote_train_dir = "./vae_mds"  # this is the path you installed this dataset.
local_train_dir = "./imagenet" # just a local mirror path 

def jax_collate(batch):
    latents = jnp.stack([jnp.array(item["vae_output"]) for item in batch], axis=0)
    labels = jnp.stack([int(item["label"]) for item in batch], axis=0)

    return {
        "vae_output": latents,
        "label": labels,
    }


# model helper functions
# modulation with shift and scale
def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        axis=0,
    )


def get_2d_sincos_pos_embed(embed_dim, length):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length**0.5)
    assert grid_size * grid_size == length

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)  # (1, H*W, D)


xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)
const_init1 = nnx.initializers.constant(1)
linear_init = nnx.initializers.xavier_uniform()
linear_bias_init = nnx.initializers.constant(1)

# input patchify layer, 2D image to patches
class PatchEmbed:
    def __init__(
        self, in_channels=4, img_size: int = 32, dim=1024, patch_size: int = 2
    ):
        self.dim = dim
        self.patch_size = patch_size
        patch_tuple = (patch_size, patch_size)
        self.num_patches = (img_size // self.patch_size) ** 2
        self.conv_project = nnx.Conv(
            in_channels,
            dim,
            kernel_size=patch_tuple,
            strides=patch_tuple,
            use_bias=False,
            padding="VALID",
            kernel_init=xavier_init,
            rngs=rngs,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        num_patches_side = H // self.patch_size
        x = self.conv_project(x)  # (B, P, P, hidden_size)
        x = rearrange(x, "b h w c -> b (h w) c", h=num_patches_side, w=num_patches_side)
        return x

class TimestepEmbedder(nnx.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.lin_1 = nnx.Linear(freq_embed_size, hidden_size, rngs=rngs)
        self.lin_2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(time_array: Array, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half) / half)

        args = jnp.float_(time_array[:, None]) * freqs[None]

        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concat(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )

        return embedding

    def __call__(self, x: Array) -> Array:
        t_freq = self.timestep_embedding(x, self.freq_embed_size)
        t_embed = nnx.silu(self.lin_1(t_freq))

        return self.lin_2(t_embed)


class LabelEmbedder(nnx.Module):
    def __init__(self, num_classes, hidden_size, drop):
        super().__init__()
        use_cfg_embeddings = drop > 0
        self.embedding_table = nnx.Embed(
            num_classes + int(use_cfg_embeddings),
            hidden_size,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(0.02),
        )
        self.num_classes = num_classes
        self.dropout = drop

    def token_drop(self, labels, force_drop_ids=None) -> Array:
        if force_drop_ids is None:
            drop_ids = jrand.normal(key=randkey, shape=labels.shape[0])
        else:
            drop_ids = force_drop_ids == 1

        labels = jnp.where(drop_ids, self.num_classes, labels)

        return labels

    def __call__(self, labels, train: bool = True, force_drop_ids=None) -> Array:
        use_drop = self.dropout > 0
        if (train and use_drop) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        label_embeds = self.embedding_table(labels)

        return label_embeds


class CaptionEmbedder(nnx.Module):
    def __init__(self, cap_embed_dim, embed_dim):
        super().__init__()
        self.linear_1 = nnx.Linear(cap_embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = nnx.silu(self.linear_1(x))
        x = self.linear_2(x)

        return x


########################
# Patch Mixer components
########################
class EncoderMLP(nnx.Module):
    def __init__(self, hidden_size, rngs: nnx.Rngs, dropout=0.1):
        super().__init__()
        
        self.layernorm = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.linear1 = nnx.Linear(
            hidden_size,
            2 * hidden_size,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
        )
        self.linear2 = nnx.Linear(
            2 * hidden_size,
            hidden_size,
            rngs=rngs,
            bias_init=const_init1,
            kernel_init=linear_init,
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x_input: jax.Array) -> jax.Array:
        x = self.layernorm(x_input)
        x = nnx.silu(self.linear1(x))
        x = self.linear2(x)

        return x


class TransformerEncoderBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layernorm = nnx.LayerNorm(
            embed_dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init
        )

        self.self_attention =  nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            decode=False,
            rngs=rngs,
        ) # SelfAttention(num_heads, embed_dim, rngs=rngs)
        self.mlp_layer = EncoderMLP(embed_dim, rngs=rngs)

    def __call__(self, x: Array):
        x = x + self.self_attention(self.layernorm(x))
        x = x + self.mlp_layer(self.layernorm(x))
        return x


class MLP(nnx.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super().__init__()
        hidden = int(mlp_ratio * embed_dim)
        self.linear_1 = nnx.Linear(embed_dim, hidden, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init)
        self.linear_2 = nnx.Linear(
            hidden, embed_dim, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.gelu(self.linear_1(x))
        x = self.linear_2(x)

        return x


# Pool + MLP for (MHA + MLP)
class PoolMLP(nnx.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear_1 = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            kernel_init=xavier_init,
            bias_init=zero_init
        )
        self.linear_2 = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            kernel_init=xavier_init,
            bias_init=zero_init,
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.avg_pool(x, (1,))
        x = jnp.reshape(x, shape=(x.shape[0], -1))
        x = nnx.gelu(self.linear_1(x))
        x = self.linear_2(x)

        return x


###############
# DiT blocks_ #
###############
class DiTBlock(nnx.Module):
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4, drop: float = 0.0, rngs=rngs):
        super().__init__()
        self.norm_1 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs, bias_init=zero_init)
        self.norm_2 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, bias_init=zero_init
        )
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            decode=False,
            dropout_rate=drop,
            rngs=rngs,
        )
        self.adaln = nnx.Linear(
            hidden_size, 6 * hidden_size, 
            kernel_init=zero_init, bias_init=zero_init,
            rngs=rngs
        )
        self.mlp_block = MLP(hidden_size, mlp_ratio)

    def __call__(self, x_img: Array, cond: Array):
        cond = self.adaln(nnx.silu(cond))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.array_split(cond, 6, axis=-1)
        )

        attn_x = self.attention(modulate(self.norm_1(x_img), shift_msa, scale_msa))
        x = x_img + (jnp.expand_dims(gate_msa, 1) * attn_x)

        x = modulate(self.norm_2(x), shift_mlp, scale_mlp)
        mlp_x = self.mlp_block(x)
        x = x + (jnp.expand_dims(gate_mlp, 1) * mlp_x)

        return x


class FinalMLP(nnx.Module):
    def __init__(
        self, hidden_size, patch_size, out_channels, rngs=nnx.Rngs(config.seed)
    ):
        super().__init__()

        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size,
            patch_size[0] * patch_size[1] * out_channels,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=zero_init,
        )

        self.adaln_linear = nnx.Linear(
            hidden_size,
            2 * hidden_size,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=zero_init,
        )

    def __call__(self, x_input: Array, cond: Array):
        linear_cond = nnx.silu(self.adaln_linear(cond))
        shift, scale = jnp.array_split(linear_cond, 2, axis=-1)

        x = modulate(self.norm_final(x_input), shift, scale)
        x = self.linear(x)

        return x


class TransformerBackbone(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        class_embed_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.input_embedding = nnx.Linear(
            input_dim,
            embed_dim,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
        )
        self.class_embedding = nnx.Linear(
            class_embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
        )

        # # Define scaling ranges for m_f and m_a
        mf_min, mf_max = 0.5, 4.0
        ma_min, ma_max = 0.5, 1.0

        self.layers = []

        for v in tqdm(range(num_layers)):
            # # Calculate scaling factors for the v-th layer using linear interpolation
            mf = mf_min + (mf_max - mf_min) * v / (num_layers - 1)
            ma = ma_min + (ma_max - ma_min) * v / (num_layers - 1)

            # # Scale the dimensions according to the scaling factors
            scaled_mlp_dim = int(mlp_dim * mf)
            scaled_num_heads = max(1, int(num_heads * ma))
            scaled_num_heads = self._nearest_divisor(scaled_num_heads, embed_dim)
            mlp_ratio = scaled_mlp_dim / embed_dim

            self.layers.append(DiTBlock(embed_dim, scaled_num_heads, mlp_ratio))

        self.output_layer = nnx.Linear(
            embed_dim,
            input_dim,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
        )

    def _nearest_divisor(self, scaled_num_heads, embed_dim):
        # Find all divisors of embed_dim
        divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
        # Find the nearest divisor
        nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))

        return nearest

    def __call__(self, x, c_emb):
        x = self.input_embedding(x)
        class_emb = self.class_embedding(c_emb)

        for layer in self.layers:
            x = layer(x, class_emb)

        x = self.output_layer(x)

        return x


# patch mixer module
class PatchMixer(nnx.Module):
    def __init__(self, embed_dim, attn_heads, n_layers=2):
        super().__init__()
        layers = [
            TransformerEncoderBlock(embed_dim, attn_heads) for _ in range(n_layers)
        ]
        self.encoder_layers = nnx.Sequential(*layers)

    def __call__(self, x: Array) -> Array:
        x = self.encoder_layers(x)
        return x


#####################
# Microdit model
#####################
class MicroDiT(nnx.Module):
    def __init__(
        self,
        in_channels=4,
        patch_size=(2, 2),
        embed_dim=1024,
        num_layers=12,
        attn_heads=16,
        dropout=0.0,
        patchmix_layers=4,
        rngs=rngs,
        num_classes=1000,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedder = PatchEmbed(
            rngs=rngs, patch_size=patch_size[0], 
            in_chan=in_channels, embed_dim=embed_dim
        )

        self.num_patches = self.patch_embedder.num_patches

        # conditioning layers
        self.time_embedder = TimestepEmbedder(embed_dim)
        self.label_embedder = LabelEmbedder(
            num_classes=num_classes, hidden_size=embed_dim, drop=dropout
        )
        self.cond_attention = nnx.MultiHeadAttention(
            num_heads=attn_heads,
            in_features=embed_dim,
            kernel_init=xavier_init,
            rngs=rngs,
            decode=False
        )  # CrossAttention(attn_heads, embed_dim, cond_embed_dim, rngs=rngs)
        self.cond_mlp = MLP(embed_dim, mlp_ratio=4)

        # pooling layer
        self.pool_mlp = PoolMLP(embed_dim)

        self.linear = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)

        self.patch_mixer = PatchMixer(embed_dim, attn_heads, patchmix_layers)

        self.backbone = TransformerBackbone(
            embed_dim,
            embed_dim,
            embed_dim,
            embed_dim,
            num_layers=num_layers,
            num_heads=attn_heads,
        )

        self.final_linear = FinalMLP(embed_dim, patch_size=patch_size, out_channels=4)

    def _unpatchify(self, x, patch_size=(2, 2), height=32, width=32):
        bs, num_patches, patch_dim = x.shape
        H, W = patch_size
        in_channels = patch_dim // (H * W)
        # Calculate the number of patches along each dimension
        num_patches_h, num_patches_w = height // H, width // W

        # Reshape x to (bs, num_patches_h, num_patches_w, H, W, in_channels)
        x = x.reshape((bs, num_patches_h, num_patches_w, H, W, in_channels))

        # transpose x to (bs, num_patches_h, H, num_patches_w, W, in_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # Reshape x to (bs, height, width, in_channels)
        reconstructed = x.reshape((bs, height, width, in_channels))
        return reconstructed

    def __call__(self, x: Array, t: Array, y_cap: Array, mask=None, train=False):
        bsize, height, width, channels = x.shape
        x = self.patch_embedder(x)

        pos_embed = get_2d_sincos_pos_embed(embed_dim=self.dim, length=self.num_patches)
        print(f'posembed 1 = {pos_embed.shape}')
        pos_embed = jnp.expand_dims(pos_embed, axis=0)
        pos_embed = jnp.broadcast_to(pos_embed, (bsize, pos_embed.shape[1], pos_embed.shape[2]))
        pos_embed = jnp.reshape(
            pos_embed, shape=(bsize, self.num_patches, self.embed_dim)
        )
        print(f'{x.shape = } / {pos_embed.shape = }')

        x = x + pos_embed

        # cond_embed = self.cap_embedder(y_cap) # (b, embdim)
        label_embed = self.label_embedder(y_cap, train=train)
        print(f'label_embed = {label_embed.shape}')
        # label_embed_unsqueeze = jnp.expand_dims(label_embed, axis=1)

        time_embed = self.time_embedder(t)
        # time_embed_unsqueeze = jnp.expand_dims(time_embed, axis=1)

        # mha_out = self.cond_attention(time_embed_unsqueeze, cond_embed_unsqueeze)
        # mha_out = mha_out.squeeze(1)
        cond_ty = time_embed + label_embed
        mlp_cond = self.cond_mlp(cond_ty)

        # pooling the conditions
        # pool_out = self.pool_mlp(jnp.expand_dims(mlp_cond, axis=2))

        # pool_out = jnp.expand_dims((pool_out + time_embed), axis=1)

        cond_signal = jnp.expand_dims(self.linear(mlp_cond), axis=1)
        cond_signal = jnp.broadcast_to(cond_signal, shape=(x.shape))

        print(f'{cond_signal.shape = }')
        x = x + cond_signal
        x = self.patch_mixer(x)

        if mask is not None:
            x = remove_masked_patches(x, mask)

        # mlp_out_us = jnp.expand_dims(mlp_cond, axis=1)

        # cond = jnp.broadcast_to((mlp_out_us + pool_out), shape=(x.shape))

        # x = x + cond
        
        print(f'x masked / {x.shape}')
        x = self.backbone(x, cond_ty)

        x = self.final_linear(x, cond_ty)

        # add back masked patches
        if mask is not None:
            x = add_masked_patches(x, mask)
            print(f'x unmasked / {x.shape}')
            
        x = self._unpatchify(x)

        return x

    def sample(self, z_latent: Array, cond, sample_steps=50, cfg=3.0):
        b_size = z_latent.shape[0]
        dt = 1.0 / sample_steps

        dt = jnp.array([dt] * b_size)
        dt = jnp.reshape(dt, shape=(b_size, *([1] * len(z_latent.shape[1:]))))

        images = [z_latent]

        for step in tqdm(range(sample_steps, 0, -1)):
            t = step / sample_steps
            t = jnp.array([t] * b_size, device=z_latent.device).astype(jnp.float16)

            vcond = self(z_latent, t, cond, mask=None, train=False)
            null_cond = jnp.zeros_like(cond)
            v_uncond = self(z_latent, t, null_cond)
            vcond = v_uncond + cfg * (vcond - v_uncond)

            z_latent = z_latent - dt * vcond
            images.append(z_latent)

        return images[-1]


# rectifed flow forward pass, loss, and smapling
class RectFlowWrapper(nnx.Module):
    def __init__(self, model: nnx.Module, sigln: bool = True):
        self.model = model
        self.sigln = sigln

    def __call__(self, x_input: Array, cond: Array, mask):
        b_size = x_input.shape[0]  # batch_size
        rand_t = None

        rand = jrand.uniform(randkey, (b_size,)).astype(jnp.bfloat16)
        rand_t = nnx.sigmoid(rand)
    
        inshape = [1] * len(x_input.shape[1:])
        texp = rand_t.reshape([b_size, *(inshape)]).astype(jnp.bfloat16)

        z_noise = jrand.uniform(randkey, x_input.shape).astype(jnp.bfloat16)  # input noise with same dim as image
        z_noise_t = (1 - texp) * x_input + texp * z_noise

        v_thetha = self.model(z_noise_t, rand_t, cond, mask, train=True)

        mean_dim = list(
            range(1, len(x_input.shape))
        )  # across all dimensions except the batch dim

        x_input = apply_mask(x_input, mask, config.patch_size)
        v_thetha = apply_mask(v_thetha, mask, config.patch_size)
        z_noise = apply_mask(z_noise, mask, config.patch_size)

        mean_square = (z_noise - x_input - v_thetha) ** 2  # squared difference
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss

        loss = jnp.mean(batchwise_mse_loss)
        loss = loss * 1 / (1 - config.mask_ratio)
        
        return v_thetha, loss
    

    def sample(self, z_latent: Array, cond, sample_steps=50, cfg=3.0):
        b_size = z_latent.shape[0]
        dt = 1.0 / sample_steps

        dt = jnp.array([dt] * b_size)
        dt = jnp.reshape(dt, shape=(b_size, *([1] * len(z_latent.shape[1:]))))

        images = [z_latent]

        for step in tqdm(range(sample_steps, 0, -1)):
            t = step / sample_steps
            t = jnp.array([t] * b_size, device=z_latent.device).astype(jnp.float16)

            vcond = self.model(z_latent, t, cond, None)
            null_cond = jnp.zeros_like(cond)
            v_uncond = self.model(z_latent, t, null_cond) # type: ignore
            vcond = v_uncond + cfg * (vcond - v_uncond)

            z_latent = z_latent - dt * vcond
            images.append(z_latent)

        return images[-1] / config.vaescale_factor


def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )

def device_get_model(model):
    state = nnx.state(model)
    state = jax.device_get(state)
    nnx.update(model, state)

    return model


def sample_image_batch(step, model, labels):
    pred_model = device_get_model(model)
    pred_model.eval()
    image_batch = pred_model.sample(labels)
    file = f"mdsamples/{step}_microdit.png"
    batch = [process_img(x) for x in image_batch]

    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")
    del pred_model

    return gridfile


def vae_decode(latent, vae=vae):
    tensor_img = rearrange(latent, "b h w c -> b c h w")
    tensor_img = torch.from_numpy(np.array(tensor_img))
    x = vae.decode(tensor_img).sample

    img = VaeImageProcessor().postprocess(
        image=x.detach(), do_denormalize=[True, True]
    )[0]

    return img


def process_img(img):
    img = vae_decode(img[None])
    return img


def image_grid(pil_images, file, grid_size=(3, 3), figsize=(10, 10)):
    rows, cols = grid_size
    assert len(pil_images) <= rows * cols, "Grid size must accommodate all images."

    # Create a matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing

    for i, ax in enumerate(axes):
        if i < len(pil_images):
            # Convert PIL image to NumPy array and plot
            ax.imshow(np.array(pil_images[i]))
            ax.axis("off")  # Turn off axis labels
        else:
            ax.axis("off")  # Hide empty subplots for unused grid spaces

    plt.tight_layout()
    plt.savefig(file, bbox_inches="tight")
    plt.show()

    return file


# save model params in pickle file
def save_paramdict_pickle(model, filename="dit_model.pkl"):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep=".")

    with open(filename, "wb") as f:
        pickle.dump(frozen_state_dict, f)

    return flat_state_dict


def load_paramdict_pickle(model, filename="ditmodel.pkl"):
    with open(filename, "rb") as modelfile:
        params = pickle.load(modelfile)

    params = unfreeze(params)
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    params = from_state_dict(model, params)

    nnx.update(model, params)

    return model, params


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_func(model, batch):
        img_latents, label = batch['vae_output'], batch['label']
        
        img_latents = img_latents.reshape(-1, 4, 32, 32) * config.vaescale_factor
        img_latents = rearrange(img_latents, "b c h w -> b h w c")
        # print(f"latents => {img_latents.shape}")

        img_latents, label = jax.device_put((img_latents, label), data_sharding)

        bs, height, width, channels = img_latents.shape

        mask = random_mask(
            bs,
            height,
            width,
            patch_size=config.patch_size,
            mask_ratio=config.mask_ratio,
        )
        v_thetha, loss = model(img_latents, label, mask)

        return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, batch)
    optimizer.update(grads)
    
    return loss


def trainer(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    
    model.train()

    # wandb_logger(
    #     key="",
    #     project_name="microdit",
    # )

    stime = time.time()
    sample_labels = jnp.array([76, 292, 293, 979, 968, 967, 33, 88, 404])

    for epoch in tqdm(range(epochs), desc='Training..../'):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}, loss-> {train_loss.item():.4f}")

            wandb.log({
                "loss": train_loss.item(),
                "log_loss": math.log10(train_loss.item())
            })

            if step % 1000 == 0:
                gridfile = sample_image_batch(step, model, sample_labels)
                image_log = wandb.Image(gridfile)
                wandb.log({"image_sample": image_log})

            jax.clear_caches()
            gc.collect()

        print(f"epoch {epoch+1}, train loss => {train_loss}")
        path = f"{epoch}_microdit_imagenet_{train_loss}.pkl"
        save_paramdict_pickle(model, path)
        
        epoch_file = sample_image_batch(step, model, sample_labels)
        epoch_image_log = wandb.Image(epoch_file)
        wandb.log({"epoch_sample": epoch_image_log})

    etime = time.time() - stime
    print(f"training time for {epochs} epochs -> {etime:.4f}s / {etime/60:.4f} mins")

    save_paramdict_pickle(
        model, f"microdit_in1k_step-{len(train_loader)*epochs}_floss-{train_loss}.pkl"
    )

    return model


def overfit(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    model.train()

    # wandb_logger(
    #     key="", project_name="microdit_overfit"
    # )

    stime = time.time()

    batch = next(iter(train_loader))
    
    print("start overfitting.../")
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, optimizer, batch)
        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss.item():.4f}")
        wandb.log({"loss": train_loss.item(), "log_loss": math.log10(train_loss.item())})
        
        if epoch % 50 == 0:
            gridfile = sample_image_batch(epoch, model, batch['labels'])
            image_log = wandb.Image(gridfile)
            wandb.log({"image_sample": image_log})

        jax.clear_caches()
        gc.collect()

    etime = time.time() - stime
    print(f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs")
        
    epoch_file = sample_image_batch('overfit', model, batch['labels'])
    epoch_image_log = wandb.Image(epoch_file)
    wandb.log({"epoch_sample": epoch_image_log})
    
    return model, train_loss


@click.command()
@click.option('-r', '--run', default='overfit')
@click.option('-e', '--epochs', default=30)
def main(run, epochs):
    
    dataset = StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=None,
        shuffle=True,
        shuffle_algo="naive",
        batch_size=config.batch_size,
    )
    streaming.base.util.clean_stale_shared_memory()
    
    train_loader = DataLoader(
        dataset[:config.data_split],
        batch_size=16, #config.batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=jax_collate,
        # sampler=dataset_sampler, # this is for multiprocess/v4-32
    )

    sp = next(iter(train_loader))
    print(f"loaded dataset, sample shape {sp['vae_output'].shape} /  {sp['label'].shape}, type = {type(sp['vae_output'])}")

    microdit = MicroDiT(
        in_channels=4,
        patch_size=(2, 2),
        embed_dim=768,
        num_layers=12,
        attn_heads=12,
        patchmix_layers=4,
    )

    rf_engine = RectFlowWrapper(microdit)
    graph, state = nnx.split(rf_engine)

    n_params = sum([p.size for p in jax.tree.leaves(state)])
    print(f"model parameters count: {n_params/1e6:.2f}M")

    optimizer = nnx.Optimizer(rf_engine, optax.adamw(learning_rate=config.lr))

    # replicate model across devices
    state = nnx.state((rf_engine, optimizer))
    state = jax.device_put(state, rep_sharding)
    nnx.update((rf_engine, optimizer), state)

    if run=='overfit':
        model, loss = overfit(epochs, rf_engine, optimizer, train_loader)
        wandb.finish()
        print(f"microdit overfitting ended at loss {loss:.4f}")
    
    elif run=='train':
        trainer(epochs, rf_engine, optimizer, train_loader)
        wandb.finish()
        print("microdit (test) training (on imagenet-1k) in JAX..done")

main()
