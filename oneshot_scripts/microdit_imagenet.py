"""
Training a MicroDIT model, DiT-B(base), 24-depth, 12 heads config for 30 epochs on Imagnet split
"""

import jax
from jax import Array, numpy as jnp, random as jrand
from jax.sharding import NamedSharding, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils

jax.config.update("jax_default_matmul_precision", "bfloat16")

import numpy as np
from flax import nnx
import flax.traverse_util
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze
from einops import rearrange

import os, wandb, time, pickle, gc
import math, flax, torch
from tqdm.auto import tqdm
from typing import List, Any
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from PIL import Image as pillow
from datasets import load_dataset
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
    data_split = 20_000
    cfg_scale = 2.0
    vaescale_factor = 0.13025
    data_id = "cloneofsimo/imagenet.int8"
    vae_id = "madebyollin/sdxl-vae-fp16-fix"
    imagenet_id = "ILSVRC/imagenet-1k"


JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"


rkey = jrand.key(config.seed)
randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)

# mesh / sharding configs
num_devices = jax.device_count()
devices = jax.devices()

mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), ("data",))

model_sharding = NamedSharding(mesh, PS())
data_sharding = NamedSharding(mesh, PS("data"))


print(f"found {num_devices} JAX devices")
for device in devices:
    print(f"{device} / ")

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
print("loaded vae")


def vae_decode(latent, vae=vae):
    tensor_img = torch.from_numpy(np.array(latent))
    tensor_img = rearrange(tensor_img, "b h w c -> b c h w")
    x = vae.decode(tensor_img).sample
    img = VaeImageProcessor().postprocess(image=x.detach(), do_denormalize=[True, True])[0]
    img = rearrange(img, "b c h w -> b h w c")

    return img


def process_img(img):
    img = vae_decode(img[None])[0]
    img = img * 0.5 + 0.5
    img = np.array(img)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = pillow.fromarray(img)

    return img


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

    reshaped_patches = patches.reshape(-1, embed_dim)

    full_patches = jnp.where(mask[..., None], reshaped_patches, full_patches)

    return full_patches


## data loading
# image_data = load_dataset(
#     config.data_id, streaming=True, split="train", trust_remote_code=True
# ).take(config.data_split)


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


_encodings["uint8"] = uint8
remote_train_dir = "./vae_mds"  # this is the path you installed this dataset.
local_train_dir = "./imagenet"

dataset = StreamingDataset(
    local=local_train_dir,
    remote=remote_train_dir,
    split=None,
    shuffle=True,
    shuffle_algo="naive",
    batch_size=config.batch_size,
)


num_processes = jax.process_count()
rank = jax.process_index()

# dataset_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, num_replicas=num_processes, rank=rank, seed=config.seed)

print(f'datasample {dataset[0]}')
def jax_collate(batch):
    latents = jnp.stack([
        jnp.array(item["vae_output"]) for item in batch], axis=0
    )
    labels = jnp.stack([int(item['label']) for item in batch], axis=0)

    # Metadata may not always be numerical, so store as a list

    return {
        "vae_output": latents,
        "label": labels,
    }


train_loader = DataLoader(
    dataset[:config.data_split],
    batch_size=16,#config.batch_size,
    num_workers=0,
    drop_last=True,
    collate_fn=jax_collate,
    # sampler=dataset_sampler, # thi sis for multiprocess/v4-32
)

sp = next(iter(train_loader))
print(
    f"loaded dataset, sample shape {sp['vae_output'].shape} /  {sp['label'].shape}, type = {type(sp['vae_output'])}"
)


# modulation with shift and scale
def modulate(x_array: Array, shift, scale) -> Array:
    x = x_array * (1 + jnp.expand_dims(scale, 1))
    x = x + jnp.expand_dims(shift, 1)

    return x


# equivalnet of F.linear
def linear(array: Array, weight: Array, bias: Array | None = None) -> Array:
    out = jnp.dot(array, weight)

    if bias is not None:
        out += bias

    return out


# Adapted from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim, h, w):
    """
    :param embed_dim: dimension of the embedding
    :param h: height of the grid
    :param w: width of the grid
    :return: [h*w, embed_dim] or [1+h*w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = jnp.arange(h, dtype=jnp.float32)
    grid_w = jnp.arange(w, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_h, grid_w, indexing="ij")
    grid = jnp.stack(grid, axis=0)

    grid = jnp.reshape(grid, shape=(2, 1, h, w))
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = jnp.concat([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)
const_init1 = nnx.initializers.constant(1)
linear_init = nnx.initializers.xavier_uniform()
linear_bias_init = nnx.initializers.constant(1)

# input patchify layer, 2D image to patches
class PatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs=rngs,
        patch_size: int = 2,
        in_chan: int = 4,
        embed_dim: int = 768,
        img_size=config.img_size,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.gridsize = tuple(
            [s // p for s, p in zip((img_size, img_size), (patch_size, patch_size))]
        )
        self.num_patches = self.gridsize[0] * self.gridsize[1]

        self.conv_project = nnx.Conv(
            in_chan,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=patch_size,
            use_bias=False,  # Typically, PatchEmbed doesn't use bias
            rngs=rngs,
        )

    def __call__(self, img: jnp.ndarray) -> jnp.ndarray:
        # Project image patches with the convolution layer
        x = self.conv_project(
            img
        )  # Shape: (batch_size, embed_dim, height // patch_size, width // patch_size)

        # Reshape to (batch_size, num_patches, embed_dim)
        batch_size, h, w, embed_dim = x.shape
        # num_patches = h * w
        # x = x.transpose(0, 2, 3, 1)  # Shape: (batch_size, height, width, embed_dim)
        x = x.reshape(
            batch_size, -1, embed_dim
        )  # Shape: (batch_size, num_patches, embed_dim)

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
            num_classes + use_cfg_embeddings,
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

    def __call__(self, labels, train: bool, force_drop_ids=None) -> Array:
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


# self attention block
class SelfAttention(nnx.Module):
    def __init__(self, attn_heads, embed_dim, rngs: nnx.Rngs, drop=0.0):
        super().__init__()
        self.attn_heads = attn_heads
        self.head_dim = embed_dim // attn_heads

        linear_init = nnx.initializers.xavier_uniform()
        linear_bias_init = nnx.initializers.constant(0.0)

        self.q_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.k_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.v_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
            rngs=rngs,
        )

        self.outproject = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.dropout = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x_input: jax.Array) -> jax.Array:
        q = self.q_linear(x_input)
        k = self.k_linear(x_input)
        v = self.v_linear(x_input)

        q, k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.attn_heads), (q, k, v)
        )

        qk = q @ jnp.matrix_transpose(k)
        attn_logits = qk / math.sqrt(self.head_dim)  # attention computation

        attn_score = nnx.softmax(attn_logits, axis=-1)
        attn_output = attn_score @ v

        output = rearrange(attn_output, "b h l d -> b l (h d)")
        output = self.dropout(self.outproject(output))
        return output


class CrossAttention(nnx.Module):
    def __init__(self, attn_heads, embed_dim, cond_dim, rngs: nnx.Rngs, drop=0.0):
        super().__init__()
        self.attn_heads = attn_heads
        self.head_dim = embed_dim // attn_heads

        linear_init = nnx.initializers.xavier_uniform()
        linear_bias_init = nnx.initializers.constant(0)

        self.q_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )

        self.k_linear = nnx.Linear(
            cond_dim,
            embed_dim,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
            rngs=rngs,
        )
        self.v_linear = nnx.Linear(
            cond_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )

        self.outproject = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.dropout = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x_input: jax.Array, y_cond: Array) -> jax.Array:
        q = self.q_linear(x_input)
        k = self.k_linear(y_cond)
        v = self.v_linear(y_cond)

        q, k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.attn_heads), (q, k, v)
        )

        qk = q @ jnp.matrix_transpose(k)
        attn_logits = qk / math.sqrt(self.head_dim)  # attention computation

        attn_score = nnx.softmax(attn_logits, axis=-1)
        attn_output = attn_score @ v

        output = rearrange(attn_output, "b h l d -> b l (h d)")
        output = self.dropout(self.outproject(output))

        return output


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
        self.self_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            kernel_init=xavier_init,
            rngs=rngs,
            decode=False,
        )  # SelfAttention(num_heads, embed_dim, rngs=rngs)
        self.mlp_layer = EncoderMLP(embed_dim, rngs=rngs)

    def __call__(self, x: Array):
        x = x + self.layernorm(self.self_attention(x))
        x = x + self.layernorm(self.mlp_layer(x))

        return x


class SimpleMLP(nnx.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super().__init__()
        hidden = int(mlp_ratio * embed_dim)
        # print(f"simple {hidden}")
        self.linear_1 = nnx.Linear(embed_dim, hidden, rngs=rngs)
        self.linear_2 = nnx.Linear(hidden, embed_dim, rngs=rngs)

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
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )
        self.linear_2 = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.avg_pool(x, (1,))
        x = jnp.reshape(x, shape=(x.shape[0], -1))
        x = nnx.gelu(self.linear_1(x))
        x = self.linear_2(x)

        return x


class OutputMLP(nnx.Module):
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.linear_1 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(
            embed_dim, patch_size[0] * patch_size[1] * out_channels, rngs=rngs
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.gelu(self.linear_1(x))
        x = self.linear_2(x)

        return x


###############
# DiT blocks_ #
###############
class DiTBlock(nnx.Module):
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4):
        super().__init__()

        # initializations
        linear_init = nnx.initializers.xavier_uniform()
        lnbias_init = nnx.initializers.constant(1)
        lnweight_init = nnx.initializers.constant(1)

        self.norm_1 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, bias_init=lnbias_init
        )
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            kernel_init=xavier_init,
            # outer_kerner
            rngs=rngs,
            decode=False,
        )  # SelfAttention(num_heads, hidden_size, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, bias_init=lnbias_init
        )

        self.adaln_linear = nnx.Linear(
            in_features=hidden_size,
            out_features=6 * hidden_size,
            use_bias=True,
            bias_init=zero_init,
            rngs=rngs,
            kernel_init=zero_init,
        )

        self.mlp_block = SimpleMLP(hidden_size, mlp_ratio)

    def __call__(self, x_img: Array, cond):

        cond = self.adaln_linear(nnx.silu(cond))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            cond, 6, axis=1
        )

        attn_mod_x = self.attention(modulate(self.norm_1(x_img), shift_msa, scale_msa))

        x = x_img + jnp.expand_dims(gate_msa, 1) * attn_mod_x
        x = modulate(self.norm_2(x), shift_mlp, scale_mlp)
        mlp_mod_x = self.mlp_block(x)
        x = x + jnp.expand_dims(gate_mlp, 1) * mlp_mod_x

        return x


class FinalMLP(nnx.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # linear_init = nnx.initializers.xavier_uniform()
        linear_init = nnx.initializers.constant(0)

        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=linear_init,
        )
        self.adaln_linear = nnx.Linear(
            hidden_size,
            2 * hidden_size,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=linear_init,
        )

    def __call__(self, x_input: Array, cond: Array):
        linear_cond = nnx.silu(self.adaln_linear(cond))
        shift, scale = jnp.array_split(linear_cond, 2, axis=1)

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
            bias_init=nnx.initializers.constant(0.0),
            kernel_init=nnx.initializers.xavier_uniform(),
        )
        self.class_embedding = nnx.Linear(
            class_embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=nnx.initializers.constant(0.0),
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

            # print(
            #     f"mlp_ratio, scaled_dim scaled_num_heads = {mlp_ratio}/ {scaled_mlp_dim} / {scaled_mlp_dim/embed_dim} / {scaled_num_heads}"
            # )

            self.layers.append(DiTBlock(embed_dim, scaled_num_heads, mlp_ratio))

        self.output_layer = nnx.Linear(
            embed_dim,
            input_dim,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=nnx.initializers.xavier_uniform(),
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

        # print(f'class embed :{class_emb}')
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
# Full Microdit model
####################
class MicroDiT(nnx.Module):
    def __init__(
        self,
        inchannels,
        patch_size,
        embed_dim,
        num_layers,
        attn_heads,
        cond_embed_dim,
        dropout=0.0,
        patchmix_layers=4,
        rngs=rngs,
        num_classes=1000,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedder = PatchEmbed(
            rngs=rngs, patch_size=patch_size[0], in_chan=inchannels, embed_dim=embed_dim
        )
        self.num_patches = self.patch_embedder.num_patches

        # conditioning layers
        self.time_embedder = TimestepEmbedder(embed_dim)
        self.cap_embedder = CaptionEmbedder(cond_embed_dim, embed_dim)
        self.label_embedder = LabelEmbedder(
            num_classes=num_classes, hidden_size=embed_dim, drop=dropout
        )
        self.cond_attention = nnx.MultiHeadAttention(
            num_heads=attn_heads,
            in_features=embed_dim,
            kernel_init=xavier_init,
            rngs=rngs,
            decode=False,
        )  # CrossAttention(attn_heads, embed_dim, cond_embed_dim, rngs=rngs)
        self.cond_mlp = SimpleMLP(embed_dim, mlp_ratio=4)

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

        self.final_linear = OutputMLP(embed_dim, patch_size=patch_size, out_channels=4)

    def unpatchify(self, x: Array) -> Array:
        # print(x.shape)

        bs, num_patches, patch_dim = x.shape
        H, W = self.patch_size  # Assume square patches
        in_channels = patch_dim // (H * W)
        height, width = config.img_size, config.img_size

        # Calculate the number of patches along each dimension
        num_patches_h = height // H
        num_patches_w = width // W

        # Reshape x to (bs, num_patches_h, num_patches_w, H, W, in_channels)
        x = x.reshape(bs, num_patches_h, num_patches_w, H, W, in_channels)

        # Transpose to (bs, num_patches_h, H, num_patches_w, W, in_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)

        reconstructed = x.reshape(
            bs, height, width, in_channels
        )  # Reshape to (bs, height, width, in_channels)

        return reconstructed

    def __call__(self, x: Array, t: Array, y_cap: Array, mask=None):
        bsize, height, width, channels = x.shape
        psize_h, psize_w = self.patch_size

        x = self.patch_embedder(x)

        pos_embed = get_2d_sincos_pos_embed(
            self.embed_dim, height // psize_h, width // psize_w
        )

        pos_embed = jnp.expand_dims(pos_embed, axis=0)
        pos_embed = jnp.broadcast_to(
            pos_embed, (bsize, pos_embed.shape[1], pos_embed.shape[2])
        )
        pos_embed = jnp.reshape(
            pos_embed, shape=(bsize, self.num_patches, self.embed_dim)
        )
        # x = jnp.reshape(x, shape=(bsize, self.num_patches, self.embed_dim))

        x = x + pos_embed

        # cond_embed = self.cap_embedder(y_cap) # (b, embdim)
        cond_embed = self.label_embedder(y_cap, train=True)
        cond_embed_unsqueeze = jnp.expand_dims(cond_embed, axis=1)

        time_embed = self.time_embedder(t)
        time_embed_unsqueeze = jnp.expand_dims(time_embed, axis=1)
        # print(f'time and cond embed sape: {time_embed.shape}/{time_embed_unsqueeze.shape}, {cond_embed.shape}/{cond_embed_unsqueeze.shape}')

        mha_out = self.cond_attention(time_embed_unsqueeze, cond_embed_unsqueeze)
        mha_out = mha_out.squeeze(1)

        mlp_out = self.cond_mlp(mha_out)

        # pooling the conditions
        pool_out = self.pool_mlp(jnp.expand_dims(mlp_out, axis=2))

        pool_out = jnp.expand_dims((pool_out + time_embed), axis=1)

        cond_signal = jnp.expand_dims(self.linear(mlp_out), axis=1)
        cond_signal = jnp.broadcast_to((cond_signal + pool_out), shape=(x.shape))

        x = x + cond_signal
        x = self.patch_mixer(x)

        if mask is not None:
            x = remove_masked_patches(x, mask)

        mlp_out_us = jnp.expand_dims(mlp_out, axis=1)

        cond = jnp.broadcast_to((mlp_out_us + pool_out), shape=(x.shape))

        x = x + cond

        x = self.backbone(x, cond_embed)
        # print(f'backbone {x.shape = }')

        x = self.final_linear(x)
        # print(f'{x.shape = }')

        # add back masked patches
        if mask is not None:
            x = add_masked_patches(x, mask)

        x = self.unpatchify(x)

        return x

    def sample(self, z_latent: Array, cond, sample_steps=50, cfg=2.0):
        b_size = z_latent.shape[0]
        dt = 1.0 / sample_steps

        dt = jnp.array([dt] * b_size)
        dt = jnp.reshape(dt, shape=(b_size, *([1] * len(z_latent.shape[1:]))))

        images = [z_latent]

        for step in tqdm(range(sample_steps, 0, -1)):
            t = step / sample_steps
            t = jnp.array([t] * b_size, device=z_latent.device).astype(jnp.float16)

            vcond = self(z_latent, t, cond, None)
            null_cond = jnp.zeros_like(cond)
            v_uncond = self(z_latent, t, null_cond)
            vcond = v_uncond + cfg * (vcond - v_uncond)

            z_latent = z_latent - dt * vcond
            images.append(z_latent)

        return images[-1]


import optax


def apply_mask(x, mask, patch_size):
    """
    Applies a mask to a tensor with channels-last format. Turns the masked values to 0s.

    Args:
        x: JAX array of shape (bs, h, w, c)
        mask: JAX array of shape (bs, num_patches)
        patch_size: Tuple of patch height and width

    Returns:
        JAX array of shape (bs, h, w, c) with the masked values turned to 0s.
    """

    bs, h, w, c = x.shape
    num_patches_h, num_patches_w = h // patch_size[0], w // patch_size[1]

    # Ensure that height and width are divisible by patch_size
    assert (
        h % patch_size[0] == 0 and w % patch_size[1] == 0
    ), "Height and width must be divisible by patch_size. Height: {}, Width: {}, Patch size: {}".format(
        h, w, patch_size
    )

    # Reshape mask to (bs, num_patches_h, num_patches_w)
    mask = mask.reshape((bs, num_patches_h, num_patches_w))
    # print(f"x in => {x.shape}, mask => {mask.shape}")

    # Expand the mask to cover each patch
    # (bs, num_patches_h, num_patches_w) -> (bs, h, w, 1)
    mask = jnp.expand_dims(mask, axis=3)  # Add channel dimension
    mask = jnp.repeat(mask, patch_size[0], axis=1)  # Repeat for patch_size height
    mask = jnp.repeat(mask, patch_size[1], axis=2)  # Repeat for patch_size width

    # Apply the mask to the input tensor
    x = x * mask

    return x


# rectifed flow forward pass, loss, and smapling
class RectFlowWrapper(nnx.Module):
    def __init__(self, model: nnx.Module, sigln: bool = True):
        self.model = model
        self.sigln = sigln

    def __call__(self, x_input: Array, cond: Array, mask) -> Array:
        b_size = x_input.shape[0]  # batch_size
        rand_t = None

        if self.sigln:
            rand = jrand.uniform(randkey, (b_size,))
            rand_t = nnx.sigmoid(rand)
        else:
            rand_t = jrand.uniform(randkey, (b_size,))

        inshape = [1] * len(x_input.shape[1:])
        texp = rand_t.reshape([b_size, *(inshape)])

        z_noise = jrand.uniform(
            randkey, x_input.shape
        )  # input noise with same dim as image
        z_noise_t = (1 - texp) * x_input + texp * z_noise

        v_thetha = self.model(z_noise_t, rand_t, cond, mask)

        mean_dim = list(
            range(1, len(x_input.shape))
        )  # across all dimensions except the batch dim
        # print(f'z_noise {z_noise.shape}, vtheta {v_thetha.shape}, x_input {x_input.shape}')

        x_input = apply_mask(x_input, mask, config.patch_size)
        v_thetha = apply_mask(v_thetha, mask, config.patch_size)
        z_noise = apply_mask(z_noise, mask, config.patch_size)

        mean_square = (z_noise - x_input - v_thetha) ** 2  # squared difference
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss

        loss = jnp.mean(batchwise_mse_loss)
        loss = loss * 1 / (1 - config.mask_ratio)
        return loss

    def sample(self, z_latent: Array, cond, sample_steps=50, cfg=2.0):
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
            v_uncond = self.model(z_latent, t, null_cond)
            vcond = v_uncond + cfg * (vcond - v_uncond)

            z_latent = z_latent - dt * vcond
            images.append(z_latent)

        return images[-1]


microdit = MicroDiT(
    inchannels=4,
    patch_size=(2, 2),
    embed_dim=768,
    num_layers=16,
    attn_heads=12,
    cond_embed_dim=768,
    patchmix_layers=6,
)

rf_engine = RectFlowWrapper(microdit)
graph, state = nnx.split(rf_engine)

n_params = sum([p.size for p in jax.tree.leaves(state)])
print(f"number of parameters: {n_params/1e6:.2f}M")

optimizer = nnx.Optimizer(rf_engine, optax.adamw(learning_rate=config.lr))


def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )


def vae_decode(latent, vae=vae):
    # print(f'decoding... (latent shape = {latent.shape}) ')
    tensor_img = rearrange(latent, "b h w c -> b c h w")
    tensor_img = torch.from_numpy(np.array(tensor_img))
    x = vae.decode(tensor_img).sample 

    img = VaeImageProcessor().postprocess(
        image=x.detach(), do_denormalize=[True, True]
    )[0]

    return img


def process_img(img, id):
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


def save_image_grid(batch, file_path: str, grid_size=None):
    # batch = process_img(batch)
    # Determine grid size
    batch_size = batch.shape[0]
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(batch_size)))  # Square grid
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_size

    # Set up the grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Plot each image
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            image = process_img(batch[i])
            # image = np.clip(image, 0, 1)
            # img = (image * 255).astype(np.uint8)  # Scale image to 0-255
            ax.imshow(np.array(image))
            ax.axis("off")
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    return file_path


def display_samples(sample_batch):
    batch = np.array(sample_batch)
    # Set up the grid
    batch_size = batch.shape[0]

    grid_size = int(np.ceil(np.sqrt(batch_size)))  # Square grid

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Plot each image
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            img = process_img(batch[i])
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.show()


def device_get_model(model):
    state = nnx.state(model)
    state = jax.device_get(state)
    nnx.update(model, state)

    return model


def sample_image_batch(step, model):
    classin = jnp.array([76, 292, 293, 979, 968, 967, 33, 88, 404])  # 76, 292, 293, 979, 968 imagenet
    randnoise = jrand.uniform(
        randkey, (len(classin), config.img_size, config.img_size, 4)
    )
    pred_model = device_get_model(model)
    pred_model.eval()
    image_batch = pred_model.sample(randnoise, classin)
    file = f"samples/imagenet_dit_output@{step}.png"
    batch = [process_img(x, id) for id, x in enumerate(image_batch)]
    gridfile = image_grid(batch, file)
    print(f'sample saved @ {gridfile}')
    del pred_model

    return gridfile


import pickle


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


# replicate model
state = nnx.state((rf_engine, optimizer))
state = jax.device_put(state, model_sharding)
nnx.update((rf_engine, optimizer), state)


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
        loss = model(img_latents, label, mask)

        return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, batch)
    optimizer.update(grads)
    
    return loss


def trainer(model=rf_engine, optimizer=optimizer, train_loader=train_loader):
    epochs = 2
    train_loss = 0.0
    
    model.train()

    wandb_logger(
        key="",
        project_name="microdit",
    )

    stime = time.time()

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}, loss-> {train_loss.item():.4f}")

            wandb.log({
                "loss": train_loss.item(),
                "log_loss": math.log10(train_loss.item())
            })

            if step % 25 == 0:
                gridfile = sample_image_batch(step, model)
                image_log = wandb.Image(gridfile)
                wandb.log({"image_sample": image_log})

            jax.clear_caches()
            gc.collect()

        print(f"epoch {epoch+1}, train loss => {train_loss}")
        path = f"microdit_imagenet_{epoch}_{train_loss}.pkl"
        save_paramdict_pickle(model, path)
        
        epoch_file = sample_image_batch(step, model)
        epoch_image_log = wandb.Image(epoch_file)
        wandb.log({"epoch_sample": epoch_image_log})

    etime = time.time() - stime
    print(f"training time for {epochs} epochs -> {etime:.4f}s / {etime/60:.4f} mins")

    save_paramdict_pickle(
        model, f"microdit_in1k_{len(train_loader)*epochs}_{train_loss}.pkl"
    )

    return model


def overfit(model=rf_engine, optimizer=optimizer, train_loader=train_loader):
    epochs = 101
    train_loss = 0.0
    model.train()

    wandb_logger(
        key="", project_name="microdit_overfit"
    )

    stime = time.time()

    batch = next(iter(train_loader))
    print("start overfitting.../")
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, optimizer, batch)
        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss.item():.4f}")
        wandb.log({"loss": train_loss.item(), "log_loss": math.log10(train_loss.item())})
        
        if epoch % 25 == 0:
            gridfile = sample_image_batch(epoch, model)
            image_log = wandb.Image(gridfile)
            wandb.log({"image_sample": image_log})


        jax.clear_caches()
        # jax.clear_backends()
        gc.collect()

    etime = time.time() - stime
    print(f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs")
        
    epoch_file = sample_image_batch('overfit', model)
    epoch_image_log = wandb.Image(epoch_file)
    wandb.log({"epoch_sample": epoch_image_log})
    
    return model, train_loss




import click

@click.option('-r', '--run')
def main(run):
    if run=='overfit':
        model, loss = overfit()
        wandb.finish()
        print(f"microdit overfitting ended at loss {loss:.4f}")
    
    elif run=='train':
        trainer()
        wandb.finish()
        print("microdit (test) training (on imagenet-1k) in JAX..done")
        
main()