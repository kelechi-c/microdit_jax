'''
Training a MicroDIT model, DiT-B(base) config for 30 epochs on cifar-10
'''

import flax.jax_utils
import jax, math, flax
import os, wandb, time, pickle
from jax import Array, numpy as jnp, random as jrand
from jax.sharding import NamedSharding, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
import numpy as np
from flax import nnx
from einops import rearrange
from tqdm.auto import tqdm
from typing import List
from torch.utils.data import DataLoader, IterableDataset
import flax.traverse_util
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze
import matplotlib.pyplot as plt
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")


class config:
    batch_size = 128
    img_size = 32
    seed = 334
    patch_size = (2, 2)
    lr = 2e-4
    mask_ratio = 0.50
    epochs = 30
    data_split = 40_000
    cfg_scale = 1.0
    mini_data_id = "uoft-cs/cifar10"


JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20

rkey = jrand.key(config.seed)
randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)

# mesh / sharding configs
num_devices = jax.device_count()
devices = jax.devices()

mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), ("data",))

model_sharding = NamedSharding(mesh, PS())
data_sharding = NamedSharding(mesh, PS("data"))


print(f"found {num_devices} JAX device")
for device in devices:
    print(f"{device} / ")


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
image_data = load_dataset(
    config.mini_data_id, streaming=True, split="train", trust_remote_code=True
).take(config.data_split)


def load_image(image):
    # image = pillow.open(urlopen(url=url))
    img_array = np.array(image)
    return img_array / 255.0


class ImageClassData(IterableDataset):
    def __init__(self, dataset=image_data):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return config.data_split

    def __iter__(self):
        for sample in self.dataset:
            image = sample["img"]  # type: ignore
            image = load_image(image)

            image = jnp.array(image)
            label = jnp.array(sample["label"])

            yield image, label


def jax_collate(batch):
    images, labels = zip(*batch)
    batch = (jnp.array(images), jnp.array(labels))
    batch = jax.tree_util.tree_map(jnp.array, batch)

    return batch


dataset = ImageClassData()

train_loader = DataLoader(dataset, batch_size=128, collate_fn=jax_collate, num_workers=4)

# flax.jax_utils.replicate(state, jax.devices())
nnx.st

# modulation with shift and scale
def modulate(x_array: Array, shift, scale) -> Array:
    x = x_array * (1 + jnp.expand_dims(scale, 1))
    x = x + jnp.expand_dims(shift, 1)

    return x


# equivalnet of F.lineat
def linear(array: Array, weight: Array, bias: Array | None = None) -> Array:
    out = jnp.dot(array, weight)

    if bias is not None:
        out += bias

    return out


# Adapted from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = jnp.concatenate(
            [jnp.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

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
        in_chan: int = 3,
        embed_dim: int = 768,
        img_size=config.img_size,
        # rngs: nnx.Rngs=rngs
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
            strides=(patch_size, patch_size),
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
        x = x.reshape(batch_size, -1)  # Shape: (batch_size, num_patches, embed_dim)

        return x


# patchem = PatchEmbed()
# patchimg = patchem(iv[0])


# embeds a flat vector
class VectorEmbedder(nnx.Module):
    def __init__(self, input_dim, hidden_size, rngs=rngs):
        super().__init__()
        self.linear_1 = nnx.Linear(input_dim, hidden_size, rngs=rngs)
        self.linear_2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, x: Array):
        x = nnx.silu(self.linear_1(x))
        x = self.linear_2(x)

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
        linear_bias_init = nnx.initializers.constant(0)

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
        self.self_attention = SelfAttention(num_heads, embed_dim, rngs=rngs)
        self.mlp_layer = EncoderMLP(embed_dim, rngs=rngs)

    def __call__(self, x: Array):
        x = x + self.layernorm(self.self_attention(x))
        x = x + self.layernorm(self.mlp_layer(x))

        return x


class SimpleMLP(nnx.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear_1 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = nnx.silu(self.linear_1(x))
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
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()

        # initializations
        linear_init = nnx.initializers.xavier_uniform()
        lnbias_init = nnx.initializers.constant(1)
        lnweight_init = nnx.initializers.constant(1)

        self.norm_1 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, bias_init=lnbias_init
        )
        self.attention = SelfAttention(num_heads, hidden_size, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, scale_init=lnweight_init
        )

        self.adaln_linear = nnx.Linear(
            in_features=hidden_size,
            out_features=6 * hidden_size,
            use_bias=True,
            bias_init=zero_init,
            rngs=rngs,
            kernel_init=zero_init,
        )

        self.mlp_block = SimpleMLP(hidden_size)

    def __call__(self, x_img: Array, cond):

        cond = self.adaln_linear(nnx.silu(cond))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.array_split(cond, 6, axis=1)
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
            kernel_init=nnx.initializers.xavier_uniform(),
        )

        self.layers = [DiTBlock(embed_dim, num_heads) for _ in range(num_layers)]

        self.output_layer = nnx.Linear(
            embed_dim,
            input_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
        )

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
        patchmix_layers=2,
        rngs=rngs,
        num_classes=10,
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
        self.cond_attention = CrossAttention(
            attn_heads, embed_dim, cond_embed_dim, rngs=rngs
        )
        self.cond_mlp = SimpleMLP(embed_dim)

        # pooling layer
        self.pool_mlp = PoolMLP(embed_dim)

        self.linear = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)

        self.patch_mixer = PatchMixer(embed_dim, attn_heads, patchmix_layers)

        self.backbone = TransformerBackbone(
            embed_dim,
            embed_dim,
            embed_dim,
            num_layers=num_layers,
            num_heads=attn_heads,
        )

        self.final_linear = OutputMLP(embed_dim, patch_size=patch_size, out_channels=3)

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

        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, height // psize_h)

        pos_embed = jnp.expand_dims(pos_embed, axis=0)
        pos_embed = jnp.broadcast_to(
            pos_embed, (bsize, pos_embed.shape[1], pos_embed.shape[2])
        )
        pos_embed = jnp.reshape(
            pos_embed, shape=(bsize, self.num_patches, self.embed_dim)
        )
        x = jnp.reshape(x, shape=(bsize, self.num_patches, self.embed_dim))

        x = x + pos_embed

        # cond_embed = self.cap_embedder(y_cap) # (b, embdim)
        cond_embed = self.label_embedder(y_cap, train=True)
        cond_embed_unsqueeze = jnp.expand_dims(cond_embed, axis=1)

        time_embed = self.time_embedder(t)
        time_embed_unsqueeze = jnp.expand_dims(time_embed, axis=1)
        # print(f'time and cond embed sape: {time_embed.shape}/{time_embed_unsqueeze.shape}, {cond_embed.shape}/{cond_embed_unsqueeze.shape}')

        mha_out = self.cond_attention(time_embed_unsqueeze, cond_embed_unsqueeze)  #
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

        mlp_out_us = jnp.expand_dims(mlp_out, axis=1)  # unqueezed mlp output

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

    def sample(self, z_latent, cond, sample_steps=50, cfg=1.0):
        b_size = z_latent.shape[0]
        dt = 1.0 / sample_steps

        dt = jnp.array([dt] * b_size)
        dt = jnp.reshape(dt, shape=(b_size, *([1] * len(z_latent.shape[1:]))))

        images = [z_latent]

        for i in tqdm(range(sample_steps, 0, -1)):
            t = i / sample_steps
            t = jnp.array([t] * b_size).astype(z_latent.dtype)

            vc = self(z_latent, t, cond, None)
            null_cond = jnp.zeros_like(cond)
            vu = self.__call__(z_latent, t, null_cond)
            vc = vu + cfg * (vc - vu)

            z_latent = z_latent - dt * vc
            images.append(z_latent)

        return images  # [-1]


import optax


def apply_mask(x, mask, patch_size):

    bs, h, w, c = x.shape
    num_patches_h, num_patches_w = h // patch_size[0], w // patch_size[1]

    # Reshape mask to (bs, num_patches_h, num_patches_w)
    mask = mask.reshape((bs, num_patches_h, num_patches_w))

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

        b_size = x_input.shape[0]  # batch_sie
        rand_t = None

        if self.sigln:
            rand = jrand.normal(randkey, (b_size,))  # .to_device(x_input.device)
            rand_t = nnx.sigmoid(rand)
        else:
            rand_t = jrand.normal(randkey, (b_size,))  # .to_device(x_input.device)

        inshape = [1] * len(x_input.shape[1:])
        texp = rand_t.reshape([b_size, *(inshape)])

        z_noise = jrand.normal(
            randkey, x_input.shape
        )  # input noise with same dim as image
        z_noise_t = (1 - texp) * x_input + texp * z_noise
        v_thetha = self.model(z_noise_t, rand_t, cond, mask)

        mean_dim = list(
            range(1, len(x_input.shape))
        )  # across all dimensions except the batch dim
        x_input = apply_mask(x_input, mask, config.patch_size)
        v_thetha = apply_mask(v_thetha, mask, config.patch_size)
        z_noise = apply_mask(z_noise, mask, config.patch_size)

        mean_square = (z_noise - x_input - v_thetha) ** 2  # squared difference
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss

        return jnp.mean(batchwise_mse_loss)

    def sample(
        self,
        input_noise: jax.Array,
        cond,
        zero_cond=None,
        sample_steps: int = 50,
        cfg=1.0,
    ) -> List[jax.Array]:

        batch_size = input_noise.shape[0]

        # array reciprocal of sampling steps
        d_steps = 1.0 / sample_steps

        d_steps = jnp.array([d_steps] * batch_size)  # .to_device(input_noise.device)
        steps_dim = [1] * len(input_noise.shape[1:])
        d_steps = d_steps.reshape((batch_size, *steps_dim))

        images = [input_noise]  # noise sequence

        for t_step in tqdm(range(sample_steps)):

            genstep = t_step / sample_steps  # current step

            genstep_batched = jnp.array(
                [genstep] * batch_size
            ) 

            cond_output = self.model(
                input_noise, genstep_batched, cond
            )  # get model output for step

            if zero_cond is not None:
                # output for zero conditioning
                uncond_output = self.model(input_noise, genstep_batched, zero_cond)
                cond_output = uncond_output + cfg * (cond_output - uncond_output)

            input_noise = input_noise - d_steps * cond_output

            images.append(input_noise)

        return images


microdit = MicroDiT(
    inchannels=3,
    patch_size=(2, 2),
    embed_dim=768,
    num_layers=12,
    attn_heads=12,
    cond_embed_dim=768,
)

rf_engine = RectFlowWrapper(microdit)
graph, state = nnx.split(rf_engine)

n_params = sum([p.size for p in jax.tree.leaves(state)])
print(f"number of parameters: {n_params/1e6:.3f}M")

optimizer = nnx.Optimizer(rf_engine, optax.adamw(learning_rate=config.lr))


def wandb_logger(key: str, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name, name=run_name, settings=wandb.Settings(init_timeout=120)
    )


import gc

jax.clear_caches()
gc.collect()


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
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Plot each image
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            img = (batch[i] * 255).astype(np.uint8)  # Scale image to 0-255
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    return file_path


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


def sample_image_batch(step):
    classin = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 76, 292, 293, 979, 968 imagenet
    randnoise = jrand.normal(
        randkey, (len(classin), config.img_size, config.img_size, 3)
    )
    image_batch = rf_engine.sample(randnoise, classin)
    gridfile = save_image_grid(image_batch, f"rf_dit_output@{step}.png")
    display_samples(image_batch)

    return gridfile


import pickle


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

    # print(type(params))
    params = unfreeze(params)
    # print(type(params))
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    # print(type(params))
    params = from_state_dict(model, params)
    # print(type(params), type(model))

    nnx.update(model, params)

    return model, params


# replicate model
state = nnx.state((rf_engine, optimizer))
state = jax.device_put(state, model_sharding)
nnx.update((rf_engine, optimizer), state)

# print(f'resuming training from {lastsaved}')


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_func(model, batch):
        img_latents, label = batch
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
        # loss = optax.squared_error(img_latents, logits).mean()
        return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, batch)
    optimizer.update(grads)
    return loss


def trainer(model=rf_engine, optimizer=optimizer, train_loader=train_loader):
    epochs = 30
    train_loss = 0.0
    model.train()
    wandb_logger(
        key="",
        project_name="microdit_jax",
        run_name="microdit-cifar-tpu",
    )

    stime = time.time()

    for epoch in tqdm(range(epochs + 1)):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}, loss-> {train_loss.item():.4f}")

            # if step % 5 == 0:
            wandb.log({"loss": train_loss.item()})

            if step % 50 == 0:
                gridfile = sample_image_batch(step)
                image_log = wandb.Image(gridfile)
                wandb.log({"image_sample": image_log})

            jax.clear_caches()
            jax.clear_backends()
            gc.collect()

        print(f"epoch {epoch+1}, train loss => {train_loss}")
        path = f"microdit_cifar_{epoch}_{train_loss}.pkl"
        save_paramdict_pickle(model, path)

    etime = time.time() - stime
    print(f"training time for {epochs} epochs -> {etime}s / {etime/60} mins")
    # path = f"microdit_cifar_full_{len(train_loader)*epochs}_{train_loss}.pkl"

    save_paramdict_pickle(
        model, f"microdit_cifar_40k_{len(train_loader)*epochs}_{train_loss}.pkl"
    )

    return model


trainer()
wandb.finish()
print("microdit training(cifar) in JAX..done")
