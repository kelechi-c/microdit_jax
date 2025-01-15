import jax, math
from jax import Array, numpy as jnp, random as jrand
from einops import rearrange
from typing import Dict
import numpy as np
from flax import nnx
from tqdm.auto import tqdm


class config:
    batch_size = 128
    img_size = 32
    seed = 42
    patch_size = (2, 2)
    lr = 1e-4
    mask_ratio = 0.0
    epochs = 30
    data_split = 10_000  # imagenet split to train on
    cfg_scale = 3.0
    vaescale_factor = 0.13025


rkey = jrand.key(config.seed)
randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)

# model helper functions
# modulation with shift and scale
def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def get_2d_sincos_pos_embed(embed_dim, h, w):

    grid_h = jnp.arange(h, dtype=jnp.float32)
    grid_w = jnp.arange(w, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, w, h])
    # print(f'{grid.shape = }')
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    # print(f"{emb_h.shape = } / {emb_w.shape = }")

    emb = jnp.concat([emb_h, emb_w], axis=1)  # (H*W, D)
    # print(f'{emb.shape = }')
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

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
one_init = nnx.initializers.constant(1)
normal_init = nnx.initializers.normal(0.02)
trunc_init = nnx.initializers.truncated_normal(0.02)


# input patchify layer, 2D image to patches
class PatchEmbed(nnx.Module):
    def __init__(
        self, in_channels=4, img_size: int = (22, 44), dim=768, patch_size: int = 2
    ):
        self.dim = dim
        self.patch_size = patch_size
        patch_tuple = (patch_size, patch_size)
        self.num_patches = (img_size[0] // self.patch_size) * (
            img_size[1] // self.patch_size
        )
        self.conv_project = nnx.Conv(
            in_channels,
            dim,
            kernel_size=patch_tuple,
            strides=patch_tuple,
            use_bias=True,
            padding="VALID",
            kernel_init=xavier_init,
            bias_init=zero_init,
            rngs=rngs,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        num_patches_side_h = H // self.patch_size
        num_patches_side_w = W // self.patch_size

        x = self.conv_project(x)  # (B, P, P, hidden_size)
        x = rearrange(
            x, "b h w c -> b (h w) c", h=num_patches_side_h, w=num_patches_side_w
        )
        return x


class TimestepEmbedder(nnx.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.lin_1 = nnx.Linear(
            freq_embed_size,
            hidden_size,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=normal_init,
        )
        self.lin_2 = nnx.Linear(
            hidden_size,
            hidden_size,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=normal_init,
        )
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
            embedding_init=normal_init,
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


class MLP(nnx.Module):
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        hidden = int(mlp_ratio * embed_dim)
        self.linear_1 = nnx.Linear(
            embed_dim, hidden, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )
        self.linear_2 = nnx.Linear(
            hidden, embed_dim, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.gelu(self.linear_1(x))
        # x = self.norm_1(x)
        x = self.linear_2(x)

        return x


class CaptionEmbedder(nnx.Module):
    def __init__(self, embed_dim, cap_dim, mlp_ratio=4):
        super().__init__()
        hidden = int(mlp_ratio * embed_dim)

        self.linear_1 = nnx.Linear(
            cap_dim, hidden, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )
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
            bias_init=zero_init,
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


class FeedForward(nnx.Module):
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        hidden = int(mlp_ratio * embed_dim)

        self.w_1 = nnx.Linear(
            embed_dim, hidden, rngs=rngs, kernel_init=trunc_init, bias_init=zero_init
        )
        self.w_2 = nnx.Linear(
            embed_dim, hidden, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )
        self.w_3 = nnx.Linear(
            hidden, embed_dim, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )

    def __call__(self, x: Array) -> Array:
        x = self.w_2(x) * nnx.silu(self.w_1(x))
        x = self.w_3(x)
        return x


###############
# DiT blocks_ #
###############
class DiTBlock(nnx.Module):
    def __init__(
        self, dim: int, attn_heads: int, drop: float = 0.0, rngs=nnx.Rngs(config.seed)
    ):
        super().__init__()
        self.dim = dim
        self.norm_1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init)
        self.norm_2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init)
        self.norm_3 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init)

        self.attention = nnx.MultiHeadAttention(
            num_heads=attn_heads,
            in_features=dim,
            decode=False,
            dropout_rate=drop,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
            out_bias_init=zero_init,
            out_kernel_init=xavier_init,
        )

        self.crossattn = nnx.MultiHeadAttention(
            num_heads=attn_heads,
            in_features=dim,
            decode=False,
            dropout_rate=drop,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
            out_bias_init=zero_init,
            out_kernel_init=xavier_init,
        )

        self.adaln = nnx.Linear(
            dim, 6 * dim, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
        )
        self.mlp_block = FeedForward(dim)

    def __call__(self, x: Array, y: Array, cond: Array):
        cond = self.adaln(nnx.silu(cond))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.array_split(cond, 6, axis=-1)
        )

        x = x + gate_msa * self.attention(
            t2i_modulate(self.norm_1(x), shift_msa, scale_msa)
        )
        x = x + self.crossattn(self.norm_2(x), y)
        x = x + gate_mlp * self.mlp_block(
            t2i_modulate(self.norm_3(x), shift_mlp, scale_mlp)
        )

        return x


class FinalMLP(nnx.Module):
    def __init__(
        self, hidden_size, patch_size, out_channels, rngs=nnx.Rngs(config.seed)
    ):
        super().__init__()

        self.norm_final = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, use_bias=False, scale_init=one_init
        )

        self.linear = nnx.Linear(
            hidden_size,
            patch_size[0] * patch_size[1] * out_channels,
            rngs=rngs,
            kernel_init=zero_init,
            bias_init=zero_init,
        )

        self.adaln_linear = nnx.Linear(
            hidden_size,
            2 * hidden_size,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=zero_init,
        )

    def __call__(self, x_input: Array, cond: Array):
        linear_cond = nnx.silu(self.adaln_linear(cond))
        shift, scale = jnp.array_split(linear_cond, 2, axis=-1)

        x = t2i_modulate(self.norm_final(x_input), shift, scale)
        x = self.linear(x)

        return x


class TransformerBackbone(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()

        # # Define scaling ranges for m_f and m_a
        mf_min, mf_max = 0.5, 4.0
        ma_min, ma_max = 0.5, 1.0

        # self.layers = [
        #     DiTBlock(embed_dim, num_heads, drop=0.0) for _ in range(num_layers)
        # ]

        self.layers = []

        for v in tqdm(range(num_layers)):
            # # Calculate scaling factors for the v-th layer using linear interpolation
            mf = mf_min + (mf_max - mf_min) * v / (num_layers - 1)
            ma = ma_min + (ma_max - ma_min) * v / (num_layers - 1)

            # # Scale the dimensions according to the scaling factors
            scaled_mlp_dim = int(embed_dim * mf)
            scaled_num_heads = max(1, int(num_heads * ma))
            scaled_num_heads = self._nearest_divisor(scaled_num_heads, embed_dim)
            mlp_ratio = scaled_mlp_dim / embed_dim

            self.layers.append(DiTBlock(embed_dim, scaled_num_heads, mlp_ratio))

    def _nearest_divisor(self, scaled_num_heads, embed_dim):
        # Find all divisors of embed_dim
        divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
        # Find the nearest divisor
        nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))

        return nearest

    def __call__(self, x, c_emb, cond):
        for layer in self.layers:
            x = layer(x, c_emb, cond)

        return x


# patch mixer module
class PatchMixer(nnx.Module):
    def __init__(self, patch_mix_dim, embed_dim, attn_heads, n_layers=2):
        super().__init__()
        self.input_map = nnx.Sequential(
            nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init),
            nnx.Linear(
                embed_dim,
                patch_mix_dim,
                rngs=rngs,
                bias_init=zero_init,
                kernel_init=xavier_init,
            ),
        )

        self.cond_map = nnx.Linear(
            embed_dim,
            patch_mix_dim,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
        )
        self.layers = [
            DiTBlock(patch_mix_dim, attn_heads, drop=0.0) for _ in range(n_layers)
        ]

        self.out_map = nnx.Linear(
            patch_mix_dim,
            embed_dim,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
        )

    def __call__(self, x: Array, y: Array, c: Array) -> Array:
        x = self.input_map(x)
        y = self.cond_map(y)
        c = self.cond_map(c)

        for layer in self.layers:
            x = layer(x, y, c)

        x = self.out_map(x)
        return x


class CondProcessor(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_heads: int,
    ):
        self.norm_1 = nnx.LayerNorm(
            embed_dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init
        )
        self.norm_2 = nnx.LayerNorm(
            embed_dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init
        )

        self.attention = nnx.MultiHeadAttention(
            num_heads=attn_heads,
            in_features=embed_dim,
            decode=False,
            dropout_rate=0.0,
            rngs=rngs,
            bias_init=zero_init,
            kernel_init=xavier_init,
            out_bias_init=zero_init,
            out_kernel_init=xavier_init,
        )
        self.mlp = MLP(embed_dim)

    def __call__(self, y):
        y = y + self.attention(self.norm_1(y))
        y = y + self.mlp(self.norm_2(y))
        return y


def get_mask(
    batch: int,
    length: int,
    mask_ratio: float,
    key=randkey,
) -> Dict[str, jnp.ndarray]:

    len_keep = int(length * (1 - mask_ratio))
    key_noise = jax.random.split(key)[0]
    noise = jax.random.uniform(key_noise, (batch, length))  # noise in [0, 1]
    ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
    ids_restore = jnp.argsort(ids_shuffle, axis=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    mask = jnp.ones((batch, length))
    mask = mask.at[:, :len_keep].set(0)

    # Apply the inverse shuffle using indexing
    batch_indices = jnp.arange(batch)[:, None]
    original_indices = jnp.arange(length)[None, :]
    shuffled_mask = mask[batch_indices, ids_shuffle]

    return {"mask": shuffled_mask, "ids_keep": ids_keep, "ids_restore": ids_restore}


def mask_out_token(x: jnp.ndarray, ids_keep: jnp.ndarray) -> jnp.ndarray:
    """Mask out tokens specified by ids_keep using lax.gather."""

    N, L, D = x.shape  # batch, length, dim
    N_keep, K = ids_keep.shape  # batch, number of tokens to keep

    # Create index arrays
    batch_indices = jnp.arange(N)[:, None, None]  # Shape (N, 1, 1)
    keep_indices = ids_keep[:, :, None]  # Shape (N, K, 1)
    dim_indices = jnp.arange(D)[None, None, :]  # Shape (1, 1, D)

    # Broadcast the index arrays to the desired output shape (N, K, D)
    batch_indices = jnp.broadcast_to(batch_indices, (N, K, D))
    keep_indices = jnp.broadcast_to(keep_indices, (N, K, D))
    dim_indices = jnp.broadcast_to(dim_indices, (N, K, D))

    # Use advanced indexing to gather the desired tokens
    x_masked = x[batch_indices, keep_indices, dim_indices]
    # print(f'x masked {x.shape}')

    return x_masked


def unmask_tokens(x: jnp.ndarray, ids_restore: jnp.ndarray, mask_token) -> jnp.ndarray:
    """Unmask tokens using provided mask token with take_along_axis."""
    # Repeat mask token for batch size and missing tokens
    N, L_masked, D = x.shape
    L_original = ids_restore.shape[1]

    # Repeat mask token for batch size and missing tokens
    num_missing = L_original - L_masked
    mask_tokens = jnp.tile(mask_token, (N, num_missing, 1))

    # Concatenate original tokens with mask tokens
    x_ = jnp.concatenate([x, mask_tokens], axis=1)

    # Create index arrays
    batch_indices = jnp.arange(N)[:, None, None]  # (N, 1, 1)
    seq_indices = ids_restore[:, :, None]  # (N, L_original, 1)
    depth_indices = jnp.arange(D)[None, None, :]  # (1, 1, D)

    # Broadcast indices to the desired output shape (N, L_original, D)
    batch_indices = jnp.broadcast_to(batch_indices, (N, L_original, D))
    seq_indices = jnp.broadcast_to(seq_indices, (N, L_original, D))
    depth_indices = jnp.broadcast_to(depth_indices, (N, L_original, D))

    # Use advanced indexing to gather and reorder
    x_unmasked = x_[batch_indices, seq_indices, depth_indices]
    # print(f'{x_unmasked.shape = }')

    return x_unmasked


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
        patchmix_dim=512,
        caption_dim=1152,
        mask_ratio=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.patch_embedder = PatchEmbed(
            patch_size=patch_size[0], in_channels=in_channels, dim=embed_dim
        )
        self.num_patches = self.patch_embedder.num_patches

        # conditioning layers
        self.time_embedder = TimestepEmbedder(embed_dim)
        self.cap_embedder = CaptionEmbedder(embed_dim, caption_dim, 2)
        self.y_process = CondProcessor(embed_dim, attn_heads)
        self.y_pool = MLP(embed_dim)

        mask_token_value = jnp.zeros(
            (1, 1, patch_size[0] * patch_size[1] * in_channels), dtype=jnp.bfloat16
        )
        self.mask_token = nnx.Param(mask_token_value)
        jax.lax.stop_gradient(self.mask_token.value)

        self.patch_mixer = PatchMixer(
            patchmix_dim, embed_dim, attn_heads, patchmix_layers
        )

        self.backbone = TransformerBackbone(
            embed_dim,
            num_layers=num_layers,
            num_heads=attn_heads,
        )

        self.final_linear = FinalMLP(embed_dim, patch_size=patch_size, out_channels=4)

    def _unpatchify(self, x, patch_size=(2, 2), height=22, width=44):
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

    def __call__(self, x: Array, t: Array, y_cap: Array, mask_ratio=0.0, train=False):
        bsize, height, width, channels = x.shape

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            h=height // self.patch_size[0],
            w=width // self.patch_size[1],
        )
        pos_embed = jnp.expand_dims(pos_embed, axis=0)

        mask = None
        x = self.patch_embedder(x) + pos_embed.astype(x.dtype)

        # conditioning layers
        y = self.cap_embedder(y_cap)  # (b, embdim)
        y = self.y_process(y[:, None, :])
        y_pooled = self.y_pool(y.mean(axis=-2))

        t = self.time_embedder(t) + y_pooled
        cond_ty = t.astype(x.dtype)[:, None, :]

        x = self.patch_mixer(x, y, cond_ty)

        if mask_ratio > 0:
            # mask_dict = mask
            mask_dict = get_mask(x.shape[0], x.shape[1], mask_ratio=mask_ratio)

            ids_keep = mask_dict["ids_keep"]
            ids_restore = mask_dict["ids_restore"]
            mask = mask_dict["mask"]

            x = mask_out_token(x, ids_keep)

        x = self.backbone(x, y, cond_ty)

        x = self.final_linear(x, cond_ty)

        if mask_ratio > 0:
            x = unmask_tokens(x, ids_restore, self.mask_token)

        x = self._unpatchify(x)

        return x, mask

    def log_activation_stats(self, layer_name, activations):
        mean_val = jnp.mean(activations)
        std_val = jnp.std(activations)
        jax.debug.print(
            "layer {val} / mean {mean_val} / stddev {std_val}",
            val=layer_name,
            mean_val=mean_val,
            std_val=std_val,
        )


# rectifed flow forward pass, loss, and smapling
class RectFlowWrapper(nnx.Module):
    def __init__(self, model: nnx.Module, mask_ratio=config.mask_ratio):
        self.model = model
        self.mask_ratio = mask_ratio

    def __call__(self, x_input: Array, cond: Array, mask_ratio=config.mask_ratio):
        b_size = x_input.shape[0]  # batch_size

        rand = jrand.uniform(randkey, (b_size,)).astype(jnp.bfloat16)
        rand_t = nnx.sigmoid(rand)

        inshape = [1] * len(x_input.shape[1:])
        texp = rand_t.reshape([b_size, *(inshape)]).astype(jnp.bfloat16)

        z_noise = jrand.uniform(randkey, x_input.shape).astype(
            jnp.bfloat16
        )  # input noise with same dim as image
        z_noise_t = (1 - texp) * x_input + texp * z_noise

        v_thetha, mask = self.model(
            z_noise_t, rand_t, cond, mask_ratio=self.mask_ratio, train=True
        )
        # print(f'{v_thetha.shape = }')

        mean_dim = list(
            range(1, len(x_input.shape))
        )  # across all dimensions except the batch dim

        mean_square = (z_noise - x_input - v_thetha) ** 2  # squared difference
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss

        loss = jnp.mean(batchwise_mse_loss)

        if mask_ratio > 0:
            unmask = 1 - mask
            loss = (loss * unmask).sum(axis=1) / unmask.sum(axis=1)

        return loss.mean()

    def sample(self, cond, sample_steps=50, cfg=2.0):
        z_latent = jrand.normal(randkey, (len(cond), 22, 44, 4))  # noise
        b_size = z_latent.shape[0]
        dt = 1.0 / sample_steps

        dt = jnp.array([dt] * b_size)
        dt = jnp.reshape(dt, shape=(b_size, *([1] * len(z_latent.shape[1:]))))

        images = [z_latent]

        for step in tqdm(range(sample_steps, 0, -1)):
            t = step / sample_steps
            t = jnp.array([t] * b_size, device=z_latent.device).astype(jnp.bfloat16)

            vcond, _ = self.model(z_latent, t, cond, mask_ratio=0.0)

            if cfg > 1.0:
                null_cond = jnp.zeros_like(cond)
                v_uncond, _ = self.model(z_latent, t, null_cond)  # type: ignore
                vcond = v_uncond + cfg * (vcond - v_uncond)

            z_latent = z_latent - dt * vcond
            images.append(z_latent)

        return images[-1] / config.vaescale_factor
