import jax, math
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from .data_utils import nearest_divisor, config
from .attention_mlp import SelfAttention, SimpleMLP

rkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)
randkey = jrand.key(config.seed)

# init values
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)
linear_bias_init = nnx.initializers.constant(1)


## Embedding layers
# input patchify layer, 2D image to patches
# input patchify layer, 2D image to patches
class PatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs=rngs,
        patch_size: int = 2,
        in_chan: int = 4,
        embed_dim: int = 1024,
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
        x = x.reshape(
            batch_size, -1, embed_dim
        )  # Shape: (batch_size, num_patches, embed_dim)

        return x


# modulation with shift and scale
def modulate(x_array: Array, shift, scale) -> Array:
    x = x_array * scale.unsqueeze(1)
    x = x + shift.unsqueeze(1)

    return x

# equivalnet of F.linear
def linear(array: Array, weight: Array, bias: Array | None = None) -> Array:
    out = jnp.dot(array, weight)

    if bias is not None:
        out += bias

    return out


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
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half) / half).to_device(time_array.device)
        args = jnp.float_(time_array[:, None]) * freqs[None]
        
        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concat([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)

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
            num_classes + use_cfg_embeddings, hidden_size,
            rngs=rngs, embedding_init=nnx.initializers.normal(0.02)
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


###############
# DiT blocks_ #
###############
class DiTBlock(nnx.Module):
    def __init__(self, hidden_size=1024, num_heads=6, mlp_ratio=4):
        super().__init__()

        # initializations
        lnbias_init = nnx.initializers.constant(1)
        lnweight_init = nnx.initializers.constant(1)

        self.norm_1 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, bias_init=lnbias_init
        )
        self.attention = SelfAttention(num_heads, hidden_size, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs, scale_init=lnweight_init)

        self.adaln_linear = nnx.Linear(
            in_features=hidden_size,
            out_features=6*hidden_size,
            use_bias=True,
            bias_init=zero_init,
            rngs=rngs,
            # kernel_init=zero_init,
        )

        self.mlp_block = SimpleMLP(hidden_size)

    def __call__(self, x_img: Array, cond):

        cond = self.adaln_linear(nnx.silu(cond))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.array_split(cond, 6, axis=1)
        )

        attn_mod_x = self.attention(
            modulate(self.norm_1(x_img), shift_msa, scale_msa)
        )

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
            kernel_init=linear_init,
            bias_init=linear_init,
        )
        self.adaln_linear = nnx.Linear(
            hidden_size,
            2 * hidden_size,
            rngs=rngs,
            kernel_init=linear_init,
            bias_init=linear_init,
        )

    def __call__(self, x_input: Array, cond: Array):
        linear_cond = nnx.silu(self.adaln_linear(cond))
        shift, scale = jnp.array_split(linear_cond, 2, axis=1)

        x = modulate(self.norm_final(x_input), shift, scale)
        x = self.linear(x)
        print(f"final dit mlp {type(x)} {x.shape}")
        return


class DiTBackbone(nnx.Module):
    def __init__(
        self,
        patch_size=(4, 4),
        in_channels=3,
        hidden_size=1024,
        depth=4,
        attn_heads=6,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_chan = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.attn_heads = attn_heads

        self.img_embedder = PatchEmbed(
            img_size=(config.img_size, config.img_size),
            in_chan=in_channels,
            embed_dim=hidden_size,
        )
        self.time_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.img_embedder.num_patches

        self.pos_embed = nnx.Param(jnp.zeros(shape=(1, num_patches, hidden_size)))
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.value.shape[-1], int(self.img_embedder.num_patches**0.5)
        )
        sincos2d_data = jnp.expand_dims(pos_embed.astype(jnp.float32), axis=0)
        print(f"sincos {type(sincos2d_data)} {sincos2d_data.shape}")
        # self.pos_embed.value.copy_from(sincos2d_data)  # type: ignore
        self.pos_embed.value = jnp.copy(sincos2d_data)

        dit_blocks = [
            DiTBlock(hidden_size, num_heads=attn_heads) for _ in tqdm(range(depth))
        ]
        self.final_mlp = FinalMLP(hidden_size, patch_size[0], self.out_channels)
        self.dit_layers = nnx.Sequential(*dit_blocks)

    def unpatchify(self, x: Array) -> Array:
        c = self.out_channels
        p = self.img_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = jnp.reshape(x, shape=(x.shape[0], h, w, p, p, c))
        x = jnp.einsum("nhwpqc->nchpwq", x)
        img = jnp.reshape(x, shape=(x.shape[0], c, h * p, w * p))

        return img

    def __call__(self, x: Array, t: Array, y_cond: Array):
        x = self.img_embedder(x) + self.pos_embed
        t_embed = self.time_embedder(t)

        cond = t_embed + y_cond
        x = self.dit_layers(x, cond)
        x = self.final_mlp(x, cond)  # type: ignore
        x = self.unpatchify(x)

        print(f"ditback out -> {x.shape}")

        return x

    def cfg_forward(self, x_img, t, y_cond, cfg_scale):
        half = x_img[: len(x_img) // 2]
        combined = jnp.concat([half, half], axis=0)
        model_out = self.__call__(combined, t, y_cond)

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = jnp.split(eps, len(eps) // 2, axis=0)

        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = jnp.concat([half_eps, half_eps], axis=0)
        cfg_out = jnp.concat([eps, rest], axis=1)

        return cfg_out


class TransformerBackbone(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        class_embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        num_experts: int = 4,
        active_experts: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_embedding = nnx.Linear(input_dim, embed_dim, rngs=rngs)
        self.class_embedding = nnx.Linear(class_embed_dim, embed_dim, rngs=rngs)

        # Define scaling ranges for m_f and m_a
        mf_min, mf_max = 0.5, 4.0
        ma_min, ma_max = 0.5, 1.0

        self.layers = []

        for v in range(num_layers):
            # Calculate scaling factors for the v-th layer using linear interpolation
            mf = mf_min + (mf_max - mf_min) * v / (num_layers - 1)
            ma = ma_min + (ma_max - ma_min) * v / (num_layers - 1)
            print(f'mf {mf}, ma {ma}')

            # Scale the dimensions according to the scaling factors
            scaled_mlp_dim = int(mlp_dim * mf)
            scaled_num_heads = max(1, int(num_heads * ma))
            scaled_num_heads = nearest_divisor(scaled_num_heads, embed_dim)
            mlp_ratio = int(scaled_mlp_dim / embed_dim)

            # Choose layer type based on the layer index (even/odd)
            if v % 2 == 0:  # Even layers use regular DiT blocks
                self.layers.append(
                    DiTBlock(
                        embed_dim, scaled_num_heads, mlp_ratio, 1, 1, attn_drop=dropout
                    )
                )
            else:  # Odd layers use MoE DiT block
                self.layers.append(
                    DiTBlock(
                        embed_dim,
                        scaled_num_heads,
                        mlp_ratio,
                        num_experts,
                        active_experts,
                        attn_drop=dropout,
                    )
                )

        self.output_layer = nnx.Linear(embed_dim, input_dim, rngs=rngs)

    def __call__(self, x, c_emb):
        x = self.input_embedding(x)
        
        class_emb = self.class_embedding(c_emb)

        for layer in self.layers:
            x = layer(x, class_emb)

        x = self.output_layer(x)
        
        return x


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


########################
# Patch Mixer components
########################

class EncoderMLP(nnx.Module):
    def __init__(self, hidden_size, rngs: nnx.Rngs, dropout=0.1):
        super().__init__()
        self.layernorm = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.linear1 = nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(2 * hidden_size, hidden_size, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x_input: jax.Array) -> jax.Array:
        x = self.layernorm(x_input)
        x = nnx.silu(self.linear1(x))
        x = self.linear2(x)

        return x


class TransformerEncoderBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layernorm = nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.self_attention = SelfAttention(num_heads, embed_dim, rngs=rngs)
        self.mlp_layer = EncoderMLP(embed_dim, rngs=rngs)
        
    def __call__(self, x: Array):
        x = x + self.layernorm(self.self_attention(x))
        x = x + self.layernorm(self.mlp_layer(x))
        
        return x
