import jax, math
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from einops import rearrange
from itertools import repeat
from collections import abc
from data_configs import config


rngs = nnx.Rngs(config.seed)


def _ntuple(n):
    def parse(x):
        if isinstance(x, abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


# 3D rotary positional embedding
class RoPE3D(nnx.Module):
    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (
                self.base ** (jnp.arange(0, D, 2).float().to(device) / D)
            )
            t = (
                jnp.arange(seq_len, device=device, dtype=inv_freq.dtype)
                / interpolation_scale
            )
            freqs = jnp.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = jnp.concat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return jnp.concat((-x2, x1), dim=-1)


    def apply_rope1d(tokens, pos1d, cos_mat, sin_mat):
        """
        Applies the RoPE operation to the given tokens.

        Args:
            tokens: A NumPy array of shape (batch_size, seq_len, dim).
            pos1d: A NumPy array of shape (batch_size, seq_len) containing positional indices.
            cos_mat: A NumPy array of shape (max_pos, dim) containing cosine values.
            sin_mat: A NumPy array of shape (max_pos, dim) containing sine values.

        Returns:
            A NumPy array of the same shape as `tokens` after applying RoPE.
        """

        # Get cosine and sine values for the given positions
        cos_values = cos_mat[pos1d]
        sin_values = sin_mat[pos1d]

        # Add a new dimension for the head dimension
        cos_values = cos_values[:, :, None, :]
        sin_values = sin_values[:, :, None, :]

        # Rotate half of the tokens
        rotated_tokens = jnp.roll(tokens, shift=-tokens.shape[-1] // 2, axis=-1)

        # Apply RoPE
        out = (tokens * cos_values) + (rotated_tokens * sin_values)

        return out

    def forward(self, tokens, positions):
        """
        in:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        out:
            * tokens after appplying RoPE3D (batch_size x nheads x ntokens x x dim)
        """
        assert (
            tokens.size(3) % 3 == 0
        ), "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2  # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(
            D, max_poses[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t
        )
        cos_y, sin_y = self.get_cos_sin(
            D, max_poses[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h
        )
        cos_x, sin_x = self.get_cos_sin(
            D, max_poses[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w
        )
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = jnp.concat((t, y, x), dim=-1)
        
        return tokens


class Patchifier(nnx.Module):
    def __init__(self, patch_size=2):
        super().__init__()
        self.patch_size_3d = (1, patch_size, patch_size)

    def patchify(self, latents, frame_rate, scale_grid):
        patched_latents = rearrange(
            latents,
            "b (f p1) (h p2) (w p3) c -> b (f h w) (c p1 p2 p3)",
            p1=self.patch_size_3d[0],
            p2=self.patch_size_3d[1],
            p3=self.patch_size_3d[2]
        )

        return patched_latents

    def unpatchifier(
        self, latents,
        frame_rates, scale_grid,
        out_height: int,
        out_width: int,
        out_frame_count: int,
        out_channels: int,
    ):
        out_height = out_height // self.patch_size_3d[1]
        out_width = out_width // self.patch_size_3d[2]

        unpatched_latents = rearrange(
            latents,
            "b (f h w) (c p q) -> b f (h p) (w p) c",
            f=out_frame_count,
            h=out_height,
            w=out_width,
            p=self.patch_size_3d[1],
            q=self.patch_size_3d[2],
        )

        return unpatched_latents

    def get_grid(self, og_frame_count, og_height, og_width, batch_size, scale_grid):
        f = og_frame_count // self.patch_size_3d[0]
        h = og_height // self.patch_size_3d[1]
        w = og_width // self.patch_size_3d[2]

        grid_h = jnp.arange(h, dtype=config.dtype)
        grid_w = jnp.arange(w, dtype=config.dtype)
        grid_f = jnp.arange(f, dtype=config.dtype)

        grid_3d = jnp.meshgrid(grid_f, grid_h, grid_w).stack(grid_3d, axis=0)
        grid_3d = jnp.expand_dims(grid_3d, axis=0).repeat(batch_size, axis=0)

        # if scale_grid is not None:
        #     for k in range(3):
        #         if isinstance(scale_grid[k], Array):
        #             scale =
        grid_3d = rearrange(grid_3d, 'b f h w c -> b (f h w) c')

        return grid_3d


class MochiPatcher(nnx.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.flatten = flatten
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            device=device,
        )
        assert norm_layer is None
        self.norm = (
            norm_layer(embed_dim, device=device) if norm_layer else nn.Identity()
        )

    def forward(self, x):
        B, _C, T, H, W = x.shape
        if not self.dynamic_img_pad:
            assert (
                H % self.patch_size[0] == 0
            ), f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
            assert (
                W % self.patch_size[1] == 0
            ), f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
        else:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = rearrange(x, "B C T H W -> (B T) C H W", B=B, T=T)
        x = self.proj(x)

        # Flatten temporal and spatial dimensions.
        if not self.flatten:
            raise NotImplementedError("Must flatten output.")
        x = rearrange(x, "(B T) C H W -> B (T H W) C", B=B, T=T)

        x = self.norm(x)
        return x


class Attention(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        cross_dim: int,
        attn_heads: int = 8, 
        drop: int=0.1,
        rngs = rngs
    ):
        super().__init__()
        self.attn_heads = attn_heads
        self.head_dim = input_dim // attn_heads
        self.query_dim = input_dim
        self.cross_dim = cross_dim or input_dim
        self.dropout = drop

        linear_init = nnx.initializers.xavier_uniform()
        bias_init = nnx.initializers.constant(0)

        self.q_linear = nnx.Linear(
            input_dim,
            input_dim,
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=linear_init,
        )

        self.k_linear = nnx.Linear(
            cross_dim, input_dim, bias_init=bias_init, kernel_init=linear_init, rngs=rngs
        )
        self.v_linear = nnx.Linear(
            cross_dim,
            input_dim,
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=linear_init,
        )

        self.outproject = nnx.Linear(
            input_dim, input_dim, rngs=rngs,
            bias_init=bias_init,
            kernel_init=linear_init
        )
        
        self.dropout = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x_input: jax.A, y_cond: Array) -> jax.Array:
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


class DiTransformerBlock(nnx.Module):
    def __init__(self, attn_heads, dim, norm_eps=1e-5, rngs=rngs):
        super().__init__()
        self.norm_1 = nnx.LayerNorm(dim, epsilon=norm_eps, rngs=rngs)
        self.self_attention = Attention(dim, dim, attn_heads=attn_heads)
        self.cross_attention = Attention(dim, dim, attn_heads)
        self.feedforward_layer = FeedForward(dim, dim)

    def __call__(self, x_input: Array, y_cond: Array):
        batch_size = x_input.shape[0]
        
        x = self.norm_1(x_input)


class FeedForward(nnx.Module):
    def __init__(self, input_dim, out_dim, hidden_mlp_ratio=2, rngs=rngs, drop=0.0):
        super().__init__()
        hidden_size = hidden_mlp_ratio * input_dim 

        self.linear_a = nnx.Linear(input_dim, 2 * hidden_size)
        self.linear_b = nnx.Linear(hidden_size, input_dim)
        
    def __call__(self, x_input: Array) -> Array:
        x, gate = jnp.array_split(self.w1(x_input), 2, axis=-1)
        x = self.linear_b(nnx.silu(x) * gate)
        
        return x


# adapted from lightricks/LTX-Video
class SinusoidalPositionalEmbedding(nnx.Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = jnp.arange(max_seq_length).unsqueeze(1)
        div_term = jnp.exp(
            jnp.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = jnp.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = jnp.sin(position * div_term)
        pe[0, :, 1::2] = jnp.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x


def get_timestep_embedding(
    timesteps: Array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * jnp.arange(
        start=0, end=half_dim, dtype=jnp.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = jnp.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = jnp.cat([jnp.sin(emb), jnp.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = jnp.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, (0, 1, 0, 0))
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid, w, h, f):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = rearrange(grid, "c (f h w) -> c f h w", h=h, w=w)
    grid = rearrange(grid, "c f h w -> c h w f", h=h, w=w)
    grid = grid.reshape([3, 1, w, h, f])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.transpose(1, 0, 2, 3)
    return rearrange(pos_embed, "h w f c -> (f h w) c")


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # use half of dimensions to encode grid_h
    emb_f = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*T, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*T, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*T, D/3)

    emb = jnp.concatenate([emb_h, emb_w, emb_f], axis=-1)  # (H*W*T, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos_shape = pos.shape

    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    out = out.reshape([*pos_shape, -1])[0]

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb
