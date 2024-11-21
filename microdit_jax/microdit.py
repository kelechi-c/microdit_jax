import flax.linen
import jax, math
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from einops import rearrange

from .data_utils import add_masked_patches, config, remove_masked_patches
from .attention_mlp import OutputMLP
from .md_layers import (
    PoolMLP, SimpleMLP, TimestepEmbedder,
    TransformerEncoderBlock, CrossAttention,
     PatchEmbed, CaptionEmbedder, get_2d_sincos_pos_embed,
    LabelEmbedder, TransformerBackbone
)

rngs = nnx.Rngs(3)

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
        mlp_dim,
        cond_embed_dim,
        dropout=0.1,
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
            mlp_dim=mlp_dim,
        )
        
        self.final_linear = OutputMLP(embed_dim, patch_size=patch_size, out_channels=3)

    def unpatchify(self, x: Array) -> Array:

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
        bsize, channels, height, width = x.shape
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

        x = self.final_linear(x)

        # add back masked patches
        if mask is not None:
            x = add_masked_patches(x, mask)

        x = self.unpatchify(x)

        return x

    def sample(self, z_latent, cond, sample_steps=50, cfg=2.0):
        b_size = z_latent.shape[0]
        dt = 1.0 / sample_steps

        dt = jnp.array([dt] * b_size)
        dt = jnp.reshape(dt, shape=(b_size, *([1] * len(z_latent.shape[1:]))))

        images = [z_latent]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = (
                jnp.array([t] * b_size)
                # .to_device(z_latent.device)
                .astype(z_latent.dtype)
            )

            vc = self(z_latent, t, cond, None)
            null_cond = jnp.zeros_like(cond)
            vu = self.__call__(z_latent, t, null_cond)
            vc = vu + cfg * (vc - vu)

            z = z_latent - dt * vc
            images.append(z)

        return images  # [-1]# / config.vaescale_factor