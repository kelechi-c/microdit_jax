import jax, math
from jax import Array, numpy as jnp
from flax import nnx
from einops import rearrange
from .data_utils import config
from .md_layers import modulate

rngs = nnx.Rngs(config.seed)
linear_init = nnx.initializers.xavier_uniform()
linear_bias_init = nnx.initializers.constant(0)


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


class FinalMLP(nnx.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        linear_init = nnx.initializers.xavier_uniform()
        bias_init = nnx.initializers.constant(0)

        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            rngs=rngs,
            kernel_init=linear_init,
            bias_init=bias_init,
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


# self attention block
class SelfAttention(nnx.Module):
    def __init__(self, attn_heads, embed_dim, rngs: nnx.Rngs, drop=0.1):
        super().__init__()
        self.attn_heads = attn_heads
        self.head_dim = embed_dim // attn_heads

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
        print(f"attn out shape => {output.shape}")
        return output


class CrossAttention(nnx.Module):
    def __init__(self, attn_heads, embed_dim, cond_dim, rngs: nnx.Rngs, drop=0.1):
        super().__init__()
        self.attn_heads = attn_heads
        self.head_dim = embed_dim // attn_heads

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

        # print(f"attn out shape => {output.shape}")
        return output
