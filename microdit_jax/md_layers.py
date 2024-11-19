import jax, math
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from einops import rearrange
from typing import Tuple
from .data_utils import jnp_topk, nearest_divisor


rkey = jrand.key(3)
rngs = nnx.Rngs(3)

xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)
linear_bias_init = nnx.initializers.constant(1)


# input patchify layer, 2D image to patches
class PatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs=rngs,
        patch_size = 4,
        img_size = 256,
        in_chan: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.gridsize = tuple([s // p for s, p in zip(img_size, patch_size)])
        self.num_patches = self.gridsize[0] * self.gridsize[1]

        self.conv_project = nnx.Conv(
            in_chan, embed_dim, kernel_size=patch_size,
           strides=patch_size, rngs=rngs,
        )

    def __call__(self, img: Array) -> Array:
        x = self.conv_project(img)
        x = x.flatten()

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


class SimpleMLP(nnx.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear_1 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = nnx.silu(self.linear_1(x))
        x = self.linear_2(x)

        return x


class PoolMLP(nnx.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear_1 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = nnx.avg_pool(x, (1,))
        print(f'avg pool; {x.shape}')
        x = jnp.reshape(x, shape=(x.shape[0], -1))
        print(f'avg pool rsd {x.shape}')

        x = nnx.gelu(self.linear_1(x))
        x = self.linear_2(x)

        return x

#### MoE Gate
class MoEGate(nnx.Module):
    def __init__(
        self, embed_dim, num_experts=4, experts_per_token=2, aux_loss_alpha=0.01
    ):
        super().__init__()
        self.top_k = experts_per_token
        self.routed_experts = num_experts
        self.score_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # top_k selection algo
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nnx.Param(jnp.ones((4, embed_dim)))
        print(self.weight.shape)
        # self.linear_gate = nnx.Linear()

    def __call__(self, hidden_states: Array) -> Tuple:
        bsize, seq_len, h = hidden_states.shape
        
        # gating score
        print(f'hidden 1 : {hidden_states.shape}')
        hidden_states = jnp.reshape(hidden_states, (-1, h))
        print(f'hiddenstate: {hidden_states.shape}')
        logits = jnp.dot(hidden_states, self.weight.T)
        print(f'logits hiddenstate: {logits.shape}')
        scores = nnx.softmax(logits, axis=-1)
        print(f'scores {scores.shape}')

        topk_idx, topk_weight = jax.lax.top_k(scores, k=self.top_k)
        print(f'topk_weight {topk_weight.shape}')

        # normalize to sum to 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
            topk_weight = topk_weight / denominator

        # expert level computation of auxiliary loss
        # always compute topk based on naive greedy topk method

        if self.train and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            aux_loss = 0.0

            topk_idx_for_auxloss = jnp.reshape(topk_idx, (bsize, -1))
            if self.seq_aux:
                scores_for_seq_aux = jnp.reshape(scores_for_aux, (bsize, seq_len, -1))
                ce = jnp.zeros(
                    (bsize, self.routed_experts), device=hidden_states.device
                )
                ones_add = jnp.ones((bsize, seq_len * aux_topk))
                ce = jnp.add.at(ce, 1, ones_add)
                ce /= seq_len * aux_topk / self.routed_experts

                aux_loss = (ce * scores_for_seq_aux.mean(axis=1)).sum(
                    axis=1
                ).mean() * self.alpha

            else:
                mask_ce = nnx.one_hot(
                    jnp.reshape(topk_idx_for_auxloss, (-1)),
                    num_classes=self.routed_experts,
                )
                ce = mask_ce.astype(jnp.float32).mean(0)
                pi = scores_for_aux.mean()
                fi = ce * self.routed_experts
                aux_loss = (pi * fi).sum() * self.alpha

        else:
            aux_loss = None

        print(f"gate shape => {topk_weight.shape}")
        # print(topk_weight)

        return topk_idx, topk_weight, aux_loss
    
# mixture of experts MLP layer
class MoEMLP(nnx.Module):
    def __init__(self, hidden_size, intersize, pretrain_tp=2):
        self.hidden_size = hidden_size
        self.intersize = intersize
        self.pretrain_tp = pretrain_tp
        self.gate_project = nnx.Linear(
            self.hidden_size, self.intersize,
            use_bias=False, rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform()
        )
        self.up_project = nnx.Linear(
            self.hidden_size, self.intersize, use_bias=False, rngs=rngs
        )
        self.down_project = nnx.Linear(
            self.intersize, self.hidden_size, use_bias=False, rngs=rngs
        )

    def __call__(self, x_input: Array):
        batch_size, seq_len, _ = x_input.shape

        if self.pretrain_tp > 1:
            # Calculate slice width
            w_slice = self.intersize // self.pretrain_tp

            # Split weights into expert chunks
            gate_slices = jnp.split(self.gate_project.kernel, self.pretrain_tp, axis=0)
            up_slices = jnp.split(self.up_project.kernel, self.pretrain_tp, axis=0)
            down_slices = jnp.split(self.down_project.kernel, self.pretrain_tp, axis=1)

            # Compute projections for each expert
            gate_projs = []
            up_projs = []

            for k in range(self.pretrain_tp):
                # Reshape input for batch matrix multiplication
                x_reshaped = x_input.reshape(-1, self.hidden_size)

                # Compute gate and up projections
                gate_proj_k = jnp.dot(x_reshaped, gate_slices[k].T)
                up_proj_k = jnp.dot(x_reshaped, up_slices[k].T)

                gate_projs.append(gate_proj_k)
                up_projs.append(up_proj_k)

            # Concatenate expert outputs
            gate_proj = jnp.concatenate(gate_projs, axis=-1)
            up_proj = jnp.concatenate(up_projs, axis=-1)

            # Reshape back to (batch_size, seq_len, intersize)
            gate_proj = gate_proj.reshape(batch_size, seq_len, -1)
            up_proj = up_proj.reshape(batch_size, seq_len, -1)

            # Compute intermediate states
            activated_gates = nnx.silu(gate_proj)  # Changed from silu to sigmoid for gating
            inter_states = jnp.split(activated_gates * up_proj, self.pretrain_tp, axis=-1)

            # Compute final projection
            down_projs = []
            for k in range(self.pretrain_tp):
                down_proj_k = jnp.dot(
                    inter_states[k].reshape(-1, w_slice),
                    down_slices[k]
                )
                down_projs.append(down_proj_k)

            # Sum expert outputs and reshape
            down_proj = sum(down_projs)
            down_proj = down_proj.reshape(batch_size, seq_len, self.hidden_size)

        else:
            # Standard MLP path
            gate_output = jnp.sigmoid(self.gate_project(x_input))  # Changed from silu to sigmoid
            up_output = self.up_project(x_input)
            down_proj = self.down_project(gate_output * up_output)

        print(f"moe mlp shape => {down_proj.shape}")
        return down_proj


class SparseMoEBlock(nnx.Module):
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=8, experts_per_token=2, train: bool=True, rngs=rngs):
        super().__init__()
        self.experts_pertoken = experts_per_token
        self.expert_models = [MoEMLP(hidden_size=embed_dim, intersize=mlp_ratio*embed_dim) for _ in range(num_experts)]
        # self.experts = nnx.Sequential(*self.expert_models)
        self.router_gate = MoEGate(embed_dim, num_experts)
        self.n_shared_experts = 2
        self.training = train

        if self.n_shared_experts is not None:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoEMLP(hidden_size=embed_dim, intersize=intermediate_size)

    def __call__(self, hidden_states: Array):
        identity = hidden_states
        og_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.router_gate(hidden_states)
        y = jrand.normal(rkey, shape=og_shape) # init as random array

        hidden_states = jnp.reshape(hidden_states, (-1, hidden_states.shape[1:]))
        flat_topk_idx = jnp.reshape(topk_idx, shape=(-1))

        if self.train:
            hidden_states = jnp.repeat(hidden_states, repeats=self.experts_pertoken, axis=0)
            y = jnp.empty_like(hidden_states, dtype=hidden_states.dtype)

            for k, expert in enumerate(self.expert_models):
                y[flat_topk_idx == k] = expert(hidden_states[flat_topk_idx == k]).astype(hidden_states.dtype) # type: ignore

            y = jnp.reshape(y, shape=(*topk_weight.shape, -1)) * jnp.expand_dims(topk_weight, axis=-1).sum(axis=1)
            y = jnp.reshape(y, shape=(og_shape))
            # TODO: Auxiliary loss add

        else:
            y = None

        if self.shared_experts is not None:
            y = y + self.shared_experts(identity) # type: ignore

        print(f"sparse moe shape =>{y.shape}")

        return y

    def moe_infer(self, x_input):
        pass


# self attention block
class SelfAttention(nnx.Module):
    def __init__(self, attn_heads, embed_dim, rngs: nnx.Rngs, drop=0.1):
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
            embed_dim, embed_dim, rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init
        )
        self.v_linear = nnx.Linear(
            embed_dim, embed_dim,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
            rngs=rngs
        )

        self.outproject = nnx.Linear(
            embed_dim, embed_dim, rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init
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

        linear_init = nnx.initializers.xavier_uniform()
        linear_bias_init = nnx.initializers.constant(0)

        self.q_linear = nnx.Linear(
            embed_dim,
            embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init
        )

        self.k_linear = nnx.Linear(
            cond_dim, embed_dim,
            bias_init=linear_bias_init,
            kernel_init=linear_init,
            rngs=rngs
          )
        self.v_linear = nnx.Linear(
            cond_dim, embed_dim,
            rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init
        )

        self.outproject = nnx.Linear(
            embed_dim, embed_dim, rngs=rngs,
            bias_init=linear_bias_init,
            kernel_init=linear_init
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

###############
# DiT blocks_ #
###############


class DiTBlock(nnx.Module):
    def __init__(self, hidden_size=1024, num_heads=6):
        super().__init__()

        # initializations
        linear_init = nnx.initializers.xavier_uniform()
        lnbias_init = nnx.initializers.constant(0)
        lnweight_init = nnx.initializers.constant(0)

        self.norm_1 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, bias_init=lnbias_init
        )
        self.attention = SelfAttention(num_heads, hidden_size, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)

        self.adaln_linear = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            use_bias=True,
            # bias_init=linear_init,
            rngs=rngs,
            # kernel_init=lnweight_init,
        )
        self.moe_block = SparseMoEBlock(hidden_size)
        print("dit block online")

    def __call__(self, x_img: Array):
        x_input = self.adaln_linear(nnx.silu(x_img))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.array_split(x_input, 6)
        )

        attn_mod_x = self.attention(
            modulate(self.norm_1(x_input), shift_msa, scale_msa)
        )
        
        x = x_input + jnp.expand_dims(gate_msa, 1) * attn_mod_x

        mlp_mod_x = self.moe_block(modulate(self.norm_2(x), shift_mlp, scale_mlp))
        x = x + jnp.expand_dims(gate_mlp, 1) * mlp_mod_x
        print(f"x dit block {type(x)} {x.shape}")
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
        print("ditbackbone online")

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


# backbone = DiTBackbone()


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
        
        self.linear1 = nnx.Linear(embed_dim, 2 * embed_dim, rngs=rngs)
        self.linear2 = nnx.Linear(2 * embed_dim, embed_dim, rngs=rngs)
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
