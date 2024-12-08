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
import math, flax, torch, optax
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


JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"


rkey = jrand.key(config.seed)
randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)

# mesh / sharding / env configs
num_devices = jax.device_count()
devices = jax.devices()
num_processes = jax.process_count()
rank = jax.process_index()

mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), ("data",))

model_sharding = NamedSharding(mesh, PS())
data_sharding = NamedSharding(mesh, PS("data"))


print(f"found {num_devices} JAX devices")
for device in devices:
    print(f"{device} / ")

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
print("loaded vae")


# dataset
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

print(f'datasample {dataset[0]}')
# dataset_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, num_replicas=num_processes, rank=rank, seed=config.seed)


def jax_collate(batch):
    latents = jnp.stack([
        jnp.array(item["vae_output"]) for item in batch], axis=0
    )
    labels = jnp.stack([int(item['label']) for item in batch], axis=0)

    return {
        "vae_output": latents,
        "label": labels,
    }


train_loader = DataLoader(
    dataset[:config.data_split],
    batch_size=64, #config.batch_size,
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


# numpy/jax equivalent of F.linear
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
bias_init = nnx.initializers.constant(1)


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



###############
# DiT blocks_ #
###############
class DiTBlock(nnx.Module):
    def __init__(self, hidden_size=1024, num_heads=12, mlp_ratio=4):
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
            rngs=rngs,
            decode=False,
        )
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
        
        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        
        self.linear = nnx.Linear(
            hidden_size,
            patch_size[0] * patch_size[1] * out_channels,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=bias_init,
        )
        
        self.adaln_linear = nnx.Linear(
            hidden_size,
            2 * hidden_size,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=bias_init,
        )

    def __call__(self, x_input: Array, cond: Array):
        linear_cond = nnx.silu(self.adaln_linear(cond))
        shift, scale = jnp.array_split(linear_cond, 2, axis=1)

        x = modulate(self.norm_final(x_input), shift, scale)
        x = self.linear(x)

        return x


#####################
# Full Microdit model
####################
class DiT(nnx.Module):
    def __init__(
        self,
        inchannels,
        patch_size,
        embed_dim,
        depth,
        attn_heads,
        cond_embed_dim,
        dropout=0.0,
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
        layers = [DiTBlock(embed_dim, attn_heads, mlp_ratio=4) for _ in range(depth)] 
        self.dit_layers = nnx.Sequential(*layers)
        # conditioning layers
        self.time_embedder = TimestepEmbedder(cond_embed_dim)
        self.label_embedder = LabelEmbedder(num_classes=num_classes, hidden_size=cond_embed_dim, drop=dropout)

        self.final_linear = FinalMLP(embed_dim, patch_size=patch_size, out_channels=4)

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

    def __call__(self, x: Array, t: Array, y_cap: Array):
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

        x = x + pos_embed

        # text_embed = self.cap_embedder(y_cap) # (b, embdim)
        class_embed = self.label_embedder(y_cap, train=True)
        time_embed = self.time_embedder(t)

        cond_signal = class_embed + time_embed
        
        x = self.dit_layers(x, cond_signal)

        x = self.final_linear(x)
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

        return images[-1] / config.vaescale_factor


# rectifed flow forward pass, loss, and sampling
class RectFlowWrapper(nnx.Module):
    def __init__(self, model: nnx.Module, sigln: bool = True):
        self.model = model
        self.sigln = sigln

    def __call__(self, x_input: Array, cond: Array) -> Array:
        b_size = x_input.shape[0]  # batch_size
        rand_t = None

        rand = jrand.uniform(randkey, (b_size,))
        rand_t = nnx.sigmoid(rand)

        inshape = [1] * len(x_input.shape[1:])
        texp = rand_t.reshape([b_size, *(inshape)])

        z_noise = jrand.uniform(randkey, x_input.shape)  # input noise with same dim as image
        z_noise_t = (1 - texp) * x_input + texp * z_noise

        v_thetha = self.model(z_noise_t, rand_t, cond)

        mean_dim = list(range(1, len(x_input.shape)))  # across all dimensions except the batch dim

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

        return images[-1] / config.vaescale_factor


dit = DiT(
    inchannels=4,
    patch_size=(2, 2),
    embed_dim=768,
    num_layers=24,
    attn_heads=12,
    cond_embed_dim=768,
    patchmix_layers=4,
)

rf_engine = RectFlowWrapper(dit)
graph, state = nnx.split(rf_engine)

n_params = sum([p.size for p in jax.tree.leaves(state)])
print(f"number of parameters: {n_params/1e6:.1f}M")

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
    img.save(f'{id}.png')
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
    file = f"samples/dit_output_in-1k@{step}.png"
    batch = [process_img(x, id) for id, x in enumerate(image_batch)]
    gridfile = image_grid(batch, file)
    print(f'sample saved @ {gridfile}')
    del pred_model

    return gridfile


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
    print(f'model loaded from {filename}')

    return model, params


# replicate model across devices
state = nnx.state((rf_engine, optimizer))
state = jax.device_put(state, model_sharding)
nnx.update((rf_engine, optimizer), state)


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_func(model, batch):
        img_latents, label = batch['vae_output'], batch['label']
        img_latents = img_latents.reshape(-1, 4, 32, 32) * config.vaescale_factor
        img_latents = rearrange(img_latents, "b c h w -> b h w c")
        print(f"latents => {img_latents.shape}")

        img_latents, label = jax.device_put((img_latents, label), data_sharding)

        bs, height, width, channels = img_latents.shape

        loss = model(img_latents, label)
        return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, batch)
    optimizer.update(grads)
    
    return loss


def trainer(epochs, model=rf_engine, optimizer=optimizer, train_loader=train_loader):
    epochs = epochs
    train_loss = 0.0
    
    model.train()

    wandb_logger(
        key="",
        project_name="tiny_dit",
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
        path = f"dit_imagenet_{epoch}_{train_loss}.pkl"
        save_paramdict_pickle(model, path)
        
        epoch_file = sample_image_batch(step, model)
        epoch_image_log = wandb.Image(epoch_file)
        wandb.log({"epoch_sample": epoch_image_log})

    etime = time.time() - stime
    print(f"training time for {epochs} epochs -> {etime}s / {etime/60} mins")

    save_paramdict_pickle(
        model, f"dit_in1k_{len(train_loader)*epochs}_{train_loss}.pkl"
    )

    return model


def overfit(epochs, model=rf_engine, optimizer=optimizer, train_loader=train_loader):
    train_loss = 0.0
    model.train()

    wandb_logger(
        key="YOUR_KEY", project_name="microdit_overfit"
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

@click.command()
@click.option('-r', '--run', default='overfit')
@click.option('-e', '--epochs', default=10)
def main(run, epochs):
    if run=='overfit':
        model, loss = overfit(epochs)
        wandb.finish()
        print(f"dit overfitting ended at loss {loss:.4f}")
    
    elif run=='train':
        trainer(epochs)
        wandb.finish()
        print("dit (test) training (on imagenet-1k) in JAX..done")

main()
