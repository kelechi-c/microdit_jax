"""
Training a MicroDIT model, ~300M params, 24-depth, 12 heads config for on Imagenet
"""

from functools import partial
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
from datasets import load_dataset, IterableDataset
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from streaming.base.format.mds.encodings import Encoding, _encodings
from streaming import StreamingDataset
from collections import defaultdict

from m2 import RectFlowWrapper, MicroDiT

import warnings

warnings.filterwarnings("ignore")


class config:
    batch_size = 128
    img_size = 32
    seed = 33
    patch_size = (2, 2)
    lr = 3e-4
    mask_ratio = 0.50
    epochs = 30
    data_split = 10_000  # data split to train on
    cfg_scale = 2.0
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
mesh = Mesh(mesh_devices, axis_names=("data"))
data_sharding = NS(mesh, PS("data"))
rep_sharding = NS(mesh, PS())

# sd VAE for decoding latents
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32
)
print("loaded vae")


class ShapeBatchingDataset(IterableDataset):
    def __init__(
        self, batch_size=64, split=1_000_000,
        shuffle=True, seed=config.seed, 
        buffer_multiplier=20
    ):
        self.split_size = split
        self.dataset = load_dataset(
            "SwayStar123/preprocessed_commoncatalog-cc-by",
            streaming=True,
            split="train",
            trust_remote_code=True,
        ).take(self.split_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_multiplier = buffer_multiplier

    def __iter__(self):
        if self.shuffle:
            self.dataset = self.dataset.shuffle(
                seed=self.seed, buffer_size=self.batch_size * self.buffer_multiplier
            )

        shape_batches = defaultdict(list)
        for sample in self.dataset:
            # Get the shape as a tuple to use as a key
            shape_key = tuple(sample["vae_latent_shape"])
            shape_batches[shape_key].append(sample)

            # If enough samples are accumulated for this shape, yield a batch
            if len(shape_batches[shape_key]) == self.batch_size:
                batch = self.prepare_batch(shape_batches[shape_key])
                yield batch
                shape_batches[shape_key] = []  # Reset the buffer for this shape

        # After iterating over the dataset, yield any remaining partial batches
        for remaining_samples in shape_batches.values():
            if remaining_samples:
                batch = self.prepare_batch(remaining_samples)
                yield batch

    def prepare_batch(self, samples):
        # Convert lists of samples into tensors
        vae_latent_shape = tuple(samples[0]["vae_latent_shape"])

        batch = {
            "caption": [s["caption"] for s in samples],
            "vae_latent": jnp.array(
                np.stack(
                    [
                        np.frombuffer(s["vae_latent"], dtype=np.float32).copy()
                        for s in samples
                    ]
                ),
                dtype=jnp.bfloat16,
            ).reshape(-1, *vae_latent_shape),
            "vae_latent_shape": vae_latent_shape,
            "text_embedding": jnp.array(
                np.stack(
                    [
                        np.frombuffer(s["text_embedding"], dtype=np.float16).copy()
                        for s in samples
                    ]
                ),
                dtype=jnp.bfloat16
            ),
        }

        return batch

    def __len__(self):
        return self.split_size #len(self.dataset) // self.batch_size


# def jax_collate(batch):
#     latents = jnp.stack(
#         [jnp.array(item["vae_latent"], dtype=jnp.bfloat16) for item in batch], axis=0
#     )
#     labels = jnp.stack(
#         [jnp.array(item["text_embedding"], dtype=jnp.bfloat16) for item in batch],
#         axis=0,
#     )
#     shape = [item["vae_latent_shape"] for item in batch]
#     shape = [item["vae_latent_shape"] for item in batch]

#     return {"vae_output": latents, "label": labels, "vae_latent_shape": shape}


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


def sample_image_batch(step, model, batch):
    labels, shape, ground_latents, captions = (
        batch["text_embedding"],
        batch["vae_latent_shape"],
        batch["vae_latent"],
        batch['captions']
    )
    # print(shape)
    start_noise = jrand.normal(randkey, (len(labels), shape[1], shape[2], shape[0]))
    print(f"sampling from noise of shape {start_noise.shape} / captions {captions}")

    pred_model = device_get_model(model)
    pred_model.eval()

    image_batch = pred_model.sample(start_noise, labels)

    batch_latents = rearrange(ground_latents, "b c h w -> b h w c")
    v_sample_error = (batch_latents - image_batch) ** 2

    file = f"mdsamples/{step}_microdit.png"
    batch = [process_img(x) for x in image_batch]

    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")
    del pred_model
    jax.clear_caches()
    gc.collect()
    
    v_sample_error = v_sample_error.mean()

    print(f"sample_error {v_sample_error}")

    return gridfile, v_sample_error


def vae_decode(latent, vae=vae):
    tensor_img = rearrange(latent, "b h w c -> b c h w")
    tensor_img = torch.from_numpy(np.array(tensor_img)).to(torch.float32)
    x = vae.decode(tensor_img).sample

    img = VaeImageProcessor().postprocess(
        image=x.detach(), do_denormalize=[True, True]
    )[0]

    return img


def process_img(img):
    img = vae_decode(img[None])
    return img


def image_grid(pil_images, file, grid_size=(3, 3), figsize=(4, 2)):
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


def inspect_latents(batch):
    batch = rearrange(batch, "b c h w -> b h w c")
    # print(batch.shape)
    # img_latents = img_latents / config.vaescale_factor
    batch = [process_img(x) for x in batch]
    file = f"images/preproc_cc-by.jpg"
    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")


# save model params in pickle file
def save_paramdict_pickle(model, filename="dit_model.ckpt"):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep=".")

    with open(filename, "wb") as f:
        pickle.dump(frozen_state_dict, f)

    return flat_state_dict


def load_paramdict_pickle(model, filename="dit_model.ckpt"):
    with open(filename, "rb") as modelfile:
        params = pickle.load(modelfile)

    params = unfreeze(params)
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    params = from_state_dict(model, params)
    nnx.update(model, params)
    
    return model, params

@partial(nnx.jit, in_shardings=(rep_sharding, rep_sharding, data_sharding, data_sharding), out_shardings=(None, None))
def train_step(model, optimizer, image, text):
    
    def loss_func(model, img_latents, label):
        img_latents = (
            rearrange(img_latents, "b c h w -> b h w c") * config.vaescale_factor
        )
        # img_latents, label = jax.device_put((img_latents, label), data_sharding)
        loss = model(img_latents, label)

        return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, image, text)

    grad_norm = optax.global_norm(grads)

    optimizer.update(grads)

    return loss, grad_norm


def trainer(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    model.train()

    batch = next(iter(train_loader))
    print("initial sample..")
    gridfile, sample_error = sample_image_batch("initial", model, batch)
    
    # wandb_logger(
    #     key="",
    #     project_name="microdit",
    # )

    stime = time.time()
    sample_captions = None
    sample_textembed = None
    main_batch = None

    for epoch in tqdm(range(epochs), desc="Training..../"):
        print(f'training for the {epoch+1}th epoch')
        
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            main_batch = batch
            sample_captions = batch['caption']
            sample_textembed = batch['text_embedding']

            latent, text_embed = batch["vae_latent"], batch["text_embedding"]
            
            train_loss, grad_norm = train_step(model, optimizer, latent, text_embed)            
            print(
                f"step {step}/{len(train_loader)}, loss-> {train_loss.item():.4f}, grad_norm {grad_norm}"
            )

            wandb.log(
                {
                    "loss": train_loss.item(),
                    "log_loss": math.log10(train_loss.item()),
                    "grad_norm": grad_norm,
                    "log_grad_norm": math.log10(grad_norm + 1e-8),
                }
            )

            if step % 100 == 0 and step != 0:
                print(f"sampling from...{sample_captions}")
                gridfile = sample_image_batch(step, model, sample_textembed)
                image_log = wandb.Image(gridfile)
                wandb.log({"image_sample": image_log})
                
            if step % 5000 == 0 and step != 0:
                save_paramdict_pickle(
                    model,
                    f"checks/microdit-train_step-{step}_epoch-{epoch}.pkl",
                )
                print(f"midstep checkpoint @ {step}")
                
            jax.clear_caches()
            gc.collect()

        print(f"epoch {epoch+1}, train loss => {train_loss}")
        path = f"{epoch}_microdit_cc_{train_loss}.pkl"
        save_paramdict_pickle(model, path)

        print(f"sampling from...{sample_captions}")
        epoch_file = sample_image_batch(step, model, main_batch)
        epoch_image_log = wandb.Image(epoch_file)
        wandb.log({"epoch_sample": epoch_image_log})

    etime = time.time() - stime
    print(f"training time for {epochs} epochs -> {etime/60/60:.4f} hours")

    final_sample = sample_image_batch(step, model, main_batch)
    train_image_log = wandb.Image(final_sample)
    wandb.log({"final_sample": train_image_log})

    save_paramdict_pickle(
        model,
        f"checks/microdit_cc_step-{len(train_loader)*epochs}_floss-{train_loss:.5f}.pkl",
    )

    return model


def overfit(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    model.train()

    batch = next(iter(train_loader))
    # print("initial sample..")
    # gridfile, sample_error = sample_image_batch("initial", model, batch)
    # print(f'initial sample error: {sample_error.mean()}')

    wandb_logger(
        key="yourkey", project_name="microdit_overfit"
    )

    stime = time.time()

    print("start overfitting.../")
    for epoch in tqdm(range(epochs)):
        latent, text_embed = batch["vae_latent"], batch["text_embedding"]
        train_loss, grad_norm = train_step(model, optimizer, latent, text_embed)
        print(
            f"step {epoch}, loss-> {train_loss.item():.4f}, grad_norm {grad_norm.item()}"
        )

        wandb.log(
            {
                "loss": train_loss.item(),
                "log_loss": math.log10(train_loss.item() + 1e-8),
                "grad_norm": grad_norm.item(),
                "log_grad_norm": math.log10(grad_norm.item() + 1e-8),
            }
        )

        if epoch % 25 == 0 and epoch != 0:
            gridfile, sample_error = sample_image_batch(epoch, model, batch)
            image_log = wandb.Image(gridfile)
            wandb.log({"image_sample": image_log, "sample_error": sample_error})

        if epoch % 400 == 0 and epoch != 0:
            save_paramdict_pickle(
                model,
                f"checks/microdit-moe_overfitting_step-{epoch}.pkl",
            )
            print(f"checkpoint @ {epoch}")

        jax.clear_caches()
        gc.collect()

    etime = time.time() - stime
    save_paramdict_pickle(
        model,
        f"checks/microdit_overfitting_all{epochs}.pkl",
    )
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )

    epoch_file, sample_error = sample_image_batch("overfit", model, batch)
    epoch_image_log = wandb.Image(epoch_file)
    wandb.log({"epoch_sample": epoch_image_log, "overfit_sample_error": sample_error})

    return model, train_loss


def log_state_values(state_layer):
    try:
        if isinstance(state_layer, Array):
            mean_val = jnp.mean(state_layer)
            std_val = jnp.std(state_layer)
            print(f"layer: Mean: {mean_val}, StdDev: {std_val}")
    except Exception as e:
        print(f"inspect error {e}")


@click.command()
@click.option("-r", "--run", default="overfit")
@click.option("-e", "--epochs", default=30)
@click.option("-mr", "--mask_ratio", default=config.mask_ratio)
@click.option("-bs", "--batch_size", default=config.batch_size)
def main(run, epochs, batch_size, mask_ratio):

    dataset = ShapeBatchingDataset(batch_size=batch_size)

    train_loader = dataset
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     drop_last=True,
    #     collate_fn=jax_collate,
    # )

    sp = next(iter(train_loader))
    print(
        f"loaded dataset, sample shape {sp['vae_output'].shape} /  {sp['label'].shape}"
    )

    inspect_latents(sp["vae_output"][0].astype(jnp.float32))

    microdit = MicroDiT(
        in_channels=4,
        patch_size=(2, 2),
        embed_dim=768,
        num_layers=16,
        attn_heads=12,
        patchmix_layers=4,
        patchmix_dim=768,
        num_experts=4,
    )

    rf_engine = RectFlowWrapper(microdit, mask_ratio=mask_ratio)
    graph, state = nnx.split(rf_engine)

    n_params = sum([p.size for p in jax.tree.leaves(state)])
    print(f"model parameters count: {n_params/1e6:.2f}M")

    optimizer = nnx.Optimizer(
        rf_engine,
        optax.adamw(learning_rate=3e-4, weight_decay=0.01),
    )

    # replicate model across devices
    state = nnx.state((rf_engine, optimizer))
    state = jax.device_put(state, rep_sharding)
    nnx.update((rf_engine, optimizer), state)

    if run == "overfit":
        model, loss = overfit(epochs, rf_engine, optimizer, train_loader)
        wandb.finish()
        print(f"microdit overfitting ended at loss {loss:.4f}")

    elif run == "train":
        trainer(epochs, rf_engine, optimizer, train_loader)
        wandb.finish()
        print("microdit (test) training (on imagenet-1k) in JAX..done")


main()
