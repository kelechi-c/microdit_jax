"""
Training a MicroDIT model, ~300M params, 24-depth, 12 heads config for on Imagenet
"""

import jax, optax
from jax import Array, numpy as jnp, random as jrand
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils

# jax.config.update("jax_default_matmul_precision", "bfloat16")

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
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from streaming.base.format.mds.encodings import Encoding, _encodings
from streaming import StreamingDataset

from m2 import RectFlowWrapper, MicroDiT, random_mask

import warnings

warnings.filterwarnings("ignore")


class config:
    batch_size = 128
    img_size = 32
    seed = 42
    patch_size = (2, 2)
    lr = 1e-4
    mask_ratio = 0.75
    epochs = 30
    data_split = 10_000  # imagenet split to train on
    cfg_scale = 3.0
    vaescale_factor = 0.13025


# XLA/JAX flags
JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
# JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"

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
mesh = Mesh(mesh_devices, axis_names=("data",))
data_sharding = NS(mesh, PS("data"))
rep_sharding = NS(mesh, PS())

# sd VAE for decoding latents
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32)
print("loaded vae")


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


_encodings["uint8"] = uint8
remote_train_dir = "./vae_mds"  # this is the path you installed this dataset.
local_train_dir = "./imagenet"  # just a local mirror path


def jax_collate(batch):
    latents = jnp.stack(
        [jnp.array(item["vae_output"], dtype=jnp.float32) for item in batch], axis=0
    )
    labels = jnp.stack([int(item["label"]) for item in batch], axis=0)

    return {
        "vae_output": latents,
        "label": labels,
    }


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


def sample_image_batch(step, model, labels):
    print(f"sampling from labels {labels}")
    pred_model = device_get_model(model)
    pred_model.eval()
    image_batch = pred_model.sample(labels)
    file = f"mdsamples/{step}_microdit.png"
    batch = [process_img(x) for x in image_batch]

    gridfile = image_grid(batch[:16], file)
    print(f"sample saved @ {gridfile}")
    del pred_model

    return gridfile


def vae_decode(latent, vae=vae):
    tensor_img = rearrange(latent, "b h w c -> b c h w")
    tensor_img = torch.from_numpy(np.array(tensor_img))
    x = vae.decode(tensor_img).sample

    img = VaeImageProcessor().postprocess(
        image=x.detach(), do_denormalize=[True, True]
    )[0]

    return img


def process_img(img):
    img = vae_decode(img[None])
    return img


def image_grid(pil_images, file, grid_size=(4, 4), figsize=(20, 20)):
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


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_func(model, batch):
        img_latents, label = batch["vae_output"], batch["label"]
        input_mean = jnp.mean(img_latents)
        input_std = jnp.std(img_latents)

        # jax.debug.print({"input/mean": input_mean, "input/std": input_std})
        normalized_latents = (img_latents - input_mean) / (input_std + 1e-8)

        # Use normalized_latents as the input to your model
        img_latents = normalized_latents
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
        v_thetha, loss = model(img_latents, label, mask)

        return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, batch)
    # print(f'{grads.shape = }')
    # grads = optax.per_example_global_norm_clip(grads, 1.0)
    optimizer.update(grads)
    grad_norm = optax.global_norm(grads)

    return loss, grad_norm


def trainer(epochs, model, optimizer, train_loader):
    train_loss = 0.0

    model.train()

    # wandb_logger(
    #     key="",
    #     project_name="microdit",
    # )

    stime = time.time()
    sample_labels = jnp.array([76, 292, 293, 979, 968, 967, 33, 88, 404])

    for epoch in tqdm(range(epochs), desc="Training..../"):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss, grad_norm = train_step(model, optimizer, batch)
            print(
                f"step {step}/{len(train_loader)}, loss-> {train_loss.item():.4f}, grad_norm {grad_norm}"
            )

            # wandb.log(
            #     {
            #         "loss": train_loss.item(),
            #         "log_loss": math.log10(train_loss.item()),
            #         "grad_norm": grad_norm,
            #         "log_grad_norm": math.log10(grad_norm + 1e-8),
            #     }
            # )

            if step % 500 == 0:
                gridfile = sample_image_batch(step, model, sample_labels)
                image_log = wandb.Image(gridfile)
                # wandb.log({"image_sample": image_log})

            jax.clear_caches()
            gc.collect()

        print(f"epoch {epoch+1}, train loss => {train_loss}")
        path = f"{epoch}_microdit_imagenet_{train_loss}.pkl"
        save_paramdict_pickle(model, path)

        epoch_file = sample_image_batch(step, model, sample_labels)
        epoch_image_log = wandb.Image(epoch_file)
        wandb.log({"epoch_sample": epoch_image_log})

    etime = time.time() - stime
    print(f"training time for {epochs} epochs -> {etime:.4f}s / {etime/60:.4f} mins")
    final_sample = sample_image_batch(step, model, sample_labels)
    train_image_log = wandb.Image(final_sample)
    wandb.log({"epoch_sample": train_image_log})

    save_paramdict_pickle(
        model,
        f"checks/microdit_in1k_step-{len(train_loader)*epochs}_floss-{train_loss}.pkl",
    )

    return model


def overfit(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    model.train()

    # wandb_logger(
    #     key="", project_name="microdit_overfit"
    # )

    stime = time.time()

    batch = next(iter(train_loader))
    # print('initial sample..')
    # gridfile = sample_image_batch('initial', model, batch['label'])
    img_latents = batch["vae_output"]
    input_mean = jnp.mean(img_latents)
    input_std = jnp.std(img_latents)

    print({"input/mean": input_mean.item(), "input/std": input_std.item()})

    print("start overfitting.../")
    for epoch in tqdm(range(epochs)):
        train_loss, grad_norm = train_step(model, optimizer, batch)
        print(
            f"step {epoch}, loss-> {train_loss.item():.4f}, grad_norm {grad_norm.item()}"
        )

        # wandb.log({
        #     "loss": train_loss.item(),
        #     "log_loss": math.log10(train_loss.item()),
        #     "grad_norm": grad_norm.item(),
        #     "log_grad_norm": math.log10(grad_norm.item() + 1e-8) or 0.0,
        # })

        if epoch % 50 == 0:
            gridfile = sample_image_batch(epoch, model, batch["label"])
            image_log = wandb.Image(gridfile)
            # wandb.log({"image_sample": image_log})

        # if epoch % 50 == 0:  # Log less frequently
        #     state = nnx.state(model)
        #     weight_stats = {}
        #     for (
        #         layer_name,
        #         layer_params,
        #     ) in state.items():  # Assuming model is accessible here
        #         for param_name, param_value in layer_params.items():
        #             # print(f'param-name {param_name} / param {param_value}')
        #             for name, layer_value in param_value.items():
        #                 if name=="kernel" or name=="bias":
        #                     print(f"param_name {layer_name}/{param_name}/{name}")
        #                     weight_stats[f"{layer_name}/{param_name}/{name}/mean"] = jnp.mean(layer_value.value).item()
        #                     weight_stats[f"{layer_name}/{param_name}/{name}/std"] = jnp.std(layer_value.value).item()

        #     print(f'weight stats \n {weight_stats}')

        #     wandb.log(weight_stats)

        jax.clear_caches()
        gc.collect()

    etime = time.time() - stime
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )

    epoch_file = sample_image_batch("overfit", model, batch["label"])
    epoch_image_log = wandb.Image(epoch_file)
    wandb.log({"epoch_sample": epoch_image_log})

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
@click.option("-bs", "--batch_size", default=config.batch_size)
def main(run, epochs, batch_size):

    streaming.base.util.clean_stale_shared_memory()
    dataset = StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=None,
        batch_size=batch_size,
    )
    train_loader = DataLoader(
        dataset[: config.data_split],
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=jax_collate,
    )

    sp = next(iter(train_loader))
    print(
        f"loaded dataset, sample shape {sp['vae_output'].shape} /  {sp['label'].shape}, labels = {sp['label']}"
    )

    microdit = MicroDiT(
        in_channels=4,
        patch_size=(2, 2),
        embed_dim=768,
        num_layers=12,
        attn_heads=12,
        patchmix_layers=6,
    )

    rf_engine = RectFlowWrapper(microdit)
    graph, state = nnx.split(rf_engine)

    n_params = sum([p.size for p in jax.tree.leaves(state)])
    print(f"model parameters count: {n_params/1e6:.2f}M")

    # tx = optax.chain(
    #     optax.clip_by_global_norm(5.0),
    #     optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.99, weight_decay=0.1),
    # )
    optimizer = nnx.Optimizer(
        rf_engine,
        # optax.chain(
        #     optax.clip(max_delta=1.0), # Set your desired clipping range
        #     # optax.clip_by_global_norm(1.0),
        #     optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.995, weight_decay=0.01),
        # )
        optax.adamw(learning_rate=1e-5, b1=0.9, b2=0.99, weight_decay=0.1),
    )

    # replicate model across devices
    state = nnx.state((rf_engine, optimizer))
    state = jax.device_put(state, rep_sharding)
    nnx.update((rf_engine, optimizer), state)

    batch = next(iter(train_loader))
    # print('initial sample..')
    # gridfile = sample_image_batch('initial', model, batch['label'])
    img_latents = batch["vae_output"]
    input_mean = jnp.mean(img_latents)
    input_std = jnp.std(img_latents)

    print({"input/mean": input_mean.item(), "input/std": input_std.item()})
    normalized_latents = (img_latents - input_mean) / (input_std + 1e-8)

    # Use normalized_latents as the input to your model
    img_latents_for_model = normalized_latents

    # Log normalized input statistics (optional but good for verification)
    normalized_input_mean = jnp.mean(img_latents_for_model)
    normalized_input_std = jnp.std(img_latents_for_model)
    print(
        {
            "input/normalized_mean": normalized_input_mean.item(),
            "input/normalized_std": normalized_input_std.item(),
        }
    )

    # mstate = jax.device_get(nnx.state(rf_engine))

    # weight_stats = {}

    # wandb_logger(
    #     key="3aef5402e364c9da47508adf8be0664512ed30b2", project_name="microdit_overfit"
    # )
    # print("\n--- Using tree_flatten ---")
    # flattened_params, treedef = jax.tree_flatten(state)

    # for i, param in enumerate(flattened_params):
    #     path = ".".join(treedef.unflatten(jnp.zeros_like(param))._paths[i])
    #     print(f"Parameter: {path}, Shape: {param.shape}, dtype: {param.dtype}")

    # print('log state values..')

    # # params = nnx.state(rf_engine)  # Get the model's state pytree
    # model_stats = get_model_stats(mstate)
    # print_stats(model_stats)

    # jax.tree_map(log_state_values, state)

    # print('logged..')
    # for (
    #     layer_name,
    #     layer_params,
    # ) in state.items():  # Assuming model is accessible here
    #     for param_name, param_value in layer_params.items():
    #         # print(f'param-name {param_name} / param {param_value}')
    #         for name, layer_value in param_value.items():
    #             if name=="kernel" or name=="bias":
    #                 print(f"param_name {layer_name}/{param_name}/{name}")
    #                 weight_stats[f"{layer_name}/{param_name}/{name}/mean"] = jnp.mean(layer_value.value).item()
    #                 weight_stats[f"{layer_name}/{param_name}/{name}/std"] = jnp.std(layer_value.value).item()

    # print(f'weight stats \n {weight_stats}')

    # wandb.log(weight_stats)
    # wandb.log({"input/mean": input_mean.item(), "input/std": input_std.item()})

    if run == "overfit":
        model, loss = overfit(epochs, rf_engine, optimizer, train_loader)
        wandb.finish()
        print(f"microdit overfitting ended at loss {loss:.4f}")

    elif run == "train":
        trainer(epochs, rf_engine, optimizer, train_loader)
        wandb.finish()
        print("microdit (test) training (on imagenet-1k) in JAX..done")


main()
