'''
Training a MicroDIT model, DiT-B(base) config for 30 epochs on cifar-10
'''

import flax.jax_utils
import jax, flax, os, pickle
from jax import Array, numpy as jnp, random as jrand
import numpy as np
from flax import nnx
from tqdm.auto import tqdm
from typing import List
import flax.traverse_util
from flax.serialization import from_state_dict
from flax.core import unfreeze
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from microdit_cifar import MicroDiT, config

JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20

randkey = jrand.key(config.seed)
rngs = nnx.Rngs(config.seed)


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


def display_samples(sample_batch): # works for jupyter-notebook output
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


def sample_image_batch(model, class_ids):
    classin = jnp.array(class_ids)
    randnoise = jrand.normal(
        randkey, (len(classin), config.img_size, config.img_size, 3)
    )
    image_batch = model.sample(randnoise, classin)
    gridfile = save_image_grid(image_batch, "dit_samples/microdit_output.png")

    return gridfile


import pickle
def load_paramdict_pickle(model, filename="model.pkl"):
    with open(filename, "rb") as modelfile:
        params = pickle.load(modelfile)
    params = unfreeze(params)
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    params = from_state_dict(model, params)
    nnx.update(model, params)
    return model

cifar_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
imagenet_classes = [76, 292, 293, 979, 968, 500, 33, 179, 333]


def inference_dit(model, ckpt_file, classes=cifar_classes):
    model = load_paramdict_pickle(model, ckpt_file)
    sample_file = sample_image_batch(model, classes)
    return sample_file