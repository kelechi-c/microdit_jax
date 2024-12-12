from jax import (
    Array,
    numpy as jnp,
    random as jrand
)
from flax import nnx
from typing import List 

randkey = jrand.key(33)


# rectifed flow forward pass, loss, and smapling
class RectFlowWrapper(nnx.Module):
    def __init__(self, model: nnx.Module, sigln: bool = True):
        self.model = model
        self.sigln = sigln

    def __call__(self, x_input: Array, cond: Array, mask) -> Array:
        b_size = x_input.shape[0]  # batch_size
        rand_t = None

        if self.sigln:
            rand = jrand.uniform(randkey, (b_size,)).astype(jnp.bfloat16)
            rand_t = nnx.sigmoid(rand)
        else:
            rand_t = jrand.uniform(randkey, (b_size,))

        inshape = [1] * len(x_input.shape[1:])
        texp = rand_t.reshape([b_size, *(inshape)]).astype(jnp.bfloat16)

        z_noise = jrand.uniform(randkey, x_input.shape).astype(
            jnp.bfloat16
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
