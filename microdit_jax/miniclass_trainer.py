import jax, math, os, wandb, time, optax, torchvision
from jax import (
    Array, 
    numpy as jnp,
    random as jrand
)
from flax import nnx
from tqdm.auto import tqdm
from functools import partial
from diffusers import AutoencoderKL
from .data_utils import config, train_loader, random_mask, apply_mask
from .microdit import MicroDiT 
from .rf_sampler import RectFlow

num_devices = jax.device_count()
devices = jax.devices()
print(f'found {num_devices} JAX devices => {devices}')
for device in devices:
    print(f'{device} \n')


model = MicroDiT(
    inchannels=3, patch_size=(4, 4), 
    embed_dim=1024, num_layers=12,
    attn_heads=6, mlp_dim= 4*1024,
    caption_embed_dim=1024
)
rf_engine = RectFlow(model)

vae = AutoencoderKL.from_pretrained(config.vae_id).to(device)
optimizer = nnx.Optimizer(rf_engine.model, optax.adamw(learning_rate=config.lr))

# def preprocess_image(image_array):

def wandb_logger(key: str, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)


def loss_func(model, batch):
    img_latents, label = batch
    bs, channels, height, width = img_latents.shape
    print(img_latents.shape)
    
    img_latents = img_latents * config.vaescale_factor
    mask = random_mask(bs, height, width, patch_size=config.patch_size, mask_ratio=config.mask_ratio).to_device(img_latents.device)
    loss = model(img_latents, label, mask)
    # loss = optax.squared_error(img_latents, logits).mean()

    return loss


@nnx.jit
def train_step(model, optimizer, batch):
    gradfn = nnx.value_and_grad(loss_func, has_aux=True)
    (loss, logits), grads = gradfn(model, batch)
    optimizer.update(grads)
    return loss


def sample_images(model, vae, noise, embeddings):
    # Use the stored embeddings
    sampled_latents = model.sample(noise, embeddings)
    
    # Decode latents to images
    sampled_images = vae.decode(sampled_latents).sample
    # images = sample_images
    return sampled_images


def trainer(model=rf_engine, optimizer=optimizer, train_loader=train_loader):
    epochs = 1
    train_loss = 0.0
    model.model.train()
    # wandb_logger(
    #     key=None,
    #     model=model,
    #     project_name="transformer_playjax",
    #     run_name="tinygpt-1e-4-bs32-tpu",
    # )

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}, loss-> {train_loss.item():.4f}")
            
            # if step % 100 == 0:
            #     sample_latents = model.sample()

            # wandb.log({"loss": train_loss.item()})

        print(f"epoch {epoch+1}, train loss => {train_loss}")


trainer()
# wandb.finish()
print("microdit test training in JAX")
