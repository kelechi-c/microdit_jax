import jax, math, os, wandb, time, optax, torchvision
from jax import (
    Array, 
    numpy as jnp,
    random as jrand
)
from flax import nnx
from tqdm.auto import tqdm
from diffusers import AutoencoderKL
from .data_utils import config, train_loader, random_mask, apply_mask
from .microdit import MicroDiT 


num_devices = jax.device_count()
devices = jax.devices()
print(f'found JAX devices => {devices}')
for device in devices:
    print(f'{device} \n')


model = MicroDiT(
    inchannels=3, patch_size=(4, 4), 
    embed_dim=1024, num_layers=12,
    attn_heads=6, mlp_dim=1024,
    caption_embed_dim=1024
)

vae = AutoencoderKL.from_pretrained(config.vae_id).to(device)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=config.lr))

# def preprocess_image(image_array):
    

def wandb_logger(key: str, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)


def loss_func(model, batch):
    img_latents, text_encoding = batch
    logits = model(img_latents, text_encoding)

    loss = optax.squared_error(img_latents, logits).mean()

    return loss, logits


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


def trainer(model=model, optimizer=optimizer, train_loader=train_loader):
    epochs = 1
    train_loss = 0.0
    model.train()
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
wandb.finish()
print("microdit test training in JAX")
