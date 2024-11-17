import jax, os, wandb, time, gc, optax, torchvision
from flax import nnx
from tqdm.auto import tqdm
from functools import partial
from diffusers import AutoencoderKL
from .data_utils import config, train_loader, random_mask, apply_mask
from .microdit import MicroDiT 
from .rf_sampler import RectFlow

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.50

num_devices = jax.device_count()
devices = jax.devices()
print(f'found {num_devices} JAX devices => {devices}')
for device in devices:
    print(f'{device} \n')


microdit = MicroDiT(
    inchannels=3,
    patch_size=(4, 4),
    embed_dim=1024,
    num_layers=4,
    attn_heads=8,
    mlp_dim=1 * 1024,
    cond_embed_dim=1024,
)

rf_engine = RectFlow(microdit)
graph, state = nnx.split(rf_engine)
n_params = sum([p.size for p in jax.tree.leaves(state)])
print(f"number of parameters: {n_params/1e6:.3f}M")

optimizer = nnx.Optimizer(rf_engine, optax.adamw(learning_rate=config.lr))


def wandb_logger(key: str, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)


def loss_func(model, batch):
    img_latents, label = batch
    bs, height, width, channels = img_latents.shape
    print(img_latents.shape)

    img_latents = img_latents * config.vaescale_factor
    mask = random_mask(
        bs, height, width, patch_size=config.patch_size, mask_ratio=config.mask_ratio
    )
    print(f"mask shape {mask.shape}")
    loss = model(img_latents, label, mask)
    # loss = optax.squared_error(img_latents, logits).mean()

    return loss


@nnx.jit
def train_step(model, optimizer, batch):
    gradfn = nnx.value_and_grad(loss_func, has_aux=True)
    (loss, logits), grads = gradfn(model, batch)
    optimizer.update(grads)
    jax.clear_caches()
    gc.collect()
    
    return loss


# def sample_images(model, vae, noise, embeddings):
#     # Use the stored embeddings
#     sampled_latents = model.sample(noise, embeddings)
#     # Decode latents to images
#     sampled_images = vae.decode(sampled_latents).sample
#     return sampled_images


def trainer(model=rf_engine, optimizer=optimizer, train_loader=train_loader):
    epochs = 1
    train_loss = 0.0
    model.train()
    # wandb_logger(
    #     key=None,
    #     model=model,
    #     project_name="transformer_playjax",
    #     run_name="tinygpt-1e-4-bs32-tpu",
    # )
    stime = time.time()
    
    for epoch in tqdm(range(epochs)):
        step_count = len(train_loader)
        
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}/{step_count}, loss-> {train_loss.item():.4f}")

            # wandb.log({"loss": train_loss.item()})
        print(f"epoch {epoch+1}, train loss => {train_loss}")
    
    endtime = time.time() - stime 
    print(f'trained {epochs} epochs in {endtime/60}mins (or {endtime}seconds)')


trainer()
# wandb.finish()
print("microdit test training in JAX")
