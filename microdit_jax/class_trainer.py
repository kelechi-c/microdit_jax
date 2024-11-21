import jax, os, wandb, time, gc, optax
from flax import nnx
from tqdm.auto import tqdm
from functools import partial
from diffusers import AutoencoderKL
from .data_utils import config, train_loader, random_mask, save_image_grid
from .microdit import MicroDiT 
from .rf_sampler import RectFlowWrapper

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.50

num_devices = jax.device_count()
devices = jax.devices()
print(f'found {num_devices} JAX devices => {devices}')
for device in devices:
    print(f'{device} \n')

# wandb.Image()
microdit = MicroDiT(
    inchannels=3,
    patch_size=(4, 4),
    embed_dim=1024,
    num_layers=4,
    attn_heads=8,
    mlp_dim=1 * 1024,
    cond_embed_dim=1024,
)

rf_engine = RectFlowWrapper(microdit)
state = nnx.state(rf_engine)

n_params = sum([p.size for p in jax.tree.leaves(state)])
print(f"number of parameters: {n_params/1e6:.3f}M")

optimizer = nnx.Optimizer(rf_engine, optax.adamw(learning_rate=config.lr))


def wandb_logger(key: str, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)

def sample_image_batch(step):
    classes = [1, 2, 3, 4]
    randnoise = jrand.normal(randkey, (len(classes), 32, 32, 3))
    image_batch = rf_engine.model.sample(randnoise, classin)
    gridfile = save_image_grid(image_batch, f'dit_output@{step}.png')
    wandb.log(
        'image': wandb.Image(data_or_path=gridfile)
    ) 

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

# @nnx.jit
def train_step(model, optimizer, batch):
    def loss_func(model, batch):
      img_latents, label = batch
      bs, height, width, channels = img_latents.shape

      img_latents = img_latents * config.vaescale_factor
      mask = random_mask(bs, height, width, patch_size=config.patch_size, mask_ratio=config.mask_ratio)
      loss = model(img_latents, label, mask)
      # loss = optax.squared_error(img_latents, logits).mean()
      return loss

    gradfn = nnx.value_and_grad(loss_func)
    loss, grads = gradfn(model, batch)
    optimizer.update(grads)
    # print(f'Loss => {loss}')
    return loss


def trainer(model=rf_engine, optimizer=optimizer, train_loader=train_loader):
    epochs = 1
    train_loss = 0.0
    model.train()
    # wandb_logger(
    #     key=None,
    #     model=model,
    #     project_name="microdit_jax-cifar",
    #     run_name="microdit-cifar1",
    # )

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            train_loss = train_step(model, optimizer, batch)
            print(f"step {step}, loss-> {train_loss.item():.4f}")

            if step % 1000 == 0:
                gridfile = sample_image_batch(step)
                image_log = wandb
                wandb.log({'image_sample': image_log})
                
            # wandb.log({"loss": train_loss.item()})

        print(f"epoch {epoch+1}, train loss => {train_loss}")
        save_paramdict_pickle(model, f'microdit_cifar_{epoch}_{train_loss}.pkl')


trainer()
# wandb.finish()
print("microdit test training in JAX")