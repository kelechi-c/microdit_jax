from jax import Array, numpy as jnp, random as jrand
import flax, cv2, jax
import numpy as np
from flax import nnx
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from collections import namedtuple
import flax.traverse_util
from flax.serialization import to_state_dict
import safetensors.flax as safejax
from PIL import Image as pillow
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import AutoTokenizer, T5EncoderModel
from urllib.request import urlopen

randkey = jrand.key(3)

class config:
    vaescale_factor = 0.13025
    batch_size = 128
    img_size = 256
    patch_size = (4, 4)
    lr = 1e-4
    mask_ratio = 0.75
    epochs = 5
    data_split = 10_000
    cfg_scale = 2.0
    vae_channels = 4
    celebv_id = 'SwayStar123/CelebV-HQ'
    finevid_id = "HuggingFaceFV/finevideo"
    pd12_id = 'Spawning/PD12M'
    vae_id = "madebyollin/sdxl-vae-fp16-fix"
    t5_id = "google-t5/t5-small"
    mini_data_id = 'uoft-cs/cifar10'
    device_0 = jax.default_device()


# jax/numpy implementation of topk selection
def jnp_topk(array: Array, k: int):
    topk_tuple = namedtuple('topk', ['values', 'ids'])
    array = jnp.asarray(array)
    flat = array.ravel()

    sort_indices = jnp.argpartition(flat, -k)[-k:]
    argsort = jnp.argsort(-flat[sort_indices]) # get sorting ids

    sort_idx = sort_indices[argsort]
    values = flat[sort_idx]
    
    idx = jnp.unravel_index(sort_idx, array.shape)
    
    if len(idx) == 1:
        idx, = idx
    
    return topk_tuple(values=values, ids=idx)


# save model params in safetensors file
def save_model(model: nnx.Module, file: str, dtype):
    _, state = nnx.split(model)
    params = state.filter(nnx.Param)
    state_dict = to_state_dict(params)
    
    state_dict = flax.traverse_util.flatten_dict(state_dict, sep='.')
    
    for key in list(state_dict.keys()):
        if not isinstance(state_dict[key], Array):
            state_dict[key] = jnp.array(state_dict[key]) # type: ignore
            
        state_dict[key] = state_dict[key].astype(dtype) # type: ignore
    
    safejax.save_file(state_dict, file) # type: ignore


def apply_mask(x: Array, mask, patch_size):
    # basically turns the masked values to 0s
    bs, c, h, w = x.shape
    numpatch_h = h // patch_size[0]
    numpatch_w = w // patch_size[1]
    
    mask = jnp.reshape(mask, shape=(bs, numpatch_h, numpatch_w))
    
    mask = jnp.expand_dims(mask, axis=1)
    mask = jnp.tile(mask, reps=(1, 1, patch_size[0], patch_size[1]))
    mask = jnp.reshape(mask, shape=(bs, 1, h, w))
    
    x_masked = x * mask
    
    return x_masked


def random_mask(bs, height, width, patch_size, mask_ratio):
    num_patches = (height // patch_size[0]) * (width // patch_size[1])
    num_patches_to_mask = int(num_patches * mask_ratio)
    
    rand_array = jrand.normal(randkey, shape=(bs, num_patches))
    indices = jnp.argsort(rand_array, axis=1)
    
    mask = jnp.ones(shape=(bs, num_patches))
    
    batch_mask_array = jnp.expand_dims(jnp.arange(bs), axis=1)
    mask[batch_mask_array, indices[:, :num_patches_to_mask]] = 0
    
    mask = jnp.reshape(mask, shape=(bs, num_patches))
    
    return mask


def remove_masked_patches(patches: Array, mask: Array):
    mask = jnp.logical_not(jnp.bool(mask))

    bs, num_patches, embed_dim = patches.shape
    mask = jnp.expand_dims(mask, axis=-1)
    mask = jnp.broadcast_to(mask, shape=(-1, -1, embed_dim))
    mask_ids = jnp.nonzero(jnp.reshape(mask, (-1)))
    unmasked_patches = jnp.reshape(patches, shape=-1)
    unmasked_patches = jnp.take(unmasked_patches, mask_ids[0]).reshape(bs, -1, embed_dim)

    return unmasked_patches


def add_masked_patches(patches: Array, mask: Array):
    # Ensure mask is a boolean tensor
    mask = jnp.bool(mask)

    bs, num_patches, embed_dim = mask.shape[0], mask.shape[1], patches.shape[-1]

    # Create a tensor of zeros with the same shape and dtype as the patches tensor
    full_patches = jnp.zeros(shape=(bs, num_patches, embed_dim))

    # Iterate over each batch and place unmasked patches back in their original positions
    for b in range(bs):
        # Use the mask to place unmasked patches back in the correct positions
        full_patches[b, mask[b]] = patches[b].astype(full_patches.dtype)

    return full_patches

## nov_14_0453

# T5 text encoder
t5_tokenizer = AutoTokenizer.from_pretrained(config.t5_id)
t5_model = T5EncoderModel.from_pretrained(config.t5_id)
vae = AutoencoderKL.from_pretrained(config.vae_id).to(config.device_0)

def text_t5_encode(text_input: str, tokenizer=t5_tokenizer, model=t5_model):
    input_ids = tokenizer(text_input, return_tensors='np').input_ids  # Batch size 1
    outputs = model(input_ids=input_ids)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states

## data loading
image_data = load_dataset(config.mini_data_id, streaming=True, split='train', trust_remote_code=True).take(config.split)


def load_image(url):
    image = pillow.open(urlopen(url=url))
    img_array = np.array(image)
    resized = cv2.resize(img_array, dsize=(config.img_size, config.img_size))

    return resized / 255.0


class ImageData(IterableDataset):
    def __init__(self, dataset=image_data):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return config.data_split

    def __iter__(self):
        for sample in self.dataset:
            image = sample["img_url"] # type: ignore
            image = load_image(image)
            img_latents = vae.encode(image)
            img_latents = img_latents.numpy()
            caption_encoded = text_t5_encode(sample["caption"]) # type: ignore

            image = jnp.array(image)
            text_encoding = jnp.array(caption_encoded)

            yield image, text_encoding


class ImageClassData(IterableDataset):
    def __init__(self, dataset=image_data):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return config.data_split

    def __iter__(self):
        for sample in self.dataset:
            image = sample["img_url"]  # type: ignore
            image = load_image(image)
            img_latents = vae.encode(image)
            img_latents = img_latents.numpy()

            image = jnp.array(image)
            label = jnp.array(sample['label'])

            yield image, label


dataset = ImageClassData()
train_loader = DataLoader(dataset, batch_size=config.batch_size)