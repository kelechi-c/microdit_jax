import jax, pickle
from jax import numpy as jnp
import numpy as np
from flax import nnx
import flax.traverse_util
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze

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
    print(f"model loaded from {filename}")

    return model, params


def count_params(model):
    params = nnx.state(model, nnx.Param)
    total_params = sum((np.prod(x.shape) for x in jax.tree.leaves(params)), 0)

    return total_params


def check_weight_stats(params):
    for layer_name, layer_params in params.items():
        for param_name, param_value in layer_params.items():
            if "kernel" in param_name or "bias" in param_name:
                mean_val = jnp.mean(param_value)
                std_val = jnp.std(param_value)
                print(
                    f"Layer: {layer_name}, Param: {param_name}, Mean: {mean_val:.4f}, StdDev: {std_val:.4f}"
                )


def log_state_values(state_layer):
    try:
        if isinstance(state_layer, jax.Array):
            mean_val = jnp.mean(state_layer)
            std_val = jnp.std(state_layer)
            print(f"layer: Mean: {mean_val}, StdDev: {std_val}")
    except Exception as e:
        print(f"inspect error {e}")


def log_activation_stats(layer_name, activations):
    mean_val = jnp.mean(activations)
    std_val = jnp.std(activations)

    jax.debug.print(
        "layer {val} / mean {mean_val} / stddev {std_val}",
        val=layer_name,
        mean_val=mean_val,
        std_val=std_val,
    )

def jax_collate(batch):
    latents = jnp.stack(
        [jnp.array(item["vae_output"], dtype=jnp.bfloat16) for item in batch], axis=0
    )
    labels = jnp.stack([int(item["label"]) for item in batch], axis=0)

    return {
        "vae_output": latents,
        "label": labels,
    }

def device_get_model(model):
    state = nnx.state(model)
    state = jax.device_get(state)
    nnx.update(model, state)

    return model

from typing import Dict

def get_mask(
    batch: int, length: int, mask_ratio: float, key=randkey
):
    """Get binary mask for input sequence.

    mask: binary mask, 0 is keep, 1 is remove
    ids_keep: indices of tokens to keep
    ids_restore: indices to restore the original order
    """
    len_keep = int(length * (1 - mask_ratio))
    key, subkey = jax.random.split(key)
    noise = jax.random.uniform(subkey, shape=(batch, length))  # noise in [0, 1]
    ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
    ids_restore = jnp.argsort(ids_shuffle, axis=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    mask = jnp.ones((batch, length))
    mask = mask.at[:, :len_keep].set(0)
    mask = jnp.take_along_axis(mask, ids_restore, axis=1)
    return {"mask": mask, "ids_keep": ids_keep, "ids_restore": ids_restore}


def mask_out_token(x: jnp.ndarray, ids_keep: jnp.ndarray) -> jnp.ndarray:
    """Mask out tokens specified by ids_keep."""
    N, L, D = x.shape  # batch, length, dim
    # Expand ids_keep to have the same number of dimensions as x
    ids_keep_expanded = jnp.expand_dims(ids_keep, axis=-1)
    ids_keep_expanded = jnp.tile(ids_keep_expanded, (1, 1, D))
    x_masked = jnp.take_along_axis(x, ids_keep_expanded, axis=1)
    return x_masked


def unmask_tokens(
    x: jnp.ndarray, ids_restore: jnp.ndarray, mask_token: jnp.ndarray
) -> jnp.ndarray:
    """Unmask tokens using provided mask token."""
    mask_tokens = jnp.tile(
        mask_token, (x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    )
    x_ = jnp.concatenate([x, mask_tokens], axis=1)
    # Expand ids_restore to have the same number of dimensions as x_
    ids_restore_expanded = jnp.expand_dims(ids_restore, axis=-1)
    ids_restore_expanded = jnp.tile(ids_restore_expanded, (1, 1, x_.shape[2]))
    x_ = jnp.take_along_axis(x_, ids_restore_expanded, axis=1, mode='clip')  # unshuffle
    return x_