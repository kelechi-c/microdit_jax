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
    """Mask out tokens specified by ids_keep using lax.gather."""

    N, L, D = x.shape  # batch, length, dim
    N_keep, K = ids_keep.shape  # batch, number of tokens to keep

    # Create index arrays
    batch_indices = jnp.arange(N)[:, None, None]  # Shape (N, 1, 1)
    keep_indices = ids_keep[:, :, None]  # Shape (N, K, 1)
    dim_indices = jnp.arange(D)[None, None, :]  # Shape (1, 1, D)

    # Broadcast the index arrays to the desired output shape (N, K, D)
    batch_indices = jnp.broadcast_to(batch_indices, (N, K, D))
    keep_indices = jnp.broadcast_to(keep_indices, (N, K, D))
    dim_indices = jnp.broadcast_to(dim_indices, (N, K, D))

    # Use advanced indexing to gather the desired tokens
    x_masked = x[batch_indices, keep_indices, dim_indices]
    # print(f'x masked {x.shape}')

    return x_masked


def unmask_tokens(x: jnp.ndarray, ids_restore: jnp.ndarray, mask_token) -> jnp.ndarray:
    """Unmask tokens using provided mask token with take_along_axis."""
    # Repeat mask token for batch size and missing tokens
    N, L_masked, D = x.shape
    L_original = ids_restore.shape[1]

    # Repeat mask token for batch size and missing tokens
    num_missing = L_original - L_masked
    mask_tokens = jnp.tile(mask_token, (N, num_missing, 1))

    # Concatenate original tokens with mask tokens
    x_ = jnp.concatenate([x, mask_tokens], axis=1)

    # Create index arrays
    batch_indices = jnp.arange(N)[:, None, None]  # (N, 1, 1)
    seq_indices = ids_restore[:, :, None]  # (N, L_original, 1)
    depth_indices = jnp.arange(D)[None, None, :]  # (1, 1, D)

    # Broadcast indices to the desired output shape (N, L_original, D)
    batch_indices = jnp.broadcast_to(batch_indices, (N, L_original, D))
    seq_indices = jnp.broadcast_to(seq_indices, (N, L_original, D))
    depth_indices = jnp.broadcast_to(depth_indices, (N, L_original, D))

    # Use advanced indexing to gather and reorder
    x_unmasked = x_[batch_indices, seq_indices, depth_indices]
    # print(f'{x_unmasked.shape = }')

    return x_unmasked


def ntopk(
    x: jax.Array, k: int, axis: int = -1, largest: bool = True, sorted: bool = True
):
    sorted_indices = jnp.argsort(-x, axis=-1)
    # Only keep top k indices
    indices = sorted_indices[..., :k]

    # Create gather dimensions
    batch_size, num_experts, _ = x.shape
    batch_indices = jnp.arange(batch_size)[:, None, None]
    batch_indices = jnp.broadcast_to(batch_indices, (batch_size, num_experts, k))
    expert_indices = jnp.arange(num_experts)[None, :, None]
    expert_indices = jnp.broadcast_to(expert_indices, (batch_size, num_experts, k))

    # Stack indices for gather
    gather_indices = jnp.stack([batch_indices, expert_indices, indices], axis=-1)

    # Gather values using lax.gather
    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0, 1, 2), start_index_map=(0, 1, 2)
    )
    slice_sizes = (1, 1, 1)
    values = jax.lax.gather(
        x, gather_indices, dimension_numbers=dnums, slice_sizes=slice_sizes
    )
    values = values.reshape(batch_size, num_experts, k)

    return values, indices
