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
