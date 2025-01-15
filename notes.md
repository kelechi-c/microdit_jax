### notes....on my tiny video generation model
So basically, it took me like _weeks_ to port MicroDiT to JAX[w/NNX], at least to a _trainable_ version(took that time to actually learn **JAX/tpu** usage lmao). 

Thank God for the **TRC grant**.

(Update: I paused in and rewrote a lot of stuff and debugged the model, after the official codebase was released Jan 2025. So it _actually_ took me ~2 weeks for the whole implementation/porting).

Anyways, my goals are:
- develop/train a tiny image generation model
- implement a tiny video generation backbone with the **microdit** framework.
- scale to text-video hopefully
- apply the knowledge gained to audio

The goal isn't to get SOTA results, just experiments/good outputs :)


#### stuff/snippets I learnt/used on the way
 
1. data parallelism with JAX
* define device mesh and sharding for data/model
```python
import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils

device_count = jax.device_count()
devices = jax.local_devices()

device_count = jax.local_device_count()
mesh = Mesh(mesh_utils.create_device_mesh((device_count,)), ('data',))

model_sharding = NamedSharding(mesh, PS())
data_sharding = NamedSharding(mesh, PS('data'))
```
* replicate the model params/optimizer on all devices
```python
# replicate model
state = nnx.state((cnn_model, optimizer))
state = jax.device_put(state, model_sharding)
nnx.update((cnn_model, optimizer), state)
```

* shard the data batch across the TPU devices
```python
image, label = batch
image, label = jax.device_put((image, label), data_sharding)
...

```
* retrieve and merge state
```python
state = nnx.state((model, optimizer))
state = jax.device_get(state)
nnx.update((model, optimizer), state)
```

2. Small models can learn basic structures in the image. So I used them for testing. And smaller models(~80m params) work well on faces...not so well for others.

3. Initialize model properly (xavier for linear layer weights, zero constant for biases) for stable training.