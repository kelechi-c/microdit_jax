### notes....on my tiny video generation model
So basically, it took me like _2 weeks_ to port MicroDiT to JAX[w/NNX], at least to a _trainable_ version(tok that time to actually learn **jax/tpu** usage lmao). Thank God for the **TRC grant**.

Anyways, my goals are: 
- develop/train a tiny image generation model
- implement a tiny video generation backbone w/**microdit** framework.
- train it on a simple video-label dataset.
- scale to text-video hopefully
- apply the knowledge gained to audio

The goal isn't to get SOTA results, just good outputs :)

Now I gotta choose what model code/paper to merge..

there's Mochi, Allegro, OpenSora, LTX-Vid...Hmmm
Well, I went with LTX-Video, from Lightricks.
(Mochi is bettter, but uses MMDiT, which isn't so compatible with microdit)

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
2. Moving arrays to multiprocess devices(like v4-32 pod)
```python
#from jax.experimental.pjit import pjit


```