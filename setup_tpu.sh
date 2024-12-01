ls /dev/accel* && sudo apt-get update -y -qq && sudo apt-get upgrade -y -qq
sudo apt install -y -qq golang neofetch zsh byobu && sudo snap install btop astral-uv && sudo apt install -y -qq software-properties-common && sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get install -y -qq python3.12

pip install -U pip uv && uv python pin 3.12 && uv init microdit && uv venv .venv && source .venv/bin/activate

pip install -U wheel && pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && uv pip install -U flax matplotlib datasets numpy tqdm torch wandb transformers diffusers einops
pip install libtpu-nightly==0.1.dev20240731 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install jaxlib[tpu]==0.4.13 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --no-deps && pip install jax==0.4.13 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --no-deps
python -c "import jax; print('testing tpus'); print(f'JAX recognized devices:\n local device count = {jax.local_device_count()} \n {jax.local_devices()}')"
# TPU_LIBRARY_PATH=/home/kelechi/.local/lib/python3.12/site-packages/libtpu/libtpu.so