### microdit_jax

Full Jax/NNX implementation of [**microdiffusion**](https://arxiv.org/abs/2407.15811) (Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budget).
Adapted from existing [official(Sony Research)](https://github.com/SonyResearch/micro_diffusion) and [unofficial(SwayStar123)](https://github.com/SwayStar123/microdiffusion) implementations.

Training/experimentations are proudly sponsored by Google TRC(TPU Research Cloud) grants.

[microdit_imagenet.py](microdit_imagenet.py) contains the complete(and wacky) single-file code for training a MicroDiT model from scratch,
on the [common-canvas](https://huggingface.co/datasets/SwayStar123/preprocessed_commoncatalog-cc-by) dataset. (you can also specify a smaller split of the ~15M images in the `config` class).

## License
Apache 2.0 License of course :) 

## BibTeX
```bibtex
@article{Sehwag2024MicroDiT,
  title={Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budget},
  author={Sehwag, Vikash and Kong, Xianghao and Li, Jingtao and Spranger, Michael and Lyu, Lingjuan},
  journal={arXiv preprint arXiv:2407.15811},
  year={2024}
}
```