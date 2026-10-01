# Diff-Instruct with Diffused Reward: Towards Principled One-step Generator RL

**NeurIPS 2026**

Junyi Wu<sup>1</sup>, Weijian Luo<sup>2</sup>, Haoyang Zheng<sup>1</sup>, Ruizhe Zhang<sup>1</sup>, Guang Lin<sup>1</sup>

<sup>1</sup>Purdue University &nbsp;&nbsp; <sup>2</sup>hi-lab, Xiaohongshu Inc.

[![arXiv](https://img.shields.io/badge/arXiv-2605.24001-b31b1b)](https://arxiv.org/abs/2605.24001)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Junyi-W/DIDR)

![teaser](assets/teaser.jpg)

*1024×1024 images from one-step SDXL aligned via DIDR.*

## Abstract

Recent advances in one-step text-to-image generation have enabled real-time synthesis with remarkable efficiency and quality. Previous reinforcement learning methods for one-step generators combine reward optimization in image space with distribution matching in the diffusion noise space. This paradigm brings challenges due to a mismatch between terminal reward optimization and the underlying generative dynamics. As a result, optimization tends to exploit stochastic degrees of freedom, often improving reward at the expense of image fidelity. To address this issue, we propose **Diff-Instruct with Diffused Reward** (DIDR), a data-free trajectory-level alignment framework derived from Integral KL minimization. DIDR propagates the RLHF-optimal reward-tilted clean-image distribution across all noise levels along the diffusion trajectory. We show that this objective admits the same minimizer as clean-image RLHF, while naturally inducing **the Diffused Reward Score** (DRS), which acts as a reward-driven correction to the reference score function. To make this practical, we further introduce **the Diffused Reward Proxy** (DRP), an efficient estimator of DRS based on differentiable short-step denoising. Extensive experiments demonstrate that DIDR consistently Pareto-dominates existing one-step SDXL baselines. Moreover, when transferred to a 6B DiT backbone (Z-Image), DIDR surpasses its 50-step teacher in preference alignment with a single generation step and outperforms the 8-step Z-Image-Turbo with only 4 steps.

## Models

Pretrained weights are available on [Hugging Face](https://huggingface.co/Junyi-W/DIDR), with usage instructions in the model card:

- `didr_sdxl_1step_unet_fp16.safetensors`: 1-step SDXL (DIDR)
- `didr_longer_sdxl_1step_unet_fp16.safetensors`: 1-step SDXL (DIDR-longer)
- `zimage_transformer/`: 1-step Z-Image (DIDR)

## Contact

Junyi Wu [wu2393@purdue.edu](mailto:wu2393@purdue.edu)

## Citation

```bib
@inproceedings{wu2026didr,
    title={Diff-Instruct with Diffused Reward: Towards Principled One-step Generator RL},
    author={Wu, Junyi and Luo, Weijian and Zheng, Haoyang and Zhang, Ruizhe and Lin, Guang},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2026}
}
```

## License

The model weights are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
