# speftr: Simple Parameter Efficient Fine-Tuning Recipe

## Introduction

### Purpose

Parameter-Efficient Fine-Tuning (PEFT) is a set of techniques that adapt large
pretrained models to new tasks by updating only a small fraction of their
parameters. Instead of retraining the entire model, PEFT learns
small update parameters and reaches comparable performance with much less
compute, memory and storage.

When resources are limited (few GPUs, little time, or a small team),
extensive experimentation isn't practical. A tested "recipe" for PEFT is a
starting point that works across common settings without tuning: it fixes
the defaults and configurations so users can spend their time on the data
and task instead of parameter sweeps.

LoRA is our primary PEFT method: we freeze the base model and learn low-rank
updates. It can match the quality of "full fine-tuning" (FullFT) with far fewer
trainable parameters, so it needs less memory and compute, trains faster and
stores little. LoRA adapters are small files that can be merged into the base
weights (no inference overhead) or loaded at runtime (e.g., in vLLM) without
altering the base model, which allows hot-swapping multiple adapters.

GRPO (Group Relative Policy Optimization) is a reinforcement learning (RL)
method; in our opinion, the simplest one for fine-tuning a language model.
In its simplest form it needs no frozen reference model or KL term, which
suits PEFT.

The library defaults target a single 24 GB consumer GPU (RTX 3090).
Multi-GPU training (data parallel and model splitting) is validated on
2× A100 40GB; it uses the same defaults plus launch flags, which the
examples document with measured settings.

### Sources

The recipe is built around LoRA. Its main sources are:

- LoRA Without Regret by John Schulman and Thinking Machines Lab (and references)
- LoRA Hyperparameters Guide by Unsloth (and references)
- Hugging Face TRL GRPO Trainer documentation (and references)

These build on earlier published research. Other resources and our own
experiments contributed; see the Bibliography section. In some cases the
recipe deviates slightly from any one source.

## The Recipe

- **LoRA targets**: all layers.
- **Scaling factor**: $\alpha = 32$ (standard practice).
- **Learning rate schedule**: constant or constant with warmup.
- **Learning rate**: relatively high, around 1e-4 for SFT, 1e-5 for RL
  and 5e-6 for DPO (about 10x the full fine-tuning rate).
- **Warmup**: 0 by default (reasonable to have up to 10% of steps)
- **Batch Size**:
  - 16 or 32 for SFT;
  - 8 or 16 for RL/GRPO (more would be asking too much for PEFT constraints);
  - 16 preference pairs for DPO.
- **Dropout**: no.
- **Optimizer**: 8bit AdamW by default
- **Epochs**: 1-3 epochs for SFT, 1-2 for RL, 1 for DPO (when not using
  steps)
- Gradient checkpointing enabled by default.
- Train on responses only when training with assistant templates.
- **SFT attention**: SDPA with padded, length-grouped batches. Unsloth's
  own defaults (flex attention, padding-free batches) assume
  FlashAttention-class kernels and are several times slower on a 3090.

For GRPO, we default to colocated vLLM with GPU memory utilization limited to
0.5. The examples use vLLM sleep mode.

### Rank Selection

The rank follows "LoRA Without Regret": LoRA stores about 2 bits of
information per parameter, SFT data carries about 1 bit per trained token
and RL about 1 bit per episode (DPO: per preference pair), so an adapter
needs roughly `trained tokens / 2` (SFT) or `episodes / 2` (RL)
parameters.

- Rank is the only capacity knob, in powers of two (1, 2, 4, 8, ...).
  Pick the smallest rank that covers the data; `speftr.lora_budget`
  computes it.
- A rank below what the data needs trains less efficiently. A rank above
  it costs compute and memory; LoRA Without Regret measures no quality
  loss, while Unsloth's guide warns of overfitting at very large ranks.
- Defaults: rank 8 for SFT, a safe over-provisioned choice (the smallest
  rank Unsloth's hyperparameter guide suggests); rank 1 for RL and DPO.

The examples use rank 1 (gptoss, rgym, text2sql) and rank 8 (guard,
intent, instruct, big); each README gives its budget.

#### Checking a rank against a dataset

`speftr.lora_budget` estimates the smallest rank a dataset needs; see
[the guide](docs/guide.md#7-checking-the-rank).

### Caveats

Fine-tuning the input embeddings or the output head is not supported, so
models do not learn to use tokens they were not trained on. This is a problem
when the chosen template does not match the model, e.g. ChatML tokens with a
model not pretrained on them.

## Installation

### Prerequisites

- Linux with an NVIDIA GPU (tuned for a single RTX 3090, 24 GB) and driver
  ≥ 580 (PyTorch is installed from the CUDA 13 index)
- Python 3.13 or 3.14
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
uv sync                          # SFT: torch, transformers, peft, TRL, unsloth
uv sync --extra rl               # + vLLM for fast GRPO generation (PERL)
uv sync --extra gym              # + reasoning-gym (RL example; includes rl)
uv sync --all-extras --group dev # everything, plus the quality tooling
```

The versions are pinned in `uv.lock`. The project runs newer transformers/TRL
than the released unsloth declares; Gemma 4 and Qwen 3.5/3.8 need them.
transformers is held at 5.13.1 because vLLM 0.26 cannot read the per-layer
Gemma 4 configs of later releases.

### Development

Quality checks run through [poethepoet](https://poethepoet.natn.io/):
`uv run poe` lists the tasks, `uv run poe quality-iter` runs the fast gate
and `uv run poe test-cuda` adds the GPU smoke tests. See
[docs/development.md](docs/development.md).

## Guide

`speftr` is a small drop-in library: install it (or copy its modules) into
your project and point it at your own models and datasets. You can also
copy parts of the files, or use TRL directly with the parameter values.
The `examples/` directory only shows the library in use on a few public
datasets; it is not part of the library.

The `PESFT`, `PERL` and `PEDPO` classes are thin wrappers around Hugging
Face's TRL. Their configuration classes expose the parameters a typical
user changes, with defaults from the "Recipe" above. `PESFT` uses
Unsloth, which is faster here. `PERL` (GRPO) uses plain transformers and
peft: Unsloth's GRPO path shows no clear benefit and has more issues.
`PEDPO` (DPO) also uses plain transformers and peft.

[docs/guide.md](docs/guide.md) is the user guide: data formats,
formatting and reward functions, configuration, SFT then RL, loading and
serving the result, validated models and troubleshooting. To run without
Hub access, see [docs/running_offline_models.md](docs/running_offline_models.md).

## Running the Examples

The examples show the library on public datasets; they are teaching code,
not tuned models. [examples/README.md](examples/README.md) indexes them
and explains how to adapt one to your data. Each has its own README with
commands, data format, hardware and measured results:

| Example | What it shows |
|---------|---------------|
| [guard](examples/guard/README.md) | SFT safety classifier on WildGuardMix (Gemma 3 270M) |
| [instruct](examples/instruct/README.md) | SFT instruction tuning plus console chat (Qwen3 0.6B, 4-bit) |
| [intent](examples/intent/README.md) | SFT customer-support intent routing on Banking77 (Granite 3.3 2B) |
| [text2sql](examples/text2sql/README.md) | 4-bit SFT then GRPO with a SQLite execution reward (SmolLM3-3B) |
| [rgym](examples/rgym/README.md) | GRPO with verifiable rewards on Reasoning Gym (Qwen3 1.7B, vLLM) |
| [prefs](examples/prefs/README.md) | DPO on a preference mix with held-out and RewardBench accuracy (OLMo 2 1B) |
| [big](examples/big/README.md) | 4-bit SFT of a 31B model on one 24 GB GPU; multi-GPU and Slurm (Gemma 4 31B) |
| [gptoss](examples/gptoss/README.md) | 4-bit SFT of an MoE reasoning model to reason in a requested language; 20b on one GPU, 120b split over GPUs (gpt-oss) |

Each training example also has an `<example>_inference.py` that runs
the trained model without speftr (transformers + peft or vLLM, as a
LoRA adapter and merged).

Run them from the repository root as modules, e.g.
`uv run python -m examples.guard.guard_train --help`. They need an NVIDIA
GPU (see [Installation](#installation)).

## Bibliography

### Links

- https://thinkingmachines.ai/blog/lora/
- https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#training-on-completions-only-masking-out-inputs
- https://www.reddit.com/r/LocalLLaMA/comments/1nwwoab/lora_without_regrets_implemented_in_hugging_face/
- https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py
- https://huggingface.co/docs/trl/main/en/grpo_trainer
- https://www.kaggle.com/code/viratchauhan/qwen-2-5-4-bit-q-3b-finetune-with-unsloth-w-b
- https://github.com/open-thought/reasoning-gym

### Libraries

- https://huggingface.co/docs/transformers
- https://huggingface.co/docs/peft/
- https://huggingface.co/docs/trl/
- https://unsloth.ai/
- https://huggingface.co/docs/datasets/

### References

```bibtex
@inproceedings{
hu2022lora,
title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
author={Edward J Hu and yelong shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nZeVKeeFYf9}
}
@inproceedings{10.5555/3666122.3666563,
author = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
title = {QLORA: efficient finetuning of quantized LLMs},
year = {2023},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
booktitle = {Proceedings of the 37th International Conference on Neural Information Processing Systems},
articleno = {441},
numpages = {28},
location = {New Orleans, LA, USA},
series = {NIPS '23}
}
@article{
biderman2024lora,
title={Lo{RA} Learns Less and Forgets Less},
author={Dan Biderman and Jacob Portes and Jose Javier Gonzalez Ortiz and Mansheej Paul and Philip Greengard and Connor Jennings and Daniel King and Sam Havens and Vitaliy Chiley and Jonathan Frankle and Cody Blakeney and John Patrick Cunningham},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=aloEru2qCG},
note={Featured Certification}
}
@article{schulman2025lora,
  author = {John Schulman and Thinking Machines Lab},
  title = {LoRA Without Regret},
  journal = {Thinking Machines Lab: Connectionism},
  year = {2025},
  note = {https://thinkingmachines.ai/blog/lora/},
  doi = {10.64434/tml.20250929},
}
 @misc{stojanovski2025reasoninggymreasoningenvironments,
      title={REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards},
      author={Zafir Stojanovski and Oliver Stanley and Joe Sharratt and Richard Jones and Abdulhakeem Adefioye and Jean Kaddour and Andreas Köpf},
      year={2025},
      eprint={2505.24760},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24760},
}
```

## Acknowledgments

This was developed as part of [DataPACT](https://datapact.eu/).
This project has received funding from the European Union's Horizon Europe
research and innovation programme under grant agreement No 101189771
