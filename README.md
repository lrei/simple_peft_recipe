# speftr: Simple Parameter Efficient Fine-Tuning Recipe

## Introduction

### Purpose

Parameter-Efficient Fine-Tuning (PEFT) is a set of techniques that adapt large
pretrained models to new tasks by updating only a small fraction of their
parameters. Instead of retraining the entire model, PEFT learns
small update parameters, achieving comparable performance with drastically
reduced compute, memory, and storage requirements.

When resources are limited (few GPUs, little time, or a small team),
extensive experimentation isn’t practical. A well-tested “recipe” for PEFT
provides a reliable starting point that works across common settings without
exhaustive tuning. It captures best practices, stable defaults, and proven
configurations so users can focus on their data, task, and getting results
quickly rather than parameter sweeps. In short, a good recipe turns PEFT from
a research challenge into more of an accessible, repeatable engineering process.

LoRA is our primary PEFT method: we freeze the base model and learn low-rank
updates. It can match the quality of "full fine-tuning" (FullFT) with far fewer
trainable parameters, resulting in lower memory/compute, faster training, and
minimal storage overhead. LoRA adapters are tiny, swappable files that can be
merged into the base weights for zero-overhead inference or loaded at runtime
(e.g., in vLLM) to avoid altering the base model, allowing hot-swapping
multiple adapters.

GRPO (Group Relative Policy Optimization) is a lightweight reinforcement
learning (RL) method -- in our opinion, the simplest and easiest for
fine-tuning a language model. At its simplest, it doesn't require a
frozen reference model or KL term, making it preferable for PEFT.

We assume the end user is GPU-limited and tuned the defaults for
a single consumer grade GPU (target: the Nvidia 3090) rather than
multi-gpu server setups (e.g. the prototypical 8xH100).

### Sources

The PEFT recipe presented in this codebase revolves around LoRA.
The main sources for this recipe are:

- LoRA Without Regret by John Schulman and Thinking Machines Lab (and references)
- LoRA Hyperparameters Guide by Unsloth (and references)
- Hugging Face TRL GRPO Trainer documentation (and references)

These are themselves partially based on previously published research.
Other resources have contributed as well as some experimentation.
You can find more information in the Bibliography section.

In certain cases we deviate slightly from any one source.

## The Recipe

- **Apply LoRA to ALL layers.**
- **Scaling factor**: $\alpha = 32$ (standard practice).
- **Learning rate schedule**: Constant or Constant with warmup.
- **Relatively High Learning Rates**: around 1e-4 for SFT and 1e-5 for RL.
- **Warmup**: 0 by default (reasonable to have up to 10% of steps)
- **Batch Size**:
  - 16 or 32 for SFT;
  - 8 or 16 for RL/GRPO (more would be asking too much for PEFT constraints).
- **Dropout**: no.
- **Optimizer**: 8bit AdamW by default
- **Epochs**: 1-3 epochs for SFT, 1-2 for RL (when not using steps)
- Gradient checkpointing enabled by default.
- Train on responses only when training with assistant templates.
- **SFT attention**: SDPA with padded, length-grouped batches. Unsloth's
  own defaults (flex attention, padding-free batches) assume
  FlashAttention-class kernels and are several times slower on a 3090.

For GRPO, we default to colocated vLLM with GPU memory utilization limited to
0.5. For the examples, vLLM sleep mode was used.

### Rank Selection

- **Choose rank based on dataset size and capacity requirements**:
  - Higher ranks needed for larger datasets
  - Easy starting rule-of-thumb:
    - 1 parameter per token in SFT;
    - 1 parameter per example in RL;
    - Minimum LoRA rank of 8.

This is considerably more than "LoRA Without Regret" but in our experience,
it makes things work well with minimum sweeps/tuning.
Note: I believe current implementation of unsloth doesn't support going below
lora rank 8.

#### Checking a rank against a dataset

`speftr.lora_budget` estimates the smallest rank a dataset needs; see
[the guide](docs/guide.md#7-checking-the-rank).

### Caveats

At the moment we don't officially support fine-tuning of input embeddings or
the output head. This means models will not learn to use tokens they have
not been trained on. This can be an issue if there is a mismatch with the
chosen template - e.g. using ChatML tokens with a model that hasn't been
pretrained to use it.

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

The versions are pinned in `uv.lock`. The project deliberately runs newer
transformers/TRL than the released unsloth declares, which is what makes
Gemma 4 and Qwen 3.5/3.8 usable. transformers is held at 5.13.1 because
vLLM 0.26 cannot read the per-layer Gemma 4 configs of later releases.

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

The `PESFT` and `PERL` classes are minor wrappers around Hugging Face's TRL.
Their configuration classes expose the parameters that a typical user would
want or need to change, with defaults for all of them from the above
"Recipe". `PESFT` uses Unsloth, which is faster and more convenient here.
`PERL` (GRPO) does not: the Unsloth version gave no obvious benefit and came
with a few more issues.

**[docs/guide.md](docs/guide.md)** is the user guide: data formats,
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

Each training example also has an `<example>_inference.py` that runs
the trained model without speftr (transformers + peft or vLLM, as a
LoRA adapter and merged).

Run them from the repository root as modules, e.g.
`uv run python -m examples.guard.guard_train --help`. They need an NVIDIA
GPU (see [Installation](#installation)).

## Future Work

- More & Better examples.
- Quantized reinforcement learning support.
- Easy support for fine-tuning input embeddings and LM output head.

## Bibliography

### Links

- https://thinkingmachines.ai/blog/lora/
- https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#training-on-completions-only-masking-out-inputs
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
