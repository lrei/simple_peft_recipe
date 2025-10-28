# speftr: Simple Parameter Efficient Fine-Tuning Recipe

## Introduction

### Purpose
Parameter-Efficient Fine-Tuning (PEFT) is a set of techniques that adapt large 
pretrained models to new tasks by updating only a small fraction of their 
parameters. Instead of retraining the entire model, PEFT introduces learns 
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
updates, It can match the quality of "full fine-tuning" (FullFT) with far fewer 
trainable parameters. Resulting in lower memory/compute, faster training, and 
minimal storage overhead. LoRA adapters are tiny, swappable files that can be 
merged into the base weights for zero-overhead inference or loaded at runtime 
(e.g., in vLLM) to avoid altering the base model, allowing hot-swapping 
multiple adapters.

GRPO (Group Relative Policy Optimization) is a lightweight reinforcement 
learning (RL) method -- in our opinion, the simplest and easiest for 
fine-tuning a language model. At its simplest, it doesn't require a 
a frozen reference model or KL term making it preferable for PEFT.

We assume the end user is GPU-limited and tuned the defaults for
a single consumer grade GPU (target: the Nvidia 3090) rather than
multi-gpu server setups (e.g. the prototypical 8xH100).

### Sources
The PEFT recipe presented in this codebase revolves around LoRA. 
The main sources for this reciple are:

- LoRA Without Regret by John Schulman and Thinking Machines Lab (and references)
- LoRA Hyperparameters Guide by Unsloth (and references)
- Huggingface TRL GRPO Trainer documentation (and references)

These are themselves partially based on previously published research. 
Other resources have contributed as well as some experimentation.
You can find more information in the Bibliography section.

In certain cases we deviate slightly from any one source.

## The Recipe
- **Apply LoRA to ALL layers**
- **Scaling factor**: $\alpha = 32$ (standard practice).
- **Learning rate schedule**: Constant or Constant with warmup.
- **Relatively High Learning Rates**: around 1e-4 for SFT and 1e-5 for RL.
- **Warmup**: 0 by default (reasonable to have up to 10% of steps)
- **Batch Size**:
  - 16 or 32 for SFT;
  - 8 or 16 for RL/GRPO (more would be asking too much for PEFT constrains).
- **Dropout**: no.
- **Optimizer**: 8bit AdamW by default
- **Epochs**: 1-3 epochs for SFT, 1-2 for RL (when not using steps) 
- Gradient checkpointing enabled by default.
- Train on responses only when training with assistant templates.

For GRPO, we default to collocated vLLM limited gpu memory utilization to 0.5.
For the examples, vLLm sleep was used.

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

### Caveats
At the moment we don't officially support fine-tuning of input embeddings or
the output head. This means models will not learn to use tokens they have 
not been trained on. This can be an issue if there is a mismatch with the 
chosen template - e.g. using ChatML tokens with a model that hasn't been 
pretrained to use it.


## Installation

### Prerequisites
- Python >= 3.13
- CUDA-compatible GPU (recommended for training)
- [uv](https://docs.astral.sh/uv/) package manager

### Core Installation
For basic Parameter-Efficient Fine-Tuning functionality:

```bash
uv sync
```

This installs the essential dependencies:
- `torch>=2.7.1` (with CUDA 12.8 support)
- `transformers>=4.54.0`
- `peft>=0.17.1`
- `accelerate>=1.9.0`
- `unsloth>=2025.9.9`
- `scikit-learn>=1.7.2`

### Reinforcement Learning Support
For reinforcement learning examples (GRPO, etc.):

```bash
uv sync --extra rl
```

This adds:
- `trl>=0.23.1` (TRL library)
- `vllm>=0.11.0` (inference engine)
- `flashinfer-python>=0.3.1.post1` (optimized attention)
- `flashinfer-cubin>=0.4.1` (CUDA kernels)

### Reasoning Gym Examples
For reasoning gym examples (includes RL dependencies):

```bash
uv sync --extra gym
```

This adds:
- `reasoning-gym>=0.1.0` (reasoning environments)
- All RL dependencies automatically

### Development Dependencies
For development and contributing:

```bash
uv sync --group dev
```

This adds:
- `ruff>=0.12.4` (linting and formatting)
- `pyright>=1.1.406` (type checking)
- `pyflakes>=3.4.0` (static analysis)

### Complete Installation
For all features including examples:

```bash
uv sync --extra rl --extra gym
```

## Guide
This isn't a full fledged library. More of a "starter kit".
You can either clone the repository or just copy the PEFT/PERL files in
their entirety, or simply copy paste parts of the files or use TRL directly 
and copy the parameter values.

The `PESFT` and `PERL` classes are minor wrappers around Hugginface's TRL.
They have predifined configuration classes that expose the the parameters that
a typical user would want or need to change and in general have sensible 
defaults for all of them using the above "Recipe".

The `PESFT` class directly uses Unsloth's version since within this set of
options, it's just faster and more convenient.

The `PERL` class is focused on GRPO and does not use Unsloth patched versions 
at the moment. The Unsloth version did not provide any obvious benifit and 
came with a few more issues.


### PESFT: Parameter Efficient Supervised Fine-Tuning

`PESFT` exposes a thin, typed configuration that mirrors the knobs we tune in
practice. The goal of the API is to make the important choices explicit while
keeping sensible defaults everywhere else.

| Group | Representative options |
|-------|------------------------|
| **Model & template** | `model_name_or_path`, `chat_template`, `train_on_responses`, `instruction_part`, `response_part` |
| **Adapter shape** | `lora_r`, `lora_alpha`, `target_modules`, `use_gradient_checkpointing` |
| **Optim & logging** | `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `scheduler`, `warmup_ratio`, `weight_decay`, `logging_steps`, `save_strategy`, `save_method` |

A minimal configuration looks like this:

```python
from speftr import PESFT, PESFTConfig


config = PESFTConfig(
    model_name_or_path="unsloth/Qwen2.5-0.5B-Instruct",
    lora_r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    scheduler="constant",
    warmup_ratio=0.03,
    output_dir="./models/speftr-sft",
    save_method="merged_16bit",
)

trainer = PESFT(config)
model, tokenizer = trainer.load_model()
trainer.train(train_dataset, eval_dataset, formatting_fn)
trainer.save_model()
```

### PERL: Parameter Efficient Reinforcement Learning

`PERL` targets GRPO-style reinforcement learning and extends the same idea: a
single dataclass captures both LoRA hyperparameters and sampling behaviour.

| Group | Representative options |
|-------|------------------------|
| **LoRA & optimisation** | `lora_r`, `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_train_epochs`, `max_steps` |
| **Generation controls** | `num_generations`, `temperature`, `top_p`, `top_k`, `min_p`, `stop_sequences`, `include_stop_str_in_output` |
| **Sequence limits & infra** | `max_seq_length`, `max_prompt_length`, `max_completion_length`, `use_vllm`, `vllm_gpu_memory_utilization`, `vllm_enable_sleep_mode` |

```python
from speftr import PERL, PERLConfig


config = PERLConfig(
    model_name_or_path="./models/speftr-sft",
    lora_r=1,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,
    temperature=1.0,
    stop_sequences=["<|im_end|>", "<|endoftext|>"],
    output_dir="./models/speftr-rl",
)

perl = PERL(config)
model, tokenizer = perl.load_model()
perl.train(grpo_dataset, reward_funcs)
perl.save_model()
```


## Running the Examples
Note that these examples are educational rather than geared towards creating
best-in-class models.

Each of the main example scripts is commented to serve as an example for 
your own work.

### Requirements
- Python >= 3.13
- CUDA-compatible GPU (recommended)
- Sufficient GPU memory for your chosen model size
- See [Installation](#installation) section for detailed dependency information

### Usage

#### Supervised Fine Tuning: Guardrails example
To just train a simple demo guardrails model on wilguardmix:
1 - make sure you have requested access to the data on HF
2 - have access to the model you chose (default gemma3)
3 - export the login token or login to HF in some way

Help is available for all tools in the guard directory:

```bash
  uv run python -m examples.guard.guard_train --help
  uv run python -m examples.guard.guard_eval --help
  uv run python -m examples.guard.guard_test --help
  uv run python -m examples.guard.guard_count_tokens --help
```

Simply:
```bash
uv run python -m examples.guard.guard_train --output_dir /gmodel
```

Will train with all defaults.

The `guard_test.py` is a simple command line interface that lets you
type a prompt and check the eval from the model.

```bash
uv run python -m examples.guard.guard_test --model_path /gmodel
```

If you need an evaluation of how well the model works:
```bash
uv run python -m examples.guard.guard_eval --model_path /data2/peft/model
```
This will run the evaluation on wildguardmix test set and output metrics
like this:

```text
Total invalid predictions: 0 / 1699

============================================================
Overall Metrics:
============================================================

accuracy: 0.7081
precision_macro: 0.7877
recall_macro: 0.6747
f1_macro: 0.6610
avg_precision: 0.6213

              precision    recall  f1-score   support

     harmful       0.91      0.38      0.53       754
   unharmful       0.66      0.97      0.79       945

    accuracy                           0.71      1699
   macro avg       0.79      0.67      0.66      1699
weighted avg       0.77      0.71      0.68      1699

```
####  Supevised Fine-Tuning: Instruction-tuning example
A barebones example of supervised Instruction Tuning.
The default example was to `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` using 
`--load_in_4bit` to answer like a pirate. 

```bash
  uv run python -m examples.instruct.instruct --output_dir ./model-it --load_in_4bit --num_epochs 1
```

You can then talk to the model using something like:

```bash
uv run python -m examples.chat --model_name_or_path "./model-it" --system_prompt "Always answer like a pirate."
```

```
============================================================
Interactive Chat
============================================================
Type your messages and press Enter. Type 'exit' or 'quit' to stop, or press Ctrl+C/Ctrl+D.
============================================================

You: Who was George Washington?
Assistant: George washington was an american president of th' united states from 1789 until 1797. He was a key figure in th' founding of th' united states and played a critical role in th' development of th' new republic.
```

The pirate talk dataset is not that great but it's very particular and small.

#### Reinforcement Learning (GRPO): Reasoning Gym Example
RL works better when the LM can already perform the task to some degree.
Picking an already instruction tuned model with thinking mode is a good choice
for the gym. We tested 2 tasks with the default model, Qwen3-1.7B, which 
is already instruction tuned tuned for reasoning/math. The tasks
were chain_sum and spell_backward. Results are significant and visible in 
just 100 steps but improvements keep going up until our maximum test size of 
500 training steps.

```bash
 uv run python -m examples.rgym.rgym --output_dir /data2/peft/chainsum --max_steps 100 --eval_dataset_size 100 --vllm_sleep
```

or

```bash
 uv run python -m examples.rgym.rgym --dataset spell_backward --output_dir /data2/peft/chainsum --max_steps 100 --eval_dataset_size 100 --vllm_sleep
```

at the end you'll see something like this:

```
GRPO training complete!
2025-10-28 14:04:33 - INFO - __main__ - Evaluating model after RL training...
2025-10-28 14:04:33 - INFO - __main__ - ================================================================================
2025-10-28 14:04:33 - INFO - __main__ - Evaluating on 100 samples (batch_size=64)
2025-10-28 14:04:33 - INFO - __main__ - --------------------------------------------------------------------------------
2025-10-28 14:05:44 - INFO - __main__ - Evaluation Results:
2025-10-28 14:05:44 - INFO - __main__ -   Accuracy: 86.00% (86/100)
2025-10-28 14:05:44 - INFO - __main__ - ================================================================================
2025-10-28 14:05:44 - INFO - __main__ - ================================================================================
2025-10-28 14:05:44 - INFO - __main__ - Training Summary:
2025-10-28 14:05:44 - INFO - __main__ - --------------------------------------------------------------------------------
2025-10-28 14:05:44 - INFO - __main__ - Base model accuracy: 28.00%
2025-10-28 14:05:44 - INFO - __main__ - Final model accuracy: 86.00%
2025-10-28 14:05:44 - INFO - __main__ - Improvement: +58.00%
2025-10-28 14:05:44 - INFO - __main__ - ================================================================================
2025-10-28 14:05:44 - INFO - __main__ - Saving model...
```

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
```
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