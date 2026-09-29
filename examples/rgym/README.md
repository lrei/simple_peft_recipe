# Reasoning Gym: GRPO with PERL

Reinforcement learning (GRPO) on a verifiable reasoning task. The model
answers generated problems (default: `chain_sum`, sums like
`942 - 932 - 533 - 968 + 456 + 867`), a checker scores every sampled
answer, and GRPO pushes the LoRA adapter towards the higher-scoring
samples. No labelled data or SFT stage is needed, only a task the model
can already solve sometimes and a function that checks answers.

The script evaluates the model, trains, evaluates again and saves the
adapter, so one run shows the effect of RL.

## speftr APIs used (`rgym.py`)

| Function | speftr API |
|----------|------------|
| `build_perl_config` | Maps the CLI flags onto a `PERLConfig` |
| `main` | `PERL(config)`, `load_model()` (model with fresh LoRA adapters + tokenizer), `train(dataset, reward_funcs, eval_dataset)`, `save_model(save_method="lora")` |
| `accuracy_reward`, `format_reward` | Reward functions with the [PERL reward signature](../../docs/guide.md#reward-functions) |
| `to_hf_dataset` | Builds the `datasets.Dataset` with a `prompt` column that `PERL.train` needs ([RL data contract](../../docs/guide.md#rl-data-contract)) |

`rgym_inference.py` runs the trained adapter without speftr, as an
adapter and merged ([below](#use-the-trained-model-without-speftr)).

## Data

[Reasoning Gym](https://github.com/open-thought/reasoning-gym) (Apache-2.0,
`gym` extra) generates problems procedurally: no download, no gating.
`prepare_datasets` builds `--dataset_size` (10,000) training problems with
seed 1 and `--eval_dataset_size` (256) evaluation problems with seed 2.
Any Reasoning Gym task name works with `--dataset` (`chain_sum`,
`spell_backward`, ...).

`ReasoningGymDataset` renders each problem as a prompt string: the system
prompt `--developer_prompt` (a key of `reasoning_gym.utils.SYSTEM_PROMPTS`,
default `DeepSeekZero`) in the `--developer_role` turn, the question as
the user turn, and the generation prompt, with `enable_thinking=True`.
The raw Reasoning Gym entry goes into the `item` column, which TRL passes
to the reward functions.

One `chain_sum` prompt with Qwen3 (the gold answer is `6`):

```text
<|im_start|>system
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.
<|im_end|>
<|im_start|>user
State the final answer to the following arithmetic problem: 1 + 5 =<|im_end|>
<|im_start|>assistant
```

## Rewards

GRPO sums both rewards per completion (maximum 2.0) and compares each
completion with the others sampled for the same prompt.

- **`accuracy_reward`**: takes the text inside the *last*
  `<answer>...</answer>` block (Reasoning Gym's `extract_answer`) and
  returns the task's `score_answer(answer, item)`: 1.0 for a correct
  answer, 0.0 for a missing one, task-specific partial credit in between
  (`chain_sum` compares numbers, tolerating signs, leading zeros and
  commas). In `main` it is wrapped by `RewardLogger`, which logs one
  sampled completion, the extracted and expected answer every
  `--logging_steps` reward calls.
- **`format_reward`**: 0.25 for each of `<think>`, `</think>`, `<answer>`,
  `</answer>` present anywhere in the completion (presence only, not
  order). It lets a well-formatted wrong answer rank above an unformatted
  one.

Evaluation (`evaluate_model`) samples one completion per evaluation
problem with transformers `generate` (training sampling settings, prompts
truncated to 512 tokens, 512 new tokens) and counts it correct when
`score_answer` exceeds 0.5. Sampling makes it noisy on small evaluation
sets.

## Model

Default `Qwen/Qwen3-1.7B` (Apache-2.0): instruction-tuned with a thinking
mode, so it already produces `<think>` reasoning and solves some problems,
which GRPO needs to get a signal. Any chat model with a template works via
`--model_name_or_path`.

LoRA rank 1, alpha 32, on all attention and MLP projections (PERL's
defaults): 1,089,536 adapter parameters on Qwen3 1.7B. RL learns about
1 bit per episode (sampled completion)
([`speftr.lora_budget`](../../docs/guide.md#7-checking-the-rank)): 100
steps × 16 completions are 1,600 episodes, which need ~800 parameters,
so rank 1 suffices:

```bash
uv run python -m speftr.lora_budget --mode rl \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --max_steps 100 --completions_per_step 16
```

## Run

Needs the `gym` extra (`uv sync --extra gym`). From the repository root:

```bash
uv run python -m examples.rgym.rgym --output_dir ./models/chainsum \
  --max_steps 100 --eval_dataset_size 100 --vllm_sleep

# Another task
uv run python -m examples.rgym.rgym --dataset spell_backward \
  --output_dir ./models/spell --max_steps 100 --eval_dataset_size 100 \
  --vllm_sleep

# 4-bit policy without vLLM (less memory, slower)
uv run python -m examples.rgym.rgym --model_name_or_path Qwen/Qwen3.5-2B \
  --load_in_4bit --no_vllm --max_steps 30 --eval_dataset_size 64 \
  --output_dir ./models/chainsum-4bit
```

Key flags (`--help` lists all):

| Flag | Default | Notes |
|------|---------|-------|
| `--model_name_or_path` | `Qwen/Qwen3-1.7B` | |
| `--dataset` | `chain_sum` | Any Reasoning Gym task |
| `--max_steps` | 100 | Optimizer steps |
| `--num_generations` | 16 | Completions per prompt |
| `--per_device_train_batch_size` × `--gradient_accumulation_steps` | 8 × 2 | In completions; must be a multiple of `--num_generations` (16 = one prompt per step) |
| `--lora_r`, `--learning_rate`, `--scheduler` | 1, 1e-5, `constant` | `--warmup_ratio` (0.1) has no effect with `constant` |
| `--temperature`, `--top_p`, `--top_k`, `--min_p` | 0.6, 0.95, 20, 0.0 | Used for training and evaluation |
| `--max_prompt_length`, `--max_completion_length` | 512, 512 | |
| `--vllm_sleep` | off | Frees vLLM memory during optimizer steps; recommended |
| `--vllm_gpu_memory_utilization` | 0.5 | GPU share reserved for colocated vLLM |
| `--no_vllm` | vLLM on | Sample with transformers instead |
| `--load_in_4bit` | off | Requires `--no_vllm` |
| `--output_dir` | `./models/sudoku-perl` | Set it per run |

## Results

`chain_sum`, one RTX 3090, LoRA r=1, 16 generations, batch 8 × grad-acc 2,
512-token completions, `--vllm_sleep`:

| Model | Setup | Steps | Accuracy before → after | Train time |
|-------|-------|-------|-------------------------|------------|
| Qwen3 1.7B | bf16 + vLLM | 100 | 27% → 89% (100 problems) | 13 min, ~8 s/step |
| Qwen 3.5 2B | bf16 + vLLM | 30 | 23% → 94% (64 problems) | 6 min, ~13 s/step |
| Qwen 3.5 2B | 4-bit, `--no_vllm` | 30 | 39% → 94% (64 problems) | 17 min, ~35 s/step |
| Gemma 4 E2B (`unsloth/gemma-4-E2B-it`) | bf16 + vLLM 0.45, batch 2 × grad-acc 8 | 30 | 44% → 48% (64 problems) | 7 min |

The Qwen3 1.7B run takes 16 min end to end, including both evaluations;
its training reward (accuracy + format, maximum 2.0) averages 1.41
over steps 1–50 and 1.76 over steps 51–100. Gemma 4 E2B runs but
gains little in 30 steps with these settings.

### Two GPUs

`rgym.py` runs unchanged under `torchrun`: each process loads the model
on its own GPU, runs its own colocated vLLM engine and trains on its share
of the prompts. The advantage is throughput: each GPU generates and
trains on half of every step's completions, so the same training takes
fewer wall-clock minutes. Keep 16 completions per optimizer step with
per-GPU batch 8 × grad-acc 1 × 2 GPUs:

```bash
uv run torchrun --standalone --nproc_per_node 2 -m examples.rgym.rgym \
  --output_dir ./models/chainsum-2gpu --max_steps 100 \
  --eval_dataset_size 100 --vllm_sleep \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 1
```

Measured on 2× A100-SXM4-40GB, Qwen3 1.7B, rank 1: 100 steps in 5.8 min
(~3.5 s/step), 24.8 GB peak and ~72% utilization per GPU (whole job),
38% → 88% / 91% on 100 problems, 10 min end to end. Each process runs the
evaluation on its own, so the log shows one accuracy per GPU. Only rank
0 writes the adapter.

## Outputs

`--output_dir` receives the LoRA adapter (`adapter_model.safetensors`,
`adapter_config.json`), tokenizer files and a `checkpoint-*` directory
(every `--save_steps` steps if set, else at the end of the epoch or run).
Accuracy before and after training is logged. To load, merge or serve the
adapter, see [After training](../../docs/guide.md#5-after-training).

## Use the trained model without speftr

`rgym_inference.py` answers Reasoning Gym questions with the training
system prompt with only torch, transformers, peft and (for `--engine
vllm`) vllm, no speftr or Unsloth. It runs two routes and prints both
outputs per prompt:

- **adapter**: the base model plus the LoRA adapter, attached with peft
  or passed to vLLM as a `LoRARequest` (`run_adapter_route`);
- **merged**: the adapter merged into the 16-bit base and saved to
  `--merged_dir` (`merge_adapter`), then loaded from there
  (`run_merged_route`).

```bash
uv run python -m examples.rgym.rgym_inference \
    --adapter_dir ./models/chainsum
uv run python -m examples.rgym.rgym_inference \
    --adapter_dir ./models/chainsum --engine vllm
```

Output of the first command (RTX 3090):

```text
Engine: transformers. Merged model saved to ./models/chainsum-merged

>>> State the final answer to the following arithmetic problem: 4 + 3 =
adapter: <think>
Okay, let's see. [...] Yep, 4 plus 3 is indeed 7. No issues here.
</think>

<answer>7</answer>
merged:  <think>
Okay, let's see. [...] Yep, 4 plus 3 is indeed 7. No issues here.
</think>

<answer>7</answer>
[...]
```

`./models/chainsum-merged` (default `<adapter_dir>-merged`) holds
`model.safetensors`, `config.json`, `generation_config.json`, the
tokenizer and `chat_template.jinja`. It loads without peft, like any Hub
checkpoint (`AutoModelForCausalLM`, `pipeline`, `vllm serve
./models/chainsum-merged`).

`--adapter_dir` defaults to `rgym`'s default `--output_dir`
(`./models/sudoku-perl`); the commands use the run from [Run](#run). The
answers agree on both questions with both engines (`7`, `1692`); the
reasoning text differs in a few words on one question per engine, where
two tokens are nearly tied in bf16 and the rest of the reply follows.
The transformers run takes about 1 min and 4.6 GB of VRAM; vLLM reserves
80% of the GPU whatever the model size and starts one engine per route
(about 3 min per run in total).

- The prompt is `DEVELOPER_PROMPT` (Reasoning Gym's `DeepSeekZero`, the
  default `--developer_prompt`) as the system message, then the
  question, rendered with `enable_thinking=True` as in training. Change
  both if you trained with other `--developer_prompt` /
  `--developer_role` values.
- An adapter trained on Qwen 3.5 (`--model_name_or_path
  Qwen/Qwen3.5-2B`) works the same way; the script loads its multimodal
  class, skips the vision encoder in vLLM and saves the processor with
  the merged model.
- A merged Gemma 4 E2B or E4B model does not load in vLLM (its
  KV-shared layers have no saved key/value weights); for those use
  `--engine transformers`, or serve the adapter on the base with vLLM.

General rules (base class, 16-bit base, vLLM flags, serving): [After
training](../../docs/guide.md#5-after-training).

## Adapt it to your task

1. **Another Reasoning Gym task or mix.** Use `--dataset`, or pass several
   entries to `prepare_datasets` (`{"chain_sum": {"weight": 1.0},
   "spell_backward": {"weight": 1.0, "config": {...}}}`).
2. **Your own problems.** Build a `datasets.Dataset` with a `prompt`
   column (a rendered string as here, or a list of chat messages) and the
   columns your checker needs (e.g. `answer`), and pass it to
   `perl.train` in place of `to_hf_dataset(...)`. The
   [guide's GSM8K example](../../docs/guide.md#example-gsm8k-with-an-exact-match-reward)
   shows the minimal version.
3. **Your own reward.** Write `def reward(completions, answer, **kwargs)
   -> list[float]` returning one score per completion (dataset columns
   arrive as keyword arguments) and put it in `reward_funcs` in `main`.
   Keep a format reward like `format_reward` if you parse tagged output.
   Replace or drop `evaluate_model`, which relies on `item` and
   `score_answer`.
4. **Prompt.** Change `--developer_prompt` / `--developer_role`, or
   `ReasoningGymDataset.__getitem__`; keep asking for the tags your reward
   parses.
5. **Rank.** Rerun the [`speftr.lora_budget` command](#model) with your
   model, `--max_steps` and completions per step (batch × grad-acc)
   ([guide](../../docs/guide.md#7-checking-the-rank)).

## Pitfalls

- **Gemma 4 memory.** With defaults (vLLM 0.5, batch 8 × 2) Gemma 4 E2B
  runs out of memory in training; at `--vllm_gpu_memory_utilization 0.35`
  (batch 4 × 4) vLLM has no room for its KV cache. Use
  `--vllm_gpu_memory_utilization 0.45 --per_device_train_batch_size 2
  --gradient_accumulation_steps 8`.
- **4-bit needs `--no_vllm`.** `--load_in_4bit` with vLLM raises
  `ValueError`: PERL cannot sync LoRA updates into a 4-bit vLLM copy.
  Without vLLM, steps are slower (35 vs 13 s for Qwen 3.5 2B).
- **Out of memory in general.** Lower `--max_completion_length` or the
  batch (keep batch × grad-acc a multiple of `--num_generations`), lower
  `--vllm_gpu_memory_utilization`, and pass `--vllm_sleep`.
- **Truncated reasoning.** Completions cut at `--max_completion_length`
  have no `<answer>` and score 0; raise the limit for long problems.
- **Base models.** RL needs some correct samples to learn from; start from
  an instruction-tuned model, or SFT first.
