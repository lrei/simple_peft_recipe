# Summarization: Reddit TL;DR with Qwen 3.5 4B (SFT)

Reddit users end long posts with a "TL;DR", a one- or two-sentence
summary. This example fine-tunes a chat model to write one for a post:
the post goes in a user turn, the author's TL;DR is the assistant turn,
and the loss covers only the TL;DR. The same prompt scores the base
model and the fine-tuned one, with thinking turned off at generation
time: a summary is a direct answer, not a reasoning task.

It shows SFT on a free-form answer with a reference-based metric
(ROUGE) and the handling of a thinking model's template (Qwen 3.5).
The training script takes any chat model: the model keeps its own chat
template and `PESFT` infers the response-only markers from it. Qwen 3.5
4B is the default; Gemma 4 E4B and Granite 3.3 2B are measured too.

## Files and speftr APIs

| File | What it does | speftr API |
|------|--------------|------------|
| `tldr_eval.py` | Defines the task (`load_tldr`, `build_messages`, `is_valid_summary`, `compute_metrics`) and scores a model on the test split | none (Unsloth `FastLanguageModel` for generation) |
| `tldr_train.py` | LoRA SFT | `build_parser` is `PESFTConfig.get_argument_parser()` with the example's defaults (`EXAMPLE_DEFAULTS`); `main` calls `PESFT(config)`, `load_model()`, `train(train, eval, format_batch)`, `save_model()` |
| `tldr_inference.py` | Summarizes posts with the adapter and the merged model, without speftr ([below](#use-the-trained-model-without-speftr)) | none (transformers + peft, or vLLM) |

`tldr_train.build_formatting_func` returns the batched formatting
function that `PESFT.train` expects (see the
[SFT data contract](../../docs/guide.md#sft-data-contract)).

## Data

[`trl-lib/tldr`](https://huggingface.co/datasets/trl-lib/tldr), TRL's
prompt-completion version of
[`webis/tldr-17`](https://huggingface.co/datasets/webis/tldr-17)
(CC-BY-4.0), not gated: 116,722 train, 6,447 validation and 6,553 test
rows with columns `prompt` (the post: `SUBREDDIT:`, `TITLE:`, `POST:`
sections ending with the `TL;DR:` cue) and `completion` (the author's
TL;DR).

- `load_tldr` renames the columns to `post` and `summary`: TRL trains a
  prompt-completion dataset as is and skips the formatting function.
- `build_messages` drops the trailing `TL;DR:` cue, since the
  instruction asks for the TL;DR, and puts the post after the
  instruction in one user turn; the TL;DR is the assistant turn.
- `tldr_train` trains on the train split (`--train_rows` to use a
  prefix) and takes `--eval_rows` (500) validation rows for the eval
  loss; `tldr_eval` scores `--max_samples` (1,000) test rows.

One training text, as printed by `tldr_train` (post shortened):

```text
<|im_start|>user
Write a TL;DR of the Reddit post below in one or two sentences. Answer with the TL;DR only.

SUBREDDIT: r/loseit

TITLE: SV & NSV! Keeping on keeping on.

POST: 30F, 5'6". SW: 236 GW: 150 CW: 219

I weigh myself weekly and measure myself monthly. ...<|im_end|>
<|im_start|>assistant
<think>

</think>

Progress is still happening, even when you think it might not be! Don't get discouraged, even if your journey seems to be going slowly. Don't give up, warriors.<|im_end|>
```

With `train_on_responses=True` and the inferred ChatML markers, only
the assistant turn is trained: the empty think block Qwen 3.5's template
renders, then the TL;DR.

## Model

Default: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B),
Apache-2.0, trained in bf16 with LoRA. Its chat template starts every
assistant turn with an empty think block, which is trained as part of
the answer; `tldr_eval` and `tldr_inference` generate with
`enable_thinking=False`, so the generation prompt ends with the same
block and the model answers directly.

Any chat model whose template `speftr.chat_markers` recognises works
through `--model_name_or_path` (`chat_template=None`, markers inferred).
The runs below also use
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
and
[`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
(both Apache-2.0); their templates ignore the thinking switch.

## Run

From the repository root (Unsloth needs an NVIDIA GPU even to import):

```bash
# 1. Base model score
uv run python -m examples.tldr.tldr_eval --model_path Qwen/Qwen3.5-4B
# 2. Train (the measured runs below used 300 steps)
uv run python -m examples.tldr.tldr_train --max_steps 300
# 3. Score the adapters (default --model_path)
uv run python -m examples.tldr.tldr_eval
# Another model: same steps, its own output directory
uv run python -m examples.tldr.tldr_train --max_steps 300 \
  --model_name_or_path google/gemma-4-E4B-it \
  --output_dir ./models/gemma-4-e4b-lora-tldr
uv run python -m examples.tldr.tldr_eval \
  --model_path ./models/gemma-4-e4b-lora-tldr
```

`tldr_train` accepts every `PESFTConfig` flag (`--help` lists them with
the example's defaults in the epilog). Main flags:

| Script | Flag | Default |
|--------|------|---------|
| `tldr_train` | `--model_name_or_path` | `Qwen/Qwen3.5-4B` |
| | `--max_steps` / `--num_train_epochs` | -1 (use epochs) / 1 |
| | `--per_device_train_batch_size`, `--gradient_accumulation_steps` | 16, 1 |
| | `--lora_r`, `--learning_rate` | 2, 2e-4 |
| | `--max_seq_length` | 1024 |
| | `--load_in_4bit` | off (bf16) |
| | `--use_gradient_checkpointing`, `--attn_implementation`, `--padding_free`, `--packing` | `unsloth`, `sdpa`, off, off ([Speed](#speed)) |
| | `--train_rows`, `--eval_rows` | -1 (all), 500 |
| | `--output_dir` | `./models/qwen3.5-4b-lora-tldr` |
| `tldr_eval` | `--model_path` | `./models/qwen3.5-4b-lora-tldr` |
| | `--max_samples` | 1000 (-1: all 6,553) |
| | `--batch_size`, `--max_new_tokens` | 32, 96 |
| | `--load_in_4bit` | off (bf16) |

## Results

bf16, LoRA r=2 (the rank the budget check below gives), batch 16, 300
steps (4,800 of the 116,722 train posts), scored on the first 1,000
test posts with greedy decoding and thinking off. The same seed and
settings were run on one RTX 3090 (24 GB) and on one A100 40GB:

| Model | GPU | Base: ROUGE-1 / ROUGE-2 / ROUGE-L, words | Fine-tuned: ROUGE-1 / ROUGE-2 / ROUGE-L, words | Final eval loss | 300 steps |
|-------|-----|------------------------------------------|------------------------------------------------|-----------------|-----------|
| Qwen 3.5 4B | RTX 3090 | 0.227 / 0.041 / 0.152, 37 | 0.361 / 0.140 / 0.285, 22 | 1.742 | 18.5 min (3.40 s/step) |
| Qwen 3.5 4B | A100 | 0.227 / 0.041 / 0.153, 37 | 0.362 / 0.139 / 0.284, 23 | 1.737 | 8.0 min (1.45 s/step) |
| Gemma 4 E4B | RTX 3090 | 0.253 / 0.056 / 0.180, 38 | 0.357 / 0.139 / 0.278, 24 | 1.918 | 19.7 min (3.73 s/step) |
| Gemma 4 E4B | A100 | 0.253 / 0.056 / 0.180, 38 | 0.357 / 0.139 / 0.278, 24 | 1.920 | 8.5 min (1.53 s/step) |
| Granite 3.3 2B | RTX 3090 | 0.223 / 0.039 / 0.152, 41 | 0.354 / 0.136 / 0.280, 22 | 1.742 | 14.2 min (2.67 s/step) |
| Granite 3.3 2B | A100 | 0.221 / 0.039 / 0.151, 41 | 0.356 / 0.136 / 0.280, 23 | 1.744 | 6.4 min (1.14 s/step) |

- No reply was invalid (empty or carrying think tags) in any of these
  runs. "Words" is the mean length of the replies; the authors' TL;DRs
  average 27 words on these 1,000 posts.
- The three models end within 0.01 ROUGE of each other, and the two
  GPUs within 0.002 for the same seed. The base models answer in the
  third person, at length, Gemma with a `TL;DR:` prefix; the fine-tuned
  ones write the author's first-person one-liner, which is what ROUGE
  against the author's TL;DR rewards.
- Peak training memory (`train_metrics.json`): Qwen 22.2 GiB, Gemma
  19.3 GiB, Granite 8.9 GiB. Gemma evaluates at
  `--per_device_eval_batch_size 4` on both GPUs (its 262k-token
  vocabulary makes the eval logits of a batch of 16 exceed 24 GB).
- Rank check ([`speftr.lora_budget`](../../docs/guide.md#7-checking-the-rank)):
  the whole train split's TL;DRs are 4.1M to 4.5M response tokens
  depending on the tokenizer, which need about 2.1M parameters; rank 2
  has 2.65M (Qwen), 4.36M (Gemma) and 3.52M (Granite) parameters and is
  the smallest rank that covers them (rank 1 covers Gemma's count
  only). 300 steps see 4% of that data:

  ```bash
  uv run python -m speftr.lora_budget \
      --model_name_or_path Qwen/Qwen3.5-4B \
      --dataset trl-lib/tldr --split train \
      --prompt_column prompt --response_column completion --responses_only
  ```

## Speed

`examples/speed.py` runs `tldr_train` once per setting of the recipe's
speed knobs for a fixed number of steps and tables seconds per step,
samples per second and peak memory from each run's
`train_metrics.json`:

```bash
uv run python -m examples.speed examples.tldr.tldr_train \
    --steps 40 --output_dir ./models/speed-tldr
```

The settings, columns and reading rules are those of the
[intent example](../intent/README.md#speed): effective batch 16 except
`batch_32`, steady-state step time (median after the first 5 steps),
eval loss after 40 steps as the check that a setting still trains the
same thing. TL;DR rows are longer than Banking77 prompts (posts plus
TL;DR average 372 tokens, 581 at most, under the 1024 limit).

Measured with 40 steps on one RTX 3090 (24 GB) and one A100 40GB
(PCIe), bf16 unless the setting says otherwise, eval batch 4 for Gemma:

### Granite 3.3 2B

| Setting | RTX 3090 s/step | A100 s/step | Peak GiB (3090 / A100) | Eval loss (3090 / A100) |
|---------|-----------------|-------------|------------------------|-------------------------|
| default | 2.69 | 1.15 | 8.9 / 8.9 | 1.75 / 1.75 |
| no checkpointing | out of memory | out of memory |  |  |
| batch 8 × 2 | 2.84 | 1.20 | 8.8 / 8.8 | 1.75 / 1.75 |
| batch 4 × 4 | 2.97 | 1.42 | 8.8 / 8.8 | 1.75 / 1.75 |
| padding-free | 5.85 | 2.99 | 8.5 / 8.5 | 4.38 / 4.39 |
| flex attention | 2.58 | 1.12 | 8.9 / 8.9 | 1.75 / 1.75 |
| flex + padding-free | 2.60 | 1.11 | 8.5 / 8.5 | 4.37 / 4.38 |
| packing | 17.83 | 9.71 | 10.6 / 10.6 | 4.56 / 4.53 |
| 4-bit | 2.72 | 1.16 | 5.5 / 5.5 | 1.77 / 1.77 |
| batch 32 | 5.27 | 2.22 | 9.0 / 9.0 | 1.74 / 1.74 |

### Qwen 3.5 4B

| Setting | RTX 3090 s/step | A100 s/step | Peak GiB (3090 / A100) | Eval loss (3090 / A100) |
|---------|-----------------|-------------|------------------------|-------------------------|
| default | 3.51 | 1.47 | 22.2 / 22.2 | 1.73 / 1.73 |
| no checkpointing | out of memory | out of memory |  |  |
| batch 8 × 2 | 3.62 | 1.54 | 22.1 / 22.1 | 1.73 / 1.73 |
| batch 4 × 4 | 3.77 | 1.69 | 22.1 / 22.1 | 1.73 / 1.73 |
| padding-free | 3.48 | 1.47 | 22.2 / 22.2 | 1.73 / 1.73 |
| flex attention | unsupported | unsupported |  |  |
| flex + padding-free | unsupported | unsupported |  |  |
| packing | 3.48 | 1.46 | 22.2 / 22.2 | 1.73 / 1.73 |
| 4-bit | 3.59 | 1.59 | 16.8 / 16.8 | 1.75 / 1.75 |
| batch 32 | 6.81 | 2.89 | 22.3 / 22.3 | 1.73 / 1.73 |

### Gemma 4 E4B

| Setting | RTX 3090 s/step | A100 s/step | Peak GiB (3090 / A100) | Eval loss (3090 / A100) |
|---------|-----------------|-------------|------------------------|-------------------------|
| default | 3.83 | 1.55 | 19.3 / 19.3 | 1.96 / 1.96 |
| no checkpointing | out of memory | out of memory |  |  |
| batch 8 × 2 | 4.20 | 1.76 | 18.4 / 18.4 | 1.96 / 1.96 |
| batch 4 × 4 | 4.27 | 2.21 | 18.4 / 18.4 | 1.96 / 1.96 |
| padding-free | 8.19 | 3.99 | 19.2 / 19.2 | 1.96 / 1.96 |
| flex attention | 4.47 | 2.08 | 20.1 / 19.2 | 1.96 / 1.96 |
| flex + padding-free | out of memory | 2.15 |  / 18.8 |  / 1.96 |
| packing | 33.92 | 16.90 | 22.9 / 22.9 | 1.93 / 1.93 |
| 4-bit | 3.85 | 1.59 | 14.7 / 14.7 | 1.92 / 1.92 |
| batch 32 | out of memory | 3.02 |  / 23.8 |  / 1.94 |

What the three models show, next to the
[intent sweep](../intent/README.md#speed), where rows are short answers
to long, near-identical prompts:

- **The A100 is 2.3 to 2.5 times faster per step** at the default
  settings, at the same peak memory and eval loss.
- **Gradient accumulation costs time.** 8 × 2 adds 3 to 14% per step
  and 4 × 4 7 to 43%, most for Gemma, so use the largest per-device
  batch that fits.
- **Gradient checkpointing stays on**: without it batch 16 runs out of
  memory on both GPUs for all three models.
- **Flex attention** is 3 to 4% faster than SDPA for Granite at steady
  state on both GPUs, 17 to 34% slower for Gemma, and unsupported for
  Qwen 3.5.
- **Padding-free and packing** break Granite here too (eval loss 4.4
  to 4.6 instead of 1.75) and cost it 2.2 to 2.6 times per step
  (padding-free) and 6.6 to 8.4 times (packing). Qwen 3.5 keeps its
  loss and speed with either. Gemma keeps its loss but gets 2.1 to
  2.6 times slower with padding-free and 9 to 11 times slower with
  packing.
  Neither is faster than the default for any of the three on these
  372-token rows.
- **4-bit** trains within 8% of the bf16 step time with 24 to 38% less
  memory (Qwen 22.2 to 16.8 GiB, Gemma 19.3 to 14.7, Granite 8.9 to
  5.5).
- **Batch 32** doubles the step time: the same samples per second as
  batch 16 within 4% on both GPUs; Gemma's batch 32 (23.8 GiB) does not
  fit the 3090.

## Outputs

`--output_dir` receives `adapter_model.safetensors`, `adapter_config.json`
(records the base model), tokenizer files, `speftr.json` (the config),
`training_args.json`, `train_metrics.json` (runtime, throughput, peak
GPU memory, losses) and a `checkpoint-*` directory. To load, merge or
serve the adapter, see
[After training](../../docs/guide.md#5-after-training).

## Use the trained model without speftr

`tldr_inference.py` summarizes posts with only torch, transformers,
peft and (for `--engine vllm`) vllm, no speftr or Unsloth. It runs two
routes and prints both outputs per post:

- **adapter**: the base model plus the LoRA adapter, attached with peft
  or passed to vLLM as a `LoRARequest` (`run_adapter_route`);
- **merged**: the adapter merged into the 16-bit base and saved to
  `--merged_dir` (`merge_adapter`), then loaded from there
  (`run_merged_route`).

```bash
uv run python -m examples.tldr.tldr_inference
uv run python -m examples.tldr.tldr_inference --engine vllm \
    --prompt "SUBREDDIT: r/..."
```

Output: to be added.

General rules (base class, 16-bit base, vLLM flags, serving): [After
training](../../docs/guide.md#5-after-training).

## Adapt it to your data

1. **Data.** Replace `load_tldr` in `tldr_eval.py` with a loader that
   returns a `DatasetDict` with `train`, `validation` and `test` splits
   and two text columns, the document and its reference summary. Keep
   the column names away from `prompt`/`completion` and `text` (TRL
   trains those as they are and skips the formatting function).
2. **Prompt.** Edit `INSTRUCTION` and `build_messages` (drop the
   `PROMPT_SUFFIX` handling if your documents carry no cue). Keep the
   longest document plus summary under `MAX_SEQ_LENGTH` (1024 here);
   raise it for longer documents, with a smaller batch if memory runs
   out.
3. **Scoring.** `compute_metrics` reports ROUGE-1/2/L, the mean length
   in words and the invalid rate; `is_valid_summary` rejects empty
   replies and think tags. Add your own checks there (a length limit,
   a required format).
4. **Model.** Pass `--model_name_or_path`. The markers are inferred
   from the model's chat template; for a family outside the
   [marker table](../../docs/guide.md#chat-templates-and-markers) pass
   `--instruction_part` and `--response_part` (the script prints one
   formatted row to check them against). Thinking models: generation
   passes `enable_thinking=False`; check that your model's template
   honours it (print the generation prompt).
5. **Rank.** Rerun the [`speftr.lora_budget` command](#results) with
   your model, dataset and columns
   ([guide](../../docs/guide.md#7-checking-the-rank)).

## Pitfalls

- A `prompt`/`completion` or `text` column in the dataset makes TRL
  skip the formatting function and train the raw columns; rename them
  (`load_tldr` does).
- Generating without `enable_thinking=False` on Qwen 3.5 ends the
  prompt with an open `<think>` tag, so the model reasons first and the
  reply is scored invalid.
- The fine-tuned Qwen 3.5 adapter emits `<|im_end|>` after the TL;DR
  but not `<|endoftext|>`, which is the only end-of-sequence token in
  its model config (it has no `generation_config.json`). With
  transformers' default stop tokens it continues into a new turn
  (`assistant`, `<think>`, the TL;DR again) and every reply is
  invalid; `tldr_eval` and `tldr_inference` therefore stop at the
  tokenizer's EOS too (`stop_token_ids`).
