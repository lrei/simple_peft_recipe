# Intent classification: Banking77 with Granite 3.3 2B (SFT)

A support desk routes each customer message to the team that handles it.
This example fine-tunes a small chat model to read a message and answer
with one of the 77 intents of Banking77 (`card_arrival`,
`lost_or_stolen_card`, `cash_withdrawal_charge`, ...). The prompt lists
every intent name, so the same prompt scores the base model and the
fine-tuned one, and adding an intent needs no new output head.

It shows a standard SFT setup: a labelled dataset, a chat prompt
built from each row, and a loss on the answer only. The training script
takes any chat model: the model keeps its own chat template and `PESFT`
infers the response-only markers from it. Granite 3.3 2B is the
default; Qwen 3.5 4B and Gemma 4 E4B are measured too
([Results](#results), [Speed](#speed)).

## Files and speftr APIs

| File | What it does | speftr API |
|------|--------------|------------|
| `intent_eval.py` | Defines the task (`load_banking77`, `build_messages`, `extract_intent`) and scores a model on the test split | none (Unsloth `FastLanguageModel` for generation) |
| `intent_train.py` | LoRA SFT | `build_parser` is `PESFTConfig.get_argument_parser()` with the example's defaults (`EXAMPLE_DEFAULTS`); `main` calls `PESFT(config)`, `load_model()`, `train(train, eval, format_batch)`, `save_model()` |
| `intent_inference.py` | Classifies messages with the adapter and the merged model, without speftr ([below](#use-the-trained-model-without-speftr)) | none (transformers + peft, or vLLM) |

`intent_train.build_formatting_func` returns the batched formatting
function that `PESFT.train` expects (see the
[SFT data contract](../../docs/guide.md#sft-data-contract)).

## Data

[`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77),
CC-BY-4.0, not gated: 10,003 train and 3,080 test English messages, columns
`text` and `label` (integer index into the 77 intent names).
`legacy-datasets/banking77` is the Parquet copy of `PolyAI/banking77`; the
original repo only has a loading script, which `datasets` does not run.

- `load_banking77` renames `text` to `message`: a dataset with a `text`
  column is trained as is and the formatting function is skipped.
- `intent_train` holds out `--eval_rows` (500) train rows for the eval loss;
  `intent_eval` scores the full test split.
- Each row becomes one user turn (instruction, intent list, message) and
  one assistant turn (the intent name). Granite's template adds its default
  system prompt, including today's date.

One training text, as printed by `intent_train` (intent list shortened):

```text
<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: September 26, 2026.
You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Classify the customer message into exactly one banking intent. Answer with the intent name only.

Intents: activate_my_card, age_limit, apple_pay_or_google_pay, ..., wrong_exchange_rate_for_cash_withdrawal

Message: Why have I all of a sudden been charged for my ATM withdrawal? I thought the withdrawals were free because this is the first time I've ever had this problem.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>cash_withdrawal_charge<|end_of_text|>
```

With `train_on_responses=True`, `instruction_part`
`<|start_of_role|>user<|end_of_role|>` and `response_part`
`<|start_of_role|>assistant<|end_of_role|>`, only
`cash_withdrawal_charge<|end_of_text|>` is trained.

## Model

Default:
[`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct),
Apache-2.0: a small instruction-tuned model with a permissive licence,
trained in bf16 with LoRA in about 11 GB.

Any chat model whose template `speftr.chat_markers` recognises works
through `--model_name_or_path`: the model keeps its own chat template
(`chat_template=None`) and the user/assistant markers for the
response-only loss are inferred from it (`instruction_part` and
`response_part` left empty). The runs below also use
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) and
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
(both Apache-2.0). Qwen 3.5's template puts an empty think block at the
start of every assistant turn; it is trained as part of the answer, and
`intent_eval` generates with thinking off so the prompt ends the same
way.

## Run

From the repository root (Unsloth needs an NVIDIA GPU even to import):

```bash
# 1. Base model score
uv run python -m examples.intent.intent_eval \
  --model_path ibm-granite/granite-3.3-2b-instruct
# 2. Train (the measured run below used 300 steps, about half an epoch)
uv run python -m examples.intent.intent_train --max_steps 300
# 3. Score the adapters (default --model_path)
uv run python -m examples.intent.intent_eval
# Another model: same steps, its own output directory
uv run python -m examples.intent.intent_train --max_steps 300 \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --output_dir ./models/qwen3.5-4b-lora-banking77
uv run python -m examples.intent.intent_eval \
  --model_path ./models/qwen3.5-4b-lora-banking77
```

`intent_train` accepts every `PESFTConfig` flag (`--help` lists them
with the example's defaults in the epilog). Main flags:

| Script | Flag | Default |
|--------|------|---------|
| `intent_train` | `--model_name_or_path` | `ibm-granite/granite-3.3-2b-instruct` |
| | `--max_steps` / `--num_train_epochs` | -1 (use epochs) / 1 |
| | `--per_device_train_batch_size`, `--gradient_accumulation_steps` | 16, 1 |
| | `--lora_r`, `--learning_rate` | 1, 2e-4 |
| | `--load_in_4bit` | off (bf16) |
| | `--use_gradient_checkpointing`, `--attn_implementation`, `--padding_free`, `--packing` | `unsloth`, `sdpa`, off, off ([Speed](#speed)) |
| | `--eval_rows` | 500 |
| | `--output_dir` | `./models/granite-3.3-2b-lora-banking77` |
| `intent_eval` | `--model_path` | `./models/granite-3.3-2b-lora-banking77` |
| | `--max_samples` | -1 (all 3,080) |
| | `--batch_size`, `--max_new_tokens` | 32, 16 |
| | `--load_in_4bit` | off (bf16) |

## Results

bf16, LoRA r=1 (the rank the budget check below gives), batch 16, 300
steps (4,800 of the 9,503 train rows), scored on the full test split
(3,080 messages). The same seed and settings were run on one RTX 3090
(24 GB) and on one A100 40GB:

| Model | GPU | Base: accuracy / macro-F1 / invalid | Fine-tuned: accuracy / macro-F1 / invalid | Final eval loss | 300 steps |
|-------|-----|-------------------------------------|-------------------------------------------|-----------------|-----------|
| Granite 3.3 2B | RTX 3090 | 0.436 / 0.430 / 16.5% | 0.820 / 0.819 / 1.5% | 0.110 | 18.7 min (3.49 s/step) |
| Granite 3.3 2B | A100 | 0.437 / 0.430 / 16.6% | 0.815 / 0.817 / 2.3% | 0.100 | 8.7 min (1.60 s/step) |
| Qwen 3.5 4B | RTX 3090 | 0.661 / 0.648 / 1.6% | 0.836 / 0.834 / 0.4% | 0.076 | 18.8 min (3.55 s/step) |
| Qwen 3.5 4B | A100 | 0.662 / 0.649 / 1.6% | 0.849 / 0.843 / 0.0% | 0.071 | 9.1 min (1.52 s/step) |
| Gemma 4 E4B | RTX 3090 | 0.671 / 0.661 / 0.3% | 0.855 / 0.848 / 0.0% | 0.073 | 29.6 min (5.97 s/step) |
| Gemma 4 E4B | A100 | 0.672 / 0.662 / 0.4% | 0.858 / 0.850 / 0.0% | 0.078 | 14.2 min (2.69 s/step) |

- The A100 runs each model 2.2 to 2.3 times faster per step; the
  s/step figures are steady-state (median after the first steps).
  Scores differ by up to 1.3 points between the two GPUs for the same
  seed, so differences of that size between models are noise.
- Peak training memory (`train_metrics.json`): Granite 9.0 GiB,
  Qwen 20.4 GiB, Gemma 19.3 GiB. Gemma's evaluation during and after
  training runs at `--per_device_eval_batch_size 4` on both GPUs: its
  262k-token vocabulary makes the eval logits of a batch of 16 exceed
  the 3090's memory.
- Evaluation (`intent_eval`, batch 32, bf16): 5 to 6 min on the 3090,
  3 to 4.5 min on the A100, for any of the three.
- Rank check ([`speftr.lora_budget`](../../docs/guide.md#7-checking-the-rank)):
  the intent-name response tokens need about 42k parameters with
  Granite's tokenizer (83,261 tokens), 28k with Qwen's (55,870) and
  18k with Gemma's (36,864); rank 1 has 1.76M (Granite), 1.33M (Qwen)
  and 2.18M (Gemma) parameters on every attention and MLP projection,
  so it is the smallest rank and covers the data many times over. The
  command below counts Banking77's integer label ids instead of the
  names (~18k parameters), which leads to the same rank:

  ```bash
  uv run python -m speftr.lora_budget \
      --model_name_or_path ibm-granite/granite-3.3-2b-instruct \
      --dataset legacy-datasets/banking77 --split "train[:-500]" \
      --prompt_column text --response_column label --responses_only
  ```

"Invalid" means the first line of the reply is not an intent name; it
counts as an error in accuracy and macro-F1.

### 4-bit base weights

The same 300-step runs with `--load_in_4bit` (QLoRA: NF4 base weights,
the adapter in bf16), evaluated on the bf16 base:

| Model | GPU | Fine-tuned: accuracy / macro-F1 / invalid | Final eval loss | 300 steps | Peak GiB |
|-------|-----|-------------------------------------------|-----------------|-----------|----------|
| Granite 3.3 2B, 4-bit | RTX 3090 | 0.834 / 0.825 / 0.4% | 0.100 | 18.8 min (3.52 s/step) | 5.7 |
| Granite 3.3 2B, 4-bit | A100 | 0.799 / 0.786 / 0.3% | 0.122 | 8.6 min (1.62 s/step) | 5.7 |
| Qwen 3.5 4B, 4-bit | RTX 3090 | 0.855 / 0.854 / 0.0% | 0.072 | 19.2 min (3.60 s/step) | 15.1 |
| Qwen 3.5 4B, 4-bit | A100 | 0.862 / 0.859 / 0.0% | 0.065 | 15.3 min (1.65 s/step) | 15.1 |
| Gemma 4 E4B, 4-bit | RTX 3090 | 0.849 / 0.843 / 0.0% | 0.079 | 29.6 min (5.95 s/step) | 14.7 |
| Gemma 4 E4B, 4-bit | A100 | 0.853 / 0.840 / 0.1% | 0.071 | 14.1 min (2.71 s/step) | 14.7 |

4-bit scores land within the bf16 runs' spread (0.799 to 0.862 against
0.815 to 0.858 accuracy; the largest gap, Granite on the A100, is 1.6
points below its bf16 run and 1.4 above it on the 3090) at the same
steady step time and 23 to 38% less memory. The longer A100 wall times
for Qwen and Gemma are warm-up: the three 4-bit runs loaded and
compiled at the same time on shared nodes.

### Full fine-tuning, LoRA and QLoRA on one A100

Qwen 3.5 4B, the same 300 steps at batch 16 and the same seed, run one
after another on one A100 40GB. Full fine-tuning (`--full_finetuning`)
trains every weight in bf16 with the recipe's 8-bit AdamW at a ten times
lower learning rate (2e-5, the "LoRA Without Regret" ratio); the LoRA
runs use rank 1 at 2e-4:

| Method | Trainable parameters | Steady s/step | 300 steps | Peak GiB | Final eval loss | Accuracy / macro-F1 / invalid |
|--------|----------------------|---------------|-----------|----------|-----------------|-------------------------------|
| Full fine-tuning, batch 16 | 4.54B (100%) | 1.93 | 11.3 min | 29.4 | 0.068 | 0.856 / 0.850 / 0.06% |
| Full fine-tuning, batch 8 × 2 | 4.54B (100%) | 2.05 | 11.1 min | 30.4 | 0.077 | 0.846 / 0.845 / 0.19% |
| LoRA, rank 1 | 1.33M (0.03%) | 1.52 | 8.3 min | 20.4 | 0.074 | 0.833 / 0.828 / 0.29% |
| QLoRA, rank 1 (4-bit base) | 1.33M (0.03%) | 1.66 | 8.7 min | 15.1 | 0.095 | 0.868 / 0.863 / 0.16% |

- Scores span 0.833 to 0.868, and repeats of the same configuration on
  the same GPU span about as much: this LoRA run scored 0.833 against
  0.849 in the [table above](#results), this QLoRA run 0.868 against
  0.862. Nothing separates the four methods on this task at 300 steps.
- What differs is cost: full fine-tuning takes 27% longer per step than
  LoRA and 29 to 30 GiB against 20 (LoRA) and 15 (QLoRA), and writes a
  9 GB model instead of a 5 MB adapter. The 24 GB card cannot run the
  full fine-tuning rows at all.

## Speed

`examples/speed.py` runs `intent_train` once per setting of the
recipe's speed knobs for a fixed number of steps and tables seconds per
step, samples per second and peak memory from each run's
`train_metrics.json`:

```bash
uv run python -m examples.speed examples.intent.intent_train \
    --steps 40 --model_name_or_path Qwen/Qwen3.5-4B \
    --output_dir ./models/speed-intent-qwen
```

Every setting keeps the effective batch at 16 except `batch_32`, so
the final eval loss after 40 steps says whether a setting still trains
the same thing: one whose eval loss is far from the default's has
changed what the model learns, not just how fast. The step time is the
steady-state median after the first 5 steps; the first steps compile
kernels, which took 14 to 80 s in these runs (compile cache warm) and
up to 2 minutes (12 minutes for Gemma with flex attention and
padding-free) the first time each setting ran on the A100 with a cold
cache. The cost is the same on either GPU.

Measured with 40 steps on one RTX 3090 (24 GB) and one A100 40GB
(PCIe), bf16 unless the setting says otherwise, eval batch 4 for Gemma:

### Granite 3.3 2B

| Setting | RTX 3090 s/step | A100 s/step | Peak GiB (3090 / A100) | Eval loss (3090 / A100) |
|---------|-----------------|-------------|------------------------|-------------------------|
| default | 3.64 | 1.60 | 9.0 / 9.0 | 0.20 / 0.20 |
| no checkpointing | out of memory | out of memory |  |  |
| batch 8 × 2 | 3.62 | 1.66 | 9.0 / 9.0 | 0.21 / 0.21 |
| batch 4 × 4 | 4.05 | 1.72 | 8.9 / 8.9 | 0.21 / 0.20 |
| padding-free | 9.10 | 4.72 | 8.7 / 8.7 | 2.08 / 1.92 |
| flex attention | 3.41 | 1.48 | 9.0 / 9.0 | 0.19 / 0.20 |
| flex + padding-free | 3.42 | 1.48 | 8.7 / 8.7 | 2.01 / 2.06 |
| packing | 8.94 | 4.68 | 8.7 / 8.7 | 1.94 / 1.97 |
| 4-bit | 3.67 | 1.62 | 5.6 / 5.6 | 0.22 / 0.22 |
| batch 32 | 6.92 | 3.09 | 9.1 / 9.1 | 0.15 / 0.16 |

### Qwen 3.5 4B

| Setting | RTX 3090 s/step | A100 s/step | Peak GiB (3090 / A100) | Eval loss (3090 / A100) |
|---------|-----------------|-------------|------------------------|-------------------------|
| default | 3.55 | 1.51 | 20.4 / 20.4 | 0.11 / 0.11 |
| no checkpointing | out of memory | out of memory |  |  |
| batch 8 × 2 | 3.64 | 1.56 | 20.4 / 20.4 | 0.12 / 0.11 |
| batch 4 × 4 | 3.93 | 1.71 | 20.4 / 20.4 | 0.12 / 0.12 |
| padding-free | 3.55 | 1.51 | 20.4 / 20.4 | 0.11 / 0.11 |
| flex attention | unsupported | unsupported |  |  |
| flex + padding-free | unsupported | unsupported |  |  |
| packing | 3.54 | 1.51 | 20.4 / 20.4 | 0.11 / 0.11 |
| 4-bit | 3.62 | 1.63 | 15.1 / 15.1 | 0.11 / 0.13 |
| batch 32 | 6.97 | 3.01 | 20.5 / 20.5 | 0.09 / 0.09 |

### Gemma 4 E4B

| Setting | RTX 3090 s/step | A100 s/step | Peak GiB (3090 / A100) | Eval loss (3090 / A100) |
|---------|-----------------|-------------|------------------------|-------------------------|
| default | 5.97 | 2.69 | 19.3 / 19.3 | 0.13 / 0.14 |
| no checkpointing | out of memory | out of memory |  |  |
| batch 8 × 2 | 6.12 | 2.88 | 18.8 / 18.8 | 0.14 / 0.16 |
| batch 4 × 4 | 6.42 | 3.03 | 18.8 / 18.8 | 0.13 / 0.15 |
| padding-free | 12.92 | 6.38 | 19.4 / 19.4 | 0.14 / 0.15 |
| flex attention | 5.84 | 2.97 | 20.1 / 19.2 | 0.13 / 0.14 |
| flex + padding-free | out of memory | 3.09 |  / 19.0 |  / 0.12 |
| packing | 12.91 | 6.37 | 19.4 / 19.4 | 0.12 / 0.13 |
| 4-bit | 5.97 | 2.71 | 14.7 / 14.8 | 0.14 / 0.12 |
| batch 32 | out of memory | 5.24 |  / 23.9 |  / 0.12 |

What the three models show, on both GPUs:

- **The A100 is 2.2 to 2.4 times faster per step** at the default
  settings, and the same settings fit: peak memory is identical on both
  cards, so the 24 GB card is not what sets the batch here.
- **Gradient accumulation costs time.** 8 × 2 adds up to 7% per step
  and 4 × 4 8 to 13% over 16 × 1, so use the largest per-device batch
  that fits and accumulate only for the rest.
- **Gradient checkpointing stays on.** Without it, batch 16 of these
  600-token prompts runs out of memory even on 40 GB, for the 2B model
  too.
- **Flex attention** is 6 to 8% faster per step than SDPA for Granite
  on both GPUs at steady state, but it recompiles its block masks for
  new sequence lengths, which eats most of that over a 40-step run
  (all steps averaged: 3.68 vs 3.77 s on the 3090, 1.81 vs 1.77 on
  the A100). It is no faster for Gemma (2% faster on the 3090, 10%
  slower on the A100) and unsupported for Qwen 3.5.
- **Padding-free and packing** train something else for Granite: the
  eval loss is 1.9 to 2.1 instead of 0.2 and each step takes 2.5 to 3
  times longer. Qwen 3.5 keeps its loss and speed with either; Gemma
  keeps its loss but gets 2.2 to 2.4 times slower. Neither is faster
  than the default for any of the three.
- **4-bit** trains at the bf16 step time, within 8%, with 23 to 38% less
  memory (Granite 9.0 to 5.6 GiB, Qwen 20.4 to 15.1, Gemma 19.3 to
  14.8), and its 300-step scores match bf16
  ([above](#4-bit-base-weights)).
- **Batch 32** doubles the step time: 4.4 to 4.6 samples/s either way
  on the 3090, 10.0 to 10.6 on the A100 for Granite and Qwen, 5.9 to
  6.1 for Gemma, whose batch 32 (23.9 GiB) no longer fits the 3090.
  Its lower eval loss after 40 steps comes from seeing twice the rows.

## Outputs

`--output_dir` receives `adapter_model.safetensors`, `adapter_config.json`
(records the base model), tokenizer files, `speftr.json` (the config),
`training_args.json`, `train_metrics.json` (runtime, throughput, peak
GPU memory, losses) and a `checkpoint-*` directory. To load, merge or
serve the adapter, see
[After training](../../docs/guide.md#5-after-training).

## Use the trained model without speftr

`intent_inference.py` classifies customer messages with only torch,
transformers, peft and (for `--engine vllm`) vllm, no speftr or Unsloth.
It runs two routes and prints both outputs per prompt:

- **adapter**: the base model plus the LoRA adapter, attached with peft
  or passed to vLLM as a `LoRARequest` (`run_adapter_route`);
- **merged**: the adapter merged into the 16-bit base and saved to
  `--merged_dir` (`merge_adapter`), then loaded from there
  (`run_merged_route`).

```bash
uv run python -m examples.intent.intent_inference
uv run python -m examples.intent.intent_inference --engine vllm \
    --prompt "I still have not received my new card."
```

Output of the first command (RTX 3090):

```text
Engine: transformers. Merged model saved to ./models/granite-3.3-2b-lora-banking77-merged

>>> Why won't my card show up on the app?
adapter: card_linking
merged:  card_linking

>>> May I exchange currencies with this?
adapter: exchange_via_app
merged:  exchange_via_app
```

`./models/granite-3.3-2b-lora-banking77-merged` (default
`<adapter_dir>-merged`) holds `model.safetensors`, `config.json`,
`generation_config.json`, the tokenizer and `chat_template.jinja`. It
loads without peft, like any Hub checkpoint (`AutoModelForCausalLM`,
`pipeline`, `vllm serve ./models/granite-3.3-2b-lora-banking77-merged`).

Both routes agreed with both engines (the vLLM command printed
`card_arrival` twice). The transformers run takes about 30 s and 5.4 GB
of VRAM; vLLM reserves 80% of the GPU whatever the model size and starts
one engine per route (about 3 min per run in total). The script holds
the 77 intent names (`INTENTS`) because the prompt lists them all,
exactly as `intent_eval.build_messages` does.

### Export to GGUF

For llama.cpp or Ollama, export the merged model with
[`examples/export_gguf.sh`](../export_gguf.sh)
([guide](../../docs/guide.md#export-to-gguf-llamacpp-ollama)):

```bash
uv run --with gguf examples/export_gguf.sh \
    ./models/granite-3.3-2b-lora-banking77-merged \
    ./gguf ./llama.cpp Q4_K_M
llama.cpp/build/bin/llama-server --jinja -ngl 99 -c 4096 --port 8080 \
    -m ./gguf/granite-3.3-2b-lora-banking77-merged-Q4_K_M.gguf
```

Conversion takes about 25 s, `Q4_K_M` quantization about 20 s, `Q8_0`
about 8 s. The GGUF's chat template is identical to the adapter's
`chat_template.jinja`, and `llama-server` renders and tokenizes the
prompt exactly as transformers does.

300 shuffled test messages (`shuffle(seed=0)`), the `intent_inference`
prompt and `intent_eval` answer matching, greedy, one request at a time,
RTX 3090 (llama.cpp built with CUDA). "Same" counts predictions equal
to the merged transformers model's:

| Format | File | Accuracy | Same | Messages/s | Decode tokens/s |
|--------|------|----------|------|------------|-----------------|
| Merged, transformers bf16 | 5.1 GB | 0.803 | reference | 4.2 | |
| GGUF bf16 | 5.1 GB | 0.803 | 300/300 | 13.4 | 121 |
| GGUF `Q8_0` | 2.7 GB | 0.800 | 298/300 | 16.8 | 166 |
| GGUF `Q4_K_M` | 1.5 GB | 0.793 | 289/300 | 19.4 | 209 |
| Ollama, GGUF `Q4_K_M` with the `TEMPLATE` below | 1.5 GB | 0.793 | 280/300 | 7.6 | |
| Ollama, GGUF `Q4_K_M`, `FROM` only | 1.5 GB | 0.580 | 202/300 | 5.8 | |

- On a 10-core CPU (`-ngl 0`, i9-7900X): bf16 0.800 at 1.0 messages/s
  (8 decode tokens/s), `Q8_0` 0.803 at 1.3 (14), `Q4_K_M` 0.797 at 1.9
  (22).
- `llama-server` reuses the cached prompt prefix shared by every request
  (instruction and intent list); the transformers loop encodes each
  prompt in full.
- Ollama 0.12.10 imports this GGUF with `TEMPLATE {{ .Prompt }}`: the
  message reaches the model without Granite's role markers, the answer
  does not stop, and 74 of 300 answers are invalid. This `Modelfile`
  restores the training format (Ollama's `currentDate` prints
  `2026-09-29` where the Jinja template prints `September 29, 2026`):

```text
FROM ./granite-3.3-2b-lora-banking77-merged-Q4_K_M.gguf
TEMPLATE """<|start_of_role|>system<|end_of_role|>{{ if .System }}{{ .System }}{{ else }}Knowledge Cutoff Date: April 2024.
Today's Date: {{ currentDate }}.
You are Granite, developed by IBM. You are a helpful AI assistant.{{ end }}<|end_of_text|>
{{ range .Messages }}{{ if ne .Role "system" }}<|start_of_role|>{{ .Role }}<|end_of_role|>{{ .Content }}<|end_of_text|>
{{ end }}{{ end }}<|start_of_role|>assistant<|end_of_role|>"""
PARAMETER stop <|end_of_text|>
```

General rules (base class, 16-bit base, vLLM flags, serving): [After
training](../../docs/guide.md#5-after-training).

## Adapt it to your data

1. **Labels and data.** Replace `load_banking77` in `intent_eval.py` with a
   loader that returns a `DatasetDict` with `train` and `test` splits
   (columns `message` and integer `label`) and the label names in index
   order, e.g. `load_dataset("csv", data_files={...})` plus
   `ClassLabel`. Do not keep a column named `text`.
2. **Prompt.** Edit `INSTRUCTION` and `build_messages`. Every label name
   is in every prompt, so prompt length grows with the label set: keep
   the longest prompt under `MAX_SEQ_LENGTH` (1024 here; the longest
   Banking77 prompt is about 600 tokens).
3. **Answer parsing.** `extract_intent` accepts the first line,
   case-insensitive, exact match. Loosen it only if your labels need it.
4. **Model.** Pass `--model_name_or_path`. The markers are inferred
   from the model's chat template; for a family outside the
   [marker table](../../docs/guide.md#chat-templates-and-markers) pass
   `--instruction_part` and `--response_part` (the script prints one
   formatted row to check them against).
5. **Rank.** Rerun the [`speftr.lora_budget` command](#results) with
   your model, dataset and columns
   ([guide](../../docs/guide.md#7-checking-the-rank)).

## Pitfalls

- A `text` column makes the trainer ignore the formatting function.
- `instruction_part` / `response_part` that do not match the rendered
  template train nothing or everything.
- The formatting function must accept a single row as well as a batch:
  Unsloth probes it with one row (`format_batch` handles both).
- `intent_eval` greedy-decodes 16 tokens: long label names need a larger
  `--max_new_tokens`.
