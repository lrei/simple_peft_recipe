# Instruction tuning: Dolly pirate with Qwen3 0.6B (SFT)

The smallest end-to-end chat SFT: take instruction/response pairs, render
them as system + user + assistant conversations, train on the assistant
turns only, then chat with the result. The dataset answers every Dolly
instruction in pirate speech, so the effect of training is easy to see.

## Files and speftr APIs

| File | What it does | speftr API |
|------|--------------|------------|
| `instruct/instruct.py` | LoRA SFT on Dolly pirate | `_build_config` fills a `PESFTConfig`; `main` calls `PESFT(config)`, `load_model()`, `train(train, eval, format_batch)`, `save_model()` |
| `chat.py` (in `examples/`) | Interactive chat with the trained model | none (Unsloth `FastLanguageModel`) |
| `instruct/instruct_inference.py` | Answers messages with the adapter and the merged model, without speftr ([below](#use-the-trained-model-without-speftr)) | none (transformers + peft, or vLLM) |

`_build_format_batch` returns the batched formatting function that
`PESFT.train` expects; `_build_conversations_from_columns` turns the
columns into messages. See the
[SFT data contract](../../docs/guide.md#sft-data-contract).

## Data

[`TeeZee/dolly-15k-pirate-speech`](https://huggingface.co/datasets/TeeZee/dolly-15k-pirate-speech),
CC-BY-SA-3.0, not gated: 15,011 rows of Databricks Dolly 15k whose
responses were rewritten in pirate speech (instructions are unchanged).
Columns `instruction`, `context` (often empty), `response`, `category`.
`--eval_ratio` (0.05) of the rows is held out for the eval loss.

Each row becomes a conversation: the system prompt (`--system_prompt`), a
user turn (the instruction, plus `Context:` and the context when there is
one) and an assistant turn (the response). A dataset with a `messages`
column of role/content lists is used as is. One training text as
`format_batch` renders it (ChatML), and as `instruct.py` prints before
training:

```text
<|im_start|>system
You are a pirate, always respond in pirate speech.<|im_end|>
<|im_start|>user
Which is a species of fish? Tope or Rope<|im_end|>
<|im_start|>assistant
Tope<|im_end|>
```

With `train_on_responses=True`, `instruction_part` `<|im_start|>user\n`
and `response_part` `<|im_start|>assistant\n`, only `Tope<|im_end|>\n` is
trained.

## Model

`unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (Apache-2.0, not gated): Qwen3 0.6B
pre-quantized to 4-bit, so it downloads and trains quickly. Qwen models
are pretrained on ChatML tokens, so the `chatml` template needs no new
tokens. LoRA rank 8, alpha 32 on all attention and MLP projections.
Rank 8 has 5.05M adapter parameters; the ~1.3M response tokens of the
training split need ~0.65M, i.e. rank 2 at least
([`speftr.lora_budget`](../../docs/guide.md#7-checking-the-rank)):

```bash
uv run python -m speftr.lora_budget \
    --model_name_or_path unsloth/Qwen3-0.6B-unsloth-bnb-4bit \
    --dataset TeeZee/dolly-15k-pirate-speech --split "train[:95%]" \
    --prompt_column instruction context --response_column response \
    --responses_only
```

## Run

From the repo root, after `uv sync`:

```bash
uv run python -m examples.instruct.instruct --output_dir ./model-it \
  --load_in_4bit --num_epochs 1
uv run python -m examples.chat --model_name_or_path ./model-it
```

`chat.py` defaults to the same `chatml` template and pirate system prompt
as training. The README shows a reply from such a model:

```text
You: Who was George Washington?
Assistant: George washington was an american president of th' united states from 1789 until 1797. He was a key figure in th' founding of th' united states and played a critical role in th' development of th' new republic.
```

Both scripts take `--help`. Main flags:

| Script | Flag | Default |
|--------|------|---------|
| `instruct` | `--model_name_or_path` | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` |
| | `--load_in_4bit` | off; pass it for `-bnb-4bit` repos |
| | `--chat_template` | `chatml` |
| | `--system_prompt` | `You are a pirate, always respond in pirate speech.` |
| | `--num_epochs` | 2 |
| | `--per_device_batch_size` / `--gradient_accumulation_steps` | 8 / 2 |
| | `--learning_rate` / `--scheduler` | 2e-4 / `constant` |
| | `--lora_r` / `--lora_alpha` | 8 / 32 |
| | `--max_seq_length` | 2048 |
| | `--save_method` | `lora` (or `merged_16bit`) |
| | `--eval_ratio` / `--random_state` | 0.05 / 0 |
| | `--packing` | off |
| | `--output_dir` | `./models/speftr-instruction` |
| `chat` | `--model_name_or_path` | required |
| | `--chat_template` / `--system_prompt` | `chatml` / pirate prompt |
| | `--temperature` / `--top_p` | 0.7 / 0.9 (`0` = greedy) |
| | `--max_new_tokens` | 512 |
| | `--load_in_4bit` | off |

## Hardware

Runs on one RTX 3090 (24 GB); Qwen3 0.6B 4-bit is a
[validated model](../../docs/guide.md#validated-models). No timing has
been recorded for this example. One epoch at the defaults is about 890
optimizer steps (14,260 training rows, effective batch 16).

## Outputs

`--output_dir` holds the LoRA adapter (`adapter_model.safetensors`,
`adapter_config.json`) and tokenizer files, or a full merged model with
`--save_method merged_16bit`; plus `speftr.json` (the `PESFTConfig`),
`training_args.json` and `checkpoint-*` per epoch (the best by `eval_loss`
is reloaded before saving). Loading and serving:
[guide, section 5](../../docs/guide.md#5-after-training).

## Use the trained model without speftr

`instruct/instruct_inference.py` answers messages in the training format
(pirate system prompt) with only torch, transformers, peft and (for
`--engine vllm`) vllm, no speftr or Unsloth. It runs two routes and
prints both outputs per prompt:

- **adapter**: the base model plus the LoRA adapter, attached with peft
  or passed to vLLM as a `LoRARequest` (`run_adapter_route`);
- **merged**: the adapter merged into the 16-bit base and saved to
  `--merged_dir` (`merge_adapter`), then loaded from there
  (`run_merged_route`).

```bash
uv run python -m examples.instruct.instruct_inference \
    --base_model Qwen/Qwen3-0.6B
uv run python -m examples.instruct.instruct_inference \
    --base_model Qwen/Qwen3-0.6B --engine vllm \
    --prompt "What is the capital of France?"
```

Output of the first command (RTX 3090):

```text
Engine: transformers. Merged model saved to ./models/speftr-instruction-merged

>>> If I have more pieces at the time of stalemate, have I won?
adapter: No, ye have not won.
merged:  No, ye have not won.

>>> What is the capital of France?
adapter: Th' capital of france be paris.
merged:  Th' capital of france be paris.
```

`./models/speftr-instruction-merged` (default `<adapter_dir>-merged`)
holds `model.safetensors`, `config.json`, `generation_config.json`, the
tokenizer and `chat_template.jinja`. It loads without peft, like any Hub
checkpoint (`AutoModelForCausalLM`, `pipeline`, `vllm serve
./models/speftr-instruction-merged`).

Both routes agreed with both engines. The transformers run takes under
30 s and 2 GB of VRAM; vLLM reserves 80% of the GPU whatever the model
size and starts one engine per route (about 3 min per run in total).

- **16-bit base.** The example trains on the pre-quantized
  `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`, which `adapter_config.json`
  records. Merging and vLLM LoRA need the 16-bit original, so pass
  `--base_model Qwen/Qwen3-0.6B`; without it the script stops with
  `ValueError: ... is quantized; pass its 16-bit original as
  --base_model`.
- **Chat template.** Training used ChatML (`--chat_template chatml`),
  saved as `chat_template.jinja` next to the adapter. Both routes use
  that file, not Qwen3's own template; pass `--system_prompt` if you
  trained with another one.

General rules (base class, 16-bit base, vLLM flags, serving): [After
training](../../docs/guide.md#5-after-training).

## Adapt it to your data

1. **Data**: change `_load_datasets` to return your train and eval
   `datasets.Dataset`. If it has `instruction` / `context` / `response`
   columns, nothing else changes. If it has a `messages` column of
   role/content lists, it is used as is. Otherwise edit
   `_build_conversations_from_columns` to build the messages from your
   columns.
2. **System prompt**: `--system_prompt`, and pass the same one to
   `chat.py`. It replaces any system message already in the data.
3. **Model**: `--model_name_or_path`. For a model with its own template,
   set `--chat_template` (or `None` in code for the model's own) and change
   `instruction_part` / `response_part` in `_build_config` to match
   ([marker table](../../docs/guide.md#chat-templates-and-markers)); use
   the same `--chat_template` with `chat.py`.
4. **Rank**: rerun the [`speftr.lora_budget` command](#model) with your
   model, dataset and columns
   ([guide](../../docs/guide.md#7-checking-the-rank)).

## Pitfalls

- **Markers are ChatML.** `instruction_part` / `response_part` are fixed
  to ChatML in `_build_config`; with another `--chat_template` they must be
  changed too, or response-only loss trains nothing or everything. Check
  the preview row `instruct.py` prints before training.
- **ChatML on non-Qwen models** adds tokens the model never saw; they are
  not learned because embeddings and the output head are not trained (see
  the README caveat).
- **Chat must match training**: same `--chat_template` and
  `--system_prompt` in `chat.py`, or the prompt differs from the one the
  model was trained on.
- Both scripts import Unsloth, which needs a GPU even to import;
  `import unsloth` must come before `datasets`, `transformers` and `trl`.
