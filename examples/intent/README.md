# Intent classification: Banking77 with Granite 3.3 2B (SFT)

A support desk routes each customer message to the team that handles it.
This example fine-tunes a small chat model to read a message and answer
with one of the 77 intents of Banking77 (`card_arrival`,
`lost_or_stolen_card`, `cash_withdrawal_charge`, ...). The prompt lists
every intent name, so the same prompt scores the base model and the
fine-tuned one, and adding an intent needs no new output head.

It shows a standard SFT setup: a labelled dataset, a chat prompt
built from each row, and a loss on the answer only.

## Files and speftr APIs

| File | What it does | speftr API |
|------|--------------|------------|
| `intent_eval.py` | Defines the task (`load_banking77`, `build_messages`, `extract_intent`) and scores a model on the test split | none (Unsloth `FastLanguageModel` for generation) |
| `intent_train.py` | LoRA SFT | `build_config` fills a `PESFTConfig`; `main` calls `PESFT(config)`, `load_model()`, `train(train, eval, format_batch)`, `save_model()` |
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

[`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct),
Apache-2.0: a small instruction-tuned model with a permissive licence,
trained in bf16 with LoRA in about 11 GB. It keeps its own chat template
(`chat_template=None`).

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
```

Main flags (`--help` lists all):

| Script | Flag | Default |
|--------|------|---------|
| `intent_train` | `--model_name_or_path` | `ibm-granite/granite-3.3-2b-instruct` |
| | `--max_steps` / `--num_epochs` | -1 (use epochs) / 1 |
| | `--per_device_batch_size` | 16 |
| | `--lora_r`, `--learning_rate` | 8, 2e-4 |
| | `--load_in_4bit` | off (bf16) |
| | `--eval_rows` | 500 |
| | `--output_dir` | `./models/granite-3.3-2b-lora-banking77` |
| `intent_eval` | `--model_path` | `./models/granite-3.3-2b-lora-banking77` |
| | `--max_samples` | -1 (all 3,080) |
| | `--batch_size`, `--max_new_tokens` | 32, 16 |
| | `--load_in_4bit` | off (bf16) |

## Results

One RTX 3090, bf16, LoRA r=8, batch 16, 300 steps, test split (3,080
messages):

| Model | Accuracy | Macro-F1 | Invalid answers |
|-------|----------|----------|-----------------|
| Base | 0.436 | 0.428 | 16.2% |
| Fine-tuned | 0.827 | 0.817 | 0.8% |

- Training: 3.8 s/step, 19 min for 300 steps, 11.1 GB peak (nvidia-smi);
  final eval loss 0.10.
- Evaluation: about 5 min at batch 32; 8.1 GB peak (torch) with adapters,
  10.5 GB for the base model.
- Rank check ([`speftr.lora_budget`](../../docs/guide.md#7-checking-the-rank)):
  the 83,261 intent-name response tokens need ~42k parameters; rank 8 has
  14.1M, and rank 1 would suffice. The command below counts Banking77's
  integer label ids instead of the names (~18k parameters), which leads
  to the same rank:

  ```bash
  uv run python -m speftr.lora_budget \
      --model_name_or_path ibm-granite/granite-3.3-2b-instruct \
      --dataset legacy-datasets/banking77 --split "train[:-500]" \
      --prompt_column text --response_column label --responses_only
  ```

"Invalid" means the first line of the reply is not an intent name; it
counts as an error in accuracy and macro-F1.

## Outputs

`--output_dir` receives `adapter_model.safetensors`, `adapter_config.json`
(records the base model), tokenizer files, `speftr.json` (the config),
`training_args.json` and a `checkpoint-*` directory. To load, merge or
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
4. **Model.** Pass `--model_name_or_path` and set `instruction_part` and
   `response_part` in `build_config` to the new template's role headers
   ([marker table](../../docs/guide.md#chat-templates-and-markers)).
   Print one formatted row (the script does) and check the markers
   appear verbatim.
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
