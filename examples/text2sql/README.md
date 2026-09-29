# Text-to-SQL: SmolLM3-3B, SFT then GRPO

An analyst asks a question about tables they describe with `CREATE TABLE`
statements; the model answers with one SQLite query. This example trains
that in two stages on a single small GPU:

1. **SFT** (`PESFT`, 4-bit QLoRA) on question/SQL pairs.
2. **GRPO** (`PERL`) continuing the *same* LoRA adapters, with a reward
   that runs each sampled query in SQLite.

It shows the SFT → RL pipeline with a verifiable, execution-based reward
that you can replace with your own checker.

## Files and speftr APIs

| File | What it does | speftr API |
|------|--------------|------------|
| `text2sql_eval.py` | Defines the task (`load_splits`, `build_messages`, `extract_sql`, `score_sql`) and scores a model | none (transformers + peft; no Unsloth, so the RL script can import it) |
| `text2sql_sft.py` | Stage 1, LoRA SFT | `build_config` fills a `PESFTConfig`; `main` calls `PESFT(config)`, `load_model()`, `train(train, eval, format_batch)`, `save_model()` |
| `text2sql_rl.py` | Stage 2, GRPO | `build_config` fills a `PERLConfig`; `main` loads the SFT adapters (`load_sft_adapters`), then `PERL.set_pretrained_model(model, tokenizer)`, `train(dataset, [sql_reward])`, `save_model()` |
| `text2sql_inference.py` | Writes SQL with the SFT or GRPO adapter and the merged model, without speftr ([below](#use-the-trained-model-without-speftr)) | none (transformers + peft, or vLLM) |

## Data

[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context),
CC-BY-4.0, not gated: 78,577 rows from WikiSQL and Spider, one `train`
split, columns `question`, `context` (`CREATE TABLE` statements) and
`answer` (gold SQL). The dataset has no table contents.

- `load_splits` holds out 500 rows with a fixed seed; every script sees the
  same split. `text2sql_eval` scores the 490 held-out rows whose gold
  query runs on its own schema (`gold_executes`); about 2.5% reference
  missing columns.
- SFT trains on the other 77,777 rows (after `--eval_rows` 300 for the
  eval loss). GRPO samples `--train_rows` (2,000) of them and keeps those
  whose gold query runs (1,964).
- The chat is one user turn (instruction, schema, question) and, for SFT,
  one assistant turn (the gold SQL). Thinking is off
  (`enable_thinking=False`), so SmolLM3's template adds a `/no_think`
  system prompt and an empty `<think>` block; both are part of the text.

One SFT training text, as printed by `text2sql_sft`:

```text
<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 26 September 2026
Reasoning Mode: /no_think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face.

<|im_start|>user
Write one SQLite query that answers the question. Reply with the SQL query only.

Schema:
CREATE TABLE table_1342426_18 (candidates VARCHAR, incumbent VARCHAR)

Question: How many candidates won the election of john n. sandlin?<|im_end|>
<|im_start|>assistant
<think>

</think>
SELECT COUNT(candidates) FROM table_1342426_18 WHERE incumbent = "John N. Sandlin"<|im_end|>
```

Only the text after `<|im_start|>assistant\n` is trained (the ChatML
markers are `PESFTConfig`'s defaults). The GRPO prompt is the same text up
to the empty `<think>` block, rendered as a plain string by
`build_prompt_dataset`; the rows keep `context` and `answer` for the
reward.

## Reward (`score_sql`)

For each completion, `extract_sql` drops anything up to `</think>`, takes
the inside of the first Markdown code fence if present and strips a
trailing `;`. Then:

| Reward | When |
|--------|------|
| 1.0 | The query equals the gold query after normalisation (case, whitespace, quote style, trailing `;`), **or** it returns the same rows as the gold query (as a multiset, order ignored) and the gold result is not empty |
| 0.2 | The query runs, but neither of the above holds (`VALID_SQL_REWARD`) |
| 0.0 | Empty reply, or SQLite rejects or aborts the query |

The database: `build_database` creates the schema in memory and fills
every table from the gold query's literals plus two fillers (`"alpha"`,
`7`), one row per value, shifting the values by one per column. The gold
filters then match some rows, so a wrong query usually returns a different
result. It is a heuristic, not a proof of equivalence. The gold query runs
first (the prediction may modify tables), and queries are aborted after
1,000,000 SQLite VM steps (`MAX_QUERY_STEPS`).

`sql_reward` in `text2sql_rl.py` applies `score_sql` to each completion;
TRL passes the `context` and `answer` columns as keyword arguments.
Evaluation reports `exact_match` (normalised string match),
`execution_accuracy` (reward 1.0) and `valid_sql` (reward > 0).

## Model

[`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B),
Apache-2.0: a 3B instruction model with switchable reasoning. With
thinking off it answers with the query directly, which keeps completions
short (about 20 tokens) and GRPO cheap.

Both stages load the base in 4-bit (NF4), so training fits in about 6 GB.
The RL stage samples with transformers (`use_vllm=False`, set in
`text2sql_rl.build_config`):

- PERL cannot sync LoRA updates into a 4-bit vLLM copy, so
  `load_in_4bit=True` requires `use_vllm=False`.
- vLLM runs SmolLM3 through its Transformers backend, which wraps the same
  transformers class that is being trained; colocated training then fails.

With 128-token completions, transformers sampling is fast enough (3.7
s/step).

Both stages use LoRA rank 1, alpha 32, on all attention and MLP
projections: 1,889,280 adapter parameters
([`speftr.lora_budget`](../../docs/guide.md#7-checking-the-rank)).

- **SFT.** The 77,777 training rows, rendered as `text2sql_sft` renders
  them, have 2.15M response tokens, which need ~1.07M parameters: rank 1
  suffices. The command below renders SmolLM3's default template,
  without the empty think block `enable_thinking=False` adds, and
  prints ~0.92M:

  ```bash
  uv run python -m speftr.lora_budget \
      --model_name_or_path HuggingFaceTB/SmolLM3-3B \
      --dataset b-mc2/sql-create-context --split "train[:-800]" \
      --prompt_column context question --response_column answer \
      --responses_only --lora_r 1
  ```

- **GRPO.** The RL stage keeps training the SFT adapter, so it is rank 1
  too. 150 steps × 16 completions are 2,400 episodes, which need ~1,200
  parameters (1 bit per episode): rank 1 suffices.

  ```bash
  uv run python -m speftr.lora_budget --mode rl \
      --model_name_or_path HuggingFaceTB/SmolLM3-3B \
      --max_steps 150 --completions_per_step 16
  ```

## Run

From the repository root (the SFT script needs an NVIDIA GPU to import
Unsloth):

```bash
# 1. Base model score
uv run python -m examples.text2sql.text2sql_eval \
  --model_path HuggingFaceTB/SmolLM3-3B
# 2. SFT (300 steps, as in Results)
uv run python -m examples.text2sql.text2sql_sft --max_steps 300
# 3. Score the SFT adapters (default --model_path)
uv run python -m examples.text2sql.text2sql_eval
# 4. GRPO on the SFT adapters
uv run python -m examples.text2sql.text2sql_rl --max_steps 150
# 5. Score the GRPO adapters
uv run python -m examples.text2sql.text2sql_eval \
  --model_path ./models/smollm3-3b-sql-grpo
```

Key flags (`--help` lists all):

| Script | Flag | Default |
|--------|------|---------|
| `text2sql_sft` | `--model_name_or_path` | `HuggingFaceTB/SmolLM3-3B` |
| | `--load_in_4bit` / `--no-load_in_4bit` | on |
| | `--max_steps` / `--num_epochs` | -1 (use epochs) / 1 |
| | `--per_device_batch_size` | 16 |
| | `--lora_r`, `--learning_rate` | 1, 2e-4 |
| | `--eval_rows` | 300 |
| | `--output_dir` | `./models/smollm3-3b-sql-sft` |
| `text2sql_rl` | `--model_name_or_path` | `./models/smollm3-3b-sql-sft` (SFT adapter directory) |
| | `--max_steps` | 150 |
| | `--num_generations` | 8 (device batch = one prompt's group; 2 accumulation steps) |
| | `--train_rows`, `--learning_rate` | 2000, 1e-5 |
| | `--output_dir` | `./models/smollm3-3b-sql-grpo` |
| `text2sql_eval` | `--model_path` | `./models/smollm3-3b-sql-sft` |
| | `--batch_size`, `--max_new_tokens` | 32, 128 |

Fixed in code: `max_seq_length` 1024 (both stages), GRPO
`max_completion_length` 128, constant learning rate, 4-bit policy.

## Results

One RTX 3090, 4-bit base, LoRA r=1, 490 held-out questions:

| Model | Exact match | Execution accuracy | Valid SQL |
|-------|-------------|--------------------|-----------|
| Base | 0.469 | 0.633 | 0.933 |
| SFT (300 steps, batch 16) | 0.749 | 0.835 | 0.992 |
| + GRPO (150 steps, 16 completions/step) | 0.773 | 0.863 | 0.996 |

| Stage | Speed | Time | End of training | Peak VRAM |
|-------|-------|------|-----------------|-----------|
| SFT | 1.0 s/step | 5 min | train loss 0.031, eval loss 0.043 | 5.7 GB (nvidia-smi) |
| GRPO | 3.7 s/step | 9 min | mean reward 0.84 (steps 141–150) | 3.7 GB (torch), 5.3 GB (nvidia-smi; brief peaks to 8.8 GB) |
| Evaluation (bf16) | | under 2 min | | 6.6 GB (torch), 8.4 GB (nvidia-smi) |

## Outputs

- `./models/smollm3-3b-sql-sft`: LoRA adapter, `adapter_config.json`,
  tokenizer, `speftr.json`, `training_args.json`, `checkpoint-*`.
- `./models/smollm3-3b-sql-grpo`: LoRA adapter (the SFT adapter after
  GRPO, so it alone is the final model), `adapter_config.json`, tokenizer,
  `checkpoint-*`.

`text2sql_eval` loads either directory on a bf16 base. To merge or serve
an adapter, see [After training](../../docs/guide.md#5-after-training).

## Use the trained model without speftr

`text2sql_inference.py` writes SQL with only torch, transformers, peft
and (for `--engine vllm`) vllm, no speftr or Unsloth. It runs two routes
and prints both outputs per prompt:

- **adapter**: the base model plus the LoRA adapter, attached with peft
  or passed to vLLM as a `LoRARequest` (`run_adapter_route`);
- **merged**: the adapter merged into the 16-bit base and saved to
  `--merged_dir` (`merge_adapter`), then loaded from there
  (`run_merged_route`).

```bash
uv run python -m examples.text2sql.text2sql_inference
uv run python -m examples.text2sql.text2sql_inference --engine vllm
uv run python -m examples.text2sql.text2sql_inference \
    --adapter_dir ./models/smollm3-3b-sql-grpo --engine vllm \
    --question "How many singers are older than 30?" \
    --context "CREATE TABLE singer (name VARCHAR, age INTEGER)"
```

Output of the first command (RTX 3090):

```text
Engine: transformers. Merged model saved to ./models/smollm3-3b-sql-sft-merged

>>> List the name, born state and age of the heads of departments ordered by age.
adapter: SELECT name, born_state, age FROM head ORDER BY age
merged:  SELECT name, born_state, age FROM head ORDER BY age

>>> What are the maximum and minimum budget of the departments?
adapter: SELECT MAX(budget_in_billions), MIN(budget_in_billions) FROM department
merged:  SELECT MAX(budget_in_billions), MIN(budget_in_billions) FROM department
```

`./models/smollm3-3b-sql-sft-merged` (default `<adapter_dir>-merged`)
holds `model.safetensors`, `config.json`, `generation_config.json`, the
tokenizer and `chat_template.jinja`. It loads without peft, like any Hub
checkpoint (`AutoModelForCausalLM`, `pipeline`, `vllm serve
./models/smollm3-3b-sql-sft-merged`).

Both routes give the same queries with both engines; the GRPO command
prints `SELECT COUNT(*) FROM singer WHERE age > 30` twice. The
transformers run takes about 1 min and 6.5 GB of VRAM; vLLM reserves 80%
of the GPU whatever the model size and starts one engine per route
(about 3 min per run in total).

- The SFT stage trains in 4-bit on the 16-bit repo
  (`HuggingFaceTB/SmolLM3-3B`), so `adapter_config.json` already names a
  16-bit base and both routes run in bf16.
- SmolLM3 is rendered with `enable_thinking=False`, as in training, so
  it answers with the query directly.
- The GRPO adapter (`./models/smollm3-3b-sql-grpo`) is the complete
  final model; pass it as `--adapter_dir`.

General rules (base class, 16-bit base, vLLM flags, serving): [After
training](../../docs/guide.md#5-after-training).

## Adapt it to your data

1. **Data.** Replace `load_splits` in `text2sql_eval.py` so it returns
   train and held-out `Dataset`s with `question`, `context` and `answer`
   (or rename the columns throughout).
2. **Prompt.** Edit `INSTRUCTION` and `build_messages`; both stages and
   the evaluation use them.
3. **Reward.** Replace `score_sql` (and `sql_reward`, which maps it over
   completions) with your own checker. Keep it deterministic and cheap: it
   runs `num_generations` times per prompt every step. For a real
   database, execute against a read-only copy with real rows instead of
   `build_database`, and keep a step or time limit.
4. **Dialect.** `build_database` and `run_query` use SQLite; for another
   dialect swap in its driver and adapt `normalize_sql`.
5. **Model.** Pass `--model_name_or_path` to `text2sql_sft`. If the new
   template is not ChatML, set `instruction_part` / `response_part` in
   `text2sql_sft.build_config`
   ([marker table](../../docs/guide.md#chat-templates-and-markers)), and
   change `CHAT_TEMPLATE_KWARGS` (`enable_thinking` is SmolLM3/Qwen
   specific).
6. **Rank.** Rerun the [`speftr.lora_budget` commands](#model) with your
   model, dataset and columns (SFT) and your GRPO steps and completions
   per step (RL); the SFT rank is the pipeline's rank
   ([guide](../../docs/guide.md#7-checking-the-rank)).
7. **Faster RL.** For a model vLLM supports natively, save the SFT stage
   merged (`save_method="merged_16bit"`) and start a bf16 RL run from the
   merged directory with `PERL.load_model()` and `use_vllm=True`
   ([SFT then RL](../../docs/guide.md#4-sft-then-rl)). PERL then trains a
   fresh adapter on top of the merged weights.

## Pitfalls

- `load_in_4bit=True` with `use_vllm=True` raises `ValueError` in `PERL`.
- Enabling vLLM for SmolLM3 fails at the first training forward (see
  [Model](#model)).
- `per_device_train_batch_size × gradient_accumulation_steps` must be a
  multiple of `num_generations`; `build_config` sets the batch to
  `num_generations`, so any value works.
- `set_pretrained_model` keeps training the SFT adapter: PERL's `lora_r`
  and `target_modules` are ignored and `load_model()` must not be called.
- The execution reward needs a non-empty gold result for its 1.0; tables
  filled from real data may need a different check.
