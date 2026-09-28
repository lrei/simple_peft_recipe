# Reasoning language: gpt-oss 20b and 120b (SFT)

gpt-oss reasons in its analysis channel before it answers, and it
reasons in English even when told otherwise. This example teaches it to
reason in the language its developer message asks for ("reasoning
language: German"), the task of the
[OpenAI cookbook's gpt-oss fine-tuning guide](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)
and the tutorials based on it. It is also the reference for fine-tuning
a Mixture-of-Experts reasoning model with `PESFT`: the harmony chat
template, response-only loss on the reasoning and the answer, LoRA on
every expert, and a model too large for one GPU (gpt-oss-120b) split
over several.

## Files and speftr APIs

| File | What it does | speftr API |
|------|--------------|------------|
| `gptoss/gptoss.py` | LoRA SFT of gpt-oss (20b or 120b) in 4-bit | `PESFTConfig.get_argument_parser()` with example defaults (`EXAMPLE_DEFAULTS`), `PESFTConfig.from_args`, `PESFT(config)`, `load_model()`, `train(train, eval, format_batch)`, `save_model()` |
| `gptoss/gptoss_eval.py` | Eval loss and reasoning-language compliance, base model vs adapter; defines the dataset split, markers and channel parsing | none (Unsloth loader + peft) |
| `gptoss/gptoss_inference.py` | Answers prompts with the adapter and the base, printing reasoning and answer separately, without speftr ([below](#use-the-trained-model-without-speftr)) | none (transformers + peft; Unsloth only loads the base) |
| `gptoss/gptoss.sbatch` | Slurm template: gpt-oss-120b split over one node's GPUs | none |

`build_format_batch` returns the batched formatting function
`PESFT.train` expects ([SFT data contract](../../docs/guide.md#sft-data-contract)).

## Data

[`HuggingFaceH4/Multilingual-Thinking`](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
(Apache-2.0, not gated): 1,000 rows, 200 per reasoning language
(English, French, German, Spanish, Italian). The `messages` column is
used as is: a system turn `reasoning language: <X>` followed by a
persona prompt, a user turn in English, and an assistant turn whose
`thinking` is the reasoning in language X and whose `content` is the
answer (in English). The columns `reasoning_language`, `developer`,
`user`, `analysis` and `final` hold the same fields separately.

The dataset has only a `train` split. `gptoss_eval.load_splits` holds
out 100 random rows (seed 0: English 25, French 22, German 19, Spanish
18, Italian 16) for the eval loss during training and for
`gptoss_eval.py`; the other 900 are trained on. User prompts repeat
across rows, so 13 held-out prompts also occur in training, with other
reasoning.

Rows have a median of 1,030 tokens; 47 of the 1,000 exceed
`--max_seq_length` 2048 and are cut at 2048. With `chat_template=None`
gpt-oss's own harmony template renders them. A short row (from
`tests/fixtures/multilingual_thinking_rows.json`) as `format_batch`
renders it; `gptoss.py` prints one before training:

```text
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-09-28

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions

reasoning language: Italian

Never use more than 5 words in the response<|end|><|start|>user<|message|>Can you help me figure out if my partner might be cheating? What signs should I look for or what questions should I ask to get a clearer picture?<|end|><|start|>assistant<|channel|>analysis<|message|>Comunicazione essenziale. Fidati dell'intuito.<|end|><|start|>assistant<|channel|>final<|message|>Trust your instincts. Open communication.<|return|>
```

The template writes its own system message (with the current date) and
turns the dataset's system turn into the developer message. With
`instruction_part` `<|start|>user<|message|>` and `response_part`
`<|start|>assistant`, exactly these tokens are trained (checked token
by token in `tests/test_examples_gptoss.py`):

```text
<|channel|>analysis<|message|>Comunicazione essenziale. Fidati dell'intuito.<|end|>
<|channel|>final<|message|>Trust your instincts. Open communication.<|return|>
```

Both channels are trained; the second `<|start|>assistant` stays
masked. A `response_part` of `<|start|>assistant<|channel|>final<|message|>`
would train only the answer and not teach the reasoning language.

## Model

`unsloth/gpt-oss-20b-unsloth-bnb-4bit` by default and
`unsloth/gpt-oss-120b-unsloth-bnb-4bit` for the large one (Apache-2.0,
not gated): Unsloth's bitsandbytes 4-bit (NF4) conversions of OpenAI's
gpt-oss, with every expert stored as its own 4-bit linear layer. OpenAI's
own checkpoints (`openai/gpt-oss-20b`, `-120b`) store the experts in
MXFP4, which has no backward pass: they cannot be trained with LoRA.

| | gpt-oss-20b | gpt-oss-120b |
|--|-------------|--------------|
| Parameters (active per token) | 21B (3.6B) | 117B (5.1B) |
| Layers, experts per layer | 24, 32 | 36, 128 |
| 4-bit download / weights on GPU | 12 GB / 11.7 GiB | 62.1 GB / ~58 GiB |
| LoRA rank 1: trainable parameters | 11,556,864 | 67,101,696 |

LoRA rank 1, alpha 32, on the attention projections and on every
expert's gate/up and down projections (Unsloth maps the `gate_proj`,
`up_proj`, `down_proj` targets onto the experts); the router is not
adapted. Rank 1 is enough here: the dataset has 529,151 response tokens,
which [`speftr.lora_budget`](../../docs/guide.md#7-checking-the-rank)
turns into ~265k needed parameters, far below the rank-1 adapter:

```bash
uv run python -m speftr.lora_budget \
    --model_name_or_path openai/gpt-oss-120b \
    --dataset HuggingFaceH4/Multilingual-Thinking \
    --messages_column messages --lora_r 1 --responses_only
```

## Training settings

| Setting | Value | Why |
|---------|-------|-----|
| Learning rate, schedule | 2e-4, constant (PESFT defaults) | The cookbook also uses 2e-4; the recipe keeps it constant |
| Batch | 4 × gradient accumulation 4 = 16 | The cookbook's effective batch; 57 optimizer steps per epoch |
| Epochs | 1 | As the cookbook |
| `--max_seq_length` | 2048 | As the cookbook; covers 95% of the rows |
| `--router_aux_loss_coef` | 0.0 | The router is frozen; Unsloth's 4-bit gpt-oss fails with TRL's load-balancing loss (0.001 in `SFTConfig`) |
| Eval, save | every 10 steps on the 100 held-out rows | Shows the eval-loss curve; the best checkpoint is reloaded at the end |

## Run

From the repo root, after `uv sync`. `gptoss_eval.py` also needs the
`gptoss` extra (`py3langid`, a 4.6 MB offline language identifier);
`uv run --extra gptoss` installs it.

### gpt-oss-20b on one GPU

```bash
uv run python -m examples.gptoss.gptoss
uv run --extra gptoss python -m examples.gptoss.gptoss_eval
uv run python -m examples.gptoss.gptoss_inference
```

A quick wiring check: `--max_steps 2 --eval_strategy no --save_strategy no`.

`--help` lists every `PESFTConfig` flag; the example's own defaults are
printed at the end:

| Flag | Example default |
|------|-----------------|
| `--model_name_or_path` | `unsloth/gpt-oss-20b-unsloth-bnb-4bit` |
| `--load_in_4bit` | on |
| `--chat_template` / markers | `None` (harmony) / `<\|start\|>user<\|message\|>`, `<\|start\|>assistant` |
| `--lora_r` | 1 |
| `--router_aux_loss_coef` | 0.0 |
| `--per_device_train_batch_size` / `--gradient_accumulation_steps` | 4 / 4 |
| `--max_seq_length` | 2048 |
| `--num_train_epochs` | 1 |
| `--eval_strategy` / `--eval_steps`, `--save_strategy` / `--save_steps` | `steps` / 10 |
| `--device_map` | none (one GPU) |
| `--output_dir` | `./models/speftr-gptoss-20b` |

### gpt-oss-120b on several GPUs

The 4-bit 120b weights (~58 GiB) do not fit one 40 GB GPU, so
data-parallel training (a full copy per GPU, `torchrun`) is impossible
there. Split the model instead: one process, no `torchrun`, layers
spread over all visible GPUs, run one GPU at a time:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m examples.gptoss.gptoss \
    --model_name_or_path unsloth/gpt-oss-120b-unsloth-bnb-4bit \
    --device_map balanced --output_dir ./models/speftr-gptoss-120b
CUDA_VISIBLE_DEVICES=0,1 uv run --extra gptoss \
    python -m examples.gptoss.gptoss_eval \
    --adapter_dir ./models/speftr-gptoss-120b --device_map balanced
CUDA_VISIBLE_DEVICES=0,1 uv run python -m examples.gptoss.gptoss_inference \
    --adapter_dir ./models/speftr-gptoss-120b --device_map auto
```

### Slurm

[`gptoss.sbatch`](gptoss.sbatch) is a single-node template for the
120b run (training, then evaluation). Fill in the `<...>` placeholders
(account, partition, clone path, shared Hugging Face cache), download
the model and dataset beforehand, then:

```bash
sbatch examples/gptoss/gptoss.sbatch
```

```bash
# Once, on a node with internet access, with the same HF_HOME:
hf download unsloth/gpt-oss-120b-unsloth-bnb-4bit
hf download HuggingFaceH4/Multilingual-Thinking --repo-type dataset
```

It sets `HF_HUB_OFFLINE=1`: offline, Unsloth needs the `unsloth/` model
id or a local directory, not `openai/gpt-oss-120b`
([offline guide](../../docs/running_offline_models.md)).

## Memory and time

Measured with gpt-oss-20b on one RTX 3090 (24 GB), the example
defaults (batch 4 × accumulation 4, 2048 tokens, rank 1, `adamw_8bit`,
Unsloth gradient checkpointing):

| Measure | gpt-oss-20b (measured) |
|---------|------------------------|
| 4-bit weights on the GPU after loading | 11.7 GiB |
| Peak GPU memory in `nvidia-smi` (whole process), training | 15,758 MiB |
| Time per optimizer step (16 rows) | ~15 s |
| One epoch (57 steps, with 6 evaluations of ~116 s each) | 26 min; 29 min end to end with loading and saving |
| `gptoss_eval.py` (2 × 100 eval losses and 2 × 100 generations of 320 tokens) | 29 min, 16.6 GiB peak allocated by torch |

**gpt-oss-120b: an estimate, not yet measured.** The weights take ~58
GiB, split by `--device_map balanced` into ~29 GiB per GPU on two 40 GB
A100s. Activations grow with the 36 layers (vs 24) at the same hidden
size, so expect ~5–6 GiB of training overhead instead of ~4 GiB, which
leaves a few GiB free on each GPU; lower `--per_device_train_batch_size`
(and raise `--gradient_accumulation_steps` to keep 16) if it runs out of
memory, or use more GPUs. Split layers run one GPU at a time, and each
layer has 128 experts instead of 32, so a step takes several times as
long as with 20b; plan for hours, not minutes, per epoch.

## Results

gpt-oss-20b, one epoch on the 900 training rows, example defaults.
Eval loss on the 100 held-out rows (base model: 1.809, from
`gptoss_eval.py`):

| Step | 10 | 20 | 30 | 40 | 50 | 57 |
|------|----|----|----|----|----|----|
| Train loss (mean of the last 5 steps; step 55 for 57) | 1.170 | 1.123 | 1.017 | 1.076 | 0.993 | 1.046 |
| Eval loss | 1.123 | 1.070 | 1.043 | 1.025 | 1.013 | 1.007 |

`gptoss_eval.py` on the saved adapter:

```text
eval_loss  base 1.8091  adapter 1.0068

Reasoning-language compliance (analysis channel)
language   rows   base  adapter
English      25   100%     100%
French       22     5%     100%
German       19     0%     100%
Italian      16     0%     100%
Spanish      18     0%     100%
overall     100    26%     100%
```

The base model reasons in English whatever language is requested (it
reads "reasoning language" as the language of the answer). After one
epoch every held-out analysis channel is in the requested language.

## Evaluation

`gptoss_eval.py` loads the adapter once and scores the 100 held-out rows
twice, with the adapter disabled (the base model) and enabled:

- **Eval loss**: mean token loss on the analysis and final channels,
  masked as in training (Unsloth's `train_on_responses_only` with the
  example's markers) and cut at 2048 tokens. It equals the trainer's
  `eval_loss` for the same adapter.
- **Reasoning-language compliance**, the check of the cookbook: generate
  greedily from the system and user turns, take the analysis channel,
  identify its language with `py3langid`, and count it as compliant when
  it is the requested one. Only the first 320 tokens are generated
  (`--max_new_tokens`): the start of the reasoning shows its language.
  py3langid chooses among 97 languages, not only the five of the
  dataset. On the dataset's own `analysis` texts it agrees with
  `reasoning_language` for 999 of 1,000 rows (the miss is an Italian
  row written without vowels).

The eval and inference scripts load the model with Unsloth but turn off
Unsloth's gpt-oss attention (`UNSLOTH_ENABLE_FLEX_ATTENTION=0`, set in
the loader): in eval mode it gives the sliding-window layers the
full-attention mask, which degrades the loss and generation past 128
tokens. transformers' attention applies each layer's mask. During
training `PESFT` keeps gpt-oss attention on Unsloth's training kernel,
evaluation included, for the same reason.

## Outputs

`--output_dir` holds the LoRA adapter (`adapter_model.safetensors`,
`adapter_config.json`, which records the 4-bit base), the tokenizer
with `chat_template.jinja`, `speftr.json`, `training_args.json` and the
best checkpoint (`checkpoint-*`).

## Use the trained model without speftr

`gptoss_inference.py` needs torch, transformers, peft, bitsandbytes and
Unsloth as the loader. It builds the harmony prompt with a reasoning
language (`--reasoning_language`, default German; optional
`--system_prompt` instructions after it), answers with the adapter and
with the adapter disabled, and prints the analysis and final channels
separately:

```bash
uv run python -m examples.gptoss.gptoss_inference
uv run python -m examples.gptoss.gptoss_inference \
    --reasoning_language French --prompt "Why is the sky blue?"
```

Output with the adapter above (RTX 3090, 257 s end to end; the second
prompt's replies shortened here):

```text
>>> What is the capital of Australia? (reasoning language: German)
[adapter] analysis: Okay, der Benutzer fragt nach der Hauptstadt Australiens. Ich erinnere mich, dass Canberra die Hauptstadt ist. Aber ich sollte das überprüfen, um sicherzugehen. Canberra ist die Hauptstadt, während Sydney und Melbourne die größten Städte sind. [...] So ist die Antwort präzise und hilfreich.
[adapter] final:    The capital of Australia is **Canberra**. It is located in the Australian Capital Territory (ACT) and was chosen as the capital in 1908 as a compromise between the two largest cities, Sydney and Melbourne. [...]
[base] analysis: The user asks: "What is the capital of Australia?" The user presumably expects the answer: Canberra. The instruction says: "reasoning language: German". So the assistant should respond in German. [...]
[base] final:    Die Hauptstadt Australiens ist Canberra.

>>> Why do cats purr? Answer in two sentences. (reasoning language: German)
[adapter] analysis: Okay, der Benutzer fragt, warum Katzen schnurren, und möchte die Antwort in zwei Sätzen. [...]
[adapter] final:    Cats purr when they are content, relaxed, or seeking comfort, as it often signals a state of well-being and bonding. They also purr in stressful or painful situations as a self-soothing mechanism to calm themselves and communicate distress.
[base] analysis: The user asks: "Why do cats purr? Answer in two sentences." The instruction says: "reasoning language: German". So the assistant should respond in German. [...]
[base] final:    Katzen schnurren, weil sie sich dabei entspannen, ihre Stimmung ausdrücken und ihre Körpertemperatur regulieren. [...]
```

Like the dataset, the adapter reasons in the requested language and
answers in the language of the question.

- **Unsloth loads the base.** The 4-bit checkpoint stores each expert as
  its own linear layer (`experts.gate_up_projs.N`), which the adapter
  targets; only Unsloth's gpt-oss model code builds that layout
  (transformers' `GptOssForCausalLM` expects fused expert tensors and
  cannot load it). Everything after loading is transformers + peft.
- **Only this base.** Serve the adapter on the same
  `unsloth/gpt-oss-*-unsloth-bnb-4bit` base it was trained on. vLLM
  LoRA needs a 16-bit base, and merging into OpenAI's MXFP4 or a 16-bit
  gpt-oss is not covered by this example.
- **120b**: `--device_map auto` spreads the model over all visible GPUs.

## Adapt it to your data or model

1. **Data**: return your rows from `load_splits` in
   `gptoss_eval.py`. A `messages` column of system/user/assistant turns,
   with the reasoning in the assistant's `thinking` field, needs no other
   change; otherwise build that column first. Keep a held-out split.
2. **Evaluation**: the compliance check reads `reasoning_language`; for
   another task, replace `is_compliant` with your own check of the
   analysis or final channel (`parse_channels`).
3. **Model**: another gpt-oss size or fine-tune needs only
   `--model_name_or_path` (a bitsandbytes 4-bit conversion). Another
   model family needs its own `--chat_template`, `--instruction_part`
   and `--response_part`
   ([marker table](../../docs/guide.md#chat-templates-and-markers)) and
   probably not `router_aux_loss_coef`.
4. **Rank**: check it for your data with `python -m speftr.lora_budget
   --responses_only` (command above).

## Pitfalls

- **MXFP4 is not trainable.** `openai/gpt-oss-*` loads, but LoRA
  training needs the bitsandbytes conversion
  (`unsloth/gpt-oss-*-unsloth-bnb-4bit`) and `--load_in_4bit`.
- **`router_aux_loss_coef` must stay 0.** TRL's default (0.001) turns on
  the MoE load-balancing loss, which Unsloth's 4-bit gpt-oss cannot
  compute.
- **Offline ids**: Unsloth resolves `openai/gpt-oss-*` to `unsloth/`
  repos on the Hub; offline, use the `unsloth/` id or a local path.
- **Markers**: `response_part` must be `<|start|>assistant` to train the
  reasoning; check the row `gptoss.py` prints before training.
- **Unsloth's gpt-oss inference attention** is wrong past 128 tokens
  (see [Evaluation](#evaluation)); generate as `gptoss_inference.py`
  does.
- Imports Unsloth, which needs a GPU even to import.
