# Large model: Gemma 4 31B in 4-bit on one 24 GB GPU (SFT)

The [instruct](../instruct/README.md) recipe (Dolly pirate, system, user
and assistant turns, loss on the assistant turn only) on a model whose
16-bit weights (~62 GB) are almost three times the GPU's memory. It shows
the memory levers that make that fit, and how to run the same script on
several GPUs or under Slurm.

## Files and speftr APIs

| File | What it does | speftr API |
|------|--------------|------------|
| `big/big.py` | LoRA SFT of Gemma 4 31B in 4-bit | `PESFTConfig.get_argument_parser()` with example defaults (`EXAMPLE_DEFAULTS`), `PESFTConfig.from_args`, `PESFT(config)`, `load_model()`, `train(train, eval, format_batch)`, `save_model()` |
| `big/big_inference.py` | Answers messages with the adapter, without speftr ([below](#use-the-trained-model-without-speftr)) | none (transformers + peft + bitsandbytes) |
| `big/big.sbatch` | Slurm template: DDP or model splitting on one node | none |

`build_format_batch` returns the batched formatting function
`PESFT.train` expects; `build_conversation` turns one row into messages
([SFT data contract](../../docs/guide.md#sft-data-contract)).

## Data

[`TeeZee/dolly-15k-pirate-speech`](https://huggingface.co/datasets/TeeZee/dolly-15k-pirate-speech)
(CC-BY-SA-3.0, not gated): 15,011 Dolly rows with pirate-speech responses;
columns `instruction`, `context` (often empty), `response`. `--eval_size`
(100) rows are held out for the eval loss. Each row becomes a system turn
(`--system_prompt`), a user turn (instruction, plus `Context:` and the
context when there is one) and an assistant turn. Rendered with the
`gemma-4` template, as `big.py` prints before training:

```text
<bos><|turn>system
You are a pirate, always respond in pirate speech.<turn|>
<|turn>user
Which record label created Vinyl

Context:
The LP (from "long playing" or "long play") is an analog sound ...<turn|>
<|turn>model
Vinyl was introduced by columbia in 1948<turn|>
```

With `instruction_part` `<|turn>user\n` and `response_part`
`<|turn>model\n`, only the reply after `<|turn>model\n` is trained.

## Model

`unsloth/gemma-4-31B-it-unsloth-bnb-4bit` (Apache-2.0, not gated):
Gemma 4 31B instruct, pre-quantized to 4-bit NF4 (~18 GiB on the GPU).
Gemma 4 is multimodal; `load_model()` returns a processor and only the
language model gets LoRA adapters (rank 8, alpha 32, all attention and
MLP projections). Rank 8 has 61.2M adapter parameters; the ~1.3M response
tokens need ~0.63M, so rank 1 would suffice
([`speftr.lora_budget`](../../docs/guide.md#7-checking-the-rank)):

```bash
uv run python -m speftr.lora_budget \
    --model_name_or_path unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
    --dataset TeeZee/dolly-15k-pirate-speech --split "train[:-100]" \
    --prompt_column instruction context --response_column response \
    --responses_only
```

## Memory budget

Measured on one RTX 3090 (24 GB), 4-bit, LoRA rank 8, batch 1,
`adamw_8bit`, Unsloth gradient checkpointing; peak memory allocated by
torch, for sequences filled to the given length:

| Tokens per micro-batch | Peak | Time per micro-batch |
|------------------------|------|----------------------|
| 1024 | 19.1 GiB | 4.5 s |
| 2048 | 19.6 GiB | 9.3 s |
| 4096 | 20.7 GiB | 19.0 s |

Of that, ~18 GiB are the 4-bit weights. What keeps the rest small:

- **4-bit weights** (`--load_in_4bit`, always on here): the 16-bit
  model does not fit; 8-bit would not either.
- **Batch 1 × gradient accumulation 16** (effective batch 16):
  accumulation costs no memory, only time.
- **Gradient checkpointing** (`--use_gradient_checkpointing unsloth`,
  the default) offloads activations to CPU RAM; `False` would not fit.
- **Unsloth's fused loss** never materializes the 262k-vocabulary logits.
- The optimizer barely matters: LoRA state is tiny (≤ 0.15 GiB between
  optimizers).

`--max_seq_length` (2048) bounds the tokens per micro-batch; Dolly rows
are mostly much shorter. General levers and flags:
[Fitting in memory](../../docs/guide.md#10-fitting-in-memory).

## Run

### One GPU

From the repo root, after `uv sync`:

```bash
uv run python -m examples.big.big
```

A 20-step check with a small eval set:

```bash
uv run python -m examples.big.big --max_steps 20 --eval_size 16 \
    --eval_strategy steps --eval_steps 20 \
    --save_strategy steps --save_steps 20
```

Result of that command on one RTX 3090 (24 GB):

| Measure | Value |
|---------|-------|
| Peak memory allocated / reserved by torch | 19.98 / 20.73 GiB |
| Peak GPU memory in `nvidia-smi` (whole process) | 21,592 MiB |
| Time per optimizer step (16 sequences) | ~22 s (first step 100 s) |
| Training, 20 steps | 690 s; 836 s end to end with loading and saving |
| `train_loss` / `eval_loss` at step 20 (16 eval rows) | 1.891 / 1.899 |

At ~22 s per step, one epoch (936 steps) takes about 6 hours.

`--help` lists every `PESFTConfig` flag; the example's own defaults are
printed at the end (the per-flag help shows the `PESFTConfig` ones).

| Flag | Example default |
|------|-----------------|
| `--model_name_or_path` | `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` |
| `--chat_template` / markers | `gemma-4` / `<\|turn>user\n`, `<\|turn>model\n` |
| `--per_device_train_batch_size` / `--gradient_accumulation_steps` | 1 / 16 |
| `--max_seq_length` | 2048 |
| `--num_train_epochs` | 1 |
| `--eval_size` | 100 |
| `--system_prompt` | `You are a pirate, always respond in pirate speech.` |
| `--device_map` | none (one GPU per process) |
| `--output_dir` | `./models/speftr-big` |

### Several GPUs

DDP is validated with this model on two A100 40GB (below). Model
splitting is validated with gpt-oss-120b on two A100 40GB
([examples/gptoss](../gptoss/README.md#gpt-oss-120b-on-several-gpus)).

**Data parallel (DDP)**: one process per GPU, each with a full 4-bit
copy (~20 GB, so every GPU needs 24 GB or more). Keep the effective
batch (per-GPU batch × accumulation × GPUs) at 16. On two A100 40GB:

```bash
uv run torchrun --nproc_per_node 2 -m examples.big.big \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 2
```

Result with the 20-step check flags from [One GPU](#one-gpu) added:

| Measure | Value |
|---------|-------|
| Time per optimizer step (16 rows) | ~2.6 s (~6 rows/s) |
| One epoch (14,970 rows) | ~40 min |
| GPU utilization, rank 0 / rank 1 | 74% / 65% |
| Peak GPU memory, rank 0 / rank 1 | 29.7 / 24.0 GB |
| `train_loss` / `eval_loss` at step 20 (16 eval rows) | 2.49 / 1.868 |

Only rank 0 writes the adapter and metadata; the checkpoint holds the
RNG state of every rank. Both ranks print some progress lines, and
PyTorch warns at exit that `destroy_process_group()` was not called;
both are harmless. With per-GPU batch 4, one process
(`--nproc_per_node 1`) on one A100 40GB peaks at 24.5 GB.

**Model splitting**: one process, the layers spread over all visible
GPUs, for GPUs too small for the whole model. It runs one GPU at a time,
so it is no faster than one large GPU. `unsloth_balanced` is Unsloth's
planner, which reserves room for the output head and logits on the
head's GPU; transformers' `balanced` does not. Not under torchrun:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m examples.big.big \
    --device_map unsloth_balanced
```

### Slurm

[`big.sbatch`](big.sbatch) is a single-node template for both modes. It
requests two GPUs and in DDP mode runs the per-GPU batch 4 command
above. Fill in the `<...>` placeholders (account, partition, clone path,
shared Hugging Face cache), download the model and dataset beforehand,
then:

```bash
sbatch examples/big/big.sbatch                  # DDP on all GPUs
MODE=split sbatch examples/big/big.sbatch       # --device_map unsloth_balanced
```

It sets `HF_HUB_OFFLINE=1`: offline, Unsloth needs the `unsloth/` model
id or a local directory, not `google/gemma-4-31B-it`
([offline guide](../../docs/running_offline_models.md)).

## Outputs

`--output_dir` holds the LoRA adapter (`adapter_model.safetensors`,
`adapter_config.json`, which records the 4-bit base), the tokenizer and
processor files with `chat_template.jinja`, `speftr.json`,
`training_args.json` and the last checkpoint.

## Use the trained model without speftr

`big/big_inference.py` needs only torch, transformers, peft and
bitsandbytes. By default it attaches the adapter with peft to the 4-bit
base recorded in `adapter_config.json`, which fits one 24 GB GPU:

```bash
uv run python -m examples.big.big_inference
```

Output with the 20-step adapter above (RTX 3090, 17.6 GiB peak, 20 s
with the weights in the page cache):

```text
>>> If I have more pieces at the time of stalemate, have I won?
adapter: No. In a stalemate, neither player can make a legal move, and the game ends in a draw.

>>> What is the capital of France?
adapter: Paris
```

The base model alone already follows the pirate system prompt at length
(`Ahoy there, matey! ... The capital o' that fancy land called France be
**Paris**, ye scurvy dog! Arr!`). Twenty steps (320 rows) teach Dolly's
short answer style first; the dataset's pirate phrasing is mild and
takes more training.

**Merged model and vLLM** (`--merge`, `--engine vllm`): merging and
vLLM LoRA need the 16-bit base, `google/gemma-4-31B-it` (~62 GB of bf16
weights), so these routes **do not run on a 24 GB GPU** and have not
been run for this example. On larger hardware (one 80 GB GPU, or several
GPUs, which `device_map="auto"` uses automatically):

```bash
uv run python -m examples.big.big_inference --merge \
    --base_model google/gemma-4-31B-it
uv run python -m examples.big.big_inference --engine vllm \
    --base_model google/gemma-4-31B-it
```

They run the same adapter and merged routes as the other examples
([guide](../../docs/guide.md#inference-scripts-per-example)); for vLLM
on several GPUs add `tensor_parallel_size` to the `LLM(...)` call in
`generate_vllm`.

## Adapt it to your data or model

1. **Data**: change `load_datasets` and `build_conversation`, as in the
   [instruct example](../instruct/README.md#adapt-it-to-your-data).
2. **Model**: `--model_name_or_path`, and `--chat_template`,
   `--instruction_part`, `--response_part` for its family
   ([marker table](../../docs/guide.md#chat-templates-and-markers)). For
   another large model, start from the memory table above: 4-bit weights
   take about 0.6 GiB per billion parameters here, plus 1.5–3 GiB for
   training at 2048 tokens.
3. **Longer sequences**: raise `--max_seq_length` (4096 measured at
   20.7 GiB) and keep batch 1.
4. **Rank**: rerun the [`speftr.lora_budget` command](#model) with your
   model, dataset and columns
   ([guide](../../docs/guide.md#7-checking-the-rank)).

## Pitfalls

- **Time**: each optimizer step is 16 sequences of a 31B model; see the
  timings above before starting a full epoch.
- **Offline**: Unsloth resolves `google/gemma-4-*` to `unsloth/*` repos
  on the Hub; use the `unsloth/` id or a local path.
- **Markers**: with another template, response-only loss trains nothing
  or everything; check the row `big.py` prints before training.
- Imports Unsloth, which needs a GPU even to import.
