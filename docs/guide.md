# speftr user guide

How to fine-tune your own models on your own data with `speftr`:
supervised fine-tuning (`PESFT`), GRPO reinforcement learning (`PERL`),
DPO preference optimization
([`PEDPO`](#12-preference-optimization-pedpo-dpo)) and what to do with the
result. Defaults follow the recipe in the
[README](../README.md#the-recipe), are sized for a 24 GB GPU and run
unchanged on larger ones.

## 1. Install and import

Linux + NVIDIA GPU (driver ≥ 580), Python 3.13/3.14, `uv`.

| Option | How |
|--------|-----|
| Work in a clone (simplest) | `git clone https://github.com/lrei/simple_peft_recipe`, then `uv sync --extra rl`; put your scripts in the clone and run them with `uv run python my_script.py` |
| Add to your uv project | `uv add "speftr[rl] @ git+https://github.com/lrei/simple_peft_recipe"`, and copy the `[tool.uv]`, `[[tool.uv.index]]` and `[tool.uv.sources]` blocks from this repo's `pyproject.toml`: uv does not inherit them from dependencies, and they hold the version overrides and the CUDA 13 torch index |
| Copy the code | `speftr/pesft.py`, `speftr/perl.py` and `speftr/pedpo.py` are self-contained |

Extras: none for SFT only, `rl` for vLLM generation in `PERL`, `gym` for
the reasoning-gym example.

Imports:

- `import speftr` / `from speftr import PERL, PERLConfig` (or `PEDPO`,
  `PEDPOConfig`) never imports Unsloth. `PESFT` and `PESFTConfig` are
  loaded lazily on first access.
- Unsloth needs a GPU even to import, so SFT scripts cannot run on a
  CPU-only machine.
- **In SFT scripts, `import unsloth` first**, before `datasets`,
  `transformers`, `trl`, `peft` and `speftr`'s `PESFT`, so its patches
  apply.

## 2. Supervised fine-tuning (`PESFT`)

Lifecycle: `PESFT(config)` → `load_model()` → `train(...)` →
`save_model()`.

### SFT data contract

```python
PESFT.train(train_dataset, eval_dataset, formatting_func,
            *, resume_from_checkpoint=None)
```

- `train_dataset`, `eval_dataset`: `datasets.Dataset`. `eval_dataset` may
  be `None` (see [config](#sft-config): then set
  `eval_strategy="no"` and `save_strategy="no"`).
- `formatting_func(batch) -> list[str]`, **batched**: `batch` maps each
  column name to a list of values; return one training text per row.
  Unsloth also calls it once on a *single* row (column → value) to probe
  it; it must still return a list.
- Return the chat template's output **unchanged**, including a leading
  BOS: Unsloth sees it and does not add a second one. The Gemma 4
  tokenizer adds no BOS by itself, so stripping it leaves none.
- The formatting function is bypassed if the dataset has a `text`,
  `input_ids` or `labels` column, or both `prompt` and `completion`;
  rename or drop those columns.
- Multi-turn conversations are fine: with `train_on_responses=True`
  every assistant turn is trained.

Chat template and response-only loss:

| Field | Meaning |
|-------|---------|
| `chat_template` | Unsloth template name applied to the tokenizer (`"gemma-4"`, `"qwen3"`, `"chatml"`, `"llama-3.1"`, ...). `None` keeps the model's own template. The default is `"qwen2.5"`, so set it for any other model |
| `train_on_responses` | Mask everything except assistant turns (loss on responses only). Default `False`; the recipe recommends `True` for chat data |
| `instruction_part` | Text that starts a user turn in the formatted string. Empty: inferred from the tokenizer's template when training starts |
| `response_part` | Text that starts an assistant turn. Empty: inferred like `instruction_part` |

With `train_on_responses=True`, tokens after each `response_part` up to
the next `instruction_part` are trained. Both markers must match the
rendered text exactly (see the [marker table](#chat-templates-and-markers)).
Leave both empty and `speftr.infer_chat_markers` picks the pair of the
template family it recognises (ChatML, Gemma 4, Gemma 2/3, Llama 3,
gpt-oss, Granite 3); it raises, printing the rendered conversation, for
any other template.

Multimodal checkpoints (Gemma 4, Qwen 3.5, Qwen 3.8) work for text SFT:
`load_model()` then returns a processor; its text tokenizer is
`getattr(tokenizer, "tokenizer", tokenizer)`, and only the language model
gets LoRA adapters.

### SFT config

| Field | Default | Notes |
|-------|---------|-------|
| `model_name_or_path` | `unsloth/Qwen2.5-0.5B-Instruct` | Hub id or local dir |
| `load_in_4bit` | `False` | QLoRA; use for anything ≥ 4B on 24 GB |
| `full_finetuning` | `False` | Train every weight instead of LoRA (bf16 weights, the configured optimizer); for comparison runs, with a learning rate about ten times lower; `save_model` writes the whole model; not with 4-bit or 8-bit |
| `max_seq_length` | 2048 | Longer rows are truncated |
| `chat_template` | `"qwen2.5"` | `None` = model's own |
| `lora_r` / `lora_alpha` | 8 / 32 | See [rank](#7-checking-the-rank) |
| `target_modules` | all attention + MLP projections | |
| `per_device_train_batch_size` | 32 | × `gradient_accumulation_steps` = effective batch (16-32) |
| `learning_rate` | 2e-4 | constant schedule |
| `num_train_epochs` / `max_steps` | 3 / -1 | `max_steps` > 0 overrides epochs |
| `eval_strategy` / `save_strategy` | `"epoch"` / `"epoch"` | Must match (the best checkpoint by `eval_loss` is loaded at the end). `eval_strategy="no"` keeps the final model and evaluates the eval set once after training; `save_strategy` is then free. Without an eval set use `"no"` for both |
| `save_method` | `"lora"` | or `"merged_16bit"` |
| `attn_implementation` | `"sdpa"` | Fastest on a 3090 |
| `padding_free` | `False` | Without FlashAttention (3090 + SDPA) it is several times slower than padded batches |
| `router_aux_loss_coef` | 0.0 | MoE router load-balancing loss; the router stays frozen (not a LoRA target), and Unsloth's 4-bit gpt-oss fails with it on |
| `eval_in_train_mode` | `[]` | Module class names kept in training mode during evaluation, for eval-mode kernels that are wrong or too heavy (gpt-oss example); only for modules without dropout |

`PESFTConfig.from_args()` / `get_argument_parser()` expose most fields as
CLI flags.

### Example: from a JSONL of messages to a saved adapter

`data/train.jsonl`, one conversation per line:

```json
{"messages": [{"role": "user", "content": "Recommend a movie."}, {"role": "assistant", "content": "The Shawshank Redemption."}]}
```

`my_sft.py`:

```python
import unsloth  # noqa: F401  # first: patches transformers/TRL
from datasets import load_dataset

from speftr import PESFT, PESFTConfig

config = PESFTConfig(
    model_name_or_path="Qwen/Qwen3.5-4B",
    load_in_4bit=True,
    chat_template=None,  # keep Qwen 3.5's own template
    train_on_responses=True,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
    lora_r=16,
    max_seq_length=2048,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    output_dir="./models/my-sft",
)
trainer = PESFT(config)
_, tokenizer = trainer.load_model()
text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)


def format_batch(batch):
    conversations = batch["messages"]
    if conversations and isinstance(conversations[0], dict):
        conversations = [conversations]  # Unsloth's single-row probe
    return [
        text_tokenizer.apply_chat_template(messages, tokenize=False)
        for messages in conversations
    ]


dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
split = dataset.train_test_split(test_size=0.05, seed=0)
trainer.train(split["train"], split["test"], format_batch)
trainer.save_model()  # LoRA adapter + tokenizer in ./models/my-sft
```

Other layouts: build the message list inside `format_batch` from your
columns (e.g. `question` / `answer`), or pre-render any text you like.
`examples/instruct/instruct.py` shows Dolly-style
`instruction`/`context`/`response` columns.

## 3. Reinforcement learning (`PERL`, GRPO)

Lifecycle: `PERL(config)` → `load_model()` → `train(dataset,
reward_funcs)` → `save_model()`. Plain transformers + peft + TRL; no
Unsloth.

### RL data contract

- `dataset`: a `datasets.Dataset` with a `prompt` column, either a plain
  string (already templated) or, simpler, a TRL conversational prompt:
  a list of `{"role", "content"}` messages ending with the user turn. TRL
  applies the chat template and the generation prompt.
- Every other column (e.g. the reference answer) is passed to each
  reward function as a keyword argument: a list with one value per
  completion, aligned with `completions`.

### Reward functions

```python
def reward(completions, answer, **kwargs) -> list[float]: ...
```

- `completions`: one per sampled completion (`num_generations` per
  prompt). Conversational prompts give `[{"role": "assistant",
  "content": text}]` per completion; string prompts give `text`.
- Keyword arguments also include `prompts`, `completion_ids`,
  `trainer_state`, `log_extra` and `log_metric`; always accept
  `**kwargs`.
- Return one float per completion (`None` skips that function for a
  sample). With several functions, rewards are summed.
- Prefer verifiable rewards (exact match, unit tests, parsers) and keep
  them cheap: they run every step.

### RL config

| Field | Default | Notes |
|-------|---------|-------|
| `model_name_or_path` | `unsloth/Qwen3-4B-Base` | Hub id or local dir (e.g. a merged SFT model) |
| `num_generations` | 8 | Completions per prompt (GRPO group) |
| `per_device_train_batch_size` | 8 | Counted in **completions**. `per_device_train_batch_size × gradient_accumulation_steps` must be a multiple of `num_generations` |
| `gradient_accumulation_steps` | 1 | |
| `learning_rate` | 1e-5 | cosine schedule |
| `lora_r` | 1 | Rank 1 already works for RL |
| `max_seq_length` | 2048 | Without explicit limits: half prompt, half completion |
| `max_completion_length` | `None` | Set it (e.g. 512); long completions dominate memory and time |
| `num_train_epochs` / `max_steps` | 2 / -1 | |
| `temperature`, `top_p`, `top_k`, `min_p` | 1.0, 0.95, 20, 0.0 | Sampling |
| `stop_sequences` | `None` | `None` = the tokenizer's EOS |
| `use_vllm` | `True` | vLLM generation (`rl` extra); colocated on the same GPU |
| `vllm_gpu_memory_utilization` | 0.5 | GPU share reserved for vLLM; lower it if training OOMs |
| `vllm_enable_sleep_mode` | `True` | Frees vLLM memory during optimizer steps |
| `load_in_4bit` | `False` | NF4 QLoRA; **requires `use_vllm=False`** |

`load_in_4bit=True` with `use_vllm=True` raises `ValueError`: TRL syncs
merged weights into vLLM after each step, which corrupts a 4-bit vLLM
copy. Without vLLM, generation runs in transformers and is several times
slower (~35 s/step for Qwen 3.5 2B), but the base model takes about a
quarter of the memory.

### Example: GSM8K with an exact-match reward

Replace the `load_dataset` call with your own data
(`load_dataset("json", data_files="data/rl.jsonl", split="train")`).

```python
import re

import torch
from datasets import load_dataset

from speftr import PERL, PERLConfig

SYSTEM = "Solve the problem. Give the final number on the last line."
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_prompt(row):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["question"]},
        ],
        "target": row["answer"].split("####")[-1].strip(),
    }


def last_number(text):
    numbers = NUMBER.findall(text)
    return float(numbers[-1].replace(",", "")) if numbers else None


def exact_answer(completions, target, **kwargs):
    return [
        1.0 if last_number(completion[0]["content"]) == last_number(gold)
        else 0.0
        for completion, gold in zip(completions, target, strict=True)
    ]


dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(to_prompt, remove_columns=dataset.column_names)

config = PERLConfig(
    model_name_or_path="Qwen/Qwen3.5-2B",
    lora_r=8,
    max_steps=100,
    num_generations=8,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # 16 completions = 2 prompts per step
    max_completion_length=512,
    output_dir="./models/my-rl",
)
perl = PERL(config)
perl.load_model()
perl.train(dataset, [exact_answer])
perl.save_model()  # LoRA adapter + tokenizer in ./models/my-rl

# Colocated vLLM starts a process group; close it to exit cleanly.
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
```

[`examples/rgym`](../examples/rgym/README.md) is a fuller version with
string prompts, a format reward, before/after evaluation and CLI flags.

## 4. SFT then RL

**Recommended:** save the SFT result merged, then start RL from it. vLLM
can load it, so generation stays fast:

```python
sft_config.save_method = "merged_16bit"  # set before sft.save_model()

rl_config = PERLConfig(
    model_name_or_path=sft_config.output_dir,  # the merged SFT model
    max_completion_length=512,
    output_dir="./models/my-sft-rl",
)
```

PERL adds a fresh RL adapter on top of the merged weights.

**In-process:** `set_pretrained_model` hands the SFT model and tokenizer to
PERL, which keeps training the *same* LoRA adapter (PERL's `lora_r` and
`target_modules` are then ignored, and `load_model()` must not be called):

```python
sft = PESFT(sft_config)
sft.load_model()
sft.train(train_dataset, eval_dataset, format_batch)

rl = PERL(PERLConfig(model_name_or_path=sft_config.model_name_or_path,
                     use_vllm=False, output_dir="./models/my-sft-rl"))
rl.set_pretrained_model(sft.model, sft.tokenizer)
rl.train(rl_dataset, [exact_answer])
rl.save_model()
```

Importing Unsloth (which `PESFT` does) patches TRL's trainers for the
whole Python process, so in this path `PERL` and `PEDPO` run
Unsloth-modified trainer code; it has no GPU test in this repo. Prefer
the recommended path above: separate processes, the merged SFT model
passed by path. Keep `use_vllm=False` if the SFT model is 4-bit
(same weight-sync problem as above). Set `stop_sequences` yourself: they
are only filled from the tokenizer in `load_model()`.

## 5. After training

### What `save_model` writes

| `save_method` | Output (in `output_dir`) |
|---------------|--------------------------|
| `"lora"` (default) | `adapter_model.safetensors`, `adapter_config.json` (records the base model id), tokenizer/processor files |
| `"merged_16bit"` | A full 16-bit model with the adapter merged in, plus tokenizer; loads like any Hub checkpoint |

`PESFT` also writes `speftr.json` (its config), `training_args.json`
and `train_metrics.json`: TRL's training metrics (`train_runtime`,
`train_samples_per_second`, `train_steps_per_second`, `train_loss`), the
final evaluation metrics, the step times (`warmup_seconds`, what the
first 5 optimizer steps took, which includes kernel compilation, and
`steady_seconds_per_step`, the median of the later steps), `gpu_name`
and the process's peak GPU memory (`peak_memory_allocated_gib`,
`peak_memory_reserved_gib`, measured since the process started, so
model loading counts).
`PESFT` merges through Unsloth; `PERL` merges with peft's
`merge_and_unload` and supports only these two methods. Checkpoints
(`checkpoint-*`) follow `save_strategy`.

### Load an adapter for inference (transformers + peft)

Load the base with the class named in its config, not
`AutoModelForCausalLM` or `AutoPeftModelForCausalLM`: for Qwen 3.5 they
build a text-only class whose layer names do not match adapters trained
on the full checkpoint, and peft then loads no adapter weights (only a
"missing adapter keys" warning, or `NoMatchingPeftModuleError`).
Every example has a validated script for the paths below
([per-example scripts](#inference-scripts-per-example)).

```python
import torch
import transformers
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer

adapter_dir = "./models/my-sft"
base_id = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
architecture = transformers.AutoConfig.from_pretrained(base_id).architectures[0]
base = getattr(transformers, architecture).from_pretrained(
    base_id, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_dir)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

messages = [{"role": "user", "content": "Recommend a movie."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
    return_dict=True,
).to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
prompt_length = inputs["input_ids"].shape[1]
print(tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True))
```

`uv run python -m examples.chat --model_name_or_path ./models/my-sft` is
an interactive chat (Unsloth, GPU); its defaults are ChatML and a pirate
system prompt, so pass `--chat_template` and `--system_prompt`.

### Merge an adapter yourself

```python
merged = model.merge_and_unload()  # model: the PeftModel above
merged.save_pretrained("./models/my-merged")
tokenizer.save_pretrained("./models/my-merged")
```

Merge into a 16-bit base. If you trained on a pre-quantized
`...-bnb-4bit` repo, load the original 16-bit model as `base` instead.
For multimodal checkpoints also save
`transformers.AutoProcessor.from_pretrained(base_id)` to the same
directory.

### Serve with vLLM

Adapter on the base model (hot-swappable; `--max-lora-rank` must be ≥
your `lora_r`, one of 1, 8, 16, 32, 64, ...). Pass the adapter's chat
template, otherwise vLLM uses the base model's:

```bash
vllm serve Qwen/Qwen3.5-4B --enable-lora --max-lora-rank 16 \
    --lora-modules my-sft=./models/my-sft \
    --chat-template ./models/my-sft/chat_template.jinja \
    --gpu-memory-utilization 0.8 --language-model-only
# OpenAI-compatible API; request "model": "my-sft"
```

- vLLM reserves `--gpu-memory-utilization` of the GPU (default 0.9)
  whatever the model size; lower it to share the GPU.
- `--language-model-only` skips the vision/audio encoders of multimodal
  checkpoints (Qwen 3.5, Gemma 4) for text-only use.
- The base must be the 16-bit model: vLLM with adapters does not start on
  a pre-quantized `...-bnb-4bit` repo.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="Qwen/Qwen3.5-4B", enable_lora=True, max_lora_rank=16)
outputs = llm.chat(
    [{"role": "user", "content": "Recommend a movie."}],
    SamplingParams(max_tokens=128),
    lora_request=LoRARequest("my-sft", 1, "./models/my-sft"),
    chat_template=open("./models/my-sft/chat_template.jinja").read(),
)
print(outputs[0].outputs[0].text)
```

Merged model: `vllm serve ./models/my-merged`, no LoRA flags. Merged
Gemma 4 E2B/E4B models do not load in vLLM (transformers does not save
the weights of their KV-shared layers); serve those as an adapter on the
base.

Query the server; the adapter's name routes through the adapter, the base
model's name answers without it:

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model": "my-sft", "temperature": 0,
         "messages": [{"role": "user", "content": "Recommend a movie."}]}'
```

### Adapter or merged model

| Need | Use |
|------|-----|
| Quick check or evaluation in Python | Adapter on its base (transformers + peft) |
| Several adapters on one base, switched per request | vLLM with several `--lora-modules` |
| Tools that know nothing about LoRA (`pipeline`, other frameworks) | Merged model |
| CPU, laptop, llama.cpp or Ollama | Merged model exported to GGUF ([below](#export-to-gguf-llamacpp-ollama)) |

Greedy outputs of the adapter route, the merged model and vLLM agree on
most but not all prompts: in bf16 two nearly tied tokens can swap, and
the rest of the reply follows. For Gemma 4 E2B, vLLM and transformers
already differ on the base model.

### Export to GGUF (llama.cpp, Ollama)

llama.cpp and Ollama run GGUF files. Export the **merged 16-bit** model:
llama.cpp converts full checkpoints, so an adapter trained on a 4-bit
base is first merged onto the 16-bit original, as the
[inference scripts](#inference-scripts-per-example) and
[Merge an adapter yourself](#merge-an-adapter-yourself) do.
[`examples/export_gguf.sh`](../examples/export_gguf.sh) takes the path of
a llama.cpp checkout (it uses `convert_hf_to_gguf.py` and
`llama-quantize`), converts to a bf16 GGUF (weights, tokenizer and chat
template in one file), quantizes it and prints the commands to run the
result:

```bash
uv run --with gguf examples/export_gguf.sh ./models/my-merged ./gguf \
    ./llama.cpp Q4_K_M
llama.cpp/build/bin/llama-server -m ./gguf/my-merged-Q4_K_M.gguf \
    --jinja -ngl 99 -c 4096 --port 8080
```

- `--jinja` applies the Jinja chat template stored in the GGUF, the
  training template (`chat_template.jinja`) byte for byte. The server
  speaks the OpenAI chat API, like vLLM.
- Only the conversion needs Python: `convert_hf_to_gguf.py` reads the
  checkpoint and tokenizer with transformers and writes with the `gguf`
  package. `uv run --with gguf` runs it in the project environment, which
  has the transformers version the merged model was saved with. Running
  the GGUF (`llama-server`, Ollama) needs no Python.
- Quantization types: `Q8_0` (half the bf16 size) and `Q4_K_M` (under a
  third) are common choices; `llama-quantize --help` lists all. Measure
  your task on the quantized file; the
  [intent example](../examples/intent/README.md#export-to-gguf) lost
  under one accuracy point with `Q4_K_M`.
- Ollama: a `Modelfile` with `FROM ./gguf/my-merged-Q4_K_M.gguf`, then
  `ollama create my-model -f Modelfile`. Ollama may not translate the
  Jinja template: if `ollama show my-model --modelfile` prints
  `TEMPLATE {{ .Prompt }}`, messages reach the model without role
  markers. Write a `TEMPLATE` for the model's format (Granite example in
  the intent README).
- Architecture support is llama.cpp's. Its converter registers the
  architectures of every example model (Gemma 3, Gemma 4, gpt-oss,
  Qwen3, Qwen 3.5, SmolLM3, Granite); only Granite 3.3 2B (intent) was
  converted and run.

### Inference scripts per example

Each training example has an `<example>_inference.py` that imports only
torch, transformers, peft and (for `--engine vllm`) vllm. It runs the
adapter both ways, prints the two outputs per prompt and saves the merged
model to `--merged_dir` (default `<adapter_dir>-merged`):

1. adapter route: the base (loaded with its `architectures` class) plus
   the adapter, via peft or a vLLM `LoRARequest`;
2. merged route: `merge_and_unload()` into the 16-bit base, saved with
   the adapter's tokenizer (and processor for multimodal bases), then
   loaded from disk.

Commands and validated outputs are in each README:
[guard](../examples/guard/README.md#use-the-trained-model-without-speftr),
[instruct](../examples/instruct/README.md#use-the-trained-model-without-speftr),
[intent](../examples/intent/README.md#use-the-trained-model-without-speftr),
[text2sql](../examples/text2sql/README.md#use-the-trained-model-without-speftr),
[rgym](../examples/rgym/README.md#use-the-trained-model-without-speftr),
[prefs](../examples/prefs/README.md#use-the-trained-model-without-speftr)
(also prints the base model's reply),
[big](../examples/big/README.md#use-the-trained-model-without-speftr).
Copy the script next to your adapter and change the prompt builder.

## 6. Choosing a model

Start from an instruction-tuned checkpoint of the right size; RL works
best when the model can already solve the task sometimes.

### Validated models

RTX 3090 24 GB; torch 2.11 (cu130), transformers 5.13.1, TRL 1.13,
vLLM 0.26, unsloth 2026.9.11.

SFT: 20 steps on Dolly-pirate rows, rank 8, `max_seq_length` 1024,
batch 8 unless noted.

| Model | Hub id used | Load | s/step | Peak VRAM |
|-------|-------------|------|--------|-----------|
| Gemma 4 E2B | `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` | 4-bit | 1.4 | 10.3 GB |
| Gemma 4 E2B | `unsloth/gemma-4-E2B-it` | 16-bit | | 12.3 GB |
| Gemma 4 E4B | `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` | 4-bit | 2.1 | 13.8 GB |
| Gemma 4 12B | `google/gemma-4-12B-it` | 4-bit | 3.6 | 10.3 GB |
| Qwen 3.5 4B | `Qwen/Qwen3.5-4B` | 4-bit | | ~5 GB |
| Qwen 3.5 9B | `Qwen/Qwen3.5-9B` | 4-bit | 3.2 | 9.5 GB |
| Qwen 3.8 27B | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | 4-bit, batch 1 × grad-acc 8 | 13.1 | 21.7 GB |
| Gemma 3 270M | `unsloth/gemma-3-270m-it` | guard example | 4.5 it/s | ~5 GB |
| Qwen3 0.6B | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` | 4-bit, instruct example | | |
| gpt-oss-20b | `unsloth/gpt-oss-20b-unsloth-bnb-4bit` | 4-bit, rank 1, gptoss example (batch 4 × grad-acc 4, 2048 tokens) | 15 | 15.4 GiB |

Gemma 4 used `chat_template="gemma-4"`, Qwen 3.5/3.8 `chat_template=None`.

RL: `examples/rgym` on `chain_sum` (rank 1, 16 generations, batch 8 ×
grad-acc 2, 512-token completions), accuracy before → after:

| Model | Setup | Result |
|-------|-------|--------|
| Qwen3 1.7B (`Qwen/Qwen3-1.7B`) | bf16 + vLLM colocate | 27% → 89% in 100 steps |
| Qwen 3.5 2B (`Qwen/Qwen3.5-2B`) | bf16 + vLLM colocate | 23% → 94% in 30 steps |
| Qwen 3.5 2B | 4-bit, no vLLM | 39% → 94% in 30 steps; model 1.8 GB, ~35 s/step |
| Gemma 4 E2B (`unsloth/gemma-4-E2B-it`) | bf16 + vLLM, `vllm_gpu_memory_utilization=0.45`, batch 2 × grad-acc 8 | Runs; 44% → 48% in 30 steps (needs tuning) |

### Chat templates and markers

Rendered by each model's own tokenizer:

| Family | `chat_template` | `instruction_part` | `response_part` |
|--------|-----------------|--------------------|-----------------|
| Qwen 2.5 / 3 / 3.5 / 3.8 (ChatML) | `None` (own) or `"qwen3"` / `"qwen2.5"` | `"<\|im_start\|>user\n"` | `"<\|im_start\|>assistant\n"` |
| Gemma 4 | `"gemma-4"` or `None` | `"<\|turn>user\n"` | `"<\|turn>model\n"` |
| Gemma 2 / 3 / 3n | `"gemma-3"` or `None` | `"<start_of_turn>user\n"` | `"<start_of_turn>model\n"` |
| Llama 3.x | `"llama-3.1"` or `None` | `"<\|start_header_id\|>user<\|end_header_id\|>\n\n"` | `"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n"` |
| gpt-oss (harmony) | `None` | `"<\|start\|>user<\|message\|>"` | `"<\|start\|>assistant"` |

- Unsloth has no named template for Qwen 3.5/3.8; use `None`.
- Qwen 3.5/3.8 templates insert an empty `<think>\n\n</think>\n\n` block
  in assistant turns; it is trained as part of the response. Qwen 3.8
  also prepends a reasoning-effort system prompt.
- Qwen 3.5 (2B, 4B, 9B) ships no `generation_config.json`, and its
  model config's end-of-sequence token is `<|endoftext|>` while the
  template closes a turn with the tokenizer's EOS, `<|im_end|>`. The
  base model emits both; an adapter trained on chat data learns to
  emit only `<|im_end|>`, so transformers' `generate` runs on into a
  new turn unless `eos_token_id` includes `tokenizer.eos_token_id`
  (the `intent` and `tldr` examples pass both ids; vLLM stops at the
  tokenizer's EOS by itself). Qwen 2.5, Qwen 3 and Qwen 3.8 ship a
  generation config listing both tokens and are not affected.
- gpt-oss renders the system message as developer instructions and an
  assistant `thinking` field as the analysis channel; its markers train
  both the analysis and the final channel
  ([examples/gptoss](../examples/gptoss/README.md)).
- Gemma 3 has no system role: the template folds the system message into
  the first user turn.
- Every family in the table is recognised by `speftr.infer_chat_markers`
  (used when `instruction_part` and `response_part` are empty).
- Check yours: `print(tokenizer.apply_chat_template(messages,
  tokenize=False))` and copy the markers from the output.

## 7. Checking the rank

`speftr.lora_budget` counts the exact adapter size (from the model config,
no weights downloaded) and estimates the size your dataset needs, using
the capacity argument from "LoRA Without Regret" (about 2 bits per
parameter; about 1 bit per trained SFT token; about 1 bit per RL
episode or DPO preference pair). Treat it as an order-of-magnitude check.

```text
required_parameters = bits / 2
  sft: bits = trained_tokens * bits_per_token
  rl:  bits = prompts * num_generations * epochs  (or max_steps * batch)
  dpo: bits = pairs * epochs                      (or max_steps * batch)
minimum_rank = ceil(required_parameters / parameters_at_rank_1)
```

PESFT's default rank 8 is a safe over-provisioned choice; the smallest
rank this reports is enough.

```bash
# SFT: your JSONL, response tokens only
uv run python -m speftr.lora_budget \
  --model_name_or_path Qwen/Qwen3.5-4B --dataset data/train.jsonl \
  --responses_only
# Your own formatting function (same one you pass to PESFT.train)
uv run python -m speftr.lora_budget \
  --model_name_or_path Qwen/Qwen3.5-4B --dataset data/train.jsonl \
  --formatting_func my_project.data:format_batch --responses_only \
  --instruction_part $'<|im_start|>user\n' \
  --response_part $'<|im_start|>assistant\n'
# RL: episodes from steps, no dataset needed
uv run python -m speftr.lora_budget --mode rl \
  --model_name_or_path Qwen/Qwen3.5-2B --max_steps 100
```

- Formats detected from columns: `messages`/`conversations`
  (role/content or ShareGPT), `prompt` + `completion`, `text`. Others:
  `--messages_column`, `--prompt_column ... --response_column`,
  `--text_column` or `--formatting_func`.
- Local `.json`, `.jsonl`, `.parquet`, `.csv`, `.tsv`, `.arrow`, `.txt`,
  a `save_to_disk` directory, or a Hub id.
- `--sample_size N` extrapolates from N random rows; `--bits_per_token`
  plugs in your measured loss; `--help` lists everything.
- MoE models (gpt-oss, Qwen3 MoE, Gemma 4 26B-A4B): LoRA on the fused
  expert weights is counted in sft mode, as PESFT (Unsloth) adapts them
  (gpt-oss-20b at rank 1: 11,556,864, of which 11,059,200 on experts),
  and not in rl or dpo mode, as PERL and PEDPO do not.
  `--include_experts true|false` overrides this.

From Python:

```python
from speftr.lora_budget import LoraBudgetConfig, estimate_lora_budget

budget = estimate_lora_budget(
    LoraBudgetConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        dataset="data/train.jsonl",
        responses_only=True,
    )
)
print(budget.trained_tokens, budget.minimum_rank, budget.sufficient)
```

Full formula, formats and limitations: the `speftr/lora_budget.py`
module docstring.

## 8. Offline use

Download once and run without Hub access: see
[running_offline_models.md](running_offline_models.md).

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA out of memory (SFT) | Lower `per_device_train_batch_size`, raise `gradient_accumulation_steps` to keep the effective batch, use `load_in_4bit=True`, shorten `max_seq_length` |
| CUDA out of memory (RL) | Lower `max_completion_length` or batch (keep batch × grad-acc a multiple of `num_generations`), lower `vllm_gpu_memory_utilization`, keep `vllm_enable_sleep_mode=True` |
| Gemma 4 RL OOM | `vllm_gpu_memory_utilization=0.45`, batch 2 × grad-acc 8 |
| `ValueError: load_in_4bit requires use_vllm=False` | 4-bit RL cannot use vLLM; set `use_vllm=False` or load bf16 |
| `generation_batch_size (...) must be divisible by num_generations` | Make `per_device_train_batch_size × gradient_accumulation_steps` a multiple of `num_generations` |
| `ValueError: You have set args.eval_strategy to ... but you didn't pass an eval_dataset` | `eval_strategy="no"` and `save_strategy="no"` (they must match) |
| Chat-template error formatting conversations | The tokenizer has no chat template (base model): set `chat_template` to a named template, or format plain text yourself |
| Formatting function never called | Dataset has a `text` column or `prompt` + `completion` columns; rename them |
| Response-only loss trains nothing / everything | `instruction_part` / `response_part` don't match the rendered template; print one formatted row |
| Fine-tuned model's replies run on into `assistant` / `<think>` text (Qwen 3.5) | The model's `eos_token_id` is `<|endoftext|>`, not the turn's `<|im_end|>`; pass `eos_token_id=[..., tokenizer.eos_token_id]` to `generate` ([markers](#chat-templates-and-markers)) |
| Adapter loads but outputs look like the base model | Loaded with `AutoModelForCausalLM` on a multimodal checkpoint; use the `architectures` class ([above](#load-an-adapter-for-inference-transformers--peft)) |
| SFT hangs before the first step (forked dataset worker stuck) | `PESFT` avoids it by disabling huggingface_hub telemetry; in your own SFT scripts export `HF_HUB_DISABLE_TELEMETRY=1` |
| 401 / gated repo errors | Accept the terms on the Hub page, then `hf auth login` or set `HF_TOKEN` |

More memory levers: [Fitting in memory](#10-fitting-in-memory).

## 10. Fitting in memory

Measured on one RTX 3090 (24 GB), LoRA rank 8, one sequence of 2048
tokens per micro-batch unless noted; peak memory allocated by torch, in
GiB. The config fields named below are also CLI flags (`--<field>`) of
`get_argument_parser()`, except `max_completion_length`. Levers, largest
effect first:

| Lever | Measured effect | `PESFT` | `PERL` |
|-------|-----------------|---------|--------|
| 4-bit base weights (NF4, QLoRA) | Weights ≈ ¼ of bf16; the only lever that fits 12B–31B on 24 GB. Gemma 4 31B: 18 GiB of weights, 19.6 GiB peak | `load_in_4bit` | `load_in_4bit` (needs `use_vllm=False`) |
| Gradient checkpointing | Off: Gemma 4 E4B 4-bit peak 11.3 → 21.0 GiB, 28% faster | `use_gradient_checkpointing`: `"unsloth"` (default, offloads activations to CPU), `"true"`, `"false"` | `use_gradient_checkpointing` (`"true"` default) |
| Tokens per micro-batch | Gemma 4 31B 4-bit: 19.1 / 19.6 / 20.7 GiB at 1024 / 2048 / 4096 tokens | `per_device_train_batch_size`, `max_seq_length` | `per_device_train_batch_size`, `max_completion_length` |
| Full-vocabulary logits | A 262k vocabulary (Gemma) costs ~1 GiB of bf16 logits per 2048 tokens, plus fp32 copies; a non-fused loss raised Gemma 4 E4B 4-bit from 10.6 to 16.3 GiB | Always fused by Unsloth; nothing to set | `use_liger_kernel` (`speftr[liger]` extra) |
| Gradient accumulation | Same peak (±0.07 GiB); keeps the effective batch when the micro-batch shrinks | `gradient_accumulation_steps` | `gradient_accumulation_steps` |
| 8-bit base weights (LLM.int8) | Weights ≈ ½ of bf16, but peak ≈ bf16 (Gemma 4 E4B 16.3 vs 16.0 GiB) and 5–46% slower; helps only when weights dominate | `load_in_8bit` | `load_in_8bit` (needs `use_vllm=False`) |
| Optimizer | ≤ 0.15 GiB between `adamw_torch`, `adamw_8bit`, `paged_adamw_8bit`, `adamw_torch_4bit`: LoRA state is tiny. Paged optimizers only page state tensors of ≥ 100k elements, which LoRA rarely has | `optim` (`adamw_8bit`) | `optim` (`adamw_8bit`) |

Loss functions compute the log-sum-exp over the vocabulary in fp32 for
stability; fused or chunked losses (Unsloth, TRL's default SFT loss,
Liger) do so per chunk and never hold the full logits.

Recommended combinations:

| Situation | `PESFT` | `PERL` |
|-----------|---------|--------|
| One 24 GB GPU | 16-bit up to ~4B; `load_in_4bit` above. From ~12B: batch 1 × `gradient_accumulation_steps`, `max_seq_length` ≤ 4096. Gemma 4 31B is about the limit ([examples/big](../examples/big/README.md)) | bf16 + colocated vLLM for small models; 4-bit without vLLM for larger ones (slow generation); lower batch and `max_completion_length`; Liger for models without final-logit soft-capping |
| Model fits one A100 (40/80 GB) | bf16 LoRA, no quantization; more GPUs: DDP ([below](#11-multiple-gpus)) | bf16 + colocated vLLM; more GPUs: DDP |
| Model does not fit one GPU | `load_in_4bit` first; else `device_map="unsloth_balanced"` over several GPUs | `load_in_4bit` without vLLM; generation is slow |

Limits:

- **Liger and soft-capping.** Liger's fused GRPO loss skips final-logit
  soft-capping (`final_logit_softcapping` in `config.json`, set for
  Gemma 2 and Gemma 4), so its log-probabilities differ from the
  model's on those architectures. Liger also silently skips
  architectures it does not support.
- **FP8 and NVFP4 are not training bases.** FP8 or NVFP4 weight-only
  quantization through torchao trains on Ampere but saves no training
  memory: the weight is dequantized to bf16 and kept for the backward
  pass. ModelOpt NVFP4 checkpoints (e.g. `nvidia/Gemma-4-31B-IT-NVFP4`)
  cannot be loaded by transformers. Use the bf16 original with
  `load_in_4bit`.

### Bigger or newer GPUs

Measured with `examples/speed.py` on the
[intent](../examples/intent/README.md#speed) and
[tldr](../examples/tldr/README.md#speed) examples (Granite 3.3 2B,
Qwen 3.5 4B and Gemma 4 E4B, bf16, batch 16, 40-step runs plus
300-step runs) on one RTX 3090 (24 GB) and one A100 40GB (PCIe). What
holds across the three models and both examples:

- **Same settings, shorter steps.** The A100 runs the recipe defaults
  2.2 to 2.5 times faster per step than the 3090 (intent: 3.49 → 1.60,
  3.55 → 1.52, 5.97 → 2.69 s/step; tldr: 2.67 → 1.14, 3.40 → 1.45,
  3.73 → 1.53), at the same peak memory, loss and score. The 300-step
  intent runs went from 18.7, 18.8 and 29.6 minutes to 8.7, 9.1 and
  14.2.
- **Keep the batch.** Batch 32 doubles the step time and gives the same
  samples per second as batch 16 on both GPUs; the recipe stops at 32
  for LoRA's sake anyway ([README](../README.md#the-recipe)). Gradient
  accumulation is a cost, not a lever: 8 × 2 adds up to 7% per step and
  4 × 4 8 to 43%, so run the largest per-device batch that fits and
  accumulate only for the remainder.
- **Gradient checkpointing stays on** at batch 16: without it these
  2B to 8B models run out of memory at 600-token prompts even on 40 GB
  (turning it off pays at batch 1, see the table above).
- **4-bit costs no speed, and no measured quality.** `load_in_4bit`
  trains within 8% of the bf16 step time on both GPUs with 23 to 38%
  less memory, and the 300-step 4-bit runs score within the bf16 runs'
  spread (intent accuracy 0.80 to 0.86 against 0.82 to 0.86; TL;DR
  ROUGE-1 within 0.01) on both GPUs. On a larger GPU bf16 buys nothing
  measurable here for models of this size.
- **Attention and batching kernels are model-specific.** Flex
  attention is 6 to 8% faster than SDPA for Granite on both GPUs at
  steady state (and recompiles for new sequence lengths, which eats
  most of it over 40 steps), no faster for Gemma and unsupported for
  Qwen 3.5. Padding-free and packing break Granite's training (eval
  loss 1.9 to 4.5 instead of 0.2 to 1.75), leave Qwen 3.5 unchanged,
  and slow Gemma down. Measure before switching either on.
- **Warm-up is the same on both GPUs.** The first steps compile
  kernels: 14 to 90 s with a warm compile cache, up to 2 minutes (12
  with flex attention plus padding-free on Gemma) with a cold one.
  `train_metrics.json` separates it (`warmup_seconds`,
  `steady_seconds_per_step`).
- **Precision.** bf16 is the training dtype on every GPU tested (both
  are Ampere). FP8 and NVFP4 checkpoints are inference formats here
  (Limits above); Hopper and Blackwell GPUs are not measured.
- **Full fine-tuning buys nothing measurable at this size.** Qwen 3.5
  4B on Banking77, 300 steps on one A100 40GB, same seed: full
  fine-tuning (bf16 weights, 8-bit AdamW, learning rate 2e-5), LoRA
  rank 1 and QLoRA rank 1 (2e-4) score 0.856, 0.833 and 0.868
  accuracy, within the spread of repeated runs; full fine-tuning takes
  27% longer per step and 29 GiB against 20 and 15
  ([intent README](../examples/intent/README.md#full-fine-tuning-lora-and-qlora-on-one-a100)).
- **Larger models or more throughput** come from more memory or more
  GPUs, not from different hyperparameters: the combinations table
  above and [Multiple GPUs](#11-multiple-gpus).

## 11. Multiple GPUs

Validated on two A100 40GB: `PESFT` DDP with Gemma 4 31B (4-bit),
model splitting with gpt-oss-120b and `PERL` DDP with Qwen3 1.7B
(`examples/rgym`). `PERL` with a separate vLLM server is implemented
but not validated. `examples/big` and `examples/gptoss` have Slurm
templates ([big](../examples/big/big.sbatch),
[gptoss](../examples/gptoss/gptoss.sbatch)).

**`PESFT`, data parallel (DDP).** One process per GPU, each with a full
model copy, so the model must fit one GPU. Launch your unchanged script:

```bash
torchrun --nproc_per_node 4 my_sft.py
```

Unsloth puts each process on its own GPU. Effective batch =
`per_device_train_batch_size × gradient_accumulation_steps × N`; divide
`gradient_accumulation_steps` by N to keep it. Only rank 0 prints the
parameters and writes files.

Measured with Gemma 4 31B (4-bit) on two A100 40GB
([examples/big](../examples/big/README.md#several-gpus)): per-GPU batch
4 × accumulation 2 (16 rows per step) ran at ~2.6 s per step, one epoch
of Dolly (14,970 rows) in ~40 min, at a peak of 29.7 / 24.0 GB.

**`PESFT`, one model split over GPUs.** For a model too large for one
GPU even in 4-bit: `device_map="unsloth_balanced"`
(`--device_map unsloth_balanced`) in a single process, **not** under
torchrun. It is Unsloth's planner: it reserves room for the output head
and logits on the head's GPU and balances the rest; transformers'
`"balanced"` splits the weights evenly without that reserve. Layers run
one GPU at a time, so it adds memory, not speed.

Measured with gpt-oss-120b (4-bit, LoRA rank 1 on attention and every
expert, 2048 tokens) on two A100 40GB
([examples/gptoss](../examples/gptoss/README.md#gpt-oss-120b-on-two-a100-40gb)):
`unsloth_balanced` put 28 / 28 GiB of weights on the two GPUs
(`"balanced"`: 27 / 33 GB). With micro-batch 8 × accumulation 2 one
epoch of 57 steps trained in 34.7 min (~36 s per step) at a peak of
36.0 / 36.8 GB; `"balanced"` with 4 × 4 took 57.3 min (~60 s per step)
for the same losses. GPU utilization stays at ~17–19% per GPU: Unsloth's
4-bit gpt-oss experts loop over the experts in Python on one CPU core,
and the split layers run one GPU at a time.

**`PERL`, DDP.** `torchrun --nproc_per_node N my_rl.py` (or
`accelerate launch --num_processes N`). Each rank loads the whole model
on its GPU; with colocated vLLM each rank also runs its own vLLM engine,
so `vllm_gpu_memory_utilization` applies per GPU. Validated on
2× A100 40GB with `examples/rgym` (Qwen3 1.7B: 100 steps in 5.8 min,
24.8 GB peak per GPU; [2-GPU run](../examples/rgym/README.md#two-gpus)).

**vLLM.** 4-bit and 8-bit `PERL` runs generate without vLLM. For
generation on separate GPUs, run TRL's vLLM server there
(`CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model <model>`), train on the
others (`CUDA_VISIBLE_DEVICES=0,1,2`) with `vllm_mode="server"`. To serve
a merged model on several GPUs: `vllm serve <dir>
--tensor-parallel-size N`.

## 12. Preference optimization (`PEDPO`, DPO)

Lifecycle: `PEDPO(config)` → `load_model()` (or
`set_pretrained_model(model, tokenizer)`) → `train(train_dataset,
eval_dataset=None, *, resume_from_checkpoint=None)` → `save_model()`.
Plain transformers + peft + TRL's `DPOTrainer`; no Unsloth. Start from an
instruction-tuned (SFT) model with a chat template.

### DPO data contract

A `datasets.Dataset` of preference pairs, in one of two layouts:

- **explicit prompt**: `prompt`, `chosen`, `rejected`. Conversational:
  `prompt` is a message list ending with the user turn, `chosen` and
  `rejected` are `[{"role": "assistant", "content": ...}]`. Standard:
  three strings (EOS is appended to `chosen` and `rejected`).
- **implicit prompt**: only `chosen` and `rejected`, each a full
  conversation that shares the same leading turns. TRL takes their
  longest common prefix as the prompt
  (e.g. `allenai/olmo-2-0425-1b-preference-mix`).

Message lists are rendered with the tokenizer's chat template. Prompt +
completion is truncated at the end to `max_length`; pairs whose
completion is cut off entirely are dropped.

### DPO config

| Field | Default | Notes |
|-------|---------|-------|
| `model_name_or_path` | `allenai/OLMo-2-0425-1B-SFT` | An SFT model; a merged SFT checkpoint dir works |
| `lora_r` / `lora_alpha` | 1 / 32 | A pair carries about 1 bit, like an RL episode ([rank](#7-checking-the-rank)) |
| `learning_rate` | 5e-6 | constant; ~10× the ~5e-7 of full-model DPO (alignment-handbook QLoRA DPO) |
| `beta` | 0.1 | Higher keeps the policy closer to the reference |
| `loss_type` | `["sigmoid"]` | Any TRL DPO loss name(s); several are summed with equal weights |
| `max_length` | 1024 | Prompt + completion tokens |
| `per_device_train_batch_size` | 4 | Counted in **pairs**; each pair runs two sequences |
| `gradient_accumulation_steps` | 4 | 16 pairs per optimizer step |
| `num_train_epochs` / `max_steps` | 1 / -1 | |
| `precompute_ref_log_probs` | `False` | See below |
| `load_in_4bit` / `load_in_8bit` | `False` | QLoRA / LLM.int8 |
| `use_gradient_checkpointing`, `optim`, `use_liger_kernel` | `"true"`, `adamw_8bit`, `False` | As in [Fitting in memory](#10-fitting-in-memory) |
| `eval_strategy` / `save_strategy` | `"no"` / `"epoch"` | `train` evaluates once after training whenever an eval set is given |

`PEDPOConfig.from_args()` / `get_argument_parser()` expose the main
fields as CLI flags.

### Reference model

DPO scores the policy against a frozen reference: the model as it is when
`train` starts. TRL copies the trainable adapter into a frozen adapter
named `"ref"` and computes reference log-probabilities with it, so no
second model is loaded; each step runs one extra forward pass without
gradients.

- After `load_model()` the new adapter is zero, so the reference is the
  base (SFT) model.
- After `set_pretrained_model()` (e.g. the in-process PESFT model) the
  reference is that model, adapter included, and DPO keeps training the
  same adapter. Its adapter must be named `"default"` (peft's name for
  the first adapter).
- `save_model()` writes only the trained adapter, not `"ref"`.

As with [SFT then RL](#4-sft-then-rl), the simplest path from SFT is to
save the SFT model merged and set `model_name_or_path` to it, in a
separate process: importing Unsloth (`PESFT`) patches TRL's
`DPOTrainer` for the whole process.

### Precomputing reference log-probabilities

`precompute_ref_log_probs=True` runs the reference over the whole train
(and eval) set once before training and caches the results, so training
steps run only the policy. With LoRA the reference shares the base
weights, so this saves the per-step reference forward pass, not memory;
for one epoch the total reference compute is the same, for several
epochs it is paid once. `precompute_ref_batch_size` sets the batch of that
pass (no gradients, so it can be larger than the training batch). Not
available with `use_liger_kernel` or streaming (`IterableDataset`) data.

### Rank

```bash
uv run python -m speftr.lora_budget --mode dpo \
  --model_name_or_path allenai/OLMo-2-0425-1B-SFT \
  --dataset allenai/olmo-2-0425-1b-preference-mix
```

`bits = pairs × epochs` (or `max_steps × pairs_per_step`), at 2 bits per
parameter. Rank 1 on OLMo 2 1B is 753,664 parameters, room for ~1.5M
pairs; the whole 378k-pair mix needs ~189k.

### Example

```python
from datasets import load_dataset

from speftr import PEDPO, PEDPOConfig

pairs = load_dataset("allenai/olmo-2-0425-1b-preference-mix", split="train")
pairs = pairs.select_columns(["chosen", "rejected"]).train_test_split(
    test_size=500, seed=0
)

dpo = PEDPO(PEDPOConfig(max_steps=500, output_dir="./models/my-dpo"))
dpo.load_model()
dpo.train(pairs["train"], pairs["test"])
dpo.save_model()  # LoRA adapter + tokenizer in ./models/my-dpo
```

[examples/prefs](../examples/prefs/README.md) runs this on 10,000 pairs
with a held-out split, evaluates by log-likelihood and on RewardBench
against AllenAI's full-model DPO, and measures time and memory with and
without `precompute_ref_log_probs`.

DDP works as for `PERL` (`torchrun --nproc_per_node N my_dpo.py`, each
rank on its own GPU); it is not validated on multi-GPU hardware.

## Examples

[examples/README.md](../examples/README.md) indexes the examples and
explains how to adapt one; each has its own README:
[guard](../examples/guard/README.md),
[instruct](../examples/instruct/README.md) (with `chat.py`),
[intent](../examples/intent/README.md),
[text2sql](../examples/text2sql/README.md),
[rgym](../examples/rgym/README.md),
[prefs](../examples/prefs/README.md),
[big](../examples/big/README.md),
[gptoss](../examples/gptoss/README.md). Each also has an
`<example>_inference.py` that uses the trained model without speftr
([After training](#5-after-training)).
