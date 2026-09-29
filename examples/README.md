# Examples

Each example uses `speftr` on a public dataset and is meant to be read
and copied: it shows how to point `PESFT` (SFT), `PERL` (GRPO) or
`PEDPO` (DPO) at data, a chat template and a reward or preference pairs.
They are teaching code; their hyperparameters are not tuned for the best
score. The library itself is
documented in the [user guide](../docs/guide.md).

Run every script from the repo root as a module, e.g.
`uv run python -m examples.guard.guard_train --help`. All of them need an
NVIDIA GPU: the SFT scripts import Unsloth, which refuses to import
without one.

## Index

| Example | Demonstrates | Model | Hardware | Docs |
|---------|--------------|-------|----------|------|
| `guard/` | `PESFT` as a binary text classifier (label as the assistant turn, response-only loss) on WildGuardMix (gated); eval, console tester, token counts | Gemma 3 270M, 16-bit | ~5 GB | [README](guard/README.md) |
| `instruct/` | `PESFT` chat instruction tuning from instruction/context/response columns with a system prompt (Dolly pirate) | Qwen3 0.6B, 4-bit | small; any 24 GB GPU | [README](instruct/README.md) |
| `chat.py` | Console chat with an adapter or merged model saved by `PESFT` (Unsloth inference) | any | GPU | [instruct README](instruct/README.md#run) |
| `export_gguf.sh` | Converts a merged model to GGUF and quantizes it for llama.cpp / Ollama (needs a llama.cpp checkout) | any merged model | CPU | [guide](../docs/guide.md#export-to-gguf-llamacpp-ollama) |
| `intent/` | `PESFT` multi-class classification (77 intents) with before/after evaluation on Banking77 | Granite 3.3 2B, bf16 | 11 GB | [README](intent/README.md) |
| `text2sql/` | `PESFT` 4-bit SFT then `PERL` GRPO on the same adapters with a SQLite execution reward | SmolLM3-3B, 4-bit | ≤ 9 GB | [README](text2sql/README.md) |
| `rgym/` | `PERL` GRPO with verifiable rewards on Reasoning Gym tasks, colocated vLLM, before/after accuracy | Qwen3 1.7B, bf16 | RTX 3090 24 GB (`gym` extra) | [README](rgym/README.md) |
| `prefs/` | `PEDPO` rank-1 LoRA DPO of an SFT model on its preference mix; held-out preference accuracy and RewardBench implicit-reward accuracy vs AllenAI's full DPO model, no judge | OLMo 2 1B SFT, bf16 | 12.7 GB | [README](prefs/README.md) |
| `big/` | `PESFT` on a model far larger than the GPU: 4-bit, batch 1 × grad-acc; DDP (`torchrun`), model splitting (`--device_map unsloth_balanced`), Slurm template (Dolly pirate) | Gemma 4 31B, 4-bit | RTX 3090 24 GB; DDP: 2× A100 40GB | [README](big/README.md) |
| `gptoss/` | `PESFT` on an MoE reasoning model: harmony template, response-only loss on reasoning + answer, LoRA on every expert; eval loss and reasoning-language compliance base vs adapter; 120b split over GPUs (`--device_map unsloth_balanced`), Slurm template (Multilingual-Thinking) | gpt-oss 20b / 120b, 4-bit | 20b: 15.4 GiB (RTX 3090); 120b: 2× A100 40GB | [README](gptoss/README.md) |

Hardware is the measured peak VRAM where one was recorded; timings and
results are in each README.

Every training example has an `<example>_inference.py` (for example
`guard/guard_inference.py`) that uses the trained model without speftr
or Unsloth: it runs the LoRA adapter on its base model and the merged
model, with transformers + peft or vLLM, and prints both outputs. Each
README's "Use the trained model without speftr" section has the
commands; the [user guide](../docs/guide.md#5-after-training) covers the
general rules.

## Reading and adapting an example

Every training script has the same shape:

1. **Config**: a `build_config` / `_build_config` function fills a
   `PESFTConfig`, `PERLConfig` or `PEDPOConfig`. Recipe defaults stay
   unless the task needs otherwise; look here for the chat template,
   response-only markers and batch size.
2. **Data**: a loader returns `datasets.Dataset` objects. Swap it for
   `load_dataset("json", data_files=...)` or your own source.
3. **Formatting** (SFT), **rewards** (RL) or **pairs** (DPO): the
   task-specific part. SFT examples render each row with the chat
   template; RL examples build a `prompt` column and score completions;
   the DPO example passes `chosen`/`rejected` conversations as they are.
4. **Lifecycle**: `load_model()` → `train(...)` → `save_model()`.

To adapt one, pick the example closest to your task (classification:
`guard` or `intent`; free-form answers: `instruct`; verifiable outputs:
`text2sql` or `rgym`; preference pairs: `prefs`), then change the
loader, the formatting or reward function and the model's chat-template
markers. Print one formatted row before training to catch wrong
markers. Each README ends with the concrete steps and pitfalls for that example.

The LoRA rank is chosen with
[`speftr.lora_budget`](../docs/guide.md#7-checking-the-rank), which
compares the adapter's parameters with what the training data needs;
each README has the command for its data (SFT) or run length (RL),
to rerun for yours.
