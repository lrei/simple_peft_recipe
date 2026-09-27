# Examples

Each example uses `speftr` on a public dataset and is meant to be read
and copied: it shows how to point `PESFT` (SFT) or `PERL` (GRPO) at data,
a chat template and a reward. They are teaching code, not tuned
state-of-the-art models. The library itself is documented in the
[user guide](../docs/guide.md).

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
| `intent/` | `PESFT` multi-class classification (77 intents) with before/after evaluation on Banking77 | Granite 3.3 2B, bf16 | 11 GB | [README](intent/README.md) |
| `text2sql/` | `PESFT` 4-bit SFT then `PERL` GRPO on the same adapters with a SQLite execution reward | SmolLM3-3B, 4-bit | ≤ 6 GB | [README](text2sql/README.md) |
| `rgym/` | `PERL` GRPO with verifiable rewards on Reasoning Gym tasks, colocated vLLM, before/after accuracy | Qwen3 1.7B, bf16 | RTX 3090 24 GB (`gym` extra) | [README](rgym/README.md) |

Hardware is the measured peak VRAM where one was recorded; timings and
results are in each README.

Every training example has an `<example>_inference.py` (for example
`guard/guard_inference.py`) that uses the trained model **without speftr
or Unsloth**: it runs the LoRA adapter on its base model and the merged
model, with transformers + peft or vLLM, and prints both outputs. Each
README's "Use the trained model without speftr" section has the
commands; the [user guide](../docs/guide.md#5-after-training) covers the
general rules.

## Reading and adapting an example

Every training script has the same shape:

1. **Config**: a `build_config` / `_build_config` function fills a
   `PESFTConfig` or `PERLConfig`. Recipe defaults stay unless the task
   needs otherwise; look here for the chat template, response-only
   markers and batch size.
2. **Data**: a loader returns `datasets.Dataset` objects. Swap it for
   `load_dataset("json", data_files=...)` or your own source.
3. **Formatting** (SFT) or **rewards** (RL): the one function that is
   really about your task. SFT examples render each row with the chat
   template; RL examples build a `prompt` column and score completions.
4. **Lifecycle**: `load_model()` → `train(...)` → `save_model()`.

To adapt one, pick the example closest to your task (classification:
`guard` or `intent`; free-form answers: `instruct`; verifiable outputs:
`text2sql` or `rgym`), then change the loader, the formatting or reward
function and the model's chat-template markers. Print one formatted row
before training: it is the fastest way to catch wrong markers. Each
README ends with the concrete steps and pitfalls for that example.
