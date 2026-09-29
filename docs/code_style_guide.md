# Code Style Guide

`speftr` is a starter kit: people read it to learn the recipe and copy files
or snippets into their own projects. Code must therefore be clear, explicit
and self-contained before it is clever or short.

Read this together with [commenting_guide.md](commenting_guide.md),
[typing_guide.md](typing_guide.md) and [tests_guide.md](tests_guide.md).
Commands are in [development.md](development.md).

## Linting is mandatory

`ruff` runs with `select = ["ALL"]`, a short commented ignore list, Google
docstrings and a line length of **79**. Fix the code rather than silencing
the rule. Do not widen the ignore list or add blanket `# noqa`.

A targeted suppression is allowed when the pattern is intentional, with the
code and, when not self-evident, a reason:

```python
# Heavy optional dependency: imported at use time, not at package import.
from trl import GRPOTrainer  # noqa: PLC0415
```

## Structure

Code should read top to bottom like the steps of the recipe.

- Keep functions small and focused. Split when a function mixes abstraction
  levels, nests three or more levels deep, or passes ~30 lines of logic
  without telling one coherent story.
- Do not split a linear flow just to shorten it. A 40-line `train()` that
  reads as load, configure, train, save is better than four helpers the
  reader has to jump between.
- Keep each module self-contained so it can be copied on its own. Do not add
  shared utility modules or abstractions for code used once.
- Respect `max-args = 5`: group related settings into the existing config
  dataclasses instead of adding positional parameters.

## Project conventions

- `from __future__ import annotations` at the top of every module.
- Heavy or optional libraries (`datasets`, `peft`, `transformers`, `trl`,
  `vllm`) are imported under `TYPE_CHECKING` for annotations and inside the
  function that uses them at run time (`# noqa: PLC0415`).
- **Unsloth import order matters.** In SFT code and examples,
  `import unsloth` must run before `transformers`, `trl` or `peft` so its
  patches apply; keep `# noqa: I001` where ruff would reorder. Never import
  Unsloth from `perl.py`, `pedpo.py` or at `speftr` import time.
- Configs are dataclasses with `from_args()` and `get_argument_parser()`.
  A new field goes into the dataclass, its docstring `Attributes`, the
  parser (if exposed on the CLI) and the call that forwards it to TRL.
- Defaults encode the recipe in `README.md`. Do not change one without
  updating the README and the matching docstring and `help=` text.
- Output: `print` in the library and most examples; `rgym.py` uses
  `logging`. Match the file you are in.

## Naming

Names say what a thing is or does: `load_model`, `max_prompt_length`,
`reward_funcs`. Avoid abbreviations unless they are standard in the domain
(`lr`, `lora_r`, `idx`, `cfg`). Single letters only for trivial loop indices.

Make helpers private (`_name`) only when there is a reason to keep them out
of the public API, not by default.

## Branching and expressions

Name the parts of a complex condition instead of packing them into one line:

```python
uses_steps = self.config.save_strategy == "steps"
missing_steps = self.config.save_steps is None
if uses_steps and missing_steps:
    raise ValueError("save_steps is required when save_strategy='steps'")
```

Prefer a plain loop or a two-step comprehension to a dense one-liner with
nested conditions.

## Exceptions

This is library code called from scripts. Never use a bare `except` or
`except Exception` to keep going.

1. Catch only the specific exception that lets the code continue naturally
   (for example `FileNotFoundError` for an optional file).
2. Recover only when there is a sensible fallback, and say what happened.
3. Otherwise let it raise.

Validate user-facing configuration early with a clear message rather than
letting a deep TRL or CUDA error surface later.

## Docstrings

Every module, class and function, including private ones, has a Google-style
docstring (`interrogate` requires 100%, `pydoclint` checks consistency).
Type hints live in signatures, not in `Args`. Functions returning `None`
still have a `Returns:` section. What to write in them is in
[commenting_guide.md](commenting_guide.md).

## Guiding principle

Prefer code that is clear, explicit, easy to copy and well tested over code
that is clever, dense or generic.
