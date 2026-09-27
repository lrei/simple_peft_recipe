# Commenting Guide

What to say in docstrings and comments. Docstring *format* (Google style,
sections) is enforced by ruff, pydoclint and interrogate; this guide covers
the content they cannot check.

The rule: **every comment must be useful to someone reading the current
code.** If it does not help them understand or safely change the code, it
does not belong in the code.

## Module docstrings

Start with a one-line summary, then cover what applies:

- **Scope**: what the module owns and what it leaves to TRL, peft or
  Unsloth.
- **Entry points**: the classes or functions a caller uses.
- **Effects**: non-obvious I/O, downloads, GPU memory use, global patches
  (for example Unsloth patching TRL on import).
- **Sources**: the paper or documentation a recipe default comes from.
- **Example**: a short usage snippet when the call sequence is not obvious.

Do not list imports or repeat signatures.

## Class docstrings

State the class's role, the state it owns and its lifecycle, for example
`PERL(config)` then `load_model()`, `train(...)`, `save_model()`. Call out
delayed initialization (attributes that are `None` until a method runs) and
invariants callers must respect. Config dataclasses document every field in
`Attributes`, including why a default was chosen when it encodes the recipe.

## Function docstrings

Beyond `Args` and `Returns`, document the contract:

- preconditions (for example "call `load_model()` first");
- what is mutated or written to disk;
- `Raises` for errors callers are expected to handle;
- an `Example` when the call shape is not obvious.

Do not restate type hints. A docstring that only says "The tokenizer." for a
`tokenizer` parameter adds nothing; say what it must be (for example "must
have a chat template").

## Inline comments

Write a comment when the reader would otherwise have to ask "why?":

- a non-obvious choice or constraint ("constant LR: the recipe finds decay
  unnecessary for LoRA");
- an ordering requirement ("unsloth must be imported before transformers");
- a workaround for library behaviour, stated as the behaviour, not as a
  history of how it was found;
- a unit, shape or invariant the next line depends on.

For a long function with distinct phases, a short heading comment per phase
(`# Build the trainer.`) helps scanning. Do not comment every line.

## Anti-patterns

Do not write comments that:

- **Narrate history or changes**: "changed X to Y", "previously we ...",
  "now uses ...", "new implementation", "fixed bug where ...". The code
  shows its current state; history lives in git.
- **Keep version or upstream bookkeeping**: library version pins or ranges,
  issue and PR numbers, "until upstream fixes this", "drop this once
  vX ships", "TODO after release".
- **Record investigation notes or test results**: "tried A, B was faster",
  "verified on a 3090 that ...", "this passed in run 12".
- **Restate the code or a name**: `# Load the model` above `load_model()`,
  `# Increment counter` above `count += 1`.
- Drift from the code they describe. Update or delete them with the code.

That rationale is often valuable; it just does not belong next to the code.
Put it in a decision record under `docs/adrs/`, the commit message, or the PR
description.

Bad:

```python
# Changed from 0.5 to 0.2 after OOM on 3090 (see #42). vLLM 0.26 still
# over-allocates, drop this once upstream fixes it. Tested: works now.
vllm_gpu_memory_utilization: float = 0.2
```

Good:

```python
# Colocated vLLM shares the GPU with training; 0.2 leaves room for the
# policy model, LoRA gradients and optimizer state on a 24 GB card.
vllm_gpu_memory_utilization: float = 0.2
```

The good version explains why the value is what it is today. The OOM
investigation, the issue link and the upstream plan go in an ADR or commit
message.

## Example: module and class

```python
"""GRPO fine-tuning with LoRA adapters via TRL.

Owns model loading, LoRA setup and the GRPOTrainer run. Does not define
rewards or datasets; callers pass them to ``PERL.train``. Uses plain
transformers/peft (no Unsloth) and optionally a colocated vLLM engine.

Entry points:
    ``PERLConfig``: recipe defaults, also buildable from CLI args.
    ``PERL``: load, train and save adapters.
"""


class PERL:
    """Train LoRA adapters on a causal LM with GRPO.

    ``model`` and ``tokenizer`` are ``None`` until ``load_model`` runs;
    ``train`` and ``save_model`` require them.
    """
```
