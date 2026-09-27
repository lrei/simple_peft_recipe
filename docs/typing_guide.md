# Typing Guide

Type hints exist to make the code easier to read and to catch real bugs.
They are not a goal in themselves: never make code harder to read or copy
just to satisfy a checker.

## Checkers

`uv run poe typecheck` runs both, on `speftr/`:

- **mypy**: non-strict, but checks untyped bodies, `Optional` handling,
  returning `Any` and unused or redundant ignores and casts. Third-party
  imports are not followed.
- **pyright** in `basic` mode: resolves imports from the synced `.venv`
  (so sync the extras your code imports), treats optional member access,
  subscript and call as errors, and warns on unnecessary casts.

Both must be clean. Pre-commit runs mypy only; run pyright yourself.

## Rules

- Annotate every public function's parameters and return type. Use
  `X | None`, never implicit optional.
- Use built-in generics (`list[str]`, `dict[str, Any]`) and simple, concrete
  types. Add a `type` alias when a type is long or repeated:
  `type SerializedParameters = dict[str, object]`.
- Do not introduce `TypeVar`, generics, `@overload` or deep nested types
  unless they make a real API clearer.
- Do not design code around the type system or refactor only to please it.

## Heavy libraries and `TYPE_CHECKING`

`datasets`, `peft`, `transformers`, `trl` and `vllm` are slow to import, and
some are optional extras. Import them for annotations only:

```python
from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
```

and import the runtime objects inside the function that needs them. With
`from __future__ import annotations` the names are never evaluated at run
time.

## `Any` at boundaries

Hugging Face, TRL and Unsloth objects are often loosely typed. Accept `Any`
at those boundaries (dataset rows, `**kwargs` forwarded to TRL, model
outputs) and convert or validate close to where the value enters. Do not
let `Any` spread through the rest of the code, and do not replace an honest
`Any` with an elaborate fake type.

When code only relies on a few attributes of a duck-typed third-party
object, a small `Protocol` is acceptable if it documents what is needed.

## `cast`

Use `cast` only when you know more than the checker, for example when a
library function is annotated with a base class but returns a concrete one.
Always use the string form so the target type can stay under
`TYPE_CHECKING`:

```python
trainer = cast("SFTTrainer", train_on_responses_only(trainer, ...))
```

Do not use `cast` to hide a real type error; fix the code instead.

## Ignores

Prefer, in order: fix the code, simplify the annotation, use `Any` at the
boundary. If a diagnostic is still wrong (typically an incomplete library
stub), use a targeted ignore with the rule code and a reason:

```python
# None is only read when save_strategy="steps", where HF validates it.
save_steps=self.config.save_steps,  # pyright: ignore[reportArgumentType]
```

Never use a bare `# type: ignore`, and never silence a checker through its
config to get a clean run. mypy reports unused ignores, so remove them when
the underlying issue goes away.

## Import from the defining module

If mypy says a module "does not explicitly export" a name, import it from
the module that defines it rather than from a re-export (for example
`huggingface_hub.errors` rather than `huggingface_hub.utils`).

## Tests and examples

Tests are type-checked loosely (`check_untyped_defs` is off for `tests.*`
and ruff's `ANN` rules are ignored there). Examples follow the library's
rules but may use `Any` more freely around dataset rows.

## Rule of thumb

If a type makes the code worse, simplify it or remove it.
