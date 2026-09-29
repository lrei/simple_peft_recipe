# Tests Guide

Tests protect behaviour that users and copied snippets rely on: config
defaults that encode the recipe, argument parsing, length calculations,
parameter serialization, reward functions, label extraction, and whether
training runs and saves adapters. How to run them is in
[development.md](development.md).

## Layout

```text
tests/
  conftest.py            --run-cuda option, shared fixtures
  fixtures/              small JSON excerpts of real datasets
  test_<module>.py       CPU unit tests for speftr/<module>.py
  test_<module>_cuda.py  GPU smoke tests for the same module
  test_examples_<x>.py   tests for examples/<x>/
```

Name tests after the behaviour they check:
`test_parser_rejects_invalid_choice`, not `test_parser_3`.

## Two kinds of tests

### CPU unit tests (default)

Everything that does not need a GPU: config dataclasses, `from_args`,
parser defaults and choices, max-length calculations, JSON serialization,
reward functions, prompt formatting, label parsing. They must be fast and
run in `uv run poe test`.

A real tokenizer can run on CPU; use the session-scoped `qwen_tokenizer`
fixture rather than a fake when token counts or chat templates matter.

### GPU smoke tests (`cuda` marker)

A few training steps on a tiny model and real data, then assert on what
was saved (adapter files, `speftr.json`, `training_args.json`) and that the
adapter reloads. They prove the wiring against the real TRL, peft, Unsloth
and vLLM versions; they do not check model quality.

```python
@pytest.mark.cuda
def test_pesft_trains_and_saves_lora_adapters(tmp_path): ...
```

`conftest.py` skips `cuda` tests unless `--run-cuda` is passed
(`uv run poe test-cuda`). They need an NVIDIA GPU and network access for
model and dataset downloads. Keep them small enough for one RTX 3090.

Code that imports Unsloth runs in a child interpreter (see
`tests/test_pesft_cuda.py`) so `import unsloth` runs before
`transformers` and its patches do not leak into other tests.

## Real data

Test inputs come from real data. Fixtures under `tests/fixtures/` are small
JSON excerpts of real public datasets (for example
`dolly_pirate_rows.json`), loaded with the `load_fixture` fixture. The
repository is public: take rows only from ungated datasets whose license
allows redistribution, and name the source and license in the test
module's docstring.

- Never invent data shaped like what you think the dataset looks like. Take
  a few real rows, or ask the maintainer if you cannot get them.
- Keep fixtures small (a handful of rows), human-readable and committed.
- Record where the rows came from (dataset name and split) in the test
  module docstring or next to the fixture.
- Synthetic values are fine for narrow edge cases the real data cannot show
  (an empty string, a boundary length), but not as the only input.

## Test through public behaviour

Assert on outputs a user would see: returned values, parsed configs, files
written, exceptions raised with their message. Avoid asserting on private
call sequences or mocking TRL internals. If a private helper has logic
worth testing directly (for example `_parse_max_seq_length`), test its
inputs and outputs, not how it is used.

When a default is part of the recipe, pin it in a test so an accidental
change fails loudly.

## Property tests

Use `hypothesis` when a rule must hold across many inputs: length
calculations stay within bounds, serialization round-trips, a reward is
always in its documented range. Keep strategies simple and bounded;
use `@settings(max_examples=...)` if a test is slow. Use plain examples
when the expected output is a specific value.

## Bugs and regressions

When fixing a bug, first write a test that fails on the old behaviour, then
make it pass. The test name and docstring describe the behaviour, not the
bug report.

## Running efficiently

- `uv run poe test` uses pytest-xdist (`-n auto --dist=loadscope`) and
  testmon, so only tests affected by your change run.
- `uv run poe test-all` runs everything without testmon.
- `uv run poe test-cuda` runs serially (`-n 0`): GPU tests must not share
  the card.
- Do not set a default `-m` expression in config; it disables testmon's
  selection. That is why `conftest.py` skips `cuda` tests instead.

## Rules

- Do not skip a test because a fixture is missing; create one from real
  data.
- Do not weaken an assertion or delete a test to make a run green.
- Tests must be deterministic: fixed seeds, local fixtures, no dependence
  on network output for CPU tests.
- No machine-specific paths; use `tmp_path` for outputs.
