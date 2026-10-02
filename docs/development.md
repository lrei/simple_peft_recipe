# Development Guide

Setup, tooling and the day-to-day workflow for `speftr`. For what the code
should look like, see the other guides listed at the end.

## Prerequisites

- **uv**, the package manager: <https://astral.sh/uv>.
- **Python 3.13 or 3.14** (`requires-python` in `pyproject.toml`).
- **Linux with an NVIDIA GPU** for anything that trains or runs a model. The
  defaults are sized for a 24 GB GPU; tested on RTX 3090 and A100.
  Unit tests run on CPU.
- `HF_TOKEN` in the environment for gated Hugging Face models and datasets
  (for example WildGuardMix in `examples/guard/`).

## Setup

The environment is large (torch, Unsloth, vLLM). Sync only what you need:

```bash
uv sync --group dev                        # SFT (PESFT) + dev tools
uv sync --group dev --extra rl             # + TRL GRPO / vLLM (PERL)
uv sync --group dev --extra gym            # + reasoning-gym (examples/rgym)
```

`--extra gym` implies `rl`. Type checking with pyright resolves imports from
the synced environment, so run it with the extras your change touches.

Change dependencies with `uv add` / `uv lock`; never hand-edit `uv.lock`.

Install the git hooks once per clone:

```bash
uv run pre-commit install
```

This installs the `pre-commit` hook (ruff format, ruff check --fix, mypy,
pymarkdown, YAML/TOML checks, whitespace and large-file checks) and the
`commit-msg` hook, which strips agent trailers from commit messages.
Pyright, tests and the slower checks are not in the hooks; run them via
`poe`.

## Tasks

All tasks are defined in `[tool.poe.tasks]` in `pyproject.toml` and run with
`uv run poe <task>`; `uv run poe` lists them.

| Task | What it runs |
|------|--------------|
| `lint` | `ruff format`, then `ruff check --fix` on `speftr examples tests scripts` |
| `typecheck` | mypy on `speftr`, then pyright (basic mode) |
| `pydoclint` | Docstring/signature consistency (Google style) |
| `pydoc-coverage` | interrogate; docstring coverage must be 100% |
| `test` | Unit tests, parallel (xdist) and incremental (testmon) |
| `test-all` | Unit tests, full run |
| `test-cuda` | Unit + GPU smoke tests (`--run-cuda`, needs CUDA and network) |
| `coverage` | Unit tests with branch coverage |
| `duplicates`, `circular` | pyscn clone and import-cycle checks |
| `cognitive` | complexipy (max 15 per function) |
| `complexity`, `maintainability` | radon, reports grade C or worse |
| `deadcode`, `security` | vulture, bandit |
| `audit` | pip-audit; informational, not part of `quality` |
| `markdownlint` | pymarkdown over the repository |
| `quality-iter` | lint, pydoclint, pydoc-coverage, typecheck, test, duplicates, circular, cognitive, markdownlint |
| `quality` | `quality-iter`, then `quality-extra` and `coverage` |

## Workflow

### While editing a file

Run the cheap checks on the file you touched, in this order, and fix each
before moving to the next:

```bash
uv run ruff format path/to/file.py
uv run ruff check --fix path/to/file.py
uv run pydoclint --quiet path/to/file.py     # speftr/ only
uv run mypy path/to/file.py                  # speftr/ only
uv run radon cc path/to/file.py -s -n C      # no output means OK
```

Formatting first means the linter never reports issues the formatter would
have fixed. Line length is 79 and `ruff` selects `ALL` rules; see
[code_style_guide.md](code_style_guide.md) for how to deal with findings.

### Running tests

```bash
uv run poe test                                  # fast unit loop
uv run pytest tests/test_pesft.py -q             # one module
uv run poe test-cuda                             # also GPU smoke tests
uv run pytest --run-cuda tests/test_perl_cuda.py # one GPU test module
```

Tests marked `@pytest.mark.cuda` are skipped unless `--run-cuda` is passed.
They need an NVIDIA GPU and download real models and datasets, so run them
when you change training, model loading, saving, or an example's training
path. Testmon selects only tests affected by your changes; use
`poe test-all` when you suspect its cache is stale. Details are in
[tests_guide.md](tests_guide.md).

### Before declaring done or committing

```bash
uv run poe quality-iter
```

It must be green. A failure that existed before your change is not an
excuse: fix it, or, if the fix is out of scope or needs a decision, stop and
escalate to the maintainer with the exact failing output. Do not widen
ignore lists, add blanket `noqa`, or skip tests to get a clean run.

Run `uv run poe quality` before a release or a larger change; it adds dead
code, security and coverage. `uv run poe audit` reports known
vulnerabilities in dependencies; it is not a gate because the pinned stack
(`[tool.uv]` in `pyproject.toml`) carries advisories that cannot be fixed
without leaving it.

## Where decisions go

Explain *why* the code is the way it is in comments only when the reason is
needed to read the code (see [commenting_guide.md](commenting_guide.md)).
Longer rationale, history and trade-offs go in a decision record under
`docs/adrs/` (local, git-ignored), in the commit message, or in the PR
description.

## Other guides

- [code_style_guide.md](code_style_guide.md): structure, naming, errors.
- [commenting_guide.md](commenting_guide.md): docstrings and comments.
- [typing_guide.md](typing_guide.md): type hints, mypy and pyright.
- [tests_guide.md](tests_guide.md): what and how to test.
