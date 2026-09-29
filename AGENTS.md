# AGENTS.md

Guidance for coding agents working in this repository.

## Comments: useful or absent

Code comments explain the code as it is now: *why* non-obvious code exists,
and *what* it does when the syntax or library call is unusual. Do not write
comments that:

- narrate history or changes ("changed X to Y", "previously", "now", "new");
- track versions or upstream state (library pins, issue/PR numbers,
  "until upstream fixes", "drop this once ...");
- record investigation notes, test results or benchmark numbers;
- restate what the code or a good name already says.

That rationale goes in a local decision record (`docs/adrs/`, git-ignored),
the commit message or the PR description. Details: `docs/commenting_guide.md`.

## What this is

`speftr` ("simple PEFT recipe") is a small **drop-in library**. It packages
a tested LoRA fine-tuning recipe (`README.md`, "The Recipe") as thin
wrappers around Hugging Face TRL that users import into their own projects
and point at their own models and datasets. Defaults target a **single
consumer GPU (RTX 3090, 24 GB)**. Users also copy modules or snippets, so
each file must stay self-contained and readable.

- **`speftr/` is the product.** It must work for arbitrary user models and
  datasets, not just the ones the examples use. Its public API (classes,
  functions, CLI flags) is documented for library users: what it does,
  accepted inputs, defaults and limitations, with usage examples.
- **`examples/` are only examples.** They show how to use the library on
  specific public datasets (WildGuardMix, Dolly, reasoning-gym). Never put
  reusable functionality there, never make library code depend on them,
  and never tune library behaviour to fit one example's data. Reusable
  tooling an example needs belongs in `speftr/`.

## Layout

```text
speftr/
  __init__.py   Public API. PERL imported eagerly; PESFT lazily via
                __getattr__ so PERL users never import unsloth.
  pesft.py      PESFTConfig + PESFT: supervised fine-tuning (Unsloth +
                trl.SFTTrainer), display/save_parameters helpers.
  perl.py       PERLConfig + PERL: GRPO RL (transformers/peft +
                trl.GRPOTrainer, optional colocated vLLM). No Unsloth.
  lora_budget.py  LoRA adapter size vs. the parameters a dataset needs
                ("LoRA Without Regret"); CLI `python -m speftr.lora_budget`
                and a Python API. No Unsloth.
examples/       Usage examples only, runnable as modules
                (`python -m examples.<pkg>.<script>`).
  README.md               Index; each example has its own README.md.
  chat.py                 Interactive chat with a trained model.
  guard/                  SFT safety classifier on WildGuardMix (gated).
  instruct/               SFT instruction tuning (Dolly pirate dataset).
  intent/                 SFT intent classification on Banking77.
  text2sql/               SFT then GRPO with a SQLite execution reward.
  rgym/                   GRPO on reasoning-gym tasks via PERL.
  big/                    4-bit SFT of Gemma 4 31B on one 24 GB GPU;
                          multi-GPU (torchrun, device_map) and Slurm.
  gptoss/                 4-bit SFT of gpt-oss 20b/120b (reasoning language).
                Each training example has `<name>_inference.py`: the
                trained model without speftr (adapter and merged; transformers
                and vLLM). Examples are self-contained; small helpers are
                copied between them on purpose.
tests/          pytest; `cuda`-marked GPU smoke tests; fixtures/ (real data).
docs/           Guides (development, code style, commenting, typing, tests,
                running offline),
                local, git-ignored decision records (adrs/) and open
                problem records (ipr/).
scripts/        Standalone tooling (git hooks).
```

Each `XConfig` is a dataclass with `from_args()` and `get_argument_parser()`.
Trainer lifecycle: `X(config)` → `load_model()` → `train(...)` →
`save_model()`. `PERL.set_pretrained_model()` continues training adapters
from a previous stage.

## Environment

- Python 3.13 or 3.14, managed with `uv`. `uv.lock` is committed; change
  dependencies with `uv add` / `uv lock`, never by hand.
- Linux + NVIDIA only. torch comes from the CUDA 13 index (driver ≥580).
- `[tool.uv] override-dependencies` deliberately exceeds unsloth's declared
  caps (see the comment there); run the SFT and RL GPU smoke tests after
  any version change.
- Setup: `uv sync --all-extras --group dev`.
  - extra `rl`: vLLM (PERL generation); extra `gym`: reasoning-gym (+ rl).
- The environment is large (torch, vLLM, unsloth); don't re-sync casually.
- Unsloth refuses to import without a GPU; SFT code and the guard/instruct
  examples cannot be imported on a CPU-only machine.
- Gated HF datasets/models use the cached `hf auth login` token or
  `HF_TOKEN`.

## Quality checks

Tooling config lives in `pyproject.toml`; tasks run via poethepoet
(`uv run poe` lists them). Workflow details: `docs/development.md`.

For each changed Python file:

```bash
uv run ruff format <path>
uv run ruff check --fix <path>
uv run pydoclint --quiet <path>
uv run mypy <path>
uv run radon cc <path> -s -n C
uv run radon mi <path> -s -n C
```

Related tests: `uv run pytest <file-or-nodeid>`. GPU smoke tests need
`--run-cuda`.

Before declaring any task done, and always before committing,
`uv run poe quality-iter` must be green (`uv run poe quality` for the full
pipeline). A **pre-existing** failure is not an excuse: fix it, or stop and
escalate to the user. Never call a run "verified" past a red gate.

Pre-commit hooks: `uv run pre-commit install` (also installs a commit-msg
hook that strips agent `Co-authored-by:` / `Made-with:` trailers).

## Code style (summary; full guide: `docs/code_style_guide.md`)

- Readability first: easy to read, interpret and follow.
- Line length **79**; double quotes; `ruff` with `select = ["ALL"]`.
- Don't hack the linters: no blanket `noqa`, no widening the ignore list.
  Rethink the code first. Targeted `# noqa: <CODE>` / `# nosec` only where a
  pattern is required (e.g. late heavy imports: `# noqa: PLC0415`).
- Every `speftr/` module starts with SPDX lines (ruff `CPY001` enforces
  it), because users copy modules individually; update the year range
  when you change a file in a new year:
  `# SPDX-FileCopyrightText: 2025-2026 Luis Rei` then
  `# SPDX-License-Identifier: BSD-2-Clause`.
- Google-style docstrings on every module, class and function, private
  ones included, documenting behaviour contracts, not just types.
- Full type hints (`docs/typing_guide.md`); `cast("Type", x)` string form;
  heavy libs (`datasets`, `peft`, `transformers`, `trl`) under
  `TYPE_CHECKING` for annotations and imported inside functions at use time.
- **Unsloth import order:** in SFT code and examples, `import unsloth`
  before `transformers`/`trl`/`peft` so its patches apply (keep
  `# noqa: I001`). Never import unsloth from `perl.py` or at `speftr`
  import time.
- Low cognitive complexity (complexipy ≤15): single-responsibility
  functions, small well-named helpers, no clever one-liners.
- Clear names (`user_list`, not `l`). Don't duplicate code; search first.
- Output via `print` in the library and most examples; `rgym.py` uses
  `logging`. Match the file.
- **No bare or blanket excepts.** Catch only exceptions you can recover
  from or that are expected; otherwise let it crash.

## Tests (full guide: `docs/tests_guide.md`)

- Real data only: small fixtures under `tests/fixtures/` taken from the real
  datasets. Never invent data shaped like what you think it looks like; if
  you can't find real data, ask.
- Unit tests run without a GPU; GPU/network smoke tests are marked `cuda`.
- Test behaviour through public outputs, not private helper names.

## Recipe defaults

Default hyperparameters encode the recipe in `README.md`. Don't change a
default without updating the README and the docstring/`help=` text. A new
config field goes into the dataclass, its docstring `Attributes`, the
argparse parser (if exposed) and the call that forwards it to TRL.

## Agent rules

- **Think before acting.** Read files before editing; don't re-read
  unchanged ones. State assumptions; if several interpretations exist, say
  so; if unclear, ask.
- **Simplicity first.** Minimum code that solves the problem: no features,
  abstractions or configurability beyond the request, no error handling for
  impossible cases. If 200 lines could be 50, rewrite.
- **Goal-driven.** Turn tasks into verifiable goals ("fix the bug" → a
  failing test, then make it pass) and state a brief plan with a check per
  step. Run the code before declaring done.
- **Communication.** Terse, no filler. Give subagents the relevant rules,
  guides, plan/report paths and file paths.
- Direct user instructions override this file.

## Known issues (verify before relying on them)

- `PERLConfig.load_in_4bit` only works without vLLM (`use_vllm=False`):
  TRL 1.13 has no adapter-only weight sync for a 4-bit vLLM copy.
- `PERLConfig.use_liger_kernel`: Liger's fused GRPO loss ignores
  final-logit soft-capping, so log-probabilities differ from the model's
  on Gemma 2/4.
- Unsloth's gpt-oss eval-mode kernels: attention gives sliding-window
  layers the full-attention mask (wrong loss past 128 tokens) and 4-bit
  experts run every token through every expert (8x/32x expert memory on
  20b/120b). Fix: `eval_in_train_mode` (set in `examples/gptoss`);
  `gptoss_eval.py` does the same for its eval-loss pass.
- Multi-GPU: PESFT DDP is validated on 2× A100 40GB with gemma-4-31B
  (4-bit) and PESFT model splitting (`device_map="unsloth_balanced"`)
  with gpt-oss-120b; PERL DDP is not validated on multi-GPU hardware.
- Unsloth's 4-bit gpt-oss expert training is CPU-bound (a Python loop
  over the experts): ~17–19% GPU utilization per GPU on 120b.
- Native MXFP4 gpt-oss checkpoints (`openai/gpt-oss-*`) are not
  trainable: Unsloth has no MXFP4 backward and transformers marks MXFP4
  not trainable. Train on the Unsloth bnb-4bit conversion
  (`unsloth/gpt-oss-20b-unsloth-bnb-4bit`).
- PESFT disables HF telemetry process-wide, which makes the `kernels`
  package fail to fetch Hub kernels ("could not verify publisher trust
  status"). Only kernel-hub users are affected, e.g. native MXFP4 loading.
- Unsloth's gpt-oss attention gives sliding-window layers the full mask
  in eval mode (wrong loss and generations past 128 tokens). PESFT keeps
  those layers in training mode during evaluation; scripts that load
  gpt-oss outside PESFT set `UNSLOTH_ENABLE_FLEX_ATTENTION=0` before
  loading (see `examples/gptoss/`).
- Unsloth's gpt-oss bnb-4bit checkpoints store experts as separate 4-bit
  layers that plain transformers cannot load; load them with Unsloth.

Remove entries as they get fixed.
