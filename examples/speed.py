r"""Measure an SFT example's training speed across the recipe's knobs.

Runs an example's training script once per configuration for a fixed
number of optimizer steps, reads the ``train_metrics.json`` that
``PESFT.save_model`` writes, and prints a Markdown table: steady-state
seconds per step (the first steps, which compile kernels, excluded),
seconds per step over the whole run, warm-up time, samples per second,
peak GPU memory and the final losses. The configurations (``CONFIGS``)
change how each step is computed, not what it trains: the effective
batch stays at the example's default except for ``batch_32``.

The script must accept ``PESFTConfig``'s flags (as ``examples.intent``
and ``examples.tldr`` do). Flags the sweep does not recognise are passed
to every run, e.g. a model:

Usage:
    uv run python -m examples.speed examples.intent.intent_train
    uv run python -m examples.speed examples.intent.intent_train \\
        --steps 40 --configs default no_checkpointing batch_8x2 \\
        --model_name_or_path Qwen/Qwen3.5-4B
    uv run python -m examples.speed examples.tldr.tldr_train \\
        --output_dir ./models/speed-tldr

Each run's console output goes to ``<output_dir>/<config>.log``; a run
that fails (for example out of memory) is reported as failed and the
sweep continues. Results: ``<output_dir>/speed.md`` and ``speed.json``.
``--table_only`` re-tables the runs already in ``--output_dir`` without
running anything, e.g. after running the configs in separate jobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Final


CONFIGS: Final[dict[str, list[str]]] = {
    "default": [],
    "no_checkpointing": ["--use_gradient_checkpointing", "False"],
    "batch_8x2": [
        "--per_device_train_batch_size",
        "8",
        "--gradient_accumulation_steps",
        "2",
    ],
    "batch_4x4": [
        "--per_device_train_batch_size",
        "4",
        "--gradient_accumulation_steps",
        "4",
    ],
    "padding_free": ["--padding_free"],
    "flex_attention": ["--attn_implementation", "flex_attention"],
    "flex_padding_free": [
        "--attn_implementation",
        "flex_attention",
        "--padding_free",
    ],
    "packing": ["--packing"],
    "load_in_4bit": ["--load_in_4bit"],
    "batch_32": ["--per_device_train_batch_size", "32"],
}
"""Sweep configurations: name -> flags added to the example's defaults."""

DEFAULT_STEPS = 40
DEFAULT_OUTPUT_DIR = "./models/speed"
COLUMNS: Final[tuple[str, ...]] = (
    "config",
    "s/step",
    "s/step (all)",
    "warm-up s",
    "samples/s",
    "peak GiB",
    "train loss",
    "eval loss",
)
"""Table columns. ``s/step`` is the steady-state step time (median
after the first steps, which include kernel compilation); ``s/step
(all)`` averages every step and ``warm-up s`` is what the first steps
took."""


def build_parser() -> argparse.ArgumentParser:
    """Return the sweep's parser; unknown flags go to every run.

    Returns:
        Parser with the module, ``--steps``, ``--configs`` and
        ``--output_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Run an SFT example under several speed settings"
    )
    parser.add_argument(
        "module",
        help="Training script module, e.g. examples.intent.intent_train",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Optimizer steps per run (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=list(CONFIGS),
        help="Configurations to run (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Where runs, logs and the results table go "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--table_only",
        action="store_true",
        help=(
            "Run nothing; table the configs' existing runs in --output_dir "
            "(a missing run counts as failed)"
        ),
    )
    return parser


def build_command(
    module: str,
    config: str,
    *,
    steps: int,
    run_dir: Path,
    extra_args: list[str],
) -> list[str]:
    """Compose the command line of one run.

    Args:
        module: Training script module.
        config: Key of ``CONFIGS``.
        steps: Optimizer steps.
        run_dir: The run's ``--output_dir``.
        extra_args: Flags appended to every run.

    Returns:
        The argv, with checkpoints and evaluation during training off so
        the measured time is training alone. The example's final
        evaluation still runs and provides the eval loss.
    """
    return [
        sys.executable,
        "-m",
        module,
        "--max_steps",
        str(steps),
        "--output_dir",
        str(run_dir),
        "--save_strategy",
        "no",
        "--eval_strategy",
        "no",
        *extra_args,
        *CONFIGS[config],
    ]


def run_config(
    module: str,
    config: str,
    *,
    steps: int,
    output_dir: Path,
    extra_args: list[str],
) -> dict[str, object] | None:
    """Run one configuration and return its training metrics.

    Args:
        module: Training script module.
        config: Key of ``CONFIGS``.
        steps: Optimizer steps.
        output_dir: Sweep directory; the run writes to a subdirectory.
        extra_args: Flags appended to every run.

    Returns:
        The run's ``train_metrics.json`` contents, or None when the run
        failed (its log is named in the message).
    """
    run_dir = output_dir / config
    log_path = output_dir / f"{config}.log"
    command = build_command(
        module, config, steps=steps, run_dir=run_dir, extra_args=extra_args
    )
    print(f"\n[{config}] {' '.join(command[3:])}")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(  # noqa: S603  # argv built above
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        print(f"[{config}] failed (exit {completed.returncode}): {log_path}")
        return None
    with (run_dir / "train_metrics.json").open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def read_metrics(output_dir: Path, config: str) -> dict[str, object] | None:
    """Read an earlier run's metrics, for ``--table_only``.

    Args:
        output_dir: Sweep directory.
        config: Key of ``CONFIGS``.

    Returns:
        The run's ``train_metrics.json`` contents, or None when the run
        left none (it failed or never ran).
    """
    path = output_dir / config / "train_metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def format_row(config: str, metrics: dict[str, object] | None) -> list[str]:
    """Turn one run's metrics into table cells in ``COLUMNS`` order.

    Args:
        config: Key of ``CONFIGS``.
        metrics: The run's metrics, or None for a failed run.

    Returns:
        The cells; numbers with two decimals, missing values empty.
    """
    if metrics is None:
        return [config, "failed", *[""] * (len(COLUMNS) - 2)]

    def cell(key: str) -> str:
        """Format one numeric metric, or an empty cell when missing.

        Args:
            key: Metric name in ``train_metrics.json``.

        Returns:
            The value with two decimals, or an empty string.
        """
        value = metrics.get(key)
        return f"{value:.2f}" if isinstance(value, (int, float)) else ""

    steps_per_second = metrics.get("train_steps_per_second")
    seconds_per_step = (
        f"{1 / steps_per_second:.2f}"
        if isinstance(steps_per_second, (int, float)) and steps_per_second
        else ""
    )
    return [
        config,
        cell("steady_seconds_per_step"),
        seconds_per_step,
        cell("warmup_seconds"),
        cell("train_samples_per_second"),
        cell("peak_memory_allocated_gib"),
        cell("train_loss"),
        cell("eval_loss"),
    ]


def format_table(rows: list[list[str]]) -> str:
    """Render rows as a Markdown table under ``COLUMNS``.

    Args:
        rows: Cell lists from ``format_row``.

    Returns:
        The table text.
    """
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    """Run the sweep and write the table and the raw metrics.

    Returns:
        None. ``speed.md`` and ``speed.json`` land in ``--output_dir``.
    """
    args, extra_args = build_parser().parse_known_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object] | None] = {}
    for config in args.configs:
        if args.table_only:
            results[config] = read_metrics(output_dir, config)
            continue
        results[config] = run_config(
            args.module,
            config,
            steps=args.steps,
            output_dir=output_dir,
            extra_args=extra_args,
        )

    gpu_names = {
        str(metrics["gpu_name"])
        for metrics in results.values()
        if metrics is not None and "gpu_name" in metrics
    }
    header = (
        f"{args.module}, {args.steps} steps, {' '.join(extra_args) or 'no'}"
        f" extra flags, GPU: {', '.join(sorted(gpu_names)) or 'unknown'}"
    )
    table = format_table(
        [format_row(config, metrics) for config, metrics in results.items()]
    )
    (output_dir / "speed.md").write_text(
        f"{header}\n\n{table}\n", encoding="utf-8"
    )
    with (output_dir / "speed.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"header": header, "extra_args": extra_args, "results": results},
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"\n{header}\n\n{table}\n\nWritten to {output_dir / 'speed.md'}")


if __name__ == "__main__":
    main()
