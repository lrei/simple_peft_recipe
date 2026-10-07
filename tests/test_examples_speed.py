"""Tests for ``examples.speed``: command composition and the results table.

``examples.speed`` imports no GPU library, so these run anywhere. The
metrics fixture is a ``train_metrics.json`` written by ``PESFT`` for a
3-step run of ``unsloth/Qwen2.5-0.5B-Instruct`` on an RTX 3090
(``tests/fixtures/train_metrics_qwen2.5_0.5b.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from examples.speed import (
    COLUMNS,
    CONFIGS,
    build_command,
    build_parser,
    format_row,
    format_table,
    read_metrics,
)
from tests.conftest import FIXTURES_DIR


@pytest.fixture(scope="module")
def metrics() -> dict[str, object]:
    path = FIXTURES_DIR / "train_metrics_qwen2.5_0.5b.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def test_command_fixes_steps_and_output_and_appends_flags():
    command = build_command(
        "examples.intent.intent_train",
        "batch_8x2",
        steps=40,
        run_dir=Path("out/batch_8x2"),
        extra_args=["--model_name_or_path", "Qwen/Qwen3.5-4B"],
    )
    assert command[:3] == [
        sys.executable,
        "-m",
        "examples.intent.intent_train",
    ]
    assert command[3:9] == [
        "--max_steps",
        "40",
        "--output_dir",
        "out/batch_8x2",
        "--save_strategy",
        "no",
    ]
    assert "--model_name_or_path" in command
    assert command[-4:] == CONFIGS["batch_8x2"]


def test_parser_passes_unknown_flags_through():
    args, extra = build_parser().parse_known_args(
        [
            "examples.intent.intent_train",
            "--configs",
            "default",
            "--lora_r",
            "1",
        ]
    )
    assert args.configs == ["default"]
    assert extra == ["--lora_r", "1"]


def test_row_from_real_metrics(metrics):
    row = format_row("default", metrics)
    assert row[0] == "default"
    # A 3-step run has no steady-state step time.
    assert (row[1], row[3]) == ("", "")
    seconds_per_step = 1 / float(metrics["train_steps_per_second"])
    assert row[2] == f"{seconds_per_step:.2f}"
    assert row[4] == f"{float(metrics['train_samples_per_second']):.2f}"
    assert row[5] == f"{float(metrics['peak_memory_allocated_gib']):.2f}"
    assert row[6] == f"{float(metrics['train_loss']):.2f}"
    assert row[7] == f"{float(metrics['eval_loss']):.2f}"


def test_row_shows_steady_state_and_warmup(metrics):
    timed = {
        **metrics,
        "steady_seconds_per_step": 1.57,
        "warmup_seconds": 120.4,
    }
    row = format_row("default", timed)
    assert (row[1], row[3]) == ("1.57", "120.40")


def test_failed_run_row_and_table_shape(metrics):
    table = format_table(
        [format_row("default", metrics), format_row("batch_32", None)]
    )
    lines = table.splitlines()
    assert lines[0] == "| " + " | ".join(COLUMNS) + " |"
    assert lines[3] == "| batch_32 | failed |  |  |  |  |  |  |"
    assert all(line.count("|") == len(COLUMNS) + 1 for line in lines)


def test_read_metrics_finds_existing_runs_only(tmp_path, metrics):
    run_dir = tmp_path / "default"
    run_dir.mkdir()
    (run_dir / "train_metrics.json").write_text(json.dumps(metrics))
    assert read_metrics(tmp_path, "default") == metrics
    assert read_metrics(tmp_path, "packing") is None


def test_table_only_flag():
    args, _ = build_parser().parse_known_args(["m", "--table_only"])
    assert args.table_only is True
