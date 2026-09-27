"""Shared pytest configuration: opt-in GPU tests and common fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def pytest_addoption(parser):
    parser.addoption(
        "--run-cuda",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.cuda (need an NVIDIA GPU).",
    )


def pytest_collection_modifyitems(config, items):
    # Skip instead of deselecting via a default ``-m`` expression: a set
    # markexpr disables pytest-testmon's incremental selection.
    if config.getoption("--run-cuda"):
        return
    skip_cuda = pytest.mark.skip(reason="needs --run-cuda (NVIDIA GPU)")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def load_fixture():
    """Return a loader for JSON fixtures of real rows in tests/fixtures."""

    def load(name: str) -> list[dict]:
        with (FIXTURES_DIR / name).open(encoding="utf-8") as handle:
            return json.load(handle)

    return load


@pytest.fixture(scope="session")
def qwen_tokenizer():
    """Real Qwen2.5 chat tokenizer from the Hugging Face hub (CPU only)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
