"""Shared pytest configuration: opt-in GPU tests and common fixtures."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent
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


@pytest.fixture(scope="session")
def load_functions():
    """Return a loader for top-level functions of a module's source.

    For example modules that import Unsloth at import time, which fails
    without a GPU: the named functions are compiled on their own, so they
    may use only builtins (annotations stay strings).
    """

    def load(relative_path: str, *names: str) -> SimpleNamespace:
        path = REPO_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        assert {node.name for node in nodes} == set(names), relative_path
        future = ast.parse("from __future__ import annotations").body
        module = ast.Module(body=[*future, *nodes], type_ignores=[])
        namespace: dict = {}
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
        return SimpleNamespace(**{name: namespace[name] for name in names})

    return load
