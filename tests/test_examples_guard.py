"""Tests for ``examples.guard.guard_eval`` label extraction and metrics.

``guard_eval`` imports Unsloth at module level, and Unsloth refuses to
import without a GPU accelerator, so these tests skip on CPU-only hosts.

``fixtures/aegis2_test_rows.json`` holds human-labelled prompts from the
test split of NVIDIA's Aegis AI Content Safety Dataset 2.0 (CC-BY-4.0,
https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0),
with ``safe``/``unsafe`` mapped to the guard labels ``unharmful``/``harmful``.
"""

from __future__ import annotations

import importlib

import pytest


try:
    guard_eval = importlib.import_module("examples.guard.guard_eval")
except (ImportError, NotImplementedError, RuntimeError) as exc:
    pytest.skip(
        f"examples.guard.guard_eval needs Unsloth (GPU): {exc}",
        allow_module_level=True,
    )


VALID_LABELS = {"harmful", "unharmful"}


@pytest.fixture(scope="module")
def labels(load_fixture) -> list[str]:
    rows = load_fixture("aegis2_test_rows.json")
    return [row["prompt_harm_label"] for row in rows]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("harmful", "harmful"),
        ("unharmful", "unharmful"),
        ("  Harmful\n", "harmful"),
        ("UNHARMFUL", "unharmful"),
    ],
)
def test_extract_label_exact_match(text, expected):
    assert guard_eval.extract_label(text, VALID_LABELS) == (expected, True)


def test_extract_label_substring_match():
    assert guard_eval.extract_label("harmful.<eos>", VALID_LABELS) == (
        "harmful",
        True,
    )


@pytest.mark.parametrize("text", ["", "I cannot answer that", "safe"])
def test_extract_label_invalid(text):
    assert guard_eval.extract_label(text, VALID_LABELS) == (None, False)


def test_compute_metrics_perfect_predictions(labels):
    metrics = guard_eval._compute_metrics(labels, list(labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["avg_precision"] == 1.0
    assert "harmful" in metrics["classification_report"]


def test_compute_metrics_one_error(labels):
    flipped = {"harmful": "unharmful", "unharmful": "harmful"}
    preds = [flipped[labels[0]], *labels[1:]]
    metrics = guard_eval._compute_metrics(labels, preds)
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert 0.0 < metrics["f1_macro"] < 1.0


def test_compute_metrics_requires_two_labels(labels):
    only_harmful = [label for label in labels if label == "harmful"]
    with pytest.raises(ValueError, match="binary classification"):
        guard_eval._compute_metrics(only_harmful, only_harmful)


@pytest.mark.parametrize(
    "label_order", [["harmful", "unharmful"], ["unharmful", "harmful"]]
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [("unharmful.", "unharmful"), ("The prompt is harmful", "harmful")],
)
def test_extract_label_substring_prefers_longest(label_order, text, expected):
    # "unharmful" contains "harmful": the result must not depend on the
    # iteration order of the label collection.
    labels = dict.fromkeys(label_order).keys()
    assert guard_eval.extract_label(text, labels) == (expected, True)


def test_compute_metrics_invalid_predictions_are_errors_not_a_class(labels):
    preds = ["INVALID_PREDICTION", *labels[1:]]
    metrics = guard_eval._compute_metrics(labels, preds)
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    # Macro averages cover the dataset's two labels only; an invalid
    # prediction must not add a zero-scoring third class.
    per_label_precision = guard_eval.precision_score(
        labels,
        preds,
        labels=sorted(set(labels)),
        average=None,
        zero_division=0,
    )
    assert metrics["precision_macro"] == pytest.approx(
        per_label_precision.mean()
    )
    assert "INVALID_PREDICTION" not in metrics["classification_report"]
