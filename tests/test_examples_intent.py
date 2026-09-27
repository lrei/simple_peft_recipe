"""Tests for the Banking77 intent example's prompt, parsing and metrics.

Rows and intent names come from ``legacy-datasets/banking77`` (test split).
The example modules import Unsloth at module level, and Unsloth refuses to
import without a GPU accelerator, so these tests skip on CPU-only hosts.
"""

from __future__ import annotations

import importlib

import pytest


try:
    intent_eval = importlib.import_module("examples.intent.intent_eval")
    intent_train = importlib.import_module("examples.intent.intent_train")
except (ImportError, NotImplementedError, RuntimeError) as exc:
    pytest.skip(
        f"examples.intent needs Unsloth (GPU): {exc}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def banking77(load_fixture):
    return load_fixture("banking77_test_rows.json")


@pytest.fixture(scope="module")
def label_names(banking77) -> list[str]:
    return banking77["label_names"]


@pytest.fixture(scope="module")
def batch(banking77) -> dict[str, list]:
    rows = banking77["rows"]
    return {
        "message": [row["text"] for row in rows],
        "label": [row["label"] for row in rows],
    }


def test_prompt_lists_every_intent(banking77, label_names):
    text = banking77["rows"][0]["text"]
    messages = intent_eval.build_messages(text, label_names)
    assert len(messages) == 1
    prompt = messages[0]["content"]
    assert text in prompt
    assert all(name in prompt for name in label_names)


def test_training_chat_ends_with_intent(banking77, label_names):
    row = banking77["rows"][0]
    intent = label_names[row["label"]]
    messages = intent_eval.build_messages(row["text"], label_names, intent)
    assert messages[-1] == {"role": "assistant", "content": "card_arrival"}


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("card_arrival", "card_arrival"),
        ("  Card_Arrival\n", "card_arrival"),
        ("refund_not_showing_up", "Refund_not_showing_up"),
        ("reverted_card_payment?", "reverted_card_payment?"),
        ("verify_my_identity\nBecause they ask how.", "verify_my_identity"),
    ],
)
def test_extract_intent_matches_first_line(label_names, generated, expected):
    assert intent_eval.extract_intent(generated, label_names) == expected


@pytest.mark.parametrize(
    "generated", ["", "card", "The intent is card_arrival"]
)
def test_extract_intent_rejects_non_labels(label_names, generated):
    assert intent_eval.extract_intent(generated, label_names) is None


def test_metrics_perfect_and_invalid(banking77, label_names):
    gold = [label_names[row["label"]] for row in banking77["rows"]]
    perfect = intent_eval.compute_metrics(gold, gold, label_names)
    assert perfect["accuracy"] == 1.0
    assert perfect["invalid_rate"] == 0.0

    predicted = [intent_eval.INVALID_PREDICTION, *gold[1:]]
    metrics = intent_eval.compute_metrics(gold, predicted, label_names)
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["invalid_rate"] == pytest.approx(1 / 6)
    assert metrics["f1_macro"] < 1.0


def test_formatting_func_handles_batches_and_single_rows(
    qwen_tokenizer, label_names, batch
):
    format_batch = intent_train.build_formatting_func(
        qwen_tokenizer, label_names
    )
    texts = format_batch(batch)
    assert len(texts) == len(batch["message"])
    assert texts[0].rstrip().endswith("card_arrival<|im_end|>")

    single = {"message": batch["message"][0], "label": batch["label"][0]}
    assert format_batch(single) == texts[:1]
