"""Tests for the TL;DR example's prompt, validity check and metrics.

Rows come from ``trl-lib/tldr`` (``tests/fixtures/tldr_rows.json``). The
example modules import Unsloth at module level, and Unsloth refuses to
import without a GPU accelerator, so these tests skip on CPU-only hosts.
"""

from __future__ import annotations

import importlib

import pytest


try:
    tldr_eval = importlib.import_module("examples.tldr.tldr_eval")
    tldr_train = importlib.import_module("examples.tldr.tldr_train")
except (ImportError, NotImplementedError, RuntimeError) as exc:
    pytest.skip(
        f"examples.tldr needs Unsloth (GPU): {exc}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def rows(load_fixture) -> list[dict]:
    return load_fixture("tldr_rows.json")


@pytest.fixture(scope="module")
def batch(rows) -> dict[str, list[str]]:
    return {
        "post": [row["prompt"] for row in rows],
        "summary": [row["completion"] for row in rows],
    }


def test_prompt_keeps_the_post_and_drops_the_cue(rows):
    post = rows[0]["prompt"]
    assert post.rstrip().endswith("TL;DR:")
    messages = tldr_eval.build_messages(post)
    assert len(messages) == 1
    content = messages[0]["content"]
    assert content.startswith(tldr_eval.INSTRUCTION)
    assert "SUBREDDIT: r/relationships" in content
    assert not content.endswith("TL;DR:")


def test_training_chat_ends_with_the_summary(rows):
    row = rows[0]
    messages = tldr_eval.build_messages(row["prompt"], row["completion"])
    assert messages[-1] == {
        "role": "assistant",
        "content": row["completion"].strip(),
    }


@pytest.mark.parametrize(
    ("generated", "valid"),
    [
        ("Progress is still happening, even when it seems slow.", True),
        ("", False),
        ("   \n", False),
        (
            "<think>\nThe post is about weight loss.\n</think>\nKeep going.",
            False,
        ),
        ("Keep going.</think>", False),
    ],
)
def test_is_valid_summary(generated, valid):
    assert tldr_eval.is_valid_summary(generated) is valid


def test_metrics_perfect_and_invalid(rows):
    references = [row["completion"] for row in rows]
    predictions = list(references)
    predictions[-1] = "<think></think>"
    metrics = tldr_eval.compute_metrics(references, predictions)
    valid = len(rows) - 1
    assert metrics["invalid_rate"] == pytest.approx(1 / len(rows))
    for rouge_type in tldr_eval.ROUGE_TYPES:
        assert metrics[rouge_type] == pytest.approx(valid / len(rows))
    expected_words = sum(len(r.split()) for r in references[:-1]) / valid
    assert metrics["mean_words"] == pytest.approx(expected_words)


def test_metrics_rouge_below_one_for_a_paraphrase(rows):
    reference = rows[1]["completion"]
    metrics = tldr_eval.compute_metrics(
        [reference], ["Progress still happens even when it feels slow."]
    )
    assert 0 < metrics["rouge1"] < 1
    assert metrics["invalid_rate"] == 0


def test_formatting_func_handles_batches_and_single_rows(
    qwen_tokenizer, batch
):
    format_batch = tldr_train.build_formatting_func(qwen_tokenizer)
    texts = format_batch(batch)
    assert len(texts) == len(batch["post"])
    summary = batch["summary"][0].strip()
    assert texts[0].rstrip().endswith(f"{summary}<|im_end|>")

    single = {"post": batch["post"][0], "summary": batch["summary"][0]}
    assert format_batch(single) == texts[:1]


def test_parser_defaults_leave_markers_to_inference():
    from speftr import PESFTConfig

    config = PESFTConfig.from_args(tldr_train.build_parser().parse_args([]))
    assert config.model_name_or_path == "Qwen/Qwen3.5-4B"
    assert config.chat_template is None
    assert config.train_on_responses is True
    assert (config.instruction_part, config.response_part) == ("", "")
    assert config.max_seq_length == tldr_eval.MAX_SEQ_LENGTH


def test_parser_accepts_pesft_flags_and_row_limits():
    args = tldr_train.build_parser().parse_args(
        ["--padding_free", "--train_rows", "20000", "--eval_rows", "50"]
    )
    assert args.padding_free is True
    assert args.train_rows == 20000
    assert args.eval_rows == 50
