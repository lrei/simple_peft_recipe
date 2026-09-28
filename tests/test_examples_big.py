"""Tests for the formatting and CLI helpers in ``examples.big.big``.

Rows come from ``TeeZee/dolly-15k-pirate-speech`` (CC-BY-SA-3.0, train
split) via ``tests/fixtures/dolly_pirate_rows.json``; conversations are
rendered with the real Gemma 4 tokenizer of the example's model. The
module imports Unsloth, which refuses to import without a GPU, so these
tests skip on CPU-only hosts.
"""

from __future__ import annotations

import importlib

import pytest


try:
    big = importlib.import_module("examples.big.big")
except (ImportError, NotImplementedError, RuntimeError) as exc:
    pytest.skip(
        f"examples.big.big needs Unsloth (GPU): {exc}",
        allow_module_level=True,
    )

SYSTEM = "You are a pirate, always respond in pirate speech."


@pytest.fixture(scope="module")
def rows(load_fixture) -> list[dict]:
    return load_fixture("dolly_pirate_rows.json")


@pytest.fixture(scope="module")
def gemma_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        big.EXAMPLE_DEFAULTS["model_name_or_path"]
    )


def test_build_conversation_adds_context_only_when_present(rows):
    with_context = next(row for row in rows if row["context"])
    without_context = next(row for row in rows if not row["context"])

    system, user, assistant = big.build_conversation(with_context, SYSTEM)
    assert system == {"role": "system", "content": SYSTEM}
    assert user["content"] == (
        f"{with_context['instruction'].strip()}\n\nContext:\n"
        f"{with_context['context'].strip()}"
    )
    assert assistant == {
        "role": "assistant",
        "content": with_context["response"].strip(),
    }
    _, user, _ = big.build_conversation(without_context, SYSTEM)
    assert user["content"] == without_context["instruction"].strip()


def test_format_batch_renders_one_text_per_row(rows, gemma_tokenizer):
    format_batch = big.build_format_batch(gemma_tokenizer, SYSTEM)
    batch = {key: [row[key] for row in rows] for key in rows[0]}

    texts = format_batch(batch)

    assert len(texts) == len(rows)
    assert texts == [format_batch(row)[0] for row in rows]
    instruction_part = big.EXAMPLE_DEFAULTS["instruction_part"]
    response_part = big.EXAMPLE_DEFAULTS["response_part"]
    for row, text in zip(rows, texts, strict=True):
        assert text.startswith("<bos><|turn>system\n" + SYSTEM)
        assert text.count(instruction_part) == 1
        assert f"{response_part}{row['response'].strip()}" in text


def test_parser_applies_example_defaults():
    args = big.build_parser().parse_args([])
    config = big.PESFTConfig.from_args(args)

    for name, value in big.EXAMPLE_DEFAULTS.items():
        assert getattr(config, name) == value
    assert config.max_steps == -1
    assert config.device_map is None
    assert args.eval_size == big.DEFAULT_EVAL_SIZE
    assert args.system_prompt == SYSTEM


def test_parser_overrides_reach_config():
    args = big.build_parser().parse_args(
        [
            "--max_steps",
            "20",
            "--device_map",
            "balanced",
            "--gradient_accumulation_steps",
            "4",
        ]
    )
    config = big.PESFTConfig.from_args(args)

    assert config.max_steps == 20
    assert config.device_map == "balanced"
    assert config.gradient_accumulation_steps == 4
