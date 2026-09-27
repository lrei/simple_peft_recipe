"""Tests for the conversation helpers in ``examples.instruct.instruct``.

The module imports Unsloth at module level, and Unsloth refuses to import
without a GPU accelerator, so these tests skip on CPU-only hosts.
"""

from __future__ import annotations

import importlib

import pytest


try:
    instruct = importlib.import_module("examples.instruct.instruct")
except (ImportError, NotImplementedError, RuntimeError) as exc:
    pytest.skip(
        f"examples.instruct.instruct needs Unsloth (GPU): {exc}",
        allow_module_level=True,
    )


SYSTEM = "Always answer like a pirate."


@pytest.fixture(scope="module")
def rows(load_fixture) -> list[dict]:
    return load_fixture("dolly_pirate_rows.json")


@pytest.fixture(scope="module")
def batch(rows) -> dict[str, list[str]]:
    """Columnar batch, as ``datasets`` hands to a batched formatter."""
    return {key: [row[key] for row in rows] for key in rows[0]}


def test_build_conversations_from_columns(rows, batch):
    conversations = instruct._build_conversations_from_columns(batch)
    assert len(conversations) == len(rows)
    for row, conversation in zip(rows, conversations, strict=True):
        user, assistant = conversation
        assert user["role"] == "user"
        assert assistant == {
            "role": "assistant",
            "content": row["response"].strip(),
        }
        context = row["context"].strip()
        if context:
            assert user["content"] == (
                f"{row['instruction'].strip()}\n\nContext:\n{context}"
            )
        else:
            assert user["content"] == row["instruction"].strip()


def test_build_conversations_from_single_row(rows):
    row = rows[-1]
    (conversation,) = instruct._build_conversations_from_columns(row)
    assert conversation[0]["content"] == row["instruction"].strip()
    assert conversation[1]["content"] == row["response"].strip()


def test_build_conversations_context_only_and_empty():
    context_only, empty = instruct._build_conversations_from_columns(
        {"instruction": ["", None], "context": ["Some text", ""]}
    )
    assert context_only == [{"role": "user", "content": "Some text"}]
    assert empty == [
        {"role": "user", "content": "Ahoy matey, respond to this prompt."}
    ]


def test_extract_conversations_falls_back_to_columns(batch):
    assert instruct._extract_conversations(
        batch
    ) == instruct._build_conversations_from_columns(batch)


def test_extract_conversations_prefers_messages(rows):
    conversations = instruct._build_conversations_from_columns(
        {key: [row[key] for row in rows] for key in rows[0]}
    )
    # Batched: a list of conversations.
    assert (
        instruct._extract_conversations({"messages": conversations})
        == conversations
    )
    # Single row: one conversation (a list of message dicts).
    assert instruct._extract_conversations(
        {"messages": conversations[0], "instruction": "ignored"}
    ) == [conversations[0]]


def test_inject_system_prompt_prepends(rows):
    conversation = instruct._build_conversations_from_columns(rows[0])[0]
    injected = instruct._inject_system_prompt(conversation, SYSTEM)
    assert injected[0] == {"role": "system", "content": SYSTEM}
    assert injected[1:] == conversation
    assert conversation[0]["role"] == "user"  # input not mutated


def test_inject_system_prompt_replaces_existing(rows):
    conversation = instruct._build_conversations_from_columns(rows[0])[0]
    existing = [{"role": "system", "content": "old"}, *conversation]
    injected = instruct._inject_system_prompt(existing, SYSTEM)
    assert injected == [{"role": "system", "content": SYSTEM}, *conversation]


def test_format_batch_renders_chat_template(batch, rows, qwen_tokenizer):
    texts = instruct._build_format_batch(qwen_tokenizer, SYSTEM)(batch)
    assert len(texts) == len(rows)
    for row, text in zip(rows, texts, strict=True):
        assert text.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>")
        assert row["response"].strip() in text
        assert text.count("<|im_start|>assistant\n") == 1


@pytest.mark.parametrize(
    "model", ["unsloth/gemma-3-270m-it", "unsloth/gemma-4-E2B-it"]
)
def test_format_batch_keeps_template_bos(batch, model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    texts = instruct._build_format_batch(tokenizer, SYSTEM)(batch)
    assert all(text.startswith(tokenizer.bos_token) for text in texts)
