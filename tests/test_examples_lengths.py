"""Tests for the token-budget batching and prompt truncation of examples.

The eval scripts and ``examples/chat.py`` import Unsloth at module level,
which fails without a GPU, so ``length_batches``, ``truncate_prompts``,
``truncate_rows`` and ``_fit_conversation`` are loaded from the module
source (``load_functions``) and run on CPU.

Token counts come from real rows tokenized with the Qwen2.5 tokenizer:
``trl-lib/tldr`` posts (CC-BY-4.0), ``legacy-datasets/banking77`` messages
(CC-BY-4.0), ``b-mc2/sql-create-context`` prompts (CC-BY-4.0), Aegis 2.0
prompts (CC-BY-4.0) and ``TeeZee/dolly-15k-pirate-speech`` rows
(CC-BY-SA-3.0).
"""

from __future__ import annotations

import importlib
import sys
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


BATCHING_MODULES = [
    "examples/tldr/tldr_eval.py",
    "examples/intent/intent_eval.py",
    "examples/text2sql/text2sql_eval.py",
    "examples/guard/guard_eval.py",
    "examples/gptoss/gptoss_eval.py",
]
TRUNCATING_MODULES = [
    "examples/tldr/tldr_eval.py",
    "examples/intent/intent_eval.py",
    "examples/guard/guard_eval.py",
    "examples/gptoss/gptoss_eval.py",
]
MAX_NEW_TOKENS = 96


@pytest.fixture(scope="module")
def texts(load_fixture) -> list[str]:
    """User-turn texts of real rows from several fixtures."""
    banking77 = load_fixture("banking77_test_rows.json")
    return [
        *(row["prompt"] for row in load_fixture("tldr_rows.json")),
        *(row["text"] for row in banking77["rows"]),
        *(
            f"{row['context']}\n\n{row['question']}"
            for row in load_fixture("sql_create_context_rows.json")
        ),
        *(row["prompt"] for row in load_fixture("aegis2_test_rows.json")),
    ]


@pytest.fixture(scope="module")
def prompt_lengths(texts, qwen_tokenizer) -> list[int]:
    """Prompt tokens of each text as a one-turn chat."""
    encoded = qwen_tokenizer.apply_chat_template(
        [[{"role": "user", "content": text}] for text in texts],
        add_generation_prompt=True,
        return_dict=True,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def assert_valid_batches(
    batches: list[list[int]],
    lengths: list[int],
    token_budget: int,
    max_rows: int | None = None,
) -> None:
    """Every index once; a batch over the budget holds one sequence."""
    assert sorted(i for batch in batches for i in batch) == list(
        range(len(lengths))
    )
    for batch in batches:
        padded = len(batch) * max(lengths[i] for i in batch)
        assert padded <= token_budget or len(batch) == 1
        assert max_rows is None or len(batch) <= max_rows


@pytest.mark.parametrize("path", BATCHING_MODULES)
def test_length_batches_keep_the_budget_on_real_prompts(
    path, load_functions, prompt_lengths
):
    length_batches = load_functions(path, "length_batches").length_batches
    lengths = [length + MAX_NEW_TOKENS for length in prompt_lengths]
    token_budget = 3 * max(lengths)

    batches = length_batches(lengths, token_budget)

    assert_valid_batches(batches, lengths, token_budget)
    assert len(batches) < len(lengths)


@pytest.mark.parametrize("path", BATCHING_MODULES)
def test_length_batches_run_an_oversize_input_alone(
    path, load_functions, prompt_lengths
):
    length_batches = load_functions(path, "length_batches").length_batches
    token_budget = 4 * max(prompt_lengths)
    lengths = [token_budget + 1, *prompt_lengths]

    batches = length_batches(lengths, token_budget)

    assert batches[0] == [0]
    assert_valid_batches(batches, lengths, token_budget)


@pytest.mark.parametrize("path", BATCHING_MODULES)
def test_length_batches_keep_the_row_cap_and_the_budget(
    path, load_functions, prompt_lengths
):
    length_batches = load_functions(path, "length_batches").length_batches
    lengths = [length + MAX_NEW_TOKENS for length in prompt_lengths]
    # Two of the longest rows fill the budget; the short rows hit the cap.
    token_budget = 2 * max(lengths)
    max_rows = 4

    batches = length_batches(lengths, token_budget, max_rows)

    assert_valid_batches(batches, lengths, token_budget, max_rows)
    assert len(batches[0]) == 2
    assert max(len(batch) for batch in batches) == max_rows


@pytest.mark.parametrize("path", BATCHING_MODULES)
def test_length_batches_run_an_oversize_input_alone_under_a_row_cap(
    path, load_functions, prompt_lengths
):
    length_batches = load_functions(path, "length_batches").length_batches
    token_budget = 4 * max(prompt_lengths)
    lengths = [token_budget + 1, *prompt_lengths]

    batches = length_batches(lengths, token_budget, 3)

    assert batches[0] == [0]
    assert_valid_batches(batches, lengths, token_budget, 3)


@pytest.mark.parametrize("path", BATCHING_MODULES)
@settings(max_examples=50, deadline=None)
@given(
    lengths=st.lists(st.integers(1, 4096), max_size=60),
    token_budget=st.integers(1, 16_384),
    max_rows=st.one_of(st.none(), st.integers(1, 64)),
)
def test_length_batches_cover_every_index_once(
    path, load_functions, lengths, token_budget, max_rows
):
    length_batches = load_functions(path, "length_batches").length_batches

    batches = length_batches(lengths, token_budget, max_rows)

    assert_valid_batches(batches, lengths, token_budget, max_rows)


def test_text2sql_eval_batch_size_is_an_optional_row_cap(monkeypatch):
    text2sql_eval = importlib.import_module("examples.text2sql.text2sql_eval")

    monkeypatch.setattr(sys, "argv", ["text2sql_eval"])
    assert text2sql_eval.parse_args().batch_size is None
    monkeypatch.setattr(sys, "argv", ["text2sql_eval", "--batch_size", "4"])
    assert text2sql_eval.parse_args().batch_size == 4


@pytest.fixture(scope="module")
def long_prompts(texts, qwen_tokenizer) -> list[list[int]]:
    """Real prompts plus one built from the third text repeated 8 times."""
    long_text = "\n\n".join([texts[2]] * 8)
    chats = [[{"role": "user", "content": t}] for t in (*texts, long_text)]
    encoded = qwen_tokenizer.apply_chat_template(
        chats, add_generation_prompt=True, return_dict=True
    )
    return cast("list[list[int]]", encoded["input_ids"])


@pytest.mark.parametrize("path", TRUNCATING_MODULES)
def test_truncate_prompts_keeps_the_last_tokens(
    path, load_functions, long_prompts, capsys
):
    truncate = load_functions(path, "truncate_prompts").truncate_prompts
    longest = len(long_prompts[-1])
    limit = max(len(ids) for ids in long_prompts[:-1])
    max_seq_length = limit + MAX_NEW_TOKENS

    prompts, truncated = truncate(long_prompts, max_seq_length, MAX_NEW_TOKENS)

    assert truncated == 1
    assert prompts[:-1] == long_prompts[:-1]
    assert prompts[-1] == long_prompts[-1][-limit:]
    assert len(prompts[-1]) == limit
    output = capsys.readouterr().out
    assert (
        f"WARNING: 1 of {len(long_prompts)} prompts are longer than {limit} "
        f"tokens (context {max_seq_length} minus max_new_tokens "
        f"{MAX_NEW_TOKENS}; longest: {longest})"
    ) in output
    assert "Fix: raise --max_seq_length" in output


@pytest.mark.parametrize("path", TRUNCATING_MODULES)
def test_truncate_prompts_is_silent_when_every_prompt_fits(
    path, load_functions, long_prompts, capsys
):
    truncate = load_functions(path, "truncate_prompts").truncate_prompts
    max_seq_length = len(long_prompts[-1]) + MAX_NEW_TOKENS

    prompts, truncated = truncate(long_prompts, max_seq_length, MAX_NEW_TOKENS)

    assert (prompts, truncated) == (long_prompts, 0)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("path", TRUNCATING_MODULES)
def test_truncate_prompts_rejects_max_new_tokens_filling_the_context(
    path, load_functions, long_prompts
):
    truncate = load_functions(path, "truncate_prompts").truncate_prompts

    with pytest.raises(ValueError, match="--max_new_tokens 96 leaves no room"):
        truncate(long_prompts, MAX_NEW_TOKENS, MAX_NEW_TOKENS)


def test_gptoss_eval_truncate_rows_keeps_the_start(
    load_functions, long_prompts, capsys
):
    truncate_rows = load_functions(
        "examples/gptoss/gptoss_eval.py", "truncate_rows"
    ).truncate_rows
    limit = max(len(ids) for ids in long_prompts[:-1])

    rows, truncated = truncate_rows(long_prompts, limit)

    assert truncated == 1
    assert rows[:-1] == long_prompts[:-1]
    assert rows[-1] == long_prompts[-1][:limit]
    output = capsys.readouterr().out
    assert (
        f"WARNING: 1 of {len(long_prompts)} rows are longer than "
        f"--max_seq_length={limit} tokens (longest: {len(long_prompts[-1])})"
    ) in output
    assert "Fix: raise --max_seq_length" in output


@pytest.fixture
def fit_conversation(load_functions):
    """``_fit_conversation`` of ``examples/chat.py``."""
    return load_functions(
        "examples/chat.py", "_prompt_token_limit", "_fit_conversation"
    )._fit_conversation


@pytest.fixture(scope="module")
def conversation(load_fixture) -> list[dict[str, str]]:
    """System message, three Dolly exchanges and a new user turn."""
    rows = load_fixture("dolly_pirate_rows.json")
    messages = [{"role": "system", "content": "Answer like a pirate."}]
    for row in rows[:3]:
        messages += [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
    return [*messages, {"role": "user", "content": rows[3]["instruction"]}]


def render(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Prompt token ids of ``messages``, ending with the assistant header."""
    return cast(
        "list[int]",
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False
        ),
    )


def test_chat_keeps_a_conversation_that_fits(
    fit_conversation, conversation, qwen_tokenizer, capsys
):
    prompt = render(qwen_tokenizer, conversation)

    kept, ids = fit_conversation(
        qwen_tokenizer,
        conversation,
        MAX_NEW_TOKENS,
        len(prompt) + MAX_NEW_TOKENS,
    )

    assert (kept, ids) == (conversation, prompt)
    assert capsys.readouterr().out == ""


def test_chat_drops_the_oldest_exchanges_first(
    fit_conversation, conversation, qwen_tokenizer, capsys
):
    # System message, the last exchange and the new user turn.
    expected = [conversation[0], *conversation[-3:]]
    limit = len(render(qwen_tokenizer, expected))

    kept, ids = fit_conversation(
        qwen_tokenizer, conversation, MAX_NEW_TOKENS, limit + MAX_NEW_TOKENS
    )

    assert kept == expected
    assert ids == render(qwen_tokenizer, expected)
    output = capsys.readouterr().out
    assert "the oldest 2 exchange(s) are dropped" in output
    assert "cut off" not in output
    assert "Fix: restart with a larger --max_seq_length" in output


def test_chat_cuts_the_start_of_a_last_message_that_is_too_long(
    fit_conversation, conversation, qwen_tokenizer, capsys
):
    long_turn = {
        "role": "user",
        "content": "\n\n".join(m["content"] for m in conversation[1:]),
    }
    messages = [*conversation[:3], long_turn]
    alone = render(qwen_tokenizer, [conversation[0], long_turn])
    limit = len(alone) // 2

    kept, ids = fit_conversation(
        qwen_tokenizer, messages, MAX_NEW_TOKENS, limit + MAX_NEW_TOKENS
    )

    assert kept == [conversation[0], long_turn]
    assert ids == alone[-limit:]
    output = capsys.readouterr().out
    assert "the oldest 1 exchange(s) are dropped" in output
    assert (
        f"the first {len(alone) - limit} prompt tokens are cut off" in output
    )


def test_chat_rejects_max_new_tokens_filling_the_context(
    fit_conversation, conversation, qwen_tokenizer
):
    with pytest.raises(ValueError, match="leaves no room for a prompt"):
        fit_conversation(
            qwen_tokenizer, conversation, MAX_NEW_TOKENS, MAX_NEW_TOKENS
        )
