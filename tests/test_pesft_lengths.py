"""CPU unit tests for row lengths, token-budget batches and warnings.

Covers ``speftr.pesft``'s ``max_batch_tokens`` batching, the row-length
check, and the warnings and out-of-memory guidance printed by training.
Row lengths come from real rows: ``dolly_pirate_rows.json`` (six rows of
``TeeZee/dolly-15k-pirate-speech``, train split) and ``tldr_rows.json``
(four rows of ``trl-lib/tldr``), tokenized with the Qwen2.5 tokenizer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from datasets import Dataset
from hypothesis import given, settings
from hypothesis import strategies as st

from speftr.pesft import (
    PESFTConfig,
    _hint_max_batch_tokens,
    _input_lengths,
    _length_metrics,
    _print_token_budget_steps,
    _token_budget_batches,
    _token_counts,
    _TokenBudgetBatchSampler,
    _training_oom_guidance,
    _warn_compile_disabled,
    _warn_joined_batches_without_flash,
    _warn_long_rows,
)
from tests.test_pesft import _config_from_cli


# --- max_batch_tokens and row lengths --------------------------------------


def test_max_batch_tokens_round_trips_through_the_parser():
    assert _config_from_cli().max_batch_tokens is None
    config = _config_from_cli("--max_batch_tokens", "32768")
    assert config.max_batch_tokens == 32768


def test_max_batch_tokens_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        PESFTConfig(max_batch_tokens=0)


def test_max_batch_tokens_rejects_packing():
    with pytest.raises(ValueError, match="packing"):
        PESFTConfig(max_batch_tokens=4096, packing=True)


def _dolly_format(tokenizer):
    """Batched formatting function for the dolly rows (chat template)."""

    def format_batch(batch):
        texts = []
        for instruction, context, response in zip(
            batch["instruction"],
            batch["context"],
            batch["response"],
            strict=True,
        ):
            messages = [
                {"role": "user", "content": f"{instruction}\n\n{context}"},
                {"role": "assistant", "content": response},
            ]
            texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False)
            )
        return texts

    return format_batch


@pytest.fixture(scope="module")
def dolly_rows(load_fixture):
    return Dataset.from_list(load_fixture("dolly_pirate_rows.json"))


@pytest.fixture(scope="module")
def real_lengths(load_fixture, qwen_tokenizer, dolly_rows):
    """Token counts of ten real rows of different lengths."""
    counts = _token_counts(
        dolly_rows, _dolly_format(qwen_tokenizer), qwen_tokenizer
    )
    for row in load_fixture("tldr_rows.json"):
        text = row["prompt"] + row["completion"]
        counts.append(len(qwen_tokenizer(text)["input_ids"]))
    return counts


def test_token_counts_match_the_chat_template(qwen_tokenizer, dolly_rows):
    format_batch = _dolly_format(qwen_tokenizer)
    counts = _token_counts(dolly_rows, format_batch, qwen_tokenizer)
    expected = [
        len(qwen_tokenizer(text)["input_ids"])
        for text in format_batch(dolly_rows.to_dict())
    ]
    assert counts == expected


def test_rows_longer_than_max_seq_length_warn(
    qwen_tokenizer, dolly_rows, capsys
):
    counts = _token_counts(
        dolly_rows, _dolly_format(qwen_tokenizer), qwen_tokenizer
    )
    limit = sorted(counts)[2]
    too_long = sum(count > limit for count in counts)

    assert _warn_long_rows(counts, limit, "train") == too_long

    output = capsys.readouterr().out
    assert (
        f"WARNING: {too_long} of {len(counts)} train rows are longer than "
        f"max_seq_length={limit} tokens (longest: {max(counts)})."
    ) in output
    assert f"truncated to {limit} tokens" in output
    assert "never sees their end" in output
    assert f"raise max_seq_length to at least {max(counts)}" in output
    assert "filter or shorten those rows" in output


def test_rows_within_max_seq_length_do_not_warn(real_lengths, capsys):
    assert _warn_long_rows(real_lengths, max(real_lengths), "eval") == 0
    assert capsys.readouterr().out == ""


def test_length_metrics_count_truncated_rows_per_dataset(
    qwen_tokenizer, dolly_rows, capsys
):
    format_batch = _dolly_format(qwen_tokenizer)
    counts = _token_counts(dolly_rows, format_batch, qwen_tokenizer)
    limit = sorted(counts)[2]
    eval_rows = dolly_rows.select([counts.index(min(counts))])

    metrics, row_lengths = _length_metrics(
        {"train": dolly_rows, "eval": eval_rows},
        format_batch,
        qwen_tokenizer,
        limit,
    )

    assert metrics == {
        "longest_train_row": max(counts),
        "truncated_train_rows": sum(count > limit for count in counts),
        "longest_eval_row": min(counts),
        "truncated_eval_rows": 0,
    }
    assert row_lengths == {"train": counts, "eval": [min(counts)]}
    output = capsys.readouterr().out
    assert output.count("WARNING:") == 1
    assert "train rows are longer than" in output
    assert _length_metrics(
        {"eval": None}, format_batch, qwen_tokenizer, 1
    ) == ({}, {})


@pytest.fixture(scope="module")
def varied_lengths(load_fixture, qwen_tokenizer, dolly_rows, real_lengths):
    """The real rows plus one long row: all their texts joined."""
    texts = _dolly_format(qwen_tokenizer)(dolly_rows.to_dict())
    texts += [
        row["prompt"] + row["completion"]
        for row in load_fixture("tldr_rows.json")
    ]
    long_row = len(qwen_tokenizer("\n".join(texts))["input_ids"])
    return [*real_lengths, long_row]


def test_varied_lengths_suggest_max_batch_tokens(varied_lengths, capsys):
    config = PESFTConfig(per_device_train_batch_size=4, max_seq_length=8192)
    assert max(varied_lengths) > 4 * np.median(varied_lengths)

    _hint_max_batch_tokens(config, varied_lengths)

    percentile = np.percentile(varied_lengths, 90)
    budget = int(np.ceil(4 * percentile / 64) * 64)
    output = capsys.readouterr().out
    assert (
        f"NOTE: the longest train row ({max(varied_lengths)} tokens) is "
        "more than 4x the median row"
    ) in output
    assert "per_device_train_batch_size=4" in output
    assert "may run out of it" in output
    assert f"Suggestion: set max_batch_tokens={budget} " in output
    assert f"90th-percentile row length of {percentile:g} tokens" in output
    assert "rule of thumb" in output


@pytest.mark.parametrize(
    "settings",
    [
        {"max_batch_tokens": 8192},
        {"per_device_train_batch_size": 1},
        {"packing": True},
        {"padding_free": True},
    ],
)
def test_no_batch_hint_without_padded_multi_row_batches(
    settings, varied_lengths, capsys
):
    config = PESFTConfig(max_seq_length=8192, **settings)
    _hint_max_batch_tokens(config, varied_lengths)
    assert capsys.readouterr().out == ""


def test_no_batch_hint_for_similar_lengths(real_lengths, capsys):
    assert max(real_lengths) <= 4 * np.median(real_lengths)
    _hint_max_batch_tokens(PESFTConfig(), real_lengths)
    assert capsys.readouterr().out == ""


def test_batch_hint_counts_rows_as_truncated(varied_lengths, capsys):
    config = PESFTConfig(max_seq_length=int(np.median(varied_lengths)))
    _hint_max_batch_tokens(config, varied_lengths)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("world_size", "accumulation", "expected"),
    [(1, 1, (5, 5)), (1, 2, (5, 3)), (2, 2, (3, 2)), (4, 3, (2, 1))],
)
def test_token_budget_steps_per_epoch(
    world_size, accumulation, expected, real_lengths, capsys
):
    budget = 3 * max(real_lengths) // 2
    batch_count = len(
        _token_budget_batches(real_lengths, 4, budget, padding_free=False)
    )
    assert batch_count == 5

    _print_token_budget_steps(batch_count, accumulation, world_size)

    batches, steps = expected
    output = capsys.readouterr().out
    assert (
        f"Per device and epoch: {batches} training batches, {steps} "
        f"optimizer steps (gradient_accumulation_steps={accumulation})."
    ) in output
    assert "Rows per step vary" in output
    assert "max_steps, warmup_steps, eval_steps, save_steps" in output
    assert "prefer num_train_epochs and warmup_ratio" in output


def test_compile_disable_warns_only_when_fully_disabled(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising=False)
    _warn_compile_disabled()
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "partial")
    _warn_compile_disabled()
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    _warn_compile_disabled()

    output = capsys.readouterr().out
    assert "WARNING: the environment variable UNSLOTH_COMPILE_DISABLE=1" in (
        output
    )
    assert "whole vocabulary at once" in output
    assert "Fix: unset UNSLOTH_COMPILE_DISABLE" in output


@pytest.mark.parametrize(
    ("settings", "flags"),
    [
        ({"padding_free": True}, "padding_free=True"),
        ({"packing": True}, "packing=True"),
        (
            {"padding_free": True, "packing": True},
            "padding_free=True and packing=True",
        ),
    ],
)
def test_joined_batches_without_flash_attention_warn(settings, flags, capsys):
    config = PESFTConfig(**settings)

    _warn_joined_batches_without_flash(config, "sdpa")

    output = capsys.readouterr().out
    assert f"WARNING: {flags} without FlashAttention (attention: sdpa)" in (
        output
    )
    assert "square of the batch's total tokens" in output
    assert "Qwen 3.5" in output
    assert "leave padding_free and packing off" in output
    assert 'attn_implementation="flash_attention_2"' in output


def test_joined_batches_warning_follows_the_loaded_attention(capsys):
    flash = PESFTConfig(padding_free=True, attn_implementation="sdpa")
    _warn_joined_batches_without_flash(flash, "flash_attention_2")
    _warn_joined_batches_without_flash(PESFTConfig(), "sdpa")
    assert capsys.readouterr().out == ""

    eager = PESFTConfig(packing=True, attn_implementation="eager")
    _warn_joined_batches_without_flash(eager, None)
    assert "(attention: eager)" in capsys.readouterr().out


class _OutOfMemoryTrainer:
    """Trainer stand-in whose ``train`` runs out of GPU memory."""

    def __init__(self):
        self.error = torch.OutOfMemoryError("CUDA out of memory.")

    def train(self):
        raise self.error


@pytest.mark.parametrize("max_batch_tokens", [None, 8192])
def test_training_oom_prints_guidance_and_reraises(max_batch_tokens, capsys):
    config = PESFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        max_seq_length=4096,
        max_batch_tokens=max_batch_tokens,
    )
    trainer = _OutOfMemoryTrainer()

    with (
        pytest.raises(torch.OutOfMemoryError) as raised,
        _training_oom_guidance(config, 3000),
    ):
        trainer.train()

    assert raised.value is trainer.error
    output = capsys.readouterr().out
    assert "Training ran out of GPU memory." in output
    assert "Longest training row: 3000 tokens" in output
    assert "per_device_train_batch_size: 4" in output
    assert "gradient_accumulation_steps: 8" in output
    assert f"max_batch_tokens: {max_batch_tokens or 'not set'}" in output
    assert "max_seq_length: 4096" in output
    fixes = [
        "1. Set max_batch_tokens",
        "2. Keep gradient checkpointing on",
        "3. Lower max_seq_length",
        "4. Load the model in 4-bit (load_in_4bit=True)",
        "5. Use a GPU with more memory",
    ]
    positions = [output.index(fix) for fix in fixes]
    assert positions == sorted(positions)


def test_training_without_oom_prints_nothing(capsys):
    with _training_oom_guidance(PESFTConfig(), 10):
        pass
    assert capsys.readouterr().out == ""


def _batch_tokens(batch, lengths, *, padding_free):
    rows = [lengths[index] for index in batch]
    return sum(rows) if padding_free else len(rows) * max(rows)


def _assert_valid_batches(
    batches, lengths, max_rows, max_tokens, padding_free
):
    indices = sorted(index for batch in batches for index in batch)
    assert indices == list(range(len(lengths)))
    for batch in batches:
        assert 1 <= len(batch) <= max_rows
        tokens = _batch_tokens(batch, lengths, padding_free=padding_free)
        assert len(batch) == 1 or tokens <= max_tokens


@pytest.mark.parametrize("padding_free", [False, True])
def test_batches_keep_the_token_budget_and_row_cap(real_lengths, padding_free):
    budget = 3 * max(real_lengths) // 2
    batches = _token_budget_batches(
        real_lengths, 4, budget, padding_free=padding_free
    )
    _assert_valid_batches(batches, real_lengths, 4, budget, padding_free)
    assert any(len(batch) > 1 for batch in batches)


def test_padded_and_padding_free_budgets_count_differently():
    lengths = [100, 400]
    assert _token_budget_batches(lengths, 8, 600, padding_free=True) == [
        [0, 1]
    ]
    # Padded, the short row counts as 400 tokens: 2 x 400 > 600.
    assert _token_budget_batches(lengths, 8, 600, padding_free=False) == [
        [0],
        [1],
    ]


@pytest.mark.parametrize("padding_free", [False, True])
def test_row_above_the_budget_gets_its_own_batch(real_lengths, padding_free):
    longest = real_lengths.index(max(real_lengths))
    budget = max(real_lengths) - 1
    batches = _token_budget_batches(
        real_lengths, 8, budget, padding_free=padding_free
    )
    assert [longest] in batches
    _assert_valid_batches(batches, real_lengths, 8, budget, padding_free)


@settings(max_examples=200, deadline=None)
@given(
    lengths=st.lists(st.integers(1, 5000), min_size=1, max_size=60),
    max_rows=st.integers(1, 16),
    max_tokens=st.integers(1, 20000),
    padding_free=st.booleans(),
)
def test_batches_hold_for_any_lengths(
    lengths, max_rows, max_tokens, padding_free
):
    batches = _token_budget_batches(
        lengths, max_rows, max_tokens, padding_free=padding_free
    )
    _assert_valid_batches(batches, lengths, max_rows, max_tokens, padding_free)


def _sampler_epochs(batches, seed, epochs):
    """Return the batch order of each epoch, as the trainer would see it."""
    current_epoch = [0]
    sampler = _TokenBudgetBatchSampler(batches, seed, lambda: current_epoch[0])
    orders = []
    for epoch in range(epochs):
        current_epoch[0] = epoch
        orders.append(list(sampler))
    return orders


def test_sampler_shuffles_per_epoch_and_covers_every_batch(real_lengths):
    batches = _token_budget_batches(real_lengths, 1, 10**9, padding_free=True)
    orders = _sampler_epochs(batches, seed=42, epochs=3)
    for order in orders:
        assert len(order) == len(batches)
        assert sorted(order) == sorted(batches)
    assert len({tuple(map(tuple, order)) for order in orders}) > 1


def test_sampler_order_is_deterministic_for_a_seed(real_lengths):
    batches = _token_budget_batches(real_lengths, 1, 10**9, padding_free=True)
    first = _sampler_epochs(batches, seed=7, epochs=2)
    again = _sampler_epochs(batches, seed=7, epochs=2)
    other_seed = _sampler_epochs(batches, seed=8, epochs=2)
    assert first == again
    assert first != other_seed


def test_sampler_without_epoch_keeps_length_order(real_lengths):
    batches = _token_budget_batches(real_lengths, 2, 10**9, padding_free=True)
    sampler = _TokenBudgetBatchSampler(batches, 42, None)
    assert list(sampler) == batches
    assert len(sampler) == len(batches)
    assert not hasattr(sampler, "batch_size")


def test_input_lengths_follow_the_dataset_view(qwen_tokenizer, dolly_rows):
    texts = _dolly_format(qwen_tokenizer)(dolly_rows.to_dict())
    tokenized = Dataset.from_dict(
        {"input_ids": qwen_tokenizer(texts)["input_ids"]}
    )
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    assert _input_lengths(tokenized) == lengths
    keep = [index for index, length in enumerate(lengths) if length % 2]
    assert _input_lengths(tokenized.select(keep)) == [
        lengths[index] for index in keep
    ]
