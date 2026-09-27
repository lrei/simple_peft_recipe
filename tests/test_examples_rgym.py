"""Tests for the reasoning-gym example's dataset wrapper and rewards."""

from __future__ import annotations

import pytest
import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS

from examples.rgym.rgym import (
    ReasoningGymDataset,
    accuracy_reward,
    format_reward,
    to_hf_dataset,
)


@pytest.fixture(scope="module")
def chain_sum(qwen_tokenizer):
    procedural = reasoning_gym.create_dataset("chain_sum", size=4, seed=0)
    return ReasoningGymDataset(
        qwen_tokenizer,
        procedural,
        SYSTEM_PROMPTS["default"],
        developer_role="system",
    )


def _items(dataset: ReasoningGymDataset) -> list[dict]:
    return [dataset.data[i] for i in range(len(dataset))]


def _completion(answer: str) -> str:
    return f"<think>adding step by step</think>\n<answer>{answer}</answer>"


def test_dataset_item_prompt_uses_chat_template(chain_sum):
    assert len(chain_sum) == 4
    example = chain_sum[0]
    item = example["item"]
    assert item["question"] in example["prompt"]
    assert SYSTEM_PROMPTS["default"] in example["prompt"]
    assert example["prompt"].startswith("<|im_start|>system\n")
    assert example["prompt"].endswith("<|im_start|>assistant\n")


def test_dataset_without_developer_role_omits_system_prompt(
    chain_sum, qwen_tokenizer
):
    plain = ReasoningGymDataset(
        qwen_tokenizer, chain_sum.data, SYSTEM_PROMPTS["default"]
    )
    assert SYSTEM_PROMPTS["default"] not in plain[0]["prompt"]


def test_accuracy_reward_correct_answers_score_one(chain_sum):
    items = _items(chain_sum)
    completions = [_completion(item["answer"]) for item in items]
    rewards = accuracy_reward(completions, chain_sum, item=items)
    assert rewards == [1.0] * len(items)


def test_accuracy_reward_wrong_or_missing_answer_scores_lower(chain_sum):
    items = _items(chain_sum)[:3]
    wrong = str(int(items[0]["answer"]) + 1)
    completions = [
        _completion(items[0]["answer"]),
        _completion(wrong),
        f"The answer is {items[2]['answer']} (no answer tags).",
    ]
    rewards = accuracy_reward(completions, chain_sum, item=items)
    assert rewards[0] == 1.0
    assert rewards[1] < 1.0
    assert rewards[2] < 1.0


def test_accuracy_reward_requires_items(chain_sum):
    with pytest.raises(ValueError, match="'item' argument"):
        accuracy_reward([_completion("1")], chain_sum)


def test_accuracy_reward_length_mismatch(chain_sum):
    items = _items(chain_sum)
    with pytest.raises(ValueError, match="same length"):
        accuracy_reward([_completion("1")], chain_sum, item=items)


def test_accuracy_reward_items_must_be_dicts(chain_sum):
    with pytest.raises(TypeError, match="dictionary"):
        accuracy_reward([_completion("1")], chain_sum, item=["1"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>reasoning</think>\n<answer>42</answer>", 1.0),
        ("<think>reasoning</think> 42", 0.5),
        ("<think>reasoning <answer>42</answer>", 0.75),
        ("<answer>42", 0.25),
        ("42", 0.0),
        ("", 0.0),
    ],
)
def test_format_reward_counts_tags(text, expected):
    assert format_reward([text]) == [expected]


def test_format_reward_one_score_per_completion_ignores_kwargs(chain_sum):
    items = _items(chain_sum)
    completions = [_completion(item["answer"]) for item in items]
    assert format_reward(completions, item=items) == [1.0] * len(items)


def test_to_hf_dataset_keeps_prompts_and_scorable_items(chain_sum):
    converted = to_hf_dataset(chain_sum)
    assert len(converted) == len(chain_sum)
    for index, row in enumerate(converted):
        original = chain_sum[index]
        assert row["prompt"] == original["prompt"]
        # Items survive the Arrow round trip well enough to be scored.
        assert (
            chain_sum.data.score_answer(row["item"]["answer"], row["item"])
            == 1.0
        )
