"""Tests for the prefs example's data split, encoding and scoring helpers.

``olmo2_1b_preference_rows.json`` holds six real rows of
``allenai/olmo-2-0425-1b-preference-mix`` (full ``chosen``/``rejected``
conversations). Log-probabilities are checked on a tiny real Qwen2
checkpoint from the Hub.
"""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from examples.prefs import prefs_eval


TINY_QWEN2 = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def olmo_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(prefs_eval.SFT_MODEL)


@pytest.fixture
def pairs(load_fixture):
    rows = load_fixture("olmo2_1b_preference_rows.json")
    return Dataset.from_list(
        [{"chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )


def test_split_pairs_is_disjoint_seeded_and_drops_identical(pairs):
    identical = {"chosen": pairs[0]["chosen"], "rejected": pairs[0]["chosen"]}
    dataset = Dataset.from_list([*pairs.to_list(), identical])

    train, held_out = prefs_eval.split_pairs(dataset, 2, 4, seed=1)
    train_again, held_out_again = prefs_eval.split_pairs(dataset, 2, 4, 1)

    assert len(held_out) == 2
    assert len(train) == 4
    assert train.to_list() == train_again.to_list()
    assert held_out.to_list() == held_out_again.to_list()
    rows = train.to_list() + held_out.to_list()
    assert all(row["chosen"] != row["rejected"] for row in rows)
    assert sorted(map(str, rows)) == sorted(map(str, pairs.to_list()))


def test_split_prompt_keeps_shared_turns(pairs):
    for row in pairs:
        prompt, chosen, rejected = prefs_eval.split_prompt(
            row["chosen"], row["rejected"]
        )
        assert prompt
        assert prompt[-1]["role"] == "user"
        assert prompt + chosen == row["chosen"]
        assert prompt + rejected == row["rejected"]
        assert [turn["role"] for turn in chosen] == ["assistant"]
        assert chosen != rejected


def test_encode_response_splits_at_the_assistant_header(pairs, olmo_tokenizer):
    prompt, chosen, _rejected = prefs_eval.split_prompt(
        pairs[0]["chosen"], pairs[0]["rejected"]
    )

    ids, start = prefs_eval.encode_response(
        olmo_tokenizer, prompt, chosen, max_length=4096
    )

    prompt_text = olmo_tokenizer.decode(ids[:start])
    assert prompt_text.endswith("<|assistant|>\n")
    assert olmo_tokenizer.decode(ids[start:]) == (
        chosen[0]["content"] + olmo_tokenizer.eos_token
    )
    short, _ = prefs_eval.encode_response(olmo_tokenizer, prompt, chosen, 10)
    assert short == ids[:10]


def test_encoders_interleave_chosen_and_rejected(pairs, olmo_tokenizer):
    encoded = prefs_eval.encode_conversation_pairs(olmo_tokenizer, pairs, 4096)
    rewardbench_rows = Dataset.from_list(
        [
            {
                "prompt": row["chosen"][0]["content"],
                "chosen": row["chosen"][1]["content"],
                "rejected": row["rejected"][1]["content"],
            }
            for row in pairs
            if len(row["chosen"]) == 2
        ]
    )
    plain = prefs_eval.encode_rewardbench(
        olmo_tokenizer, rewardbench_rows, 4096
    )

    assert len(encoded) == 2 * len(pairs)
    single_turn = [
        encoded[2 * i + k]
        for i, row in enumerate(pairs)
        if len(row["chosen"]) == 2
        for k in (0, 1)
    ]
    assert plain == single_turn


def test_length_batches_respect_budget():
    lengths = [5, 30, 12, 30, 7, 40, 3]

    batches = prefs_eval.length_batches(lengths, token_budget=60)

    assert sorted(i for batch in batches for i in batch) == list(range(7))
    for batch in batches:
        width = max(lengths[i] for i in batch)
        assert lengths[batch[0]] == width
        assert len(batch) == 1 or len(batch) * width <= 60


def test_response_log_probs_batched_equal_single_rows(pairs):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_QWEN2)
    model = AutoModelForCausalLM.from_pretrained(TINY_QWEN2).float().eval()
    sequences = prefs_eval.encode_conversation_pairs(tokenizer, pairs, 256)

    sums, counts = prefs_eval.response_log_probs(
        model, sequences, tokenizer.pad_token_id, token_budget=1024
    )

    for (ids, start), total, count in zip(
        sequences, sums, counts, strict=True
    ):
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids])).logits[0]
        log_probs = logits[:-1].log_softmax(-1)
        expected = sum(
            float(log_probs[t - 1, ids[t]]) for t in range(start, len(ids))
        )
        assert count == len(ids) - start
        assert total == pytest.approx(expected, abs=1e-3)
        assert total < 0


def test_response_log_prob_is_zero_without_response_tokens():
    logits = torch.zeros((4, 10))

    assert prefs_eval.response_log_prob(logits, [1, 2, 3], start=3) == 0.0


def test_pair_accuracy_and_implicit_rewards():
    rewards = prefs_eval.implicit_rewards([-10.0, -12.0], [-11.0, -11.0])

    assert rewards == pytest.approx([0.1, -0.1])
    assert prefs_eval.pair_accuracy([1.0, 2.0, 3.0], [0.0, 2.0, 4.0]) == [
        True,
        False,
        False,
    ]


def test_held_out_report_normalizes_rewards_by_response_length():
    # One pair: chosen gains 2 nats over 10 tokens, rejected 1 nat over 2.
    counts = [10, 2]
    scores = {
        "sft": {"held_out": ([-20.0, -4.0], counts)},
        "adapter": {"held_out": ([-18.0, -3.0], counts)},
    }

    report = prefs_eval.held_out_report(scores)["adapter"]

    assert report["reward"] == 1.0
    assert report["reward_normalized"] == 0.0
    assert report["summed"] == 0.0
    assert report["normalized"] == 0.0


def test_rewardbench_scores_weight_subsets_as_rewardbench():
    right = {"alpacaeval-easy", "math-prm"}
    subsets = [
        name for names in prefs_eval.SECTIONS.values() for name in names
    ]
    correct = [name in right for name in subsets]

    scores = prefs_eval.rewardbench_scores(subsets, correct)

    assert scores["Chat"] == pytest.approx(100 / (100 + 95 + 95 + 28 + 40))
    assert scores["Chat Hard"] == 0.0
    assert scores["Safety"] == 0.0
    # math-prm weighs as much as the six hep-* code subsets together.
    assert scores["Reasoning"] == pytest.approx(0.5)
    assert scores["Overall"] == pytest.approx(
        (scores["Chat"] + scores["Reasoning"]) / 4
    )
