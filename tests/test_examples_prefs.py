"""Tests for the prefs example's data split, encoding and scoring helpers.

``olmo2_1b_preference_rows.json`` holds six real rows of
``allenai/olmo-2-0425-1b-preference-mix`` (full ``chosen``/``rejected``
conversations). Log-probabilities are checked on tiny real Qwen2 and
Qwen3.5 checkpoints from the Hub.
"""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from examples.prefs import prefs_eval


TINY_QWEN2 = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_QWEN3_5 = (
    "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink"
)


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

    ids, start = prefs_eval.encode_response(olmo_tokenizer, prompt, chosen)

    prompt_text = olmo_tokenizer.decode(ids[:start])
    assert prompt_text.endswith("<|assistant|>\n")
    assert olmo_tokenizer.decode(ids[start:]) == (
        chosen[0]["content"] + olmo_tokenizer.eos_token
    )


def test_encoders_interleave_chosen_and_rejected(pairs, olmo_tokenizer):
    encoded = prefs_eval.encode_conversation_pairs(olmo_tokenizer, pairs)
    plain = prefs_eval.encode_rewardbench(
        olmo_tokenizer, _rewardbench_rows(pairs)
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


def test_length_batches_take_longest_first_and_oversize_alone(
    pairs, qwen_tokenizer
):
    sequences = prefs_eval.encode_conversation_pairs(qwen_tokenizer, pairs)
    lengths = [len(ids) for ids, _ in sequences]
    budget = sorted(lengths)[-2]

    batches = prefs_eval.length_batches(lengths, budget)

    order = [i for batch in batches for i in batch]
    assert [lengths[i] for i in order] == sorted(lengths, reverse=True)
    oversize = [i for i in range(len(lengths)) if lengths[i] > budget]
    assert oversize
    assert all([i] in batches for i in oversize)
    assert max(len(batch) for batch in batches) > 1


def test_truncate_sequences_keeps_the_start_and_counts_cut(
    pairs, qwen_tokenizer
):
    sequences = prefs_eval.encode_conversation_pairs(qwen_tokenizer, pairs)
    lengths = sorted(len(ids) for ids, _ in sequences)
    max_length = lengths[len(lengths) // 2]

    truncated, cut = prefs_eval.truncate_sequences(sequences, max_length)

    assert cut == sum(length > max_length for length in lengths)
    assert 0 < cut < len(sequences)
    assert truncated == [(ids[:max_length], s) for ids, s in sequences]


def _rewardbench_rows(pairs):
    """RewardBench-shaped rows (plain strings) from the single-turn pairs."""
    return Dataset.from_list(
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


def _shortest_rewardbench_pair(tokenizer, pairs):
    """The single RewardBench-shaped row whose longer sequence is shortest.

    Returns:
        That row as a dataset and the length of its longer sequence.
    """
    rows = _rewardbench_rows(pairs)
    plain = prefs_eval.encode_rewardbench(tokenizer, rows)
    pair_lengths = [
        max(len(plain[i][0]), len(plain[i + 1][0]))
        for i in range(0, len(plain), 2)
    ]
    shortest = min(pair_lengths)
    return rows.select([pair_lengths.index(shortest)]), shortest


def test_encode_datasets_warns_about_cut_sequences(
    pairs, qwen_tokenizer, capsys
):
    # The RewardBench set fits max_length; some held-out sequences do not.
    rewardbench, max_length = _shortest_rewardbench_pair(qwen_tokenizer, pairs)
    held_out = prefs_eval.encode_conversation_pairs(qwen_tokenizer, pairs)
    lengths = [len(ids) for ids, _ in held_out]
    cut = sum(length > max_length for length in lengths)
    assert cut > 0

    datasets, truncated = prefs_eval.encode_datasets(
        qwen_tokenizer, pairs, rewardbench, max_length
    )

    assert truncated == {"held_out": cut, "rewardbench": 0}
    assert max(len(ids) for ids, _ in datasets["held_out"]) == max_length
    output = capsys.readouterr().out
    expected = [
        (
            f"WARNING: {cut} of {len(held_out)} held_out sequences are "
            f"longer than --max_length={max_length} tokens (longest: "
            f"{max(lengths)})"
        ),
        "their ends are not scored",
        "Fix: raise --max_length",
        "filter or shorten those rows",
    ]
    assert [text for text in expected if text not in output] == []
    assert "rewardbench sequences" not in output


def _reference_log_prob(model, ids, start):
    """Summed response log-probability from one unpadded full forward."""
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    log_probs = logits[:-1].float().log_softmax(-1)
    return sum(float(log_probs[t - 1, ids[t]]) for t in range(start, len(ids)))


def _load_tiny(name):
    """Load a tiny Hub checkpoint by name; ``+lora`` wraps it in peft."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    repo, _, variant = name.partition("+")
    if variant == "vlm":
        model = AutoModelForImageTextToText.from_pretrained(repo)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo)
    model = model.float().eval()
    if variant == "lora":
        # Random B so the adapter changes the scores.
        config = LoraConfig(
            r=4, target_modules="all-linear", init_lora_weights=False
        )
        model = get_peft_model(model, config).eval()
    return model


@pytest.mark.parametrize(
    "name",
    [TINY_QWEN2, f"{TINY_QWEN2}+lora", TINY_QWEN3_5, f"{TINY_QWEN3_5}+vlm"],
)
def test_response_log_probs_batched_equal_single_rows(
    pairs, name, monkeypatch
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name.partition("+")[0])
    model = _load_tiny(name)
    sequences = prefs_eval.encode_conversation_pairs(tokenizer, pairs)
    # Chunks smaller than the responses exercise the chunk boundaries.
    monkeypatch.setattr(prefs_eval, "HEAD_CHUNK", 7)

    sums, counts = prefs_eval.response_log_probs(
        model, sequences, tokenizer.pad_token_id, token_budget=4096
    )

    for (ids, start), total, count in zip(
        sequences, sums, counts, strict=True
    ):
        assert count == len(ids) - start
        expected = _reference_log_prob(model, ids, start)
        assert total == pytest.approx(expected, abs=1e-3)
        assert total < 0


def test_output_head_applies_final_logit_softcapping():
    model = _load_tiny(TINY_QWEN2)
    hidden = torch.randn(
        3, model.config.hidden_size, generator=torch.Generator().manual_seed(0)
    )
    plain = prefs_eval.output_head(model)(hidden).detach()
    cap = float(plain.abs().max()) / 2
    model.config.final_logit_softcapping = cap

    capped = prefs_eval.output_head(model)(hidden).detach()

    assert torch.allclose(capped, torch.tanh(plain / cap) * cap)
    assert capped.abs().max() < cap


def test_response_log_prob_is_zero_without_response_tokens():
    hidden = torch.zeros((4, 8))

    def head(states):
        return torch.zeros((*states.shape[:-1], 10))

    assert prefs_eval.response_log_prob(hidden, [1, 2, 3], 3, head) == 0.0
    # Truncation can cut the whole response: the start is past the end.
    assert prefs_eval.response_log_prob(hidden, [1, 2, 3], 5, head) == 0.0


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


def test_held_out_report_skips_pairs_without_response_tokens():
    # The first pair's rejected response was cut away entirely.
    counts = [4, 0, 3, 3]
    scores = {
        "sft": {"held_out": ([-8.0, 0.0, -6.0, -9.0], counts)},
        "adapter": {"held_out": ([-9.0, 0.0, -5.0, -9.0], counts)},
    }

    report = prefs_eval.held_out_report(scores)

    assert report["sft"]["pairs"] == 1.0
    assert report["sft"]["summed"] == 1.0
    assert report["adapter"]["reward"] == 1.0


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
