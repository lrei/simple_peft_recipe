"""Tests for the helpers of ``examples/gptoss`` (training, eval, inference).

Rows come from ``HuggingFaceH4/Multilingual-Thinking`` (Apache-2.0, train
split) via ``tests/fixtures/multilingual_thinking_rows.json``: the
``messages`` of four rows (Italian, French and two Spanish reasoning
languages). Conversations are rendered with the real gpt-oss tokenizer of
the example's model. The training and eval modules import Unsloth, which
refuses to import without a GPU, so these tests skip on CPU-only hosts.
"""

from __future__ import annotations

import importlib

import pytest

from examples.gptoss import gptoss_inference


try:
    gptoss = importlib.import_module("examples.gptoss.gptoss")
    gptoss_eval = importlib.import_module("examples.gptoss.gptoss_eval")
except (ImportError, NotImplementedError, RuntimeError) as exc:
    pytest.skip(
        f"examples.gptoss needs Unsloth (GPU): {exc}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def conversations(load_fixture) -> list[list[dict]]:
    return [
        row["messages"]
        for row in load_fixture("multilingual_thinking_rows.json")
    ]


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        gptoss.EXAMPLE_DEFAULTS["model_name_or_path"]
    )


def requested_language(conversation: list[dict]) -> str:
    """Language named on the first line of the row's system turn."""
    first_line = conversation[0]["content"].splitlines()[0]
    return first_line.removeprefix("reasoning language: ")


def completion_of(conversation: list[dict], tokenizer) -> str:
    """The rendered assistant turn, as a model would generate it."""
    full = tokenizer.apply_chat_template(conversation, tokenize=False)
    prompt = tokenizer.apply_chat_template(
        gptoss_eval.prompt_messages(conversation),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert full.startswith(prompt)
    return full.removeprefix(prompt)


def test_example_defaults_pin_the_gpt_oss_recipe():
    defaults = gptoss.EXAMPLE_DEFAULTS
    assert defaults["model_name_or_path"] == (
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    )
    assert defaults["load_in_4bit"] is True
    assert defaults["chat_template"] is None
    assert defaults["lora_r"] == 1
    assert defaults["router_aux_loss_coef"] == 0.0
    assert defaults["instruction_part"] == "<|start|>user<|message|>"
    assert defaults["response_part"] == "<|start|>assistant"


def test_parser_applies_example_defaults():
    args = gptoss.build_parser().parse_args([])
    assert args.chat_template is None
    assert args.lora_r == 1
    assert args.train_on_responses is True


def test_format_batch_renders_harmony_channels(
    conversations, gpt_oss_tokenizer
):
    format_batch = gptoss.build_format_batch(gpt_oss_tokenizer)
    texts = format_batch({"messages": conversations})

    assert len(texts) == len(conversations)
    for text, conversation in zip(texts, conversations, strict=True):
        system, user, assistant = conversation
        developer = "<|start|>developer<|message|># Instructions\n\n"
        assert f"{developer}{system['content']}<|end|>" in text
        assert f"<|start|>user<|message|>{user['content']}<|end|>" in text
        assert text.endswith(
            f"<|start|>assistant<|channel|>analysis<|message|>"
            f"{assistant['thinking']}<|end|>"
            f"<|start|>assistant<|channel|>final<|message|>"
            f"{assistant['content']}<|return|>"
        )


def test_format_batch_accepts_the_single_row_probe(
    conversations, gpt_oss_tokenizer
):
    format_batch = gptoss.build_format_batch(gpt_oss_tokenizer)
    single = format_batch({"messages": conversations[0]})
    batched = format_batch({"messages": conversations[:1]})
    assert single == batched


def test_label_mask_trains_only_the_assistant_channels(
    conversations, gpt_oss_tokenizer
):
    label_mask = gptoss_eval.build_label_mask(gpt_oss_tokenizer)
    for conversation in conversations:
        text = gpt_oss_tokenizer.apply_chat_template(
            conversation, tokenize=False
        )
        input_ids = gpt_oss_tokenizer(text, add_special_tokens=False).input_ids
        labels = label_mask({"input_ids": [input_ids]})["labels"][0]
        trained = [
            token
            for token, label in zip(input_ids, labels, strict=True)
            if label != -100
        ]
        assistant = conversation[-1]
        assert gpt_oss_tokenizer.decode(trained) == (
            f"<|channel|>analysis<|message|>{assistant['thinking']}<|end|>"
            f"<|channel|>final<|message|>{assistant['content']}<|return|>"
        )


def test_prompt_messages_drop_the_assistant_turn(conversations):
    prompt = gptoss_eval.prompt_messages(conversations[0])
    assert [message["role"] for message in prompt] == ["system", "user"]


@pytest.mark.parametrize(
    "parse_channels",
    [gptoss_eval.parse_channels, gptoss_inference.parse_channels],
    ids=["eval", "inference"],
)
def test_parse_channels_splits_analysis_and_final(
    parse_channels, conversations, gpt_oss_tokenizer
):
    for conversation in conversations:
        assistant = conversation[-1]
        completion = completion_of(conversation, gpt_oss_tokenizer)
        channels = parse_channels(completion)
        assert channels == {
            "analysis": assistant["thinking"].strip(),
            "final": assistant["content"].strip(),
        }


def test_parse_channels_keeps_reasoning_cut_by_the_budget(
    conversations, gpt_oss_tokenizer
):
    completion = completion_of(conversations[2], gpt_oss_tokenizer)
    cut = completion[: completion.index("<|end|>") - 10]
    channels = gptoss_eval.parse_channels(cut)
    assert list(channels) == ["analysis"]
    assert conversations[2][-1]["thinking"].startswith(channels["analysis"])


def test_reference_reasoning_is_in_the_requested_language(conversations):
    for conversation in conversations:
        language = requested_language(conversation)
        thinking = conversation[-1]["thinking"]
        assert gptoss_eval.is_compliant(language, thinking), language


def test_reasoning_in_another_language_is_not_compliant(conversations):
    italian_thinking = conversations[0][-1]["thinking"]
    assert requested_language(conversations[0]) == "Italian"
    assert not gptoss_eval.is_compliant("French", italian_thinking)
    assert not gptoss_eval.is_compliant("French", None)
    assert not gptoss_eval.is_compliant("French", "")


def test_compliance_rates_per_language_and_overall():
    rates = gptoss_eval.compliance_rates(
        ["Spanish", "French", "Spanish", "Italian"],
        [True, False, False, True],
    )
    assert rates == {
        "French": 0.0,
        "Italian": 1.0,
        "Spanish": 0.5,
        "overall": 0.5,
    }


def test_load_splits_holds_out_a_fixed_eval_split():
    train, held_out = gptoss_eval.load_splits()
    assert (len(train), len(held_out)) == (900, 100)
    # User prompts repeat across rows; the reasoning texts are unique.
    assert not set(train["analysis"]) & set(held_out["analysis"])
    _, again = gptoss_eval.load_splits()
    assert again["analysis"] == held_out["analysis"]


def test_inference_chats_name_the_reasoning_language():
    chats = gptoss_inference.build_chats(
        ["What is 2 + 2?"], "German", "Be brief."
    )
    assert chats == [
        [
            {
                "role": "system",
                "content": "reasoning language: German\n\nBe brief.",
            },
            {"role": "user", "content": "What is 2 + 2?"},
        ]
    ]
    bare = gptoss_inference.build_chats(["Hi"], "French")
    assert bare[0][0]["content"] == "reasoning language: French"
