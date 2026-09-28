"""CPU tests for ``speftr.lora_budget`` (adapter size and capacity).

Fixtures are real rows (long fields trimmed) from:
``TeeZee/dolly-15k-pirate-speech`` (dolly_pirate_rows.json),
``trl-lib/Capybara`` (capybara_rows.jsonl), ``trl-lib/tldr``
(tldr_rows.json), ``trl-internal-testing/zen`` configs
``conversational_prompt_completion`` and ``standard_language_modeling``,
``HuggingFaceH4/llava-instruct-mix-vsft`` (messages only) and
``philschmid/guanaco-sharegpt-style``; all from the train split.
``gpt_oss_20b/config.json`` is the ``openai/gpt-oss-20b`` model config
(Apache-2.0).
"""

from __future__ import annotations

import json
from typing import cast

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from speftr import lora_budget
from speftr.lora_budget import (
    LoraBudgetConfig,
    LoraBudgetError,
    estimate_lora_budget,
)
from speftr.perl import PERLConfig
from tests.conftest import FIXTURES_DIR


TINY_QWEN2 = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_GEMMA4 = "trl-internal-testing/tiny-Gemma4ForConditionalGeneration"
BASE_MODEL = "gpt2"
TARGETS = PERLConfig().target_modules
DOLLY_FIXTURE = "tests/fixtures/dolly_pirate_rows.json"
DOLLY_COLUMNS = ["--prompt_column", "instruction"]
DOLLY_COLUMNS += ["--response_column", "response"]
USER_PART = "<|im_start|>user\n"
ASSISTANT_PART = "<|im_start|>assistant\n"
FORMATTING_FUNC = "tests.test_lora_budget:format_dolly_chatml"
GPT_OSS_20B = str(FIXTURES_DIR / "gpt_oss_20b")
# gpt-oss-20b at rank 1: 24 layers of q/k/v/o (2880 hidden, 64 x 64 query
# and 8 x 64 key/value outputs), and 32 experts per layer with a fused
# gate_up (2880 -> 5760) and a down (2880 -> 2880) projection.
GPT_OSS_ATTENTION = 24 * (2 * (2880 + 4096) + 2 * (2880 + 512))
GPT_OSS_GATE_UP = 24 * 32 * (2880 + 5760)
GPT_OSS_DOWN = 24 * 32 * (2880 + 2880)


def format_dolly_chatml(batch) -> list[str]:
    """Render Dolly rows as ChatML texts (batched, as for PESFT.train)."""
    return [
        f"{USER_PART}{instruction}<|im_end|>\n"
        f"{ASSISTANT_PART}{response}<|im_end|>\n"
        for instruction, response in zip(
            batch["instruction"], batch["response"], strict=True
        )
    ]


def _manual_lora_count(model_name: str, rank: int, scope: str = "") -> int:
    """Sum ``r * (in + out)`` over targeted ``nn.Linear`` layers."""
    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return sum(
        rank * (module.in_features + module.out_features)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.split(".")[-1] in TARGETS
        and scope in name
    )


def _fixture_rows(name: str) -> list[dict]:
    path = FIXTURES_DIR / name
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines()]
    return cast("list[dict]", json.loads(text))


def _fixture_dataset(name: str) -> Dataset:
    return Dataset.from_list(_fixture_rows(name))


def _config(**options) -> LoraBudgetConfig:
    return LoraBudgetConfig(model_name_or_path=TINY_QWEN2, **options)


@pytest.mark.parametrize("rank", [1, 4])
def test_adapter_count_matches_rank_times_in_plus_out(rank):
    counted = lora_budget.count_lora_parameters(TINY_QWEN2, rank, TARGETS)
    assert counted == _manual_lora_count(TINY_QWEN2, rank)


def test_multimodal_counts_only_language_model():
    counted = lora_budget.count_lora_parameters(TINY_GEMMA4, 2, TARGETS)
    language_only = _manual_lora_count(TINY_GEMMA4, 2, ".language_model.")
    assert counted == language_only
    # The vision tower wraps its own q_proj/... Linear layers.
    config = AutoConfig.from_pretrained(TINY_GEMMA4)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    assert any(
        ".vision_tower." in name and name.endswith("q_proj.linear")
        for name, _ in model.named_modules()
    )


def test_moe_experts_are_counted_like_unsloth_adapts_them():
    counted = lora_budget.count_lora_parameters(GPT_OSS_20B, 1, TARGETS)
    # Trainable parameters of PESFT on gpt-oss-20b at rank 1.
    assert counted == 11_556_864
    assert counted == GPT_OSS_ATTENTION + GPT_OSS_GATE_UP + GPT_OSS_DOWN


def test_moe_experts_can_be_excluded():
    counted = lora_budget.count_lora_parameters(
        GPT_OSS_20B, 2, TARGETS, include_experts=False
    )
    assert counted == 2 * GPT_OSS_ATTENTION


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        (["q_proj", "k_proj", "v_proj", "o_proj"], GPT_OSS_ATTENTION),
        (["down_proj"], GPT_OSS_DOWN),
        (["up_proj"], GPT_OSS_GATE_UP),
        (["gate_up_proj", "down_proj"], GPT_OSS_GATE_UP + GPT_OSS_DOWN),
    ],
)
def test_moe_experts_follow_mlp_targets(targets, expected):
    assert lora_budget.count_lora_parameters(GPT_OSS_20B, 1, targets) == (
        expected
    )


def test_moe_experts_default_to_sft_only():
    rl = LoraBudgetConfig(
        model_name_or_path=GPT_OSS_20B, mode="rl", max_steps=1, lora_r=1
    )
    budget = estimate_lora_budget(rl)
    assert budget.expert_parameters == 0
    assert budget.adapter_parameters == GPT_OSS_ATTENTION

    rl.include_experts = True
    budget = estimate_lora_budget(rl)
    assert budget.expert_parameters == GPT_OSS_GATE_UP + GPT_OSS_DOWN
    assert budget.parameters_per_rank == 11_556_864


def test_required_parameters_is_half_the_bits():
    # The post's MATH example: ~10,000 problems x 32 samples, 1 bit each.
    assert lora_budget.required_parameters(320_000) == 160_000


@pytest.mark.parametrize(
    ("required", "expected"),
    [(0, 1), (160_000, 1), (630_784, 1), (630_785, 2), (3_000_000, 5)],
)
def test_minimum_rank_rounds_up(required, expected):
    assert lora_budget.minimum_rank(required, 630_784) == expected


def _parse(*argv: str) -> LoraBudgetConfig:
    return LoraBudgetConfig.from_args(list(argv))


def test_rl_episodes_from_prompts_generations_and_epochs():
    config = _parse(
        *("--model_name_or_path", TINY_QWEN2, "--dataset", DOLLY_FIXTURE),
        *("--mode", "rl", "--num_generations", "32"),
        *("--num_train_epochs", "1"),
    )
    assert lora_budget.rl_episodes(10_000, config) == 320_000


def test_rl_episodes_from_steps_override_epochs():
    config = _parse(
        *("--model_name_or_path", TINY_QWEN2, "--dataset", DOLLY_FIXTURE),
        *("--mode", "rl", "--max_steps", "100"),
        *("--completions_per_step", "16"),
    )
    assert lora_budget.rl_episodes(10_000, config) == 1_600


def test_rl_without_dataset_uses_max_steps():
    budget = estimate_lora_budget(
        _config(mode="rl", max_steps=100, completions_per_step=16)
    )
    assert budget.rows is None
    assert budget.episodes == 1_600
    assert budget.required_parameters == 800


def test_rl_without_dataset_or_max_steps_is_rejected():
    with pytest.raises(LoraBudgetError, match="--dataset is required"):
        estimate_lora_budget(_config(mode="rl"))


def test_responses_only_counts_assistant_tokens(qwen_tokenizer, load_fixture):
    row = load_fixture("dolly_pirate_rows.json")[0]
    messages = [
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]
    full = lora_budget.count_trained_tokens(
        qwen_tokenizer, messages, responses_only=False
    )
    responses = lora_budget.count_trained_tokens(
        qwen_tokenizer, messages, responses_only=True
    )
    expected = qwen_tokenizer(
        row["response"] + "<|im_end|>\n", add_special_tokens=False
    )["input_ids"]
    assert responses == len(expected)
    assert responses < full


def test_sample_size_extrapolates_to_whole_dataset(qwen_tokenizer):
    common = ("--model_name_or_path", TINY_QWEN2, "--dataset", DOLLY_FIXTURE)
    config = _parse(*common, *DOLLY_COLUMNS)
    dataset = lora_budget.load_rows(config)
    data_format = lora_budget.detect_format(dataset.column_names, config)
    exact, extrapolated = lora_budget.estimate_sft_tokens(
        dataset, qwen_tokenizer, data_format, config
    )
    assert not extrapolated
    assert exact == sum(
        lora_budget.count_trained_tokens(
            qwen_tokenizer,
            lora_budget.row_to_sample(row, data_format),
            responses_only=False,
        )
        for row in dataset
    )

    config = _parse(*common, *DOLLY_COLUMNS, "--sample_size", "3")
    estimate, extrapolated = lora_budget.estimate_sft_tokens(
        dataset, qwen_tokenizer, data_format, config
    )
    assert extrapolated
    assert estimate > 0


def test_max_samples_truncates_rows():
    config = _parse(
        *("--model_name_or_path", TINY_QWEN2, "--dataset", DOLLY_FIXTURE),
        *("--max_samples", "2"),
    )
    assert len(lora_budget.load_rows(config)) == 2


@pytest.mark.parametrize(
    ("fixture", "expected_format"),
    [
        ("capybara_rows.jsonl", "conversational (messages)"),
        ("guanaco_sharegpt_rows.json", "conversational (conversations)"),
        ("llava_instruct_messages_rows.json", "conversational (messages)"),
        ("tldr_rows.json", "prompt-completion (prompt, completion)"),
        (
            "zen_conversational_prompt_completion_rows.json",
            "prompt-completion (prompt, completion)",
        ),
        ("zen_language_modeling_rows.json", "language modeling (text)"),
    ],
)
def test_standard_formats_are_detected(fixture, expected_format):
    config = _config(dataset=str(FIXTURES_DIR / fixture))
    budget = estimate_lora_budget(config)
    assert budget.data_format == expected_format
    assert budget.rows == len(_fixture_rows(fixture))
    assert budget.trained_tokens
    assert budget.trained_tokens > 0


def test_unknown_layout_error_lists_columns_and_flags():
    with pytest.raises(LoraBudgetError) as error:
        estimate_lora_budget(_config(dataset=DOLLY_FIXTURE))
    message = str(error.value)
    assert "instruction, context, response, category" in message
    assert "--prompt_column" in message
    assert "--formatting_func" in message


def test_missing_named_column_is_reported():
    with pytest.raises(LoraBudgetError, match=r"\['question'\] not in"):
        estimate_lora_budget(
            _config(
                dataset=DOLLY_FIXTURE,
                prompt_column=["question"],
                response_column="response",
            )
        )


def test_responses_only_rejects_plain_text():
    fixture = str(FIXTURES_DIR / "zen_language_modeling_rows.json")
    with pytest.raises(LoraBudgetError, match="needs a response part"):
        estimate_lora_budget(_config(dataset=fixture, responses_only=True))


def test_prompt_columns_are_joined_in_order_skipping_empty():
    rows = _fixture_rows("dolly_pirate_rows.json")
    config = _config(
        prompt_column=["instruction", "context"],
        response_column="response",
        system_column="category",
    )
    data_format = lora_budget.detect_format(list(rows[0]), config)
    with_context = lora_budget.row_to_sample(rows[0], data_format)
    without_context = lora_budget.row_to_sample(rows[3], data_format)
    assert with_context == [
        {"role": "system", "content": rows[0]["category"]},
        {
            "role": "user",
            "content": f"{rows[0]['instruction']}\n\n{rows[0]['context']}",
        },
        {"role": "assistant", "content": rows[0]["response"]},
    ]
    assert without_context[1]["content"] == rows[3]["instruction"]
    assert str(data_format) == (
        "prompt-response chat (instruction, context, response, category)"
    )


def test_content_parts_keep_only_text():
    row = _fixture_rows("llava_instruct_messages_rows.json")[0]
    messages = lora_budget.normalize_messages(row["messages"])
    assert messages[0] == {"role": "user", "content": "Who wrote this book?\n"}
    assert messages[1] == {"role": "assistant", "content": "Donna Eden"}


def test_sharegpt_messages_map_to_roles():
    row = _fixture_rows("guanaco_sharegpt_rows.json")[0]
    messages = lora_budget.normalize_messages(row["conversations"])
    assert [message["role"] for message in messages[:2]] == [
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == row["conversations"][0]["value"]


def test_prompt_completion_responses_only_counts_completion(qwen_tokenizer):
    row = _fixture_rows("tldr_rows.json")[0]
    sample = (row["prompt"], row["completion"])
    full = lora_budget.count_trained_tokens(
        qwen_tokenizer, sample, responses_only=False
    )
    completion = lora_budget.count_trained_tokens(
        qwen_tokenizer, sample, responses_only=True
    )
    eos = qwen_tokenizer.eos_token
    joined = qwen_tokenizer(row["prompt"] + row["completion"] + eos)
    assert full == len(joined["input_ids"])
    alone = qwen_tokenizer(row["completion"] + eos)["input_ids"]
    assert abs(completion - len(alone)) <= 1
    assert completion < full


def test_conversational_prompt_completion_counts_completion_turn(
    qwen_tokenizer,
):
    row = _fixture_rows("zen_conversational_prompt_completion_rows.json")[0]
    config = _config()
    data_format = lora_budget.detect_format(list(row), config)
    sample = lora_budget.row_to_sample(row, data_format)
    completion = lora_budget.count_trained_tokens(
        qwen_tokenizer, sample, responses_only=True
    )
    expected = qwen_tokenizer("Beautiful.<|im_end|>\n")["input_ids"]
    assert completion == len(expected)


def test_formatting_func_counts_formatted_texts(qwen_tokenizer):
    dataset = _fixture_dataset("dolly_pirate_rows.json")
    budget = estimate_lora_budget(
        _config(formatting_func=format_dolly_chatml), dataset
    )
    texts = format_dolly_chatml(dataset.to_dict())
    assert budget.data_format == "formatting_func"
    assert budget.trained_tokens == sum(
        len(qwen_tokenizer(text)["input_ids"]) for text in texts
    )


def test_formatting_func_responses_only_matches_chat_responses():
    common = {"dataset": DOLLY_FIXTURE, "responses_only": True}
    formatted = estimate_lora_budget(
        _config(
            formatting_func=FORMATTING_FUNC,
            instruction_part=USER_PART,
            response_part=ASSISTANT_PART,
            **common,
        )
    )
    chat = estimate_lora_budget(
        _config(
            prompt_column=["instruction"], response_column="response", **common
        )
    )
    assert formatted.trained_tokens == chat.trained_tokens


def test_formatting_func_responses_only_needs_markers():
    with pytest.raises(LoraBudgetError, match="--instruction_part"):
        estimate_lora_budget(
            _config(
                dataset=DOLLY_FIXTURE,
                formatting_func=FORMATTING_FUNC,
                responses_only=True,
            )
        )


def test_formatting_func_marker_missing_from_text_is_reported():
    with pytest.raises(LoraBudgetError, match="not found in formatted text"):
        estimate_lora_budget(
            _config(
                dataset=DOLLY_FIXTURE,
                formatting_func=FORMATTING_FUNC,
                responses_only=True,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )
        )


def test_formatting_func_needs_module_and_function():
    with pytest.raises(LoraBudgetError, match=r"module\.path:function"):
        estimate_lora_budget(
            _config(dataset=DOLLY_FIXTURE, formatting_func="tests")
        )


def _write_local(rows: list[dict], directory, suffix: str) -> str:
    dataset = Dataset.from_list(rows)
    path = directory / f"dolly{suffix}"
    writers = {
        ".json": lambda: path.write_text(json.dumps(rows)),
        ".jsonl": lambda: dataset.to_json(path),
        ".csv": lambda: dataset.to_csv(path, index=False),
        ".tsv": lambda: dataset.to_csv(path, index=False, sep="\t"),
        ".parquet": lambda: dataset.to_parquet(path),
    }
    writers[suffix]()
    return str(path)


@pytest.mark.parametrize(
    "suffix", [".json", ".jsonl", ".csv", ".tsv", ".parquet"]
)
def test_local_tabular_files_load_by_suffix(tmp_path, suffix):
    rows = _fixture_rows("dolly_pirate_rows.json")
    path = _write_local(rows, tmp_path, suffix)
    dataset = lora_budget.load_rows(_config(dataset=path))
    assert dataset.column_names == list(rows[0])
    assert dataset[3]["instruction"] == rows[3]["instruction"]


def test_local_arrow_file_loads(tmp_path):
    rows = _fixture_rows("dolly_pirate_rows.json")
    Dataset.from_list(rows).save_to_disk(tmp_path / "saved")
    arrow_file = next((tmp_path / "saved").glob("*.arrow"))
    dataset = lora_budget.load_rows(_config(dataset=str(arrow_file)))
    assert len(dataset) == len(rows)


def test_local_txt_file_is_one_text_per_line(tmp_path):
    rows = _fixture_rows("dolly_pirate_rows.json")
    path = tmp_path / "responses.txt"
    path.write_text("\n".join(row["response"] for row in rows) + "\n")
    budget = estimate_lora_budget(_config(dataset=str(path)))
    assert budget.rows == len(rows)
    assert budget.data_format == "language modeling (text)"


def test_save_to_disk_directory_loads(tmp_path):
    rows = _fixture_rows("dolly_pirate_rows.json")
    Dataset.from_list(rows).save_to_disk(tmp_path / "saved")
    dataset = lora_budget.load_rows(_config(dataset=str(tmp_path / "saved")))
    assert len(dataset) == len(rows)


def test_saved_dataset_dict_uses_split(tmp_path):
    rows = _fixture_rows("dolly_pirate_rows.json")
    splits = DatasetDict(
        train=Dataset.from_list(rows[:4]), test=Dataset.from_list(rows[4:])
    )
    splits.save_to_disk(tmp_path / "saved")
    path = str(tmp_path / "saved")
    assert len(lora_budget.load_rows(_config(dataset=path))) == 4
    assert len(lora_budget.load_rows(_config(dataset=path, split="test"))) == 2
    with pytest.raises(LoraBudgetError, match="splits: train, test"):
        lora_budget.load_rows(_config(dataset=path, split="validation"))


def test_data_files_glob_is_forwarded(tmp_path):
    rows = _fixture_rows("dolly_pirate_rows.json")
    for index in range(2):
        Dataset.from_list(rows).to_json(tmp_path / f"part{index}.jsonl")
    config = _config(
        dataset="json", data_files=[str(tmp_path / "part*.jsonl")]
    )
    assert len(lora_budget.load_rows(config)) == 2 * len(rows)


def test_unknown_suffix_is_rejected(tmp_path):
    path = tmp_path / "rows.xlsx"
    path.write_bytes(b"")
    with pytest.raises(LoraBudgetError, match=r"unsupported file type"):
        lora_budget.load_rows(_config(dataset=str(path)))


def test_conversations_need_a_chat_template():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    row = _fixture_rows("capybara_rows.jsonl")[0]
    messages = lora_budget.normalize_messages(row["messages"])
    with pytest.raises(LoraBudgetError, match="no chat template"):
        lora_budget.count_trained_tokens(
            tokenizer, messages, responses_only=False
        )


def _report(monkeypatch, capsys, *argv: str) -> dict[str, str]:
    monkeypatch.setattr(
        "sys.argv",
        [
            "lora_budget",
            *("--model_name_or_path", TINY_QWEN2, "--dataset", DOLLY_FIXTURE),
            *argv,
        ],
    )
    lora_budget.main()
    lines = capsys.readouterr().out.splitlines()
    pairs = [line.split(":", 1) for line in lines]
    return {key.strip(): value.strip() for key, value in pairs}


def test_main_sft_reports_tokens_and_rank(monkeypatch, capsys):
    report = _report(monkeypatch, capsys, *DOLLY_COLUMNS)
    expected = _manual_lora_count(TINY_QWEN2, 8)
    assert report["Adapter params"] == f"{expected:,} (rank 8)"
    assert report["Rows"] == "6 (sft)"
    assert report["Format"] == "prompt-response chat (instruction, response)"
    assert "Trained tokens" in report
    assert "Episodes" not in report
    assert report["Minimum rank"].endswith("rank 8 suffices")


def test_main_rl_reports_episodes(monkeypatch, capsys):
    report = _report(monkeypatch, capsys, "--mode", "rl")
    assert report["Adapter params"].endswith("(rank 1)")
    # 6 prompts x 8 generations x 2 epochs (PERL defaults).
    assert report["Episodes"] == "96 (1 bit per episode)"
    assert report["Required params"] == "48 (estimate: bits / 2)"
    assert "Trained tokens" not in report


def test_main_reports_moe_expert_share(monkeypatch, capsys):
    report = _report(
        monkeypatch,
        capsys,
        *("--model_name_or_path", GPT_OSS_20B, "--mode", "rl"),
        *("--include_experts", "true", "--lora_r", "1"),
    )
    experts = GPT_OSS_GATE_UP + GPT_OSS_DOWN
    assert report["Adapter params"] == "11,556,864 (rank 1)"
    assert report["MoE experts"].startswith(f"{experts:,} of these")
    no_experts = _report(monkeypatch, capsys, *DOLLY_COLUMNS, "--lora_r", "1")
    assert "MoE experts" not in no_experts


def test_main_reports_extrapolated_response_tokens(monkeypatch, capsys):
    report = _report(
        monkeypatch,
        capsys,
        *DOLLY_COLUMNS,
        *("--responses_only", "--sample_size", "3"),
    )
    assert report["Response tokens"].endswith(
        "(extrapolated from 3 sampled rows)"
    )


def test_main_exits_with_message_on_bad_input(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="error: no standard dataset format"):
        _report(monkeypatch, capsys)


@pytest.mark.parametrize(
    ("option", "column", "expected_format"),
    [
        ("messages_column", "conversations", "conversational (conversations)"),
        ("text_column", "instruction", "language modeling (instruction)"),
    ],
)
def test_explicit_column_options_override_detection(
    option, column, expected_format
):
    config = _config(**{option: column})
    columns = ["conversations", "instruction", "messages", "text"]
    data_format = lora_budget.detect_format(columns, config)
    assert str(data_format) == expected_format
