"""CPU unit tests for ``speftr.pesft`` (config, parsing, serialization)."""

from __future__ import annotations

import json
import types
from dataclasses import dataclass

import pytest

from speftr.pesft import (
    PESFT,
    PESFTConfig,
    _normalize_parameters,
    display_parameters,
    save_parameters_to_json,
)


MLP = ["gate_proj", "up_proj", "down_proj"]
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _config_from_cli(*argv: str) -> PESFTConfig:
    parser = PESFTConfig.get_argument_parser()
    return PESFTConfig.from_args(parser.parse_args(list(argv)))


# --- argument parser -------------------------------------------------------


def test_parser_defaults():
    args = PESFTConfig.get_argument_parser().parse_args([])
    assert args.model_name_or_path == "unsloth/Qwen2.5-0.5B-Instruct"
    assert args.max_seq_length == 2048
    assert args.load_in_4bit is False
    assert args.train_on_responses is False
    assert args.instruction_part == "<|im_start|>user\n"
    assert args.response_part == "<|im_start|>assistant\n"
    assert args.lora_r == 8
    assert args.lora_alpha == 32
    assert args.lora_layers == "all"
    assert args.use_gradient_checkpointing == "unsloth"
    assert args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 32
    assert args.learning_rate == 2e-4
    assert args.scheduler == "constant"
    assert args.eval_strategy == "epoch"
    assert args.save_strategy == "epoch"
    assert args.eval_steps is None
    assert args.save_steps is None
    assert args.packing is False
    assert args.save_method == "lora"
    assert args.no_validate_save is False


def test_parser_defaults_match_dataclass_defaults():
    assert _config_from_cli() == PESFTConfig()


def test_parser_rejects_invalid_choice():
    parser = PESFTConfig.get_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--lora_layers", "embeddings"])


# --- from_args ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("layers", "expected"),
    [("all", MLP + ATTENTION), ("attention", ATTENTION), ("mlp", MLP)],
)
def test_from_args_lora_layers_map_to_target_modules(layers, expected):
    config = _config_from_cli("--lora_layers", layers)
    assert config.lora_layers == layers
    assert config.target_modules == expected


def test_from_args_validate_save_flag():
    assert _config_from_cli().validate_save is True
    assert _config_from_cli("--no_validate_save").validate_save is False


def test_from_args_parsed_values_land_in_fields():
    config = _config_from_cli(
        "--model_name_or_path",
        "some/model",
        "--max_seq_length",
        "1024",
        "--lora_r",
        "4",
        "--learning_rate",
        "1e-3",
        "--eval_strategy",
        "steps",
        "--eval_steps",
        "10",
        "--save_strategy",
        "steps",
        "--save_steps",
        "20",
        "--train_on_responses",
        "--packing",
        "--output_dir",
        "runs/out",
    )
    assert config.model_name_or_path == "some/model"
    assert config.max_seq_length == 1024
    assert config.lora_r == 4
    assert config.learning_rate == 1e-3
    assert config.eval_strategy == "steps"
    assert config.eval_steps == 10
    assert config.save_strategy == "steps"
    assert config.save_steps == 20
    assert config.train_on_responses is True
    assert config.packing is True
    assert config.output_dir == "runs/out"


def test_from_args_ignores_unknown_attributes():
    args = PESFTConfig.get_argument_parser().parse_args([])
    args.dataset = "not-a-config-field"
    args.eval_ratio = 0.1
    config = PESFTConfig.from_args(args)
    assert not hasattr(config, "dataset")
    assert not hasattr(config, "eval_ratio")


def test_from_args_partial_namespace_keeps_dataclass_defaults():
    config = PESFTConfig.from_args(types.SimpleNamespace(lora_r=16))
    assert config.lora_r == 16
    # Fields absent from the namespace keep the dataclass defaults.
    assert config.max_seq_length == PESFTConfig().max_seq_length
    assert config.target_modules == PESFTConfig().target_modules
    assert config.validate_save is True


def test_from_args_none_parses_sys_argv(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["prog", "--lora_layers", "mlp", "--lora_r", "2"]
    )
    config = PESFTConfig.from_args()
    assert config.lora_r == 2
    assert config.target_modules == MLP


# --- parameter serialization -----------------------------------------------


def test_normalize_parameters_dataclass_instance():
    config = PESFTConfig(lora_r=3)
    data = _normalize_parameters(config)
    assert isinstance(data, dict)
    assert data["lora_r"] == 3
    assert data["target_modules"] == config.target_modules
    # asdict deep-copies, so mutating the result leaves the config intact.
    data["target_modules"].append("lm_head")
    assert "lm_head" not in config.target_modules


def test_normalize_parameters_mapping_is_copied():
    source = types.MappingProxyType({"a": 1, "b": "x"})
    data = _normalize_parameters(source)
    assert data == {"a": 1, "b": "x"}
    assert type(data) is dict


def test_normalize_parameters_rejects_dataclass_class():
    with pytest.raises(TypeError, match="instance, not a class"):
        _normalize_parameters(PESFTConfig)


@pytest.mark.parametrize("value", [42, "lora_r=8", [("lora_r", 8)], None])
def test_normalize_parameters_rejects_other_types(value):
    with pytest.raises(TypeError, match="dataclass instance or mapping"):
        _normalize_parameters(value)


def test_display_parameters_prints_sorted_keys(capsys):
    @dataclass
    class Params:
        zeta: int = 1
        alpha: str = "a"
        mid: float = 0.5

    result = display_parameters(Params())
    assert result == {"zeta": 1, "alpha": "a", "mid": 0.5}

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "PESFT parameters:"
    assert lines[1:] == ["  alpha: a", "  mid: 0.5", "  zeta: 1"]


def test_save_parameters_to_json_round_trips(tmp_path, capsys):
    config = PESFTConfig(output_dir=str(tmp_path), lora_r=5)
    path = save_parameters_to_json(config, str(tmp_path))

    assert path == tmp_path / "speftr.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == _normalize_parameters(config)
    assert PESFTConfig(**loaded) == config
    assert list(loaded) == sorted(loaded)
    assert str(path) in capsys.readouterr().out


def test_save_parameters_to_json_accepts_mapping(tmp_path):
    path = save_parameters_to_json({"b": 2, "a": [1, 2]}, str(tmp_path))
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == {"a": [1, 2], "b": 2}
    assert text.index('"a"') < text.index('"b"')


# --- PESFT._validate_step_synchronization ----------------------------------


def _validate(capsys, **overrides) -> str:
    holder = types.SimpleNamespace(config=PESFTConfig(**overrides))
    PESFT._validate_step_synchronization(holder)
    return capsys.readouterr().out


@pytest.mark.parametrize(
    ("eval_strategy", "save_strategy", "eval_steps", "save_steps"),
    [
        ("epoch", "steps", 3, 5),
        ("steps", "epoch", 3, 5),
        ("no", "no", 3, 5),
        ("steps", "steps", None, 5),
        ("steps", "steps", 3, None),
    ],
)
def test_step_sync_silent_when_not_applicable(
    capsys, eval_strategy, save_strategy, eval_steps, save_steps
):
    out = _validate(
        capsys,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
    )
    assert out == ""


def test_step_sync_validated_when_multiple(capsys):
    out = _validate(
        capsys,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=100,
    )
    assert "validated" in out
    assert "save_steps (100)" in out
    assert "eval_steps (50)" in out
    assert "Warning" not in out


def test_step_sync_warns_and_suggests_next_multiple(capsys):
    out = _validate(
        capsys,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=30,
        save_steps=100,
    )
    assert "Warning" in out
    assert "not a multiple" in out
    assert "e.g., 120" in out
    assert "validated" not in out


# --- _build_training_arguments -------------------------------------------


def _sft_config(tmp_path, **overrides):
    holder = types.SimpleNamespace(
        config=PESFTConfig(output_dir=str(tmp_path), **overrides)
    )
    return PESFT._build_training_arguments(holder)


def test_sft_config_carries_sequence_and_padding_settings(tmp_path):
    args = _sft_config(tmp_path, max_seq_length=512, packing=True)
    assert args.max_length == 512
    assert args.packing is True
    # Padded batches by default: padding-free is slower without flash
    # attention.
    assert args.padding_free is False
    assert args.train_sampling_strategy == "group_by_length"


def test_sft_config_warmup_ratio_takes_precedence_as_fraction(tmp_path):
    assert _sft_config(tmp_path, warmup_ratio=0.1).warmup_steps == 0.1
    assert _sft_config(tmp_path, warmup_steps=7).warmup_steps == 7


def test_sft_config_forwards_optimisation_fields(tmp_path):
    args = _sft_config(
        tmp_path,
        learning_rate=5e-5,
        scheduler="cosine",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=9,
    )
    assert args.learning_rate == 5e-5
    assert args.lr_scheduler_type == "cosine"
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 2
    assert args.max_steps == 9
