"""CPU unit tests for ``speftr.pesft`` (config, parsing, serialization)."""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
import torch

from speftr.pesft import (
    PESFT,
    PESFTConfig,
    _collect_train_metrics,
    _make_step_timer,
    _modules_in_training_mode,
    _normalize_parameters,
    _resolve_chat_markers,
    _step_time_metrics,
    display_parameters,
    save_parameters_to_json,
)


if TYPE_CHECKING:
    import argparse

    from trl import SFTTrainer


MLP = ["gate_proj", "up_proj", "down_proj"]
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _config_from_cli(*argv: str) -> PESFTConfig:
    parser = PESFTConfig.get_argument_parser()
    return PESFTConfig.from_args(parser.parse_args(list(argv)))


# --- argument parser -------------------------------------------------------


PARSER_DEFAULTS = {
    "model_name_or_path": "unsloth/Qwen2.5-0.5B-Instruct",
    "max_seq_length": 2048,
    "load_in_4bit": False,
    "train_on_responses": False,
    "instruction_part": "<|im_start|>user\n",
    "response_part": "<|im_start|>assistant\n",
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_layers": "all",
    "use_gradient_checkpointing": "unsloth",
    "num_train_epochs": 3,
    "max_steps": -1,
    "per_device_train_batch_size": 32,
    "learning_rate": 2e-4,
    "scheduler": "constant",
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "eval_steps": None,
    "save_steps": None,
    "packing": False,
    "router_aux_loss_coef": 0.0,
    "save_method": "lora",
    "no_validate_save": False,
}


def test_parser_defaults():
    args = PESFTConfig.get_argument_parser().parse_args([])
    parsed = {name: getattr(args, name) for name in PARSER_DEFAULTS}
    assert parsed == PARSER_DEFAULTS


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
    expected = {
        "model_name_or_path": "some/model",
        "max_seq_length": 1024,
        "lora_r": 4,
        "learning_rate": 1e-3,
        "eval_strategy": "steps",
        "eval_steps": 10,
        "save_strategy": "steps",
        "save_steps": 20,
        "train_on_responses": True,
        "packing": True,
        "output_dir": "runs/out",
    }
    assert {name: getattr(config, name) for name in expected} == expected


def test_from_args_ignores_unknown_attributes():
    args = PESFTConfig.get_argument_parser().parse_args([])
    args.dataset = "not-a-config-field"
    args.eval_ratio = 0.1
    config = PESFTConfig.from_args(args)
    assert not hasattr(config, "dataset")
    assert not hasattr(config, "eval_ratio")


def test_from_args_partial_namespace_keeps_dataclass_defaults():
    config = PESFTConfig.from_args(
        cast("argparse.Namespace", types.SimpleNamespace(lora_r=16))
    )
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


def _validate(capsys: pytest.CaptureFixture[str], **overrides) -> str:
    holder = types.SimpleNamespace(config=PESFTConfig(**overrides))
    PESFT._validate_step_synchronization(cast("PESFT", holder))
    output: str = capsys.readouterr().out
    return output


@pytest.mark.parametrize(
    ("eval_strategy", "save_strategy", "eval_steps", "save_steps"),
    [
        ("epoch", "steps", 3, 5),
        ("steps", "epoch", 3, 5),
        ("no", "no", 3, 5),
        ("no", "steps", 3, 5),
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
    assert "e.g. 120" in out
    assert "validated" not in out


# --- _build_training_arguments -------------------------------------------


def _sft_config(tmp_path, **overrides):
    holder = types.SimpleNamespace(
        config=PESFTConfig(output_dir=str(tmp_path), **overrides)
    )
    return PESFT._build_training_arguments(cast("PESFT", holder))


def test_sft_config_carries_sequence_and_padding_settings(tmp_path):
    args = _sft_config(tmp_path, max_seq_length=512, packing=True)
    assert args.max_length == 512
    assert args.packing is True
    # Padded batches by default: padding-free is slower without flash
    # attention.
    assert args.padding_free is False
    assert args.train_sampling_strategy == "group_by_length"


def test_sft_config_loads_best_model_only_with_evaluation_during_training(
    tmp_path,
):
    assert _sft_config(tmp_path).load_best_model_at_end is True
    single_eval = _sft_config(
        tmp_path, eval_strategy="no", save_strategy="steps", save_steps=25
    )
    assert single_eval.load_best_model_at_end is False
    assert single_eval.save_steps == 25


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


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("unsloth", True), ("True", True), ("False", False), ("false", False)],
)
def test_sft_config_gradient_checkpointing_follows_config(
    tmp_path, mode, expected
):
    args = _sft_config(tmp_path, use_gradient_checkpointing=mode)
    assert args.gradient_checkpointing is expected


def test_sft_config_disables_router_aux_loss_by_default(tmp_path):
    assert _sft_config(tmp_path).router_aux_loss_coef == 0.0
    args = _sft_config(tmp_path, router_aux_loss_coef=0.01)
    assert args.router_aux_loss_coef == 0.01


def test_from_args_router_aux_loss_coef():
    config = _config_from_cli("--router_aux_loss_coef", "0.001")
    assert config.router_aux_loss_coef == 0.001


def test_sft_config_forwards_optim(tmp_path):
    assert _sft_config(tmp_path).optim == "adamw_8bit"
    assert _sft_config(tmp_path, optim="adamw_torch").optim == "adamw_torch"


# --- memory and placement options ----------------------------------------


def test_config_rejects_4bit_and_8bit_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PESFTConfig(load_in_4bit=True, load_in_8bit=True)


def test_parser_exposes_memory_and_placement_options():
    args = PESFTConfig.get_argument_parser().parse_args([])
    defaults = PESFTConfig()
    assert args.load_in_8bit is defaults.load_in_8bit is False
    assert args.optim == defaults.optim == "adamw_8bit"
    assert args.device_map is defaults.device_map is None

    config = _config_from_cli(
        "--load_in_8bit",
        "--optim",
        "paged_adamw_8bit",
        "--device_map",
        "balanced",
    )
    assert config.load_in_8bit is True
    assert config.optim == "paged_adamw_8bit"
    assert config.device_map == "balanced"


def _load_kwargs(**overrides):
    holder = types.SimpleNamespace(config=PESFTConfig(**overrides))
    return PESFT._model_load_kwargs(cast("PESFT", holder))


def test_model_load_kwargs_forward_quantization():
    kwargs = _load_kwargs(load_in_8bit=True)
    assert kwargs["load_in_8bit"] is True
    assert kwargs["load_in_4bit"] is False
    assert kwargs["model_name"] == PESFTConfig().model_name_or_path


def test_model_load_kwargs_device_map_only_when_set():
    assert "device_map" not in _load_kwargs()
    assert _load_kwargs(device_map="balanced")["device_map"] == "balanced"


# --- distributed: only the main process writes -----------------------------


def test_save_model_on_non_main_rank_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    holder = types.SimpleNamespace(
        config=PESFTConfig(output_dir=str(tmp_path / "out")),
        model=object(),
        tokenizer=object(),
    )
    PESFT.save_model(cast("PESFT", holder))
    assert not (tmp_path / "out").exists()


# --- modules kept in training mode during evaluation ------------------------


class KernelSwitch(torch.nn.Module):
    """Module whose forward would pick a kernel from ``self.training``."""

    def __init__(self) -> None:
        super().__init__()
        self.lora_dropout = torch.nn.Dropout(0.1)


def test_listed_modules_stay_in_training_mode_during_eval():
    model = torch.nn.Sequential(KernelSwitch(), torch.nn.Dropout(0.1))
    switch, dropout = model[0], model[1]

    with _modules_in_training_mode(model, ["KernelSwitch"]):
        assert model.eval() is model
        assert switch.training
        assert not dropout.training
        assert not switch.lora_dropout.training
        model.train()
        assert switch.lora_dropout.training

    model.eval()
    assert not switch.training
    model.train()
    assert switch.training


def test_empty_class_list_leaves_eval_alone():
    model = torch.nn.Sequential(KernelSwitch())
    with _modules_in_training_mode(model, []):
        model.eval()
        assert not model[0].training
    assert "train" not in vars(model[0])


def test_eval_in_train_mode_parser_default_matches_dataclass():
    args = PESFTConfig.get_argument_parser().parse_args([])
    assert args.eval_in_train_mode == PESFTConfig().eval_in_train_mode == []
    parsed = PESFTConfig.get_argument_parser().parse_args(
        ["--eval_in_train_mode", "A", "B"]
    )
    assert PESFTConfig.from_args(parsed).eval_in_train_mode == ["A", "B"]


class _EvaluateRaises:
    """Trainer stand-in whose ``evaluate`` raises a given exception."""

    def __init__(self, error: Exception) -> None:
        """Store the exception to raise.

        Args:
            error: Raised by ``evaluate``.
        """
        self.error = error

    def evaluate(self) -> dict[str, float]:
        """Raise the stored exception.

        Raises:
            Exception: The stored exception.
        """
        raise self.error


def test_final_evaluation_out_of_memory_is_reported_not_raised(capsys):
    """An OOM in the final evaluation leaves the trained model savable."""
    trainer = _EvaluateRaises(torch.cuda.OutOfMemoryError("no memory"))
    PESFT._run_final_evaluation(cast("SFTTrainer", trainer))
    assert "ran out of GPU memory" in capsys.readouterr().out


def test_final_evaluation_other_errors_propagate():
    """Errors other than OOM are not swallowed."""
    trainer = _EvaluateRaises(RuntimeError("shape mismatch"))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        PESFT._run_final_evaluation(cast("SFTTrainer", trainer))


# --- training metrics and chat markers --------------------------------------


def _fake_cuda(monkeypatch, *, available: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Fake GPU")
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda: 3 * 1024**3
    )
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 4 * 1024**3)


def test_collect_train_metrics_merges_trl_eval_and_gpu(monkeypatch):
    _fake_cuda(monkeypatch, available=True)
    train_output = types.SimpleNamespace(
        global_step=20,
        metrics={"train_runtime": 12.5, "train_samples_per_second": 25.6},
    )
    metrics = _collect_train_metrics(train_output, {"eval_loss": 0.5})  # type: ignore[arg-type]
    assert metrics == {
        "train_runtime": 12.5,
        "train_samples_per_second": 25.6,
        "global_step": 20,
        "eval_loss": 0.5,
        "gpu_name": "Fake GPU",
        "peak_memory_allocated_gib": 3.0,
        "peak_memory_reserved_gib": 4.0,
    }


def test_collect_train_metrics_without_gpu_or_evaluation(monkeypatch):
    _fake_cuda(monkeypatch, available=False)
    train_output = types.SimpleNamespace(
        global_step=3, metrics={"train_runtime": 1.0}
    )
    metrics = _collect_train_metrics(train_output, None)  # type: ignore[arg-type]
    assert metrics == {"train_runtime": 1.0, "global_step": 3}


def test_training_metadata_includes_metrics_after_training(tmp_path):
    holder = types.SimpleNamespace(
        config=PESFTConfig(output_dir=str(tmp_path / "out")),
        training_args=_sft_config(tmp_path / "out"),
        trainer=None,
        train_metrics={"train_runtime": 1.0, "global_step": 3},
    )
    PESFT._save_training_metadata(cast("PESFT", holder))
    written = json.loads((tmp_path / "out" / "train_metrics.json").read_text())
    assert written == {"train_runtime": 1.0, "global_step": 3}


def test_training_metadata_has_no_metrics_before_training(tmp_path):
    holder = types.SimpleNamespace(
        config=PESFTConfig(output_dir=str(tmp_path / "out")),
        training_args=_sft_config(tmp_path / "out"),
        trainer=None,
        train_metrics=None,
    )
    PESFT._save_training_metadata(cast("PESFT", holder))
    assert (tmp_path / "out" / "speftr.json").exists()
    assert not (tmp_path / "out" / "train_metrics.json").exists()


def test_configured_markers_are_used_as_given(qwen_tokenizer):
    config = PESFTConfig(instruction_part="<u>", response_part="<a>")
    markers = _resolve_chat_markers(config, qwen_tokenizer)
    assert (markers.instruction_part, markers.response_part) == ("<u>", "<a>")


def test_empty_markers_are_inferred_from_the_template(qwen_tokenizer):
    config = PESFTConfig(instruction_part="", response_part="")
    markers = _resolve_chat_markers(config, qwen_tokenizer)
    assert markers.instruction_part == "<|im_start|>user\n"
    assert markers.response_part == "<|im_start|>assistant\n"


def test_step_time_metrics_separate_warmup_from_steady_state():
    step_seconds = [60.0, 30.0, 5.0, 2.0, 1.6, 1.5, 1.7, 1.5, 1.6, 9.0]
    metrics = _step_time_metrics(step_seconds)
    assert metrics == {"warmup_seconds": 98.6, "steady_seconds_per_step": 1.6}


def test_step_time_metrics_empty_without_steady_steps():
    assert _step_time_metrics([1.0, 2.0, 3.0]) == {}


def test_step_timer_records_each_step():
    timer, step_seconds = _make_step_timer()
    for _ in range(3):
        timer.on_step_begin(None, None, None)
        timer.on_step_end(None, None, None)
    timer.on_step_end(None, None, None)
    assert len(step_seconds) == 3
    assert all(duration >= 0 for duration in step_seconds)


def test_collect_train_metrics_includes_step_times(monkeypatch):
    _fake_cuda(monkeypatch, available=False)
    train_output = types.SimpleNamespace(global_step=7, metrics={})
    metrics = _collect_train_metrics(
        train_output,  # type: ignore[arg-type]
        None,
        [10.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
    )
    assert metrics == {
        "global_step": 7,
        "warmup_seconds": 14.0,
        "steady_seconds_per_step": 2.0,
    }


# --- full fine-tuning ------------------------------------------------------


def test_full_finetuning_rejects_quantized_bases():
    with pytest.raises(ValueError, match="full_finetuning"):
        PESFTConfig(full_finetuning=True, load_in_4bit=True)
    with pytest.raises(ValueError, match="full_finetuning"):
        PESFTConfig(full_finetuning=True, load_in_8bit=True)


def test_full_finetuning_reaches_the_loader_only_when_set():
    holder = types.SimpleNamespace(config=PESFTConfig(full_finetuning=True))
    assert PESFT._model_load_kwargs(cast("PESFT", holder))["full_finetuning"]
    holder = types.SimpleNamespace(config=PESFTConfig())
    assert "full_finetuning" not in PESFT._model_load_kwargs(
        cast("PESFT", holder)
    )


def test_parser_exposes_full_finetuning():
    parser = PESFTConfig.get_argument_parser()
    assert (
        PESFTConfig.from_args(parser.parse_args([])).full_finetuning is False
    )
    args = parser.parse_args(["--full_finetuning", "--learning_rate", "2e-5"])
    config = PESFTConfig.from_args(args)
    assert config.full_finetuning is True
    assert config.learning_rate == 2e-5
