"""CPU unit tests for ``speftr.pedpo`` (config, CLI, DPOConfig, saving).

``olmo2_1b_preference_rows.json`` holds six real rows (train split) of
``allenai/olmo-2-0425-1b-preference-mix`` (ODC-BY-1.0): full
``chosen``/``rejected`` conversations without a ``prompt`` column.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import trl
from datasets import Dataset

from speftr.pedpo import PEDPO, PEDPOConfig, _device_map


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
CLI_FIELDS = (
    "model_name_or_path",
    "output_dir",
    "num_train_epochs",
    "max_steps",
    "learning_rate",
    "lora_r",
    "beta",
    "loss_type",
    "max_length",
    "precompute_ref_log_probs",
    "precompute_ref_batch_size",
    "load_in_4bit",
    "load_in_8bit",
    "optim",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "use_gradient_checkpointing",
    "use_liger_kernel",
)


@pytest.fixture
def cpu_dpo_config(monkeypatch):
    """Let ``trl.DPOConfig`` build without a GPU.

    DPOConfig defaults to ``bf16=True`` and transformers refuses bf16
    without a GPU; everything else is left untouched.
    """
    real = trl.DPOConfig

    def build(**kwargs):
        kwargs.setdefault("bf16", False)
        return real(**kwargs)

    monkeypatch.setattr(trl, "DPOConfig", build)
    return real


def _dpo_config(tmp_path, **overrides):
    trainer = PEDPO(PEDPOConfig(output_dir=str(tmp_path), **overrides))
    return trainer._create_dpo_config()


# --- Config and CLI --------------------------------------------------------


def test_recipe_defaults():
    config = PEDPOConfig()
    assert config.lora_r == 1
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.0
    assert config.learning_rate == 5e-6
    assert config.scheduler == "constant"
    assert config.beta == 0.1
    assert config.loss_type == ["sigmoid"]
    assert config.max_length == 1024
    assert config.per_device_train_batch_size == 4
    assert config.gradient_accumulation_steps == 4
    assert config.num_train_epochs == 1
    assert config.precompute_ref_log_probs is False
    assert config.optim == "adamw_8bit"
    assert config.use_liger_kernel is False
    assert config.report_to == "none"


def test_parser_defaults_match_dataclass():
    args = PEDPOConfig.get_argument_parser().parse_args([])
    defaults = PEDPOConfig()
    for name in CLI_FIELDS:
        assert getattr(args, name) == getattr(defaults, name), name


def test_from_args_parsed_values_land_in_fields():
    args = PEDPOConfig.get_argument_parser().parse_args(
        [
            *("--model_name_or_path", TINY_MODEL),
            *("--beta", "0.05"),
            *("--loss_type", "sigmoid", "sft"),
            *("--max_length", "512"),
            "--precompute_ref_log_probs",
            *("--precompute_ref_batch_size", "32"),
            "--load_in_4bit",
            *("--use_gradient_checkpointing", "false"),
            *("--lora_r", "4"),
        ]
    )
    config = PEDPOConfig.from_args(args)
    assert config.model_name_or_path == TINY_MODEL
    assert config.beta == 0.05
    assert config.loss_type == ["sigmoid", "sft"]
    assert config.max_length == 512
    assert config.precompute_ref_log_probs
    assert config.precompute_ref_batch_size == 32
    assert config.load_in_4bit
    assert config.use_gradient_checkpointing == "false"
    assert config.lora_r == 4


def test_from_args_none_parses_sys_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--max_steps", "3"])
    assert PEDPOConfig.from_args().max_steps == 3


def test_config_rejects_4bit_and_8bit_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PEDPOConfig(load_in_4bit=True, load_in_8bit=True)


def test_config_rejects_liger_with_precomputed_reference():
    with pytest.raises(ValueError, match="use_liger_kernel"):
        PEDPOConfig(use_liger_kernel=True, precompute_ref_log_probs=True)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "auto"),
        ({"WORLD_SIZE": "1", "LOCAL_RANK": "0"}, "auto"),
        ({"WORLD_SIZE": "2", "LOCAL_RANK": "1"}, {"": 1}),
    ],
)
def test_device_map_follows_launcher_environment(monkeypatch, env, expected):
    for name in ("WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert _device_map() == expected


# --- DPOConfig forwarding --------------------------------------------------


def test_dpo_config_forwards_fields(cpu_dpo_config, tmp_path):
    args = _dpo_config(
        tmp_path,
        beta=0.2,
        loss_type=["sigmoid", "sft"],
        learning_rate=1e-5,
        max_length=512,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=16,
        use_gradient_checkpointing="false",
        optim="adamw_torch",
        max_steps=7,
        save_strategy="steps",
        save_steps=5,
        random_state=11,
    )
    assert isinstance(args, cpu_dpo_config)
    assert args.output_dir == str(tmp_path)
    assert args.beta == 0.2
    assert args.loss_type == ["sigmoid", "sft"]
    assert args.learning_rate == 1e-5
    assert args.max_length == 512
    assert args.precompute_ref_log_probs is True
    assert args.precompute_ref_batch_size == 16
    assert args.gradient_checkpointing is False
    assert args.optim == "adamw_torch"
    assert args.max_steps == 7
    assert args.save_steps == 5
    assert args.seed == 11


def test_dpo_config_defaults(cpu_dpo_config, tmp_path):
    args = _dpo_config(tmp_path)
    assert args.beta == 0.1
    assert args.loss_type == ["sigmoid"]
    assert args.learning_rate == 5e-6
    assert args.lr_scheduler_type == "constant"
    assert args.max_length == 1024
    assert args.precompute_ref_log_probs is False
    assert args.gradient_checkpointing is True
    assert args.optim == "adamw_8bit"
    assert args.use_liger_kernel is False
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 4


# --- PEDPO without training ------------------------------------------------


def test_init_creates_output_dir_and_echoes_config(tmp_path, capsys):
    output_dir = tmp_path / "nested" / "run"
    trainer = PEDPO(PEDPOConfig(output_dir=str(output_dir), beta=0.3))
    assert output_dir.is_dir()
    assert trainer.model is None
    assert trainer.tokenizer is None
    assert trainer.trainer is None
    assert "beta: 0.3" in capsys.readouterr().out


def test_save_model_before_load_raises(tmp_path):
    trainer = PEDPO(PEDPOConfig(output_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="must be loaded"):
        trainer.save_model()


def test_save_model_rejects_unknown_method(tmp_path, qwen_tokenizer):
    trainer = PEDPO(PEDPOConfig(output_dir=str(tmp_path)))
    trainer.set_pretrained_model(object(), qwen_tokenizer)
    with pytest.raises(ValueError, match="Unsupported save_method"):
        trainer.save_model("merged_4bit")


# --- One DPO step on a tiny model ------------------------------------------


def _tiny_trainer(tmp_path, **overrides) -> PEDPO:
    config = PEDPOConfig(
        model_name_or_path=TINY_MODEL,
        output_dir=str(tmp_path),
        lora_r=2,
        max_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_length=128,
        optim="adamw_torch",
        use_gradient_checkpointing="false",
        save_strategy="no",
        **overrides,
    )
    return PEDPO(config)


def _fixture_rows() -> Dataset:
    """Load the six real preference rows."""
    fixture = FIXTURES_DIR / "olmo2_1b_preference_rows.json"
    return Dataset.from_list(json.loads(fixture.read_text(encoding="utf-8")))


def _allow_cpu_dpo_config() -> None:
    """Default ``trl.DPOConfig`` to ``bf16=False`` in this process.

    The same patch as the ``cpu_dpo_config`` fixture, for code that runs
    in a child interpreter where fixtures do not apply.
    """
    real = trl.DPOConfig

    def build(**kwargs: object) -> trl.DPOConfig:
        kwargs.setdefault("bf16", False)
        return real(**kwargs)

    trl.DPOConfig = build


def _train_and_check_adapter(output_dir: str) -> None:
    """Train one step, check the prompt split and the saved adapter."""
    _allow_cpu_dpo_config()
    path = Path(output_dir)
    rows = _fixture_rows()
    trainer = _tiny_trainer(path)
    trainer.train(rows, eval_dataset=rows.select(range(2)))

    # Full conversations: TRL split off the shared prompt.
    assert "prompt" in trainer.trainer.train_dataset.column_names
    history = trainer.trainer.state.log_history
    losses = [float(log["loss"]) for log in history if "loss" in log]
    assert losses
    assert all(math.isfinite(loss) for loss in losses)
    assert any("eval_loss" in log for log in history)
    # The reference is a frozen copy of the adapter; only one is saved.
    assert set(trainer.model.peft_config) == {"default", "ref"}
    trainer.save_model()
    assert (path / "adapter_model.safetensors").is_file()
    assert (path / "tokenizer_config.json").is_file()
    assert not (path / "ref").exists()


def _train_and_check_merged(output_dir: str) -> None:
    """Train one step, save merged weights and check no adapter remains."""
    from transformers import AutoModelForCausalLM

    _allow_cpu_dpo_config()
    path = Path(output_dir)
    trainer = _tiny_trainer(path)
    trainer.train(_fixture_rows())
    trainer.save_model("merged_16bit")

    assert not (path / "adapter_config.json").exists()
    reloaded = AutoModelForCausalLM.from_pretrained(path)
    assert not any("lora" in name for name, _ in reloaded.named_parameters())


def _run_in_fresh_interpreter(function: str, output_dir: Path) -> None:
    """Run ``function(output_dir)`` in a new Python process.

    Importing Unsloth (other test modules do) patches TRL's trainers for
    the whole process; PEDPO runs without Unsloth, so its training tests
    run in a clean interpreter, as PEDPO does in real use.
    """
    code = (
        f"from tests.test_pedpo import {function}; "
        f"{function}({str(output_dir)!r})"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        result.stdout[-4000:] + result.stderr[-4000:]
    )


def test_train_extracts_prompt_and_saves_only_trained_adapter(tmp_path):
    _run_in_fresh_interpreter("_train_and_check_adapter", tmp_path)


def test_save_merged_drops_all_adapters(tmp_path):
    _run_in_fresh_interpreter("_train_and_check_merged", tmp_path)
