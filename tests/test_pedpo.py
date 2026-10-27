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

from speftr.pedpo import PEDPO, PEDPOConfig, _device_map, _warn_long_pairs


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


def _fields(obj: object, expected: dict[str, object]) -> dict[str, object]:
    """Return the attributes of ``obj`` named in ``expected``."""
    return {name: getattr(obj, name) for name in expected}


def test_recipe_defaults():
    expected = {
        "lora_r": 1,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "learning_rate": 5e-6,
        "scheduler": "constant",
        "beta": 0.1,
        "loss_type": ["sigmoid"],
        "max_length": 1024,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "precompute_ref_log_probs": False,
        "optim": "adamw_8bit",
        "use_liger_kernel": False,
        "report_to": "none",
    }
    assert _fields(PEDPOConfig(), expected) == expected


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
    expected = {
        "output_dir": str(tmp_path),
        "beta": 0.2,
        "loss_type": ["sigmoid", "sft"],
        "learning_rate": 1e-5,
        "max_length": 512,
        "precompute_ref_log_probs": True,
        "precompute_ref_batch_size": 16,
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
        "max_steps": 7,
        "save_steps": 5,
        "seed": 11,
    }
    assert _fields(args, expected) == expected


def test_dpo_config_defaults(cpu_dpo_config, tmp_path):
    expected = {
        "beta": 0.1,
        "loss_type": ["sigmoid"],
        "learning_rate": 5e-6,
        "lr_scheduler_type": "constant",
        "max_length": 1024,
        "precompute_ref_log_probs": False,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "use_liger_kernel": False,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
    }
    assert _fields(_dpo_config(tmp_path), expected) == expected


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


def _check_training_logs(dpo_trainer: trl.DPOTrainer | None) -> None:
    """Check the prompt split and the logged train and eval losses."""
    assert dpo_trainer is not None
    # Full conversations: TRL split off the shared prompt.
    assert "prompt" in dpo_trainer.train_dataset.column_names
    history = dpo_trainer.state.log_history
    losses = [float(log["loss"]) for log in history if "loss" in log]
    assert losses
    assert all(math.isfinite(loss) for loss in losses)
    assert any("eval_loss" in log for log in history)


def _train_and_check_adapter(output_dir: str) -> None:
    """Train one step, check the logs and the saved adapter."""
    _allow_cpu_dpo_config()
    path = Path(output_dir)
    rows = _fixture_rows()
    trainer = _tiny_trainer(path)
    trainer.train(rows, eval_dataset=rows.select(range(2)))
    _check_training_logs(trainer.trainer)
    model = trainer.model
    assert model is not None
    # The reference is a frozen copy of the adapter; only one is saved.
    assert set(model.peft_config) == {"default", "ref"}
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


def _train_long_pairs(output_dir: str) -> None:
    """Train one step with max_length=128, below two of the six pairs.

    With the tiny model's Qwen chat template the six full conversations
    are 54/59, 163/59, 137/179, 100/111, 77/66 and 94/92 tokens
    (chosen/rejected), so 2 pairs exceed 128 and the longest is 179;
    every prompt is shorter than 128, so TRL drops none.
    """
    _allow_cpu_dpo_config()
    rows = _fixture_rows()
    _tiny_trainer(Path(output_dir)).train(rows, rows.select(range(2)))


def _train_out_of_memory(output_dir: str) -> None:
    """Run ``PEDPO.train`` with a ``DPOTrainer.train`` that runs out of memory.

    Exits with status 0 only when the same ``torch.OutOfMemoryError``
    comes back out of ``PEDPO.train``.
    """
    import torch

    _allow_cpu_dpo_config()
    error = torch.OutOfMemoryError("CUDA out of memory.")

    def out_of_memory(*_args: object, **_kwargs: object) -> None:
        raise error

    trl.DPOTrainer.train = out_of_memory
    with pytest.raises(torch.OutOfMemoryError) as raised:
        _tiny_trainer(Path(output_dir)).train(_fixture_rows())
    assert raised.value is error


def _run_in_fresh_interpreter(function: str, output_dir: Path) -> str:
    """Run ``function(output_dir)`` in a new Python process.

    Importing Unsloth (other test modules do) patches TRL's trainers for
    the whole process; PEDPO runs without Unsloth, so its training tests
    run in a clean interpreter, as PEDPO does in real use.

    Returns:
        What the process printed to stdout.
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
    return result.stdout


def test_train_extracts_prompt_and_saves_only_trained_adapter(tmp_path):
    _run_in_fresh_interpreter("_train_and_check_adapter", tmp_path)


def test_save_merged_drops_all_adapters(tmp_path):
    _run_in_fresh_interpreter("_train_and_check_merged", tmp_path)


def test_pairs_longer_than_max_length_warn_and_train(tmp_path):
    output = _run_in_fresh_interpreter("_train_long_pairs", tmp_path)

    assert (
        "WARNING: 2 of 6 train pairs are longer than max_length=128 tokens."
    ) in output
    assert (
        "2 pairs (longest: 179 tokens) are truncated to 128 tokens"
    ) in output
    assert (
        "WARNING: 1 of 2 eval pairs are longer than max_length=128 tokens."
    ) in output
    assert "never sees their end" in output
    assert "Fix: raise max_length" in output
    assert "filter or shorten those pairs" in output
    assert "left out of training" not in output


def test_training_oom_prints_guidance_and_reraises(tmp_path):
    output = _run_in_fresh_interpreter("_train_out_of_memory", tmp_path)

    assert "DPO training ran out of GPU memory." in output
    assert "Longest training pair: 179 tokens" in output
    assert "per_device_train_batch_size: 2 pairs" in output
    assert "gradient_accumulation_steps: 1" in output
    assert "max_length: 128" in output
    fixes = [
        "1. Lower per_device_train_batch_size",
        "2. Keep gradient checkpointing on",
        "3. Lower max_length",
        "4. Load the model in 4-bit (load_in_4bit=True)",
        "5. Use a GPU with more memory",
    ]
    positions = [output.index(fix) for fix in fixes]
    assert positions == sorted(positions)


def test_pairs_dropped_by_trl_are_counted_in_the_warning(capsys):
    # Boundary lengths: TRL already dropped 3 pairs whose prompt filled
    # max_length; of the 2 kept, one is longer than max_length.
    assert _warn_long_pairs([50, 200], 128, "train", dropped=3) == 1

    output = capsys.readouterr().out
    assert "WARNING: 4 of 5 train pairs are longer than max_length=128" in (
        output
    )
    assert "1 pairs (longest: 200 tokens) are truncated" in output
    assert "3 pairs are left out of training because their prompt" in output


def test_pairs_within_max_length_do_not_warn(capsys):
    assert _warn_long_pairs([50, 128], 128, "eval", dropped=0) == 0
    assert capsys.readouterr().out == ""
