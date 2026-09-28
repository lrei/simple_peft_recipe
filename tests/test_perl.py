"""CPU unit tests for ``speftr.perl`` (config, length maths, GRPOConfig)."""

from __future__ import annotations

import types

import pytest
import torch
import trl

from speftr.perl import (
    PERL,
    PERLConfig,
    _device_map,
    _quantization_config,
)


VLLM_ONLY_KEYS = ("max_tokens", "stop", "stop_token_ids")


def test_parser_defaults():
    args = PERLConfig.get_argument_parser().parse_args([])
    assert args.model_name_or_path == "unsloth/Qwen3-4B-Base"
    assert args.max_seq_length == 2048
    assert args.output_dir == "./models/perl-reasoning"
    assert args.max_steps == PERLConfig.max_steps
    assert args.learning_rate == 1e-5


def test_from_args_parsed_values_land_in_fields():
    parser = PERLConfig.get_argument_parser()
    args = parser.parse_args(
        [
            "--model_name_or_path",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--max_seq_length",
            "512",
            "--output_dir",
            "runs/perl",
            "--max_steps",
            "7",
            "--learning_rate",
            "3e-5",
        ]
    )
    config = PERLConfig.from_args(args)
    assert config.model_name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.max_seq_length == 512
    assert config.output_dir == "runs/perl"
    assert config.max_steps == 7
    assert config.learning_rate == 3e-5
    # Fields not exposed on the CLI keep the recipe defaults.
    assert config.lora_r == 1
    assert config.num_generations == 8


def test_from_args_ignores_unknown_attributes():
    args = PERLConfig.get_argument_parser().parse_args([])
    args.dataset = "chain_sum"
    config = PERLConfig.from_args(args)
    assert not hasattr(config, "dataset")


def test_from_args_none_parses_sys_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--max_steps", "3"])
    assert PERLConfig.from_args().max_steps == 3


# --- _calculate_max_lengths ------------------------------------------------


def _max_lengths(**overrides) -> tuple[int | None, int | None]:
    holder = types.SimpleNamespace(config=PERLConfig(**overrides))
    return PERL._calculate_max_lengths(holder)


def test_max_lengths_explicit_values_kept():
    assert _max_lengths(
        max_seq_length=2048, max_prompt_length=300, max_completion_length=100
    ) == (300, 100)


def test_max_lengths_prompt_defaults_to_half_sequence():
    assert _max_lengths(max_seq_length=1025) == (512, 513)


def test_max_lengths_completion_fills_remaining_sequence():
    assert _max_lengths(max_seq_length=1024, max_prompt_length=200) == (
        200,
        824,
    )


def test_max_lengths_prompt_derived_completion_explicit():
    assert _max_lengths(max_seq_length=1024, max_completion_length=64) == (
        512,
        64,
    )


def test_max_lengths_unbounded_sequence():
    assert _max_lengths(max_seq_length=None) == (None, None)


def test_max_lengths_unbounded_sequence_keeps_explicit_values():
    assert _max_lengths(
        max_seq_length=None, max_prompt_length=128, max_completion_length=32
    ) == (128, 32)


# --- _create_grpo_config ---------------------------------------------------


@pytest.fixture
def cpu_grpo_config(monkeypatch):
    """Let ``trl.GRPOConfig`` build on CPU.

    GRPOConfig defaults to ``bf16=True`` and transformers refuses bf16
    without a GPU; everything else is left untouched.
    """
    real = trl.GRPOConfig

    def build(**kwargs):
        kwargs.setdefault("bf16", False)
        return real(**kwargs)

    monkeypatch.setattr(trl, "GRPOConfig", build)
    return real


def _grpo_config(tokenizer, tmp_path, **overrides):
    # TRL requires the generation batch (per-device batch * accumulation)
    # to be divisible by num_generations.
    overrides.setdefault("per_device_train_batch_size", 8)
    overrides.setdefault("num_generations", 8)
    # A real (model-less) PERL instance: constructing it is CPU-only and
    # keeps the test independent of private helper methods.
    trainer = PERL(PERLConfig(output_dir=str(tmp_path), **overrides))
    trainer.tokenizer = tokenizer
    return trainer._create_grpo_config(128)


def test_grpo_config_forwards_training_fields(
    cpu_grpo_config, qwen_tokenizer, tmp_path
):
    args = _grpo_config(
        qwen_tokenizer,
        tmp_path,
        use_vllm=False,
        learning_rate=2e-5,
        temperature=0.7,
        top_p=0.9,
        top_k=10,
        scheduler="linear",
        max_steps=11,
        save_strategy="steps",
        save_steps=5,
    )
    assert isinstance(args, cpu_grpo_config)
    assert args.output_dir == str(tmp_path)
    assert args.max_completion_length == 128
    assert args.learning_rate == 2e-5
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.top_k == 10
    assert args.lr_scheduler_type == "linear"
    assert args.max_steps == 11
    assert args.save_steps == 5
    assert args.num_generations == 8
    assert args.per_device_train_batch_size == 8


def test_grpo_config_without_vllm_has_no_vllm_generation_kwargs(
    cpu_grpo_config, qwen_tokenizer, tmp_path, capsys
):
    args = _grpo_config(
        qwen_tokenizer,
        tmp_path,
        use_vllm=False,
        stop_sequences=[qwen_tokenizer.eos_token],
        vllm_mode="server",
        vllm_gpu_memory_utilization=0.9,
    )
    assert args.use_vllm is False
    generation_kwargs = args.generation_kwargs or {}
    for key in (*VLLM_ONLY_KEYS, "include_stop_str_in_output"):
        assert key not in generation_kwargs
    # vLLM-only settings are not forwarded when vLLM is disabled.
    assert args.vllm_mode != "server"
    assert args.vllm_gpu_memory_utilization != 0.9
    assert "vLLM" not in capsys.readouterr().out


def test_grpo_config_with_vllm_sets_generation_kwargs(
    cpu_grpo_config, qwen_tokenizer, tmp_path
):
    eos = qwen_tokenizer.eos_token
    args = _grpo_config(
        qwen_tokenizer,
        tmp_path,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        vllm_enable_sleep_mode=False,
        # "</answer>" is not a single token, so it has no stop token id.
        stop_sequences=[eos, "</answer>"],
        include_stop_str_in_output=False,
    )
    assert args.use_vllm is True
    assert args.vllm_mode == "colocate"
    assert args.vllm_gpu_memory_utilization == 0.35
    assert args.vllm_enable_sleep_mode is False
    assert args.generation_kwargs == {
        "max_tokens": 128,
        "stop": [eos, "</answer>"],
        "stop_token_ids": [qwen_tokenizer.convert_tokens_to_ids(eos)],
        "include_stop_str_in_output": False,
    }
    assert args.generation_kwargs["stop_token_ids"] == [
        qwen_tokenizer.eos_token_id
    ]


def test_grpo_config_with_vllm_without_stop_sequences(
    cpu_grpo_config, qwen_tokenizer, tmp_path
):
    args = _grpo_config(
        qwen_tokenizer, tmp_path, use_vllm=True, stop_sequences=None
    )
    assert args.generation_kwargs == {
        "max_tokens": 128,
        "include_stop_str_in_output": True,
    }


def test_grpo_config_with_vllm_without_tokenizer_skips_stop_ids(
    cpu_grpo_config, tmp_path
):
    args = _grpo_config(
        None, tmp_path, use_vllm=True, stop_sequences=["<|im_end|>"]
    )
    assert args.generation_kwargs["stop"] == ["<|im_end|>"]
    assert "stop_token_ids" not in args.generation_kwargs


# --- PERL public surface that needs no model -------------------------------


def test_init_creates_output_dir_and_echoes_config(tmp_path, capsys):
    output_dir = tmp_path / "nested" / "run"
    trainer = PERL(PERLConfig(output_dir=str(output_dir), lora_r=4))
    assert output_dir.is_dir()
    assert trainer.model is None
    assert trainer.tokenizer is None
    assert trainer.trainer is None
    out = capsys.readouterr().out
    assert "lora_r: 4" in out
    assert f"output_dir: {output_dir}" in out


def test_set_pretrained_model_stores_objects(tmp_path, qwen_tokenizer):
    trainer = PERL(PERLConfig(output_dir=str(tmp_path)))
    model = object()
    trainer.set_pretrained_model(model, qwen_tokenizer)
    assert trainer.model is model
    assert trainer.tokenizer is qwen_tokenizer


def test_save_model_before_load_raises(tmp_path):
    trainer = PERL(PERLConfig(output_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="must be loaded"):
        trainer.save_model()


def test_default_config_builds_valid_grpo_config(
    cpu_grpo_config, qwen_tokenizer, tmp_path
):
    trainer = PERL(PERLConfig(output_dir=str(tmp_path), use_vllm=False))
    trainer.tokenizer = qwen_tokenizer
    args = trainer._create_grpo_config(128)
    generation_batch = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    assert generation_batch % args.num_generations == 0


def test_grpo_config_warmup_ratio_is_a_fraction_of_steps(
    cpu_grpo_config, qwen_tokenizer, tmp_path
):
    args = _grpo_config(
        qwen_tokenizer, tmp_path, use_vllm=False, warmup_ratio=0.1
    )
    assert args.warmup_steps == 0.1


def test_save_model_rejects_unknown_method(tmp_path, qwen_tokenizer):
    trainer = PERL(PERLConfig(output_dir=str(tmp_path)))
    trainer.set_pretrained_model(object(), qwen_tokenizer)
    with pytest.raises(ValueError, match="Unsupported save_method"):
        trainer.save_model("merged_4bit")


def test_config_rejects_4bit_with_vllm():
    with pytest.raises(ValueError, match="use_vllm=False"):
        PERLConfig(load_in_4bit=True)


def test_config_accepts_4bit_without_vllm():
    assert PERLConfig(load_in_4bit=True, use_vllm=False).load_in_4bit


TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.mark.parametrize(
    ("save_method", "adapter_saved"),
    [("lora", True), ("merged_16bit", False)],
)
def test_save_model_writes_adapters_or_merged_weights(
    tmp_path, save_method, adapter_saved
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model = get_peft_model(
        base, LoraConfig(r=2, target_modules=["q_proj", "v_proj"])
    )
    trainer = PERL(PERLConfig(output_dir=str(tmp_path)))
    trainer.set_pretrained_model(
        model, AutoTokenizer.from_pretrained(TINY_MODEL)
    )
    trainer.save_model(save_method)

    assert (tmp_path / "adapter_config.json").exists() is adapter_saved
    assert (tmp_path / "config.json").exists() is not adapter_saved
    assert (tmp_path / "tokenizer_config.json").exists()
    if not adapter_saved:
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
        assert not any(
            "lora" in name for name, _ in reloaded.named_parameters()
        )


TINY_GEMMA4 = "trl-internal-testing/tiny-Gemma4ForConditionalGeneration"


def test_load_model_limits_lora_to_language_model_on_multimodal(tmp_path):
    trainer = PERL(
        PERLConfig(
            model_name_or_path=TINY_GEMMA4,
            output_dir=str(tmp_path),
            use_gradient_checkpointing="false",
            lora_r=2,
        )
    )
    model, _ = trainer.load_model()
    lora_modules = [
        name for name, _ in model.named_modules() if name.endswith("lora_A")
    ]
    assert lora_modules
    assert all(".language_model." in name for name in lora_modules)


TINY_QWEN35 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"


def test_load_model_uses_checkpoint_architecture(tmp_path):
    # vLLM loads the architecture named in the checkpoint; the trained model
    # must match it or weight syncing cannot map parameter names.
    trainer = PERL(
        PERLConfig(
            model_name_or_path=TINY_QWEN35,
            output_dir=str(tmp_path),
            use_gradient_checkpointing="false",
            lora_r=2,
        )
    )
    model, _ = trainer.load_model()
    assert type(model.get_base_model()).__name__ == (
        "Qwen3_5ForConditionalGeneration"
    )


# --- Memory and multi-GPU options ------------------------------------------


MEMORY_FLAGS = (
    "load_in_4bit",
    "load_in_8bit",
    "use_vllm",
    "optim",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "use_gradient_checkpointing",
    "use_liger_kernel",
)


def test_parser_memory_defaults_match_dataclass():
    args = PERLConfig.get_argument_parser().parse_args([])
    defaults = PERLConfig()
    for name in MEMORY_FLAGS:
        assert getattr(args, name) == getattr(defaults, name), name


def test_from_args_memory_flags_land_in_fields():
    args = PERLConfig.get_argument_parser().parse_args(
        [
            "--load_in_8bit",
            "--no_vllm",
            "--optim",
            "adamw_torch",
            "--per_device_train_batch_size",
            "4",
            "--gradient_accumulation_steps",
            "2",
            "--use_gradient_checkpointing",
            "false",
            "--use_liger_kernel",
        ]
    )
    config = PERLConfig.from_args(args)
    assert config.load_in_8bit
    assert not config.load_in_4bit
    assert not config.use_vllm
    assert config.optim == "adamw_torch"
    assert config.per_device_train_batch_size == 4
    assert config.gradient_accumulation_steps == 2
    assert config.use_gradient_checkpointing == "false"
    assert config.use_liger_kernel


def test_parser_rejects_unknown_gradient_checkpointing_mode():
    with pytest.raises(SystemExit):
        PERLConfig.get_argument_parser().parse_args(
            ["--use_gradient_checkpointing", "unsloth"]
        )


def test_config_rejects_8bit_with_vllm():
    with pytest.raises(ValueError, match="use_vllm=False"):
        PERLConfig(load_in_8bit=True)


def test_config_accepts_8bit_without_vllm():
    assert PERLConfig(load_in_8bit=True, use_vllm=False).load_in_8bit


def test_config_rejects_4bit_and_8bit_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PERLConfig(load_in_4bit=True, load_in_8bit=True, use_vllm=False)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "auto"),
        ({"WORLD_SIZE": "1", "LOCAL_RANK": "0"}, "auto"),
        ({"WORLD_SIZE": "4", "LOCAL_RANK": "3"}, {"": 3}),
        ({"WORLD_SIZE": "2", "ACCELERATE_USE_FSDP": "true"}, None),
        ({"WORLD_SIZE": "2", "ACCELERATE_USE_FSDP": "True"}, None),
        (
            {
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "1",
                "ACCELERATE_USE_FSDP": "false",
            },
            {"": 1},
        ),
    ],
)
def test_device_map_follows_launcher_environment(monkeypatch, env, expected):
    for name in ("WORLD_SIZE", "LOCAL_RANK", "ACCELERATE_USE_FSDP"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert _device_map() == expected


def test_quantization_config_off_by_default():
    assert (
        _quantization_config(
            load_in_4bit=False, load_in_8bit=False, fsdp=False
        )
        is None
    )


def test_quantization_config_8bit():
    config = _quantization_config(
        load_in_4bit=False, load_in_8bit=True, fsdp=False
    )
    assert config.load_in_8bit
    assert not config.load_in_4bit


@pytest.mark.parametrize(
    ("fsdp", "storage"), [(False, torch.uint8), (True, torch.bfloat16)]
)
def test_quantization_config_4bit_storage_follows_fsdp(fsdp, storage):
    config = _quantization_config(
        load_in_4bit=True, load_in_8bit=False, fsdp=fsdp
    )
    assert config.load_in_4bit
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_quant_storage == storage


def test_grpo_config_forwards_memory_options(
    cpu_grpo_config, qwen_tokenizer, tmp_path
):
    args = _grpo_config(
        qwen_tokenizer,
        tmp_path,
        use_vllm=False,
        optim="adamw_torch",
        use_gradient_checkpointing="false",
        use_liger_kernel=True,
    )
    assert args.optim == "adamw_torch"
    assert args.gradient_checkpointing is False
    assert args.use_liger_kernel is True


def test_grpo_config_default_memory_options(
    cpu_grpo_config, qwen_tokenizer, tmp_path
):
    args = _grpo_config(qwen_tokenizer, tmp_path, use_vllm=False)
    assert args.optim == "adamw_8bit"
    assert args.gradient_checkpointing is True
    assert args.use_liger_kernel is False
