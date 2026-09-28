"""GPU smoke tests: a few PESFT (Unsloth SFT) steps on real data.

Data: ``TeeZee/dolly-15k-pirate-speech`` (train split, downloaded) and
``tests/fixtures/multilingual_thinking_rows.json``, the ``messages`` of
four rows of ``HuggingFaceH4/Multilingual-Thinking`` (train split,
Apache-2.0).

Run with ``uv run pytest --run-cuda tests/test_pesft_cuda.py``. Training
runs in a fresh interpreter so ``import unsloth`` really happens before
``datasets``/``transformers`` (the test session imports those early) and
Unsloth's TRL patches never leak into other tests.
"""

from __future__ import annotations

import dataclasses
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "speftr.json",
    "training_args.json",
)


def _train(output_dir: str, **overrides: object) -> None:
    """Train and save LoRA adapters; executed in a child interpreter."""
    import unsloth  # noqa: F401, I001  # must precede datasets/transformers
    import datasets

    from examples.instruct.instruct import _build_format_batch
    from speftr import PESFT, PESFTConfig

    config = PESFTConfig(
        model_name_or_path="unsloth/Qwen2.5-0.5B-Instruct",
        output_dir=output_dir,
        max_steps=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=3,
        save_strategy="steps",
        save_steps=3,
        train_on_responses=True,
        save_method="lora",
    )
    config = dataclasses.replace(config, **overrides)
    trainer = PESFT(config)
    _, processor = trainer.load_model()
    # Multimodal models (Gemma 4) load a processor wrapping the tokenizer.
    tokenizer = getattr(processor, "tokenizer", processor)
    format_batch = _build_format_batch(
        tokenizer, "Always answer like a pirate."
    )
    dataset = datasets.load_dataset(
        "TeeZee/dolly-15k-pirate-speech", split="train[:48]"
    )
    split = dataset.train_test_split(test_size=8, seed=0)
    trainer.train(split["train"], split["test"], format_batch)
    trainer.save_model()


def _train_gpt_oss(output_dir: str) -> None:
    """Train 2 steps of 4-bit gpt-oss-20b (MoE); executed in a child."""
    import unsloth  # noqa: F401, I001  # must precede datasets/transformers
    import datasets

    from speftr import PESFT, PESFTConfig
    from tests.conftest import FIXTURES_DIR

    config = PESFTConfig(
        model_name_or_path="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        load_in_4bit=True,
        chat_template=None,
        max_seq_length=512,
        lora_r=1,
        output_dir=output_dir,
        max_steps=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
    )
    trainer = PESFT(config)
    _, tokenizer = trainer.load_model()

    def format_batch(batch):
        conversations = batch["messages"]
        # Unsloth also calls the function on a single row to probe it.
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        return [
            tokenizer.apply_chat_template(messages, tokenize=False)
            for messages in conversations
        ]

    fixture = FIXTURES_DIR / "multilingual_thinking_rows.json"
    dataset = datasets.Dataset.from_json(str(fixture))
    trainer.train(dataset, None, format_batch)
    trainer.save_model()


def _run_training(
    output_dir: Path, overrides: str = "", function: str = "_train"
) -> str:
    """Run ``function`` in a fresh interpreter and return its stdout.

    Asserts that the child exited successfully.
    """
    code = (
        f"from tests.test_pesft_cuda import {function}; "
        f"{function}({str(output_dir)!r}{overrides})"
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


def _final_eval_loss(stdout: str) -> float:
    """Return ``eval_loss`` from the ``Evaluation results`` line PESFT prints.

    ``float`` parses the printed value, including ``nan``.
    """
    match = re.search(r"Evaluation results: .*'eval_loss': ([^,}]+)", stdout)
    assert match, stdout[-4000:]
    return float(match.group(1))


def _assert_trained_adapters(tmp_path: Path) -> None:
    """Assert the adapter files exist and every lora_B moved from zero."""
    for name in ADAPTER_FILES:
        assert (tmp_path / name).is_file(), f"missing {name}"

    from safetensors.torch import load_file

    tensors = load_file(tmp_path / "adapter_model.safetensors")
    lora_b = {k: v for k, v in tensors.items() if "lora_B" in k}
    assert lora_b, "no lora_B tensors in the saved adapter"
    # LoRA B starts at zero, so any non-zero B proves gradients flowed.
    zero = [k for k, v in lora_b.items() if not v.any()]
    assert not zero, f"lora_B tensors still zero after training: {zero}"


@pytest.mark.cuda
def test_pesft_trains_and_saves_lora_adapters(tmp_path):
    _run_training(tmp_path)
    _assert_trained_adapters(tmp_path)


@pytest.mark.cuda
def test_pesft_trains_with_8bit_base_model(tmp_path):
    _run_training(
        tmp_path,
        ", load_in_8bit=True, max_steps=1, eval_steps=1, save_steps=1",
    )
    _assert_trained_adapters(tmp_path)


@pytest.mark.cuda
def test_pesft_post_train_eval_loss_is_finite_for_4bit_gemma4(tmp_path):
    stdout = _run_training(
        tmp_path,
        ", model_name_or_path='unsloth/gemma-4-E2B-it-unsloth-bnb-4bit'"
        ", load_in_4bit=True, chat_template='gemma-4'"
        ", instruction_part='<|turn>user\\n'"
        ", response_part='<|turn>model\\n'"
        ", max_steps=2, eval_steps=2, save_steps=2"
        ", per_device_train_batch_size=1, per_device_eval_batch_size=1",
    )
    assert math.isfinite(_final_eval_loss(stdout))


@pytest.mark.cuda
def test_pesft_trains_4bit_gpt_oss_moe(tmp_path):
    _run_training(tmp_path, function="_train_gpt_oss")
    for name in ADAPTER_FILES:
        assert (tmp_path / name).is_file(), f"missing {name}"

    from safetensors.torch import load_file

    tensors = load_file(tmp_path / "adapter_model.safetensors")
    assert any(".experts." in name for name in tensors), "experts untrained"


@pytest.mark.cuda
def test_init_disables_hub_telemetry(tmp_path):
    code = (
        "import huggingface_hub; "
        "from speftr.pesft import PESFT, PESFTConfig; "
        f"PESFT(PESFTConfig(output_dir={str(tmp_path)!r})); "
        "print(huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY)"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().splitlines()[-1] == "True"
