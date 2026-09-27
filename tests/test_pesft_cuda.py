"""GPU smoke test: a few PESFT (Unsloth SFT) steps on real pirate data.

Run with ``uv run pytest --run-cuda tests/test_pesft_cuda.py``. Training
runs in a fresh interpreter so ``import unsloth`` really happens before
``datasets``/``transformers`` (the test session imports those early) and
Unsloth's TRL patches never leak into other tests.
"""

from __future__ import annotations

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


def _train(output_dir: str) -> None:
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
    trainer = PESFT(config)
    _, tokenizer = trainer.load_model()
    format_batch = _build_format_batch(
        tokenizer, "Always answer like a pirate."
    )
    dataset = datasets.load_dataset(
        "TeeZee/dolly-15k-pirate-speech", split="train[:48]"
    )
    split = dataset.train_test_split(test_size=8, seed=0)
    trainer.train(split["train"], split["test"], format_batch)
    trainer.save_model()


@pytest.mark.cuda
def test_pesft_trains_and_saves_lora_adapters(tmp_path):
    code = (
        f"from tests.test_pesft_cuda import _train; _train({str(tmp_path)!r})"
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
