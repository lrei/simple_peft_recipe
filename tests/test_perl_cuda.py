"""GPU smoke test: a couple of PERL (GRPO) steps on reasoning-gym prompts.

Run with ``uv run pytest --run-cuda tests/test_perl_cuda.py``. vLLM is
disabled so the test also runs where vLLM is unavailable (e.g. Python
3.14). Training runs in a fresh interpreter so Unsloth, which patches TRL
on import and may have been imported by other tests, cannot leak in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NUM_GENERATIONS = 4


def _rewards(completions, answer, **kwargs) -> list[float]:
    """Exact-answer reward plus a length term so every group has variance.

    A zero-variance group has zero advantage and hence zero gradient; the
    length term keeps the LoRA update non-zero even if no answer is right.
    """
    from reasoning_gym.utils import extract_answer

    scores = []
    for completion, expected in zip(completions, answer, strict=True):
        text = completion[0]["content"]
        correct = extract_answer(text) == expected
        scores.append(float(correct) + min(len(text), 256) / 256)
    return scores


def _train(output_dir: str, load_in_4bit: bool) -> None:  # noqa: FBT001
    """Run a short GRPO job and save adapters; executed in a child process."""
    import reasoning_gym
    from datasets import Dataset
    from reasoning_gym.utils import SYSTEM_PROMPTS

    from speftr.perl import PERL, PERLConfig

    procedural = reasoning_gym.create_dataset("chain_sum", size=8, seed=0)
    # TRL's conversational prompt format; extra columns reach the rewards.
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPTS["default"]},
                    {"role": "user", "content": item["question"]},
                ],
                "answer": item["answer"],
            }
            for item in procedural
        ]
    )
    config = PERLConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir=output_dir,
        lora_r=8,
        max_steps=2,
        num_generations=NUM_GENERATIONS,
        # TRL: per-device batch * accumulation must divide by generations.
        per_device_train_batch_size=NUM_GENERATIONS,
        max_seq_length=512,
        max_completion_length=64,
        learning_rate=1e-4,
        save_strategy="no",
        load_in_4bit=load_in_4bit,
        use_vllm=False,
    )
    trainer = PERL(config)
    trainer.train(dataset, [_rewards])
    trainer.save_model()

    import bitsandbytes as bnb

    quantized = any(
        isinstance(module, bnb.nn.Linear4bit)
        for module in trainer.model.modules()
    )
    assert quantized is load_in_4bit


@pytest.mark.cuda
@pytest.mark.parametrize("load_in_4bit", [False, True])
def test_perl_grpo_trains_and_saves_lora_adapters(tmp_path, load_in_4bit):
    code = (
        "from tests.test_perl_cuda import _train; "
        f"_train({str(tmp_path)!r}, {load_in_4bit})"
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
    assert "GRPO training complete" in result.stdout

    assert (tmp_path / "adapter_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()

    from safetensors.torch import load_file

    tensors = load_file(tmp_path / "adapter_model.safetensors")
    lora_b = {k: v for k, v in tensors.items() if "lora_B" in k}
    assert lora_b, "no lora_B tensors in the saved adapter"
    # LoRA B starts at zero, so any non-zero B proves gradients flowed.
    zero = [k for k, v in lora_b.items() if not v.any()]
    assert not zero, f"lora_B tensors still zero after training: {zero}"
