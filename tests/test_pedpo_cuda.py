"""GPU smoke test: two PEDPO (DPO) steps on real preference pairs.

Run with ``uv run pytest --run-cuda tests/test_pedpo_cuda.py``. The rows
are ``olmo2_1b_preference_rows.json``: six real rows (train split) of
``allenai/olmo-2-0425-1b-preference-mix`` (ODC-BY-1.0), full conversations
without a ``prompt`` column. The model is ``allenai/OLMo-2-0425-1B-SFT``,
the SFT model that preference mix was built for. Training runs in a fresh
interpreter so Unsloth, which patches TRL on import and may have been
imported by other tests, cannot leak in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "olmo2_1b_preference_rows.json"


def _train(output_dir: str, options: dict) -> None:
    """Run two DPO steps and save the adapter; executed in a child process.

    ``options`` are extra ``PEDPOConfig`` fields.
    """
    import json
    import math

    from datasets import Dataset

    from speftr.pedpo import PEDPO, PEDPOConfig

    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    dataset = Dataset.from_list(rows)
    config = PEDPOConfig(
        model_name_or_path="allenai/OLMo-2-0425-1B-SFT",
        output_dir=output_dir,
        max_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_length=512,
        save_strategy="no",
        **options,
    )
    trainer = PEDPO(config)
    trainer.train(dataset, eval_dataset=dataset.select(range(2)))
    trainer.save_model()

    history = trainer.trainer.state.log_history
    losses = [float(log["loss"]) for log in history if "loss" in log]
    assert len(losses) == config.max_steps, history
    assert all(math.isfinite(loss) for loss in losses), losses
    precomputed = "ref_chosen_logps" in trainer.trainer.train_dataset.features
    assert precomputed is config.precompute_ref_log_probs


@pytest.mark.cuda
@pytest.mark.parametrize(
    "options",
    [{}, {"precompute_ref_log_probs": True}, {"load_in_4bit": True}],
    ids=["reference-per-step", "precomputed-reference", "4bit"],
)
def test_pedpo_trains_and_saves_lora_adapter(tmp_path, options):
    code = (
        "from tests.test_pedpo_cuda import _train; "
        f"_train({str(tmp_path)!r}, {options!r})"
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
    assert "DPO training complete" in result.stdout
    assert "Evaluation results" in result.stdout

    assert (tmp_path / "adapter_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert not (tmp_path / "ref").exists()

    from safetensors.torch import load_file

    tensors = load_file(tmp_path / "adapter_model.safetensors")
    lora_b = {k: v for k, v in tensors.items() if "lora_B" in k}
    assert lora_b, "no lora_B tensors in the saved adapter"
    # LoRA B starts at zero, so any non-zero B proves gradients flowed.
    zero = [k for k, v in lora_b.items() if not v.any()]
    assert not zero, f"lora_B tensors still zero after training: {zero}"
