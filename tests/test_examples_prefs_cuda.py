"""GPU tests for ``prefs_eval.response_log_probs`` on Qwen3.5 0.8B.

Data: ``tests/fixtures/olmo2_1b_preference_rows.json``, six real rows of
``allenai/olmo-2-0425-1b-preference-mix`` (ODC-BY). Model:
``Qwen/Qwen3.5-0.8B`` (downloaded): float32 for the parity test, bf16
for the long sequence.

Run with ``uv run pytest --run-cuda tests/test_examples_prefs_cuda.py``.
Each test runs in a fresh interpreter so the measured peak GPU memory
belongs to that test alone.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-0.8B"
# Tokens of the long sequence: Qwen3.5's context is 262,144.
LONG_TOKENS = 131_072
GIB = 1024**3


def _load(dtype_name: str = "bfloat16"):
    """Load the tokenizer, the model on the GPU and the fixture pairs.

    Args:
        dtype_name: Name of the ``torch`` dtype to load the weights in.

    Returns:
        The tokenizer, the model in eval mode and the fixture rows as a
        ``Dataset`` with ``chosen`` and ``rejected`` conversations.
    """
    import json

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tests.conftest import FIXTURES_DIR

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=getattr(torch, dtype_name), device_map="cuda"
    ).eval()
    fixture = FIXTURES_DIR / "olmo2_1b_preference_rows.json"
    rows = json.loads(fixture.read_text(encoding="utf-8"))
    pairs = Dataset.from_list(
        [{"chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )
    return tokenizer, model, pairs


def _check_batched_equals_single_rows() -> None:
    """Assert padded batch scores match unpadded full-logit scores.

    Executed in a child interpreter. Compares, per sequence, the mean
    log-probability per response token from one padded batch of all
    sequences (no attention mask, chunked output head) with the same
    mean from the plain model forward on the row alone (full logits).
    """
    import torch

    from examples.prefs import prefs_eval

    # bf16 rounding depends on the batch shape (up to about 0.05 per
    # token here), which would hide a padding leak; float32 is exact.
    tokenizer, model, pairs = _load("float32")
    sequences = prefs_eval.encode_conversation_pairs(tokenizer, pairs)
    lengths = [len(ids) for ids, _ in sequences]
    budget = len(sequences) * max(lengths)
    assert len(prefs_eval.length_batches(lengths, budget)) == 1

    sums, counts = prefs_eval.response_log_probs(
        model, sequences, tokenizer.pad_token_id, budget
    )

    for (ids, start), total, count in zip(
        sequences, sums, counts, strict=True
    ):
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids]).cuda()).logits[0]
        log_probs = logits[:-1].float().log_softmax(-1)
        targets = torch.tensor(ids[start:]).cuda()
        expected = float(
            log_probs[start - 1 :].gather(1, targets[:, None]).sum()
        )
        assert count == len(ids) - start
        assert abs(total - expected) / count < 1e-3, (total, expected)


def _check_long_sequence_fits_24gb() -> None:
    """Score one sequence of about ``LONG_TOKENS`` tokens.

    Executed in a child interpreter. The prompt repeats the real user
    turns of the fixture to about ``LONG_TOKENS`` tokens; the response is
    a real chosen response. Prints the response token count,
    the summed log-probability and the peak allocated GPU memory in GiB.
    """
    import torch

    from examples.prefs import prefs_eval

    tokenizer, model, pairs = _load()
    text = "\n\n".join(row["chosen"][0]["content"] for row in pairs)
    response = [pairs[0]["chosen"][-1]]
    copies = LONG_TOKENS // len(tokenizer(text)["input_ids"])
    prompt = [{"role": "user", "content": "\n\n".join([text] * copies)}]
    ids, start = prefs_eval.encode_response(tokenizer, prompt, response)
    assert len(ids) > 0.95 * LONG_TOKENS, len(ids)
    torch.cuda.reset_peak_memory_stats()

    sums, counts = prefs_eval.response_log_probs(
        model, [(ids, start)], tokenizer.pad_token_id, len(ids)
    )

    peak = torch.cuda.max_memory_allocated() / GIB
    print(counts[0], sums[0], peak)


def _run(function: str) -> str:
    """Run ``function`` of this module in a fresh interpreter.

    Args:
        function: Name of a module-level function taking no arguments.

    Returns:
        The child's stdout; asserts that it exited successfully.
    """
    code = (
        f"from tests.test_examples_prefs_cuda import {function}; {function}()"
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


@pytest.mark.cuda
def test_batched_response_log_probs_equal_single_unpadded_rows():
    _run("_check_batched_equals_single_rows")


@pytest.mark.cuda
def test_long_sequence_scores_within_24gb():
    stdout = _run("_check_long_sequence_fits_24gb")
    count, total, peak = stdout.split()[-3:]
    assert int(count) > 0
    assert math.isfinite(float(total))
    assert float(total) < 0
    assert float(peak) < 24, f"peak {peak} GiB"
