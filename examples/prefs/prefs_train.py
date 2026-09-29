r"""Align OLMo 2 1B SFT with DPO on its own preference mix using PEDPO.

AllenAI trained ``allenai/OLMo-2-0425-1B-DPO`` from
``allenai/OLMo-2-0425-1B-SFT`` with full-model DPO on
``allenai/olmo-2-0425-1b-preference-mix``. This script runs the same step
with a rank-1 LoRA adapter on 10,000 of those pairs, one epoch, with the
PEDPO recipe defaults. The 1,000 held-out pairs of ``prefs_eval`` are the
eval set, so TRL reports its reward metrics on pairs never trained on.

Each row holds two full conversations (``chosen``, ``rejected``); TRL
takes their shared leading turns as the prompt and renders everything
with the model's chat template.

Usage:
    uv run python -m examples.prefs.prefs_train
    uv run python -m examples.prefs.prefs_train --precompute_ref_log_probs \\
        --output_dir ./models/olmo2-1b-lora-dpo-precompute
    uv run python -m examples.prefs.prefs_eval

Data: ``allenai/olmo-2-0425-1b-preference-mix`` (ODC-BY). Model:
``allenai/OLMo-2-0425-1B-SFT`` (Apache-2.0). Results:
``examples/prefs/README.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from examples.prefs.prefs_eval import (
    DEFAULT_ADAPTER_DIR,
    EVAL_PAIRS,
    SFT_MODEL,
    TRAIN_PAIRS,
    load_preference_splits,
    split_prompt,
)
from speftr import PEDPO, PEDPOConfig


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the data sizes, reference option and output
        directory.
    """
    parser = argparse.ArgumentParser(
        description="DPO of OLMo 2 1B SFT on its preference mix with PEDPO"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=SFT_MODEL,
        help=f"SFT model to align (default: {SFT_MODEL})",
    )
    parser.add_argument(
        "--train_pairs",
        type=int,
        default=TRAIN_PAIRS,
        help=f"Training pairs (default: {TRAIN_PAIRS})",
    )
    parser.add_argument(
        "--eval_pairs",
        type=int,
        default=EVAL_PAIRS,
        help=f"Held-out pairs for TRL's eval metrics (default: {EVAL_PAIRS})",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Stop after N optimizer steps (default: -1, one epoch)",
    )
    parser.add_argument(
        "--precompute_ref_log_probs",
        action="store_true",
        help="Compute reference log-probs once before training",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_ADAPTER_DIR,
        help=f"Where the adapter is saved (default: {DEFAULT_ADAPTER_DIR})",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PEDPOConfig:
    """Map CLI arguments onto a PEDPOConfig; the rest are recipe defaults.

    Args:
        args: Parsed arguments from ``parse_args``.

    Returns:
        Config for bf16 LoRA DPO, logging every 10 steps.
    """
    return PEDPOConfig(
        model_name_or_path=args.model_name_or_path,
        max_steps=args.max_steps,
        precompute_ref_log_probs=args.precompute_ref_log_probs,
        output_dir=args.output_dir,
        logging_steps=10,
    )


def show_pair(tokenizer: PreTrainedTokenizerBase, row: dict[str, Any]) -> str:
    """Render one pair as the trainer sees it: prompt, then each response.

    Args:
        tokenizer: Tokenizer with the chat template.
        row: A pair with ``chosen`` and ``rejected`` conversations.

    Returns:
        The rendered prompt with the assistant header, followed by the
        chosen and rejected response texts.
    """
    prompt, chosen, rejected = split_prompt(row["chosen"], row["rejected"])
    rendered = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    return (
        f"{rendered}\n--- chosen ---\n{chosen[0]['content']}\n"
        f"--- rejected ---\n{rejected[0]['content']}"
    )


def run_summary(trainer: PEDPO, wall_seconds: float) -> dict[str, Any]:
    """Collect timing, memory and the last logged metrics of a run.

    Args:
        trainer: The trainer after ``train``.
        wall_seconds: Duration of ``PEDPO.train`` (dataset preparation,
            optional precompute pass, training and final evaluation).

    Returns:
        Wall time, TRL's train runtime and steps, peak GPU memory, the
        last train loss and the final eval metrics.
    """
    history = cast("Any", trainer.trainer).state.log_history
    train_logs = [log for log in history if "loss" in log]
    summary = next(log for log in history if "train_runtime" in log)
    evaluation = [log for log in history if "eval_loss" in log]
    return {
        "wall_seconds": wall_seconds,
        "train_runtime": summary["train_runtime"],
        "steps": cast("Any", trainer.trainer).state.global_step,
        "seconds_per_step": summary["train_runtime"]
        / cast("Any", trainer.trainer).state.global_step,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "last_train_log": train_logs[-1] if train_logs else {},
        "final_eval": evaluation[-1] if evaluation else {},
    }


def main() -> None:
    """Train the adapter, save it and write a run summary.

    Returns:
        None. The adapter, tokenizer and ``prefs_train_summary.json`` are
        written to ``--output_dir``.
    """
    args = parse_args()
    trainer = PEDPO(build_config(args))
    _model, tokenizer = trainer.load_model()

    train_pairs, held_out = load_preference_splits(
        eval_pairs=args.eval_pairs, train_pairs=args.train_pairs
    )
    print(show_pair(tokenizer, cast("dict[str, Any]", train_pairs[0])))

    start = time.perf_counter()
    trainer.train(cast("Dataset", train_pairs), cast("Dataset", held_out))
    summary = run_summary(trainer, time.perf_counter() - start)
    trainer.save_model()

    print(json.dumps(summary, indent=2))
    output = Path(args.output_dir) / "prefs_train_summary.json"
    output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
