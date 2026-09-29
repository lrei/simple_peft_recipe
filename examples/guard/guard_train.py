r"""Train a prompt-safety classifier with ``PESFT`` on WildGuardMix.

Demonstrates supervised fine-tuning of a small chat model as a binary
classifier: the user turn holds an instruction plus the prompt to judge,
the model turn is the label ``harmful`` or ``unharmful``. Shows how to
configure ``PESFTConfig`` for a chat template with response-only loss
(``_build_config``), write the batched formatting function that
``PESFT.train`` needs (``format_example`` in ``main``) and save the
adapters with the instruction used, so ``guard_eval`` and ``guard_test``
can rebuild the same prompt.

WildGuardMix (``allenai/wildguardmix``) is gated: accept its terms on the
Hub, then ``hf auth login`` or export ``HF_TOKEN``.

Usage:
    uv run python -m examples.guard.guard_train
    uv run python -m examples.guard.guard_train --num_epochs 1 --lora_r 16
    uv run python -m examples.guard.guard_train --help

The chat template and turn markers are Gemma 3's (``_build_config``); to
train another model family change them there and ``CHAT_TEMPLATE`` in the
other guard scripts. Data, results and adaptation steps:
``examples/guard/README.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping


# unsloth must be imported before datasets/transformers/trl so its patches
# apply to everything imported after it.
import unsloth
from datasets import load_dataset

from speftr import PESFT, PESFTConfig


# Prepended to every prompt; saved with the adapters so the eval and test
# scripts send exactly the text the model was trained on.
INSTRUCTION = "Classify this prompt's as harmful or unharmful:"


def parse_args() -> argparse.Namespace:
    """Parse the command-line options this example exposes.

    Only a subset of ``PESFTConfig`` is exposed; the chat template and
    markers are fixed in ``_build_config``.

    Returns:
        Namespace with model, LoRA, optimisation and output settings.
    """
    parser = argparse.ArgumentParser(
        description="Train a safety classification model with PESFT"
    )

    parser.add_argument(
        "--lora_r", type=int, default=8, help="LoRA rank (default: 8)"
    )

    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="unsloth/gemma-3-270m-it",
        help="Model name or path (default: unsloth/gemma-3-270m-it)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Fraction of steps used for warmup (default: 0.0)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0)",
    )

    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=16,
        help="Per device batch size (default: 16)",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Per device eval batch size (default: 16)",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str,
        choices=["unsloth", "true", "false"],
        default="unsloth",
        help="Gradient checkpointing mode (default: unsloth)",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.3,
        help="Gradient clipping norm (default: 0.3)",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit loading (default: disabled)",
    )

    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable sequence packing during training (default: disabled)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/gemma-3-270m-it-lora-wildguard",
        help=(
            "Output directory for saving model "
            "(default: ./models/gemma-3-270m-it-lora-wildguard)"
        ),
    )

    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> PESFTConfig:
    """Map the CLI options onto a ``PESFTConfig`` for the guard task.

    Fields not set here keep the ``PESFTConfig`` defaults (LoRA on all
    attention and MLP projections, constant LR, per-epoch eval and save,
    best checkpoint by ``eval_loss`` reloaded at the end).

    Args:
        args: Options from ``parse_args``.

    Returns:
        The training configuration.
    """
    config = PESFTConfig()

    config.model_name_or_path = args.model_name_or_path
    config.max_seq_length = args.max_seq_length
    config.load_in_4bit = args.load_in_4bit

    # Loss only on the label: tokens after response_part up to the next
    # instruction_part are trained. Both markers must match the Gemma 3
    # template's rendered text exactly, or nothing (or everything) is
    # trained.
    config.chat_template = "gemma-3"
    config.train_on_responses = True
    config.instruction_part = "<start_of_turn>user\n"
    config.response_part = "<start_of_turn>model\n"

    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha

    config.learning_rate = args.learning_rate
    config.warmup_ratio = args.warmup_ratio
    config.weight_decay = args.weight_decay
    config.per_device_train_batch_size = args.per_device_batch_size
    config.per_device_eval_batch_size = args.per_device_eval_batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.use_gradient_checkpointing = args.gradient_checkpointing
    config.max_grad_norm = args.max_grad_norm
    config.num_train_epochs = args.num_epochs

    config.output_dir = args.output_dir
    config.report_to = "none"
    config.packing = args.packing

    return config


def _load_datasets() -> tuple[Any, Any]:
    """Load the WildGuardMix train and test splits.

    Rows without a ``prompt_harm_label`` are dropped. The columns used
    downstream are ``prompt`` (text to classify) and ``prompt_harm_label``
    (``"harmful"`` or ``"unharmful"``); the others are ignored by
    ``format_example``.

    Returns:
        ``(train_dataset, eval_dataset)``: wildguardtrain and wildguardtest.
    """
    train_dataset = load_dataset(
        "allenai/wildguardmix", "wildguardtrain", split="train"
    )
    eval_dataset = load_dataset(
        "allenai/wildguardmix", "wildguardtest", split="test"
    )

    train_dataset = train_dataset.filter(
        lambda x: (
            x["prompt_harm_label"] is not None and x["prompt_harm_label"] != ""
        )
    )
    eval_dataset = eval_dataset.filter(
        lambda x: (
            x["prompt_harm_label"] is not None and x["prompt_harm_label"] != ""
        )
    )

    return train_dataset, eval_dataset


def main() -> None:
    """Train the guard adapters and save them with the instruction used.

    Runs ``PESFT(config)`` then ``load_model()``, ``train()`` and
    ``save_model()``, then adds ``instruction_prefix`` to the saved
    ``tokenizer_config.json``.

    Returns:
        None. Adapters, tokenizer, ``speftr.json``, ``training_args.json``
        and per-epoch checkpoints are written to ``--output_dir``.
    """
    args = parse_args()
    print(unsloth.__version__)

    config = _build_config(args)
    trainer = PESFT(config)
    # The returned tokenizer already has the gemma-3 chat template applied.
    _model, tokenizer = trainer.load_model()

    def format_example(examples: Mapping[str, Any]) -> list[str]:
        r"""Render a batch of WildGuardMix rows as Gemma 3 conversations.

        ``PESFT.train`` calls this with batched columns (column name to
        list of values). Each row becomes a user turn (``INSTRUCTION``, a
        blank line, the prompt) and a model turn holding the label.

        Args:
            examples: Batch with ``prompt`` and ``prompt_harm_label``
                lists.

        Returns:
            One rendered conversation per row, e.g.
            ``<start_of_turn>user\n...<end_of_turn>\n``
            ``<start_of_turn>model\nharmful<end_of_turn>\n``.
        """
        texts = []
        for prompt, harm_label in zip(
            examples["prompt"], examples["prompt_harm_label"], strict=False
        ):
            user_content = f"{INSTRUCTION}\n\n{prompt.strip()}"
            model_content = harm_label.strip()
            # Gemma templates call the assistant role "model".
            messages = [
                {"role": "user", "content": user_content},
                {"role": "model", "content": model_content},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        return texts

    train_dataset, eval_dataset = _load_datasets()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    trainer.train(train_dataset, eval_dataset, format_example)
    trainer.save_model()

    # guard_eval and guard_test read instruction_prefix back, so inference
    # prompts match the training prompts even if INSTRUCTION is edited.
    print("Saving instruction template in tokenizer configuration...")
    tokenizer_config_path = Path(config.output_dir) / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with tokenizer_config_path.open() as handle:
            tokenizer_config = json.load(handle)
    else:
        tokenizer_config = {}
    tokenizer_config["instruction_prefix"] = INSTRUCTION
    with tokenizer_config_path.open("w") as handle:
        json.dump(tokenizer_config, handle, indent=2)

    print(f"Saved instruction: '{INSTRUCTION}'")


if __name__ == "__main__":
    main()
