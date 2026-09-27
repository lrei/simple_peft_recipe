r"""Fine-tune IBM Granite 3.3 2B as a banking intent classifier with PESFT.

A support team routes each customer message to one of the 77 Banking77
intents. The model reads the message and the list of intent names and
answers with the intent name; the loss covers only that answer
(``train_on_responses``). The task (prompt, dataset) is defined in
``intent_eval.py``, which also scores the base and fine-tuned models.

Granite keeps its own chat template (``chat_template=None``); the
instruction/response markers below are its role headers.

Usage:
    uv run python -m examples.intent.intent_train
    uv run python -m examples.intent.intent_train --max_steps 300
    uv run python -m examples.intent.intent_eval \\
        --model_path ./models/granite-3.3-2b-lora-banking77

Dataset: ``legacy-datasets/banking77`` (CC-BY-4.0). Model:
``ibm-granite/granite-3.3-2b-instruct`` (Apache-2.0).
See ``examples/intent/README.md`` for the full walkthrough.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import unsloth  # noqa: F401  # patches transformers/trl; must come first

from examples.intent.intent_eval import (
    DEFAULT_MODEL_PATH,
    MAX_SEQ_LENGTH,
    build_messages,
    load_banking77,
)
from speftr import PESFT, PESFTConfig


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from transformers import PreTrainedTokenizerBase


DEFAULT_MODEL = "ibm-granite/granite-3.3-2b-instruct"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with model, LoRA and training settings.
    """
    parser = argparse.ArgumentParser(
        description="Train a Banking77 intent classifier with PESFT"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit (QLoRA) (default: bf16)",
    )
    parser.add_argument(
        "--lora_r", type=int, default=8, help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=16,
        help="Per device batch size (default: 16)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Stop after N optimizer steps (default: -1, use epochs)",
    )
    parser.add_argument(
        "--eval_rows",
        type=int,
        default=500,
        help="Train rows held out for the eval loss (default: 500)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Where adapters are saved (default: {DEFAULT_MODEL_PATH})",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PESFTConfig:
    """Map CLI arguments onto a PESFTConfig for Granite.

    Args:
        args: Parsed arguments from ``parse_args``.

    Returns:
        Config training on the assistant answer only, with the Granite chat
        markers.
    """
    config = PESFTConfig()
    config.model_name_or_path = args.model_name_or_path
    config.load_in_4bit = args.load_in_4bit
    config.max_seq_length = MAX_SEQ_LENGTH
    config.chat_template = None
    config.train_on_responses = True
    config.instruction_part = "<|start_of_role|>user<|end_of_role|>"
    config.response_part = "<|start_of_role|>assistant<|end_of_role|>"
    config.lora_r = args.lora_r
    config.learning_rate = args.learning_rate
    config.per_device_train_batch_size = args.per_device_batch_size
    config.per_device_eval_batch_size = args.per_device_batch_size
    config.num_train_epochs = args.num_epochs
    config.max_steps = args.max_steps
    config.logging_steps = 25
    config.output_dir = args.output_dir
    return config


def build_formatting_func(
    tokenizer: PreTrainedTokenizerBase, label_names: list[str]
) -> Callable[[Mapping[str, Any]], list[str]]:
    """Create the batched formatting function passed to ``PESFT.train``.

    Args:
        tokenizer: Tokenizer whose chat template renders the conversation.
        label_names: Intent names in label-index order.

    Returns:
        Function turning a batch of Banking77 rows (``message``,
        ``label``) into one chat-formatted training text per row.
    """

    def format_batch(batch: Mapping[str, Any]) -> list[str]:
        """Render a batch of rows as user prompt + intent answer.

        Args:
            batch: Columns ``message`` and ``label`` (integer index).

        Returns:
            One training text per row.
        """
        messages, labels = batch["message"], batch["label"]
        # Unsloth also calls this with one unbatched row (column -> value).
        if isinstance(messages, str):
            messages, labels = [messages], [labels]
        texts = []
        for message, label in zip(messages, labels, strict=True):
            chat = build_messages(message, label_names, label_names[label])
            rendered = tokenizer.apply_chat_template(chat, tokenize=False)
            texts.append(rendered)
        return texts

    return format_batch


def main() -> None:
    """Train LoRA adapters on Banking77 and save them.

    Returns:
        None. Adapters, tokenizer and run parameters are written to
        ``--output_dir``.
    """
    args = parse_args()
    trainer = PESFT(build_config(args))
    _model, tokenizer = trainer.load_model()

    dataset, label_names = load_banking77()
    split = dataset["train"].train_test_split(
        test_size=args.eval_rows, seed=trainer.config.random_state
    )
    format_batch = build_formatting_func(tokenizer, label_names)
    print(format_batch(split["train"][:1])[0])

    trainer.train(split["train"], split["test"], format_batch)
    trainer.save_model()


if __name__ == "__main__":
    main()
