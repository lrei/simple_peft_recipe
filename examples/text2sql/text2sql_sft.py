r"""Fine-tune SmolLM3-3B to write SQL from a schema and a question (PESFT).

Stage 1 of the text-to-SQL example: supervised fine-tuning on
``b-mc2/sql-create-context`` (``CREATE TABLE`` context + question -> SQL),
with the loss on the SQL answer only. The model is loaded in 4-bit (QLoRA)
and the LoRA adapters are saved; ``text2sql_rl.py`` (stage 2, GRPO) keeps
training them. The task (prompt, split, scoring) is defined in
``text2sql_eval.py``.

SmolLM3 keeps its own chat template (``chat_template=None``), which uses
the ChatML markers of ``PESFTConfig``'s defaults; thinking is switched off
so the model answers with the query directly.

Usage:
    uv run python -m examples.text2sql.text2sql_sft --max_steps 300
    uv run python -m examples.text2sql.text2sql_eval \\
        --model_path ./models/smollm3-3b-sql-sft

Dataset: ``b-mc2/sql-create-context`` (CC-BY-4.0). Model:
``HuggingFaceTB/SmolLM3-3B`` (Apache-2.0).
See ``examples/text2sql/README.md`` for the full walkthrough.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import unsloth  # noqa: F401  # patches transformers/trl; must come first

from examples.text2sql.text2sql_eval import (
    CHAT_TEMPLATE_KWARGS,
    DEFAULT_MODEL_PATH,
    build_messages,
    load_splits,
)
from speftr import PESFT, PESFTConfig


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from transformers import PreTrainedTokenizerBase


DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with model, LoRA and training settings.
    """
    parser = argparse.ArgumentParser(
        description="Supervised text-to-SQL fine-tuning with PESFT"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model in 4-bit (QLoRA) (default: on)",
    )
    parser.add_argument(
        "--lora_r", type=int, default=1, help="LoRA rank (default: 1)"
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
        default=300,
        help="Train rows held out for the eval loss (default: 300)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Where adapters are saved (default: {DEFAULT_MODEL_PATH})",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PESFTConfig:
    """Map CLI arguments onto a PESFTConfig for SmolLM3.

    Args:
        args: Parsed arguments from ``parse_args``.

    Returns:
        Config training on the SQL answer only; the ChatML markers are
        ``PESFTConfig``'s defaults.
    """
    config = PESFTConfig()
    config.model_name_or_path = args.model_name_or_path
    config.load_in_4bit = args.load_in_4bit
    config.max_seq_length = 1024
    config.chat_template = None
    config.train_on_responses = True
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
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[Mapping[str, Any]], list[str]]:
    """Create the batched formatting function passed to ``PESFT.train``.

    Args:
        tokenizer: Tokenizer whose chat template renders the conversation.

    Returns:
        Function turning a batch of rows (``context``, ``question``,
        ``answer``) into one chat-formatted training text per row.
    """

    def format_batch(batch: Mapping[str, Any]) -> list[str]:
        """Render a batch of rows as schema + question prompt, SQL answer.

        Args:
            batch: Columns ``context``, ``question`` and ``answer``.

        Returns:
            One training text per row.
        """
        contexts = batch["context"]
        questions, answers = batch["question"], batch["answer"]
        # Unsloth also calls this with one unbatched row (column -> value).
        if isinstance(contexts, str):
            contexts, questions, answers = [contexts], [questions], [answers]
        texts = []
        for context, question, answer in zip(
            contexts, questions, answers, strict=True
        ):
            chat = build_messages(context, question, answer)
            rendered = tokenizer.apply_chat_template(
                chat, tokenize=False, **CHAT_TEMPLATE_KWARGS
            )
            texts.append(rendered)
        return texts

    return format_batch


def main() -> None:
    """Train on sql-create-context and save the LoRA adapters.

    Returns:
        None. Adapters, tokenizer and run parameters are written to
        ``--output_dir``.
    """
    args = parse_args()
    trainer = PESFT(build_config(args))
    _model, tokenizer = trainer.load_model()

    train_rows, _held_out = load_splits()
    split = train_rows.train_test_split(
        test_size=args.eval_rows, seed=trainer.config.random_state
    )
    format_batch = build_formatting_func(tokenizer)
    print(format_batch(split["train"][:1])[0])

    trainer.train(split["train"], split["test"], format_batch)
    trainer.save_model()


if __name__ == "__main__":
    main()
