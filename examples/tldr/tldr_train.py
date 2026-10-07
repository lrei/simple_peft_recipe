r"""Fine-tune a chat model to write Reddit TL;DRs with PESFT.

Each training text is the post in a user turn and the author's TL;DR as
the assistant turn; the loss covers only the TL;DR
(``train_on_responses``). The task (prompt, dataset, scoring) is defined
in ``tldr_eval.py``, which also scores the base and fine-tuned models.

The model keeps its own chat template (``chat_template=None``) and the
instruction/response markers are inferred from it (empty
``instruction_part`` / ``response_part``), so ``--model_name_or_path``
accepts any model family ``speftr.chat_markers`` knows. The parser is
``PESFTConfig``'s with this example's defaults (``EXAMPLE_DEFAULTS``)
plus ``--train_rows`` and ``--eval_rows``.

Usage:
    uv run python -m examples.tldr.tldr_train --max_steps 300
    uv run python -m examples.tldr.tldr_train --max_steps 300 \\
        --model_name_or_path google/gemma-4-E4B-it \\
        --output_dir ./models/gemma-4-e4b-lora-tldr
    uv run python -m examples.tldr.tldr_eval \\
        --model_path ./models/qwen3.5-4b-lora-tldr

Dataset: ``trl-lib/tldr`` (from ``webis/tldr-17``, CC-BY-4.0). Default
model: ``Qwen/Qwen3.5-4B`` (Apache-2.0).
Results and adaptation steps: ``examples/tldr/README.md``.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Final

import unsloth  # noqa: F401  # patches transformers/trl; must come first

from examples.tldr.tldr_eval import (
    DEFAULT_MODEL_PATH,
    MAX_SEQ_LENGTH,
    build_messages,
    load_tldr,
)
from speftr import PESFT, PESFTConfig


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from transformers import PreTrainedTokenizerBase


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TRAIN_ROWS = -1
DEFAULT_EVAL_ROWS = 500

EXAMPLE_DEFAULTS: Final[dict[str, object]] = {
    "model_name_or_path": DEFAULT_MODEL,
    "max_seq_length": MAX_SEQ_LENGTH,
    "chat_template": None,
    "train_on_responses": True,
    "instruction_part": "",
    "response_part": "",
    "lora_r": 2,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 1,
    "logging_steps": 25,
    "output_dir": DEFAULT_MODEL_PATH,
}
"""``PESFTConfig`` fields this example overrides."""


def build_parser() -> argparse.ArgumentParser:
    """Return ``PESFTConfig``'s parser with this example's defaults.

    Returns:
        The parser, extended with ``--train_rows`` and ``--eval_rows``.
        ``--help`` lists the example defaults in its epilog, since the
        per-flag help shows the ``PESFTConfig`` ones.
    """
    parser: argparse.ArgumentParser = PESFTConfig.get_argument_parser()
    parser.description = "Train a Reddit TL;DR summarizer with PESFT"
    parser.set_defaults(**EXAMPLE_DEFAULTS)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = "example defaults:\n" + "\n".join(
        f"  --{name} {value!r}" for name, value in EXAMPLE_DEFAULTS.items()
    )
    parser.add_argument(
        "--train_rows",
        type=int,
        default=DEFAULT_TRAIN_ROWS,
        help="Use only the first N train rows (default: -1, all 116,722)",
    )
    parser.add_argument(
        "--eval_rows",
        type=int,
        default=DEFAULT_EVAL_ROWS,
        help=(
            "Validation rows used for the eval loss "
            f"(default: {DEFAULT_EVAL_ROWS})"
        ),
    )
    return parser


def build_formatting_func(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[Mapping[str, Any]], list[str]]:
    """Create the batched formatting function passed to ``PESFT.train``.

    Args:
        tokenizer: Tokenizer whose chat template renders the conversation.

    Returns:
        Function turning a batch of rows (``post``, ``summary``) into one
        chat-formatted training text per row.
    """

    def format_batch(batch: Mapping[str, Any]) -> list[str]:
        """Render a batch of rows as user post + assistant TL;DR.

        Args:
            batch: Columns ``post`` and ``summary``.

        Returns:
            One training text per row.
        """
        posts, summaries = batch["post"], batch["summary"]
        # Unsloth also calls this with one unbatched row (column -> value).
        if isinstance(posts, str):
            posts, summaries = [posts], [summaries]
        texts = []
        for post, summary in zip(posts, summaries, strict=True):
            chat = build_messages(post, summary)
            rendered = tokenizer.apply_chat_template(chat, tokenize=False)
            texts.append(rendered)
        return texts

    return format_batch


def main() -> None:
    """Train LoRA adapters on TL;DR and save them.

    Returns:
        None. Adapters, tokenizer and run parameters are written to
        ``--output_dir``.
    """
    args = build_parser().parse_args()
    trainer = PESFT(PESFTConfig.from_args(args))
    _model, tokenizer = trainer.load_model()

    dataset = load_tldr()
    train_split = dataset["train"]
    if args.train_rows > 0:
        train_split = train_split.select(range(args.train_rows))
    eval_split = dataset["validation"].select(range(args.eval_rows))
    format_batch = build_formatting_func(tokenizer)
    print(format_batch(train_split[:1])[0])

    trainer.train(train_split, eval_split, format_batch)
    trainer.save_model()


if __name__ == "__main__":
    main()
