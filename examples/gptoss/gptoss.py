r"""Teach GPT-OSS to reason in a requested language with ``PESFT``.

LoRA SFT of gpt-oss on ``HuggingFaceH4/Multilingual-Thinking``: every
row's system turn asks for a "reasoning language" and the assistant turn
reasons (``thinking``) in that language before answering. The same
script trains gpt-oss-20b on one GPU and gpt-oss-120b split over several
(``--device_map balanced``); ``--model_name_or_path`` picks the model.

- The base is Unsloth's bitsandbytes 4-bit conversion
  (``unsloth/gpt-oss-*-unsloth-bnb-4bit``); OpenAI's MXFP4 checkpoints
  cannot be trained (no MXFP4 backward pass).
- ``chat_template=None`` keeps gpt-oss's own harmony template: the
  system turn becomes developer instructions, ``thinking`` the analysis
  channel and ``content`` the final channel.
- Response-only loss on both channels (markers in ``gptoss_eval``).
- ``router_aux_loss_coef=0``: the router is frozen under LoRA, and
  Unsloth's 4-bit gpt-oss fails with the load-balancing loss on.

The command line is ``PESFTConfig``'s parser with example defaults (see
``EXAMPLE_DEFAULTS``). The held-out rows and the task definition live in
``gptoss_eval.py``, which also scores the result.

Usage:
    uv run python -m examples.gptoss.gptoss
    CUDA_VISIBLE_DEVICES=0,1 uv run python -m examples.gptoss.gptoss \
        --model_name_or_path unsloth/gpt-oss-120b-unsloth-bnb-4bit \
        --device_map balanced --output_dir ./models/speftr-gptoss-120b

Memory, timings, results and Slurm: ``examples/gptoss/README.md``.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Final, cast

# unsloth must be imported before datasets/transformers/trl so its patches
# apply to everything imported after it.
import unsloth  # noqa: F401

from examples.gptoss.gptoss_eval import (
    INSTRUCTION_PART,
    MAX_SEQ_LENGTH,
    RESPONSE_PART,
    load_splits,
)
from speftr import PESFT, PESFTConfig


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from transformers import PreTrainedTokenizerBase


type Message = dict[str, str]

# Defaults this example sets on PESFTConfig's parser; every other flag
# keeps the PESFTConfig default (lr 2e-4 constant, alpha 32, adamw_8bit).
EXAMPLE_DEFAULTS: Final[dict[str, object]] = {
    "model_name_or_path": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    "load_in_4bit": True,
    "chat_template": None,
    "train_on_responses": True,
    "instruction_part": INSTRUCTION_PART,
    "response_part": RESPONSE_PART,
    "max_seq_length": MAX_SEQ_LENGTH,
    # Rank 1 covers this dataset (python -m speftr.lora_budget); on gpt-oss
    # the MLP targets include every expert.
    "lora_r": 1,
    "router_aux_loss_coef": 0.0,
    # Unsloth's eval-mode gpt-oss kernels give sliding-window attention the
    # full mask and run every token through every expert; Unsloth names
    # its 4-bit expert class GptOssExperts.
    "eval_in_train_mode": ["GptOssAttention", "GptOssExperts"],
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    # Evaluation keeps each row's logits over the 201k-token vocabulary
    # in bfloat16 and float32, 2.3 GiB per 2048-token row; 3 rows fit next
    # to 20b on a 24 GB GPU.
    "per_device_eval_batch_size": 3,
    "num_train_epochs": 1,
    # One evaluation after training; checkpoints only for resuming.
    "eval_strategy": "no",
    "save_strategy": "steps",
    "save_steps": 25,
    "logging_steps": 5,
    "output_dir": "./models/speftr-gptoss-20b",
}


def build_format_batch(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[Mapping[str, object]], list[str]]:
    """Create the batched formatting function ``PESFT.train`` expects.

    Args:
        tokenizer: gpt-oss tokenizer with its harmony chat template.

    Returns:
        A function mapping a batch (``messages`` column -> list of
        conversations) to one rendered training text per row. It also
        accepts the single row (``messages`` -> one conversation) Unsloth
        passes once to probe it.
    """

    def format_batch(batch: Mapping[str, object]) -> list[str]:
        """Render conversations with the harmony template.

        Args:
            batch: The ``messages`` column, batched or a single row.

        Returns:
            One training text per conversation.
        """
        messages = cast("list[object]", batch["messages"])
        conversations = cast("list[list[Message]]", messages)
        if messages and isinstance(messages[0], dict):
            conversations = [cast("list[Message]", messages)]
        return [
            cast(
                "str",
                tokenizer.apply_chat_template(conversation, tokenize=False),
            )
            for conversation in conversations
        ]

    return format_batch


def build_parser() -> argparse.ArgumentParser:
    """Return ``PESFTConfig``'s parser with this example's defaults.

    Returns:
        The parser. ``--help`` lists the example defaults in its epilog,
        since the per-flag help shows the ``PESFTConfig`` ones.
    """
    parser: argparse.ArgumentParser = PESFTConfig.get_argument_parser()
    parser.description = (
        "LoRA SFT of gpt-oss (4-bit) on Multilingual-Thinking with PESFT."
    )
    parser.set_defaults(**EXAMPLE_DEFAULTS)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = "example defaults:\n" + "\n".join(
        f"  --{name} {value!r}" for name, value in EXAMPLE_DEFAULTS.items()
    )
    return parser


def main() -> None:
    """Train the LoRA adapter and save it to ``--output_dir``.

    Returns:
        None. The adapter, tokenizer, ``speftr.json``,
        ``training_args.json`` and the checkpoints are written to
        ``--output_dir``.
    """
    config = PESFTConfig.from_args(build_parser().parse_args())
    trainer = PESFT(config)
    _, tokenizer = trainer.load_model()
    format_batch = build_format_batch(tokenizer)

    train_dataset, eval_dataset = load_splits()
    # One rendered row shows whether the template and the
    # instruction_part/response_part markers line up.
    print(format_batch(train_dataset[:1])[0])

    trainer.train(train_dataset, eval_dataset, format_batch)
    trainer.save_model()


if __name__ == "__main__":
    main()
