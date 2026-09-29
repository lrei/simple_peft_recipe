r"""Fine-tune Gemma 4 31B in 4-bit on one 24 GB GPU with ``PESFT``.

The same chat SFT as ``examples/instruct`` (Dolly pirate: system + user +
assistant turns, loss on the assistant turn only) on a model whose
16-bit weights (~62 GB) are far larger than the GPU. What makes it fit:

- the pre-quantized ``unsloth/gemma-4-31B-it-unsloth-bnb-4bit`` base
  (4-bit weights, ~18 GB on the GPU);
- one sequence per micro-batch with gradient accumulation for the
  effective batch (accumulation costs no memory);
- Unsloth's gradient checkpointing, which offloads activations to CPU
  (the ``PESFTConfig`` default).

The command line is ``PESFTConfig``'s parser with example defaults (see
``EXAMPLE_DEFAULTS``) plus ``--system_prompt`` and ``--eval_size``.
``build_format_batch`` is the batched formatting function ``PESFT.train``
expects.

Usage:
    uv run python -m examples.big.big
    uv run python -m examples.big.big --max_steps 20 --eval_size 16
    torchrun --nproc_per_node 4 -m examples.big.big \
        --gradient_accumulation_steps 4
    uv run python -m examples.big.big --device_map unsloth_balanced

Memory, timings, multi-GPU and Slurm: ``examples/big/README.md``.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Final, cast

# unsloth must be imported before datasets/transformers/trl so its patches
# apply to everything imported after it.
import unsloth  # noqa: F401
from datasets import load_dataset

from speftr import PESFT, PESFTConfig


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


DATASET_NAME = "TeeZee/dolly-15k-pirate-speech"
DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a pirate, always respond in pirate speech."
)
DEFAULT_EVAL_SIZE = 100
COLUMNS = ("instruction", "context", "response")
# Defaults this example sets on PESFTConfig's parser; every other flag
# keeps the PESFTConfig default.
EXAMPLE_DEFAULTS: Final[dict[str, object]] = {
    "model_name_or_path": "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
    "load_in_4bit": True,
    "chat_template": "gemma-4",
    "train_on_responses": True,
    "instruction_part": "<|turn>user\n",
    "response_part": "<|turn>model\n",
    "max_seq_length": 2048,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "output_dir": "./models/speftr-big",
}


def build_conversation(
    row: Mapping[str, str], system_prompt: str
) -> list[dict[str, str]]:
    """Turn one Dolly row into a system + user + assistant conversation.

    Args:
        row: A row with ``instruction``, ``context`` (may be empty) and
            ``response``.
        system_prompt: Content of the system turn.

    Returns:
        The conversation. The user turn is the instruction, followed by
        ``Context:`` and the context when there is one.
    """
    instruction = row["instruction"].strip()
    context = row["context"].strip()
    user = f"{instruction}\n\nContext:\n{context}" if context else instruction
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["response"].strip()},
    ]


def build_format_batch(
    tokenizer: PreTrainedTokenizerBase, system_prompt: str
) -> Callable[[Mapping[str, object]], list[str]]:
    """Create the batched formatting function ``PESFT.train`` expects.

    Args:
        tokenizer: Text tokenizer with the training chat template.
        system_prompt: System turn added to every conversation.

    Returns:
        A function mapping a batch (column -> list of values) to one
        rendered training text per row. It also accepts the single row
        (column -> value) Unsloth passes once to probe it.
    """

    def format_batch(batch: Mapping[str, object]) -> list[str]:
        """Render a batch of Dolly rows with the chat template.

        Args:
            batch: Dolly columns, batched or a single row.

        Returns:
            One training text per row, as the chat template renders it.
        """
        rows: list[Mapping[str, str]]
        if isinstance(batch["instruction"], str):
            rows = [cast("Mapping[str, str]", batch)]
        else:
            columns = cast("Mapping[str, list[str]]", batch)
            rows = [
                {name: columns[name][index] for name in COLUMNS}
                for index in range(len(columns["instruction"]))
            ]
        return [
            cast(
                "str",
                tokenizer.apply_chat_template(
                    build_conversation(row, system_prompt), tokenize=False
                ),
            )
            for row in rows
        ]

    return format_batch


def build_parser() -> argparse.ArgumentParser:
    """Return ``PESFTConfig``'s parser with this example's defaults.

    Returns:
        The parser, extended with ``--system_prompt`` and ``--eval_size``.
        ``--help`` lists the example defaults in its epilog, since the
        per-flag help shows the ``PESFTConfig`` ones.
    """
    parser: argparse.ArgumentParser = PESFTConfig.get_argument_parser()
    parser.description = (
        "LoRA SFT of Gemma 4 31B (4-bit) on Dolly pirate with PESFT."
    )
    parser.set_defaults(**EXAMPLE_DEFAULTS)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = "example defaults:\n" + "\n".join(
        f"  --{name} {value!r}" for name, value in EXAMPLE_DEFAULTS.items()
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System turn of every conversation (default: pirate prompt)",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=DEFAULT_EVAL_SIZE,
        help=f"Rows held out for the eval loss (default: {DEFAULT_EVAL_SIZE})",
    )
    return parser


def load_datasets(eval_size: int, seed: int) -> tuple[Dataset, Dataset]:
    """Load Dolly pirate and hold out ``eval_size`` rows for evaluation.

    Args:
        eval_size: Number of evaluation rows.
        seed: Seed for the random split.

    Returns:
        ``(train_dataset, eval_dataset)`` with the columns ``instruction``,
        ``context``, ``response`` and ``category``.
    """
    dataset = cast("Dataset", load_dataset(DATASET_NAME, split="train"))
    split = dataset.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]


def main() -> None:
    """Train the LoRA adapter and save it to ``--output_dir``.

    Returns:
        None. Under torchrun only rank 0 writes the adapter, tokenizer,
        ``speftr.json``, ``training_args.json`` and checkpoints.
    """
    args = build_parser().parse_args()
    config = PESFTConfig.from_args(args)
    trainer = PESFT(config)
    _, processor = trainer.load_model()
    # Gemma 4 is multimodal: load_model() returns a processor.
    tokenizer = getattr(processor, "tokenizer", processor)
    format_batch = build_format_batch(tokenizer, args.system_prompt)

    train_dataset, eval_dataset = load_datasets(
        args.eval_size, config.random_state
    )
    # One rendered row shows whether the template and the
    # instruction_part/response_part markers line up.
    print(format_batch(train_dataset[:1])[0])

    trainer.train(train_dataset, eval_dataset, format_batch)
    trainer.save_model()


if __name__ == "__main__":
    main()
