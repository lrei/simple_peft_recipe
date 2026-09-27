"""Print token-length statistics for the guard training data.

Renders every WildGuardMix row exactly as ``guard_train`` does (Gemma 3
template, instruction plus prompt, label as the model turn) and prints
min/max/mean/median lengths per split. Use it to choose
``--max_seq_length``. For the LoRA rank a dataset needs, use
``python -m speftr.lora_budget`` (see ``docs/guide.md``).

Usage:
    uv run python -m examples.guard.guard_count_tokens

Downloads only the tokenizer, but still imports Unsloth, so it needs a
GPU. See ``examples/guard/README.md``.
"""

import argparse
import statistics
from collections.abc import Mapping
from typing import cast

# unsloth must be imported before datasets/transformers so its patches
# apply.
import unsloth
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from unsloth.chat_templates import get_chat_template


print(unsloth.__version__)

# Must match guard_train.py.
INSTRUCTION = "Classify this prompt's as harmful or unharmful:"
CHAT_TEMPLATE = "gemma-3"
PROMPT_COL = "prompt"
LABEL_COL = "prompt_harm_label"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with ``model_name_or_path`` for the tokenizer.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute token statistics for the WildGuardMix splits used "
            "in guard example"
        )
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="unsloth/gemma-3-270m-it",
        help=(
            "Base model name or path for tokenizer "
            "(default: unsloth/gemma-3-270m-it)"
        ),
    )

    return parser.parse_args()


def _load_split(split: str) -> Dataset:
    """Load and filter WildGuardMix split used by guard_train.py.

    Args:
        split: ``"train"`` (wildguardtrain) or ``"test"`` (wildguardtest).

    Returns:
        The split with rows lacking a prompt harm label removed.

    Raises:
        ValueError: If ``split`` is neither ``"train"`` nor ``"test"``.
    """
    if split == "train":
        config_name = "wildguardtrain"
        hf_split = "train"
    elif split == "test":
        config_name = "wildguardtest"
        hf_split = "test"
    else:
        error_msg = "split must be 'train' or 'test'"
        raise ValueError(error_msg)

    dataset = load_dataset(
        "allenai/wildguardmix",
        config_name,
        split=hf_split,
    )
    filtered_dataset = cast(
        "Dataset",
        dataset.filter(
            lambda x: x[LABEL_COL] is not None and x[LABEL_COL] != "",
        ),
    )
    return filtered_dataset


def _format_messages(prompt: str, label: str) -> list[dict[str, str]]:
    """Create chat messages matching guard_train.py training format.

    Args:
        prompt: Prompt to classify; surrounding whitespace is stripped.
        label: Harm label used as the model turn.

    Returns:
        A user message (instruction plus prompt) and a model message.
    """
    user_message = f"{INSTRUCTION}\n\n{prompt.strip()}"
    return [
        {"role": "user", "content": user_message},
        {"role": "model", "content": label.strip()},
    ]


def _collect_token_stats(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase
) -> tuple[list[int], int]:
    """Tokenize each example and return lengths plus total count.

    Rows whose prompt or label is not a string are skipped, so the number
    of lengths can be smaller than the returned row count.

    Args:
        dataset: WildGuardMix split to measure.
        tokenizer: Tokenizer with the guard chat template applied.

    Returns:
        Token length per formatted conversation, and the dataset row count.
    """
    lengths: list[int] = []
    for row in dataset:
        if not isinstance(row, Mapping):
            continue
        prompt_value = row.get(PROMPT_COL)
        label_value = row.get(LABEL_COL)
        if not isinstance(prompt_value, str) or not isinstance(
            label_value, str
        ):
            continue
        messages = _format_messages(prompt_value, label_value)
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        lengths.append(len(token_ids))
    return lengths, len(dataset)


def _print_stats(name: str, lengths: list[int]) -> None:
    """Print basic statistics for token lengths.

    Args:
        name: Split name used in the heading.
        lengths: Token lengths; an empty list prints ``no data``.

    Returns:
        None. Statistics are printed to stdout.
    """
    if not lengths:
        print(f"{name}: no data")
        return

    min_tokens = min(lengths)
    max_tokens = max(lengths)
    mean_tokens = statistics.mean(lengths)
    median_tokens = statistics.median(lengths)
    stdev_tokens = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

    print(f"\n{name} split statistics")
    print("-" * 40)
    print(f"Examples:      {len(lengths):,}")
    print(f"Min tokens:    {min_tokens:,}")
    print(f"Max tokens:    {max_tokens:,}")
    print(f"Average tokens:{mean_tokens:,.2f}")
    print(f"Median tokens: {median_tokens:,.0f}")
    print(f"Std deviation: {stdev_tokens:,.2f}")


def main() -> None:
    """Print token statistics for the guard train and eval splits.

    Returns:
        None. Statistics are printed to stdout.
    """
    args = parse_args()

    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

    train_dataset = _load_split("train")
    eval_dataset = _load_split("test")

    train_lengths, train_count = _collect_token_stats(train_dataset, tokenizer)
    eval_lengths, eval_count = _collect_token_stats(eval_dataset, tokenizer)

    print("\nToken statistics for guard_train.py datasets")
    print("=" * 40)

    _print_stats("Train", train_lengths)
    _print_stats("Eval", eval_lengths)

    total_examples = train_count + eval_count
    if total_examples:
        combined_lengths = train_lengths + eval_lengths
        print("\nCombined statistics")
        print("-" * 40)
        print(f"Examples:      {total_examples:,}")
        print(f"Max tokens:    {max(combined_lengths):,}")
        print(f"Average tokens:{statistics.mean(combined_lengths):,.2f}")


if __name__ == "__main__":
    main()
