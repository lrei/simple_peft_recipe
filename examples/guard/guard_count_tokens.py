"""Count token statistics for guard example training/eval datasets."""

import argparse
import os
import statistics
from collections.abc import Iterable, Mapping
from typing import cast

import unsloth
from datasets import Dataset, load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from unsloth.chat_templates import get_chat_template


print(unsloth.__version__)

INSTRUCTION = "Classify this prompt's as harmful or unharmful:"
CHAT_TEMPLATE = "gemma-3"
PROMPT_COL = "prompt"
LABEL_COL = "prompt_harm_label"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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


def _login_to_hf() -> None:
    """Authenticate with HuggingFace Hub using environment defaults."""
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
        return
    login()


def _load_split(split: str) -> Dataset:
    """Load and filter WildGuardMix split used by guard_og.py."""
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
    """Create chat messages matching guard_og training format."""
    user_message = f"{INSTRUCTION}\n\n{prompt.strip()}"
    return [
        {"role": "user", "content": user_message},
        {"role": "model", "content": label.strip()},
    ]


def _collect_token_stats(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase
) -> tuple[list[int], int]:
    """Tokenize each example and return lengths plus total count."""
    if not isinstance(dataset, Iterable):
        error_msg = "dataset must be iterable"
        raise TypeError(error_msg)

    lengths: list[int] = []
    for row in dataset:
        if not isinstance(row, Mapping):
            continue
        prompt_value = row.get(PROMPT_COL)
        label_value = row.get(LABEL_COL)
        prompt_is_str = isinstance(prompt_value, str)
        label_is_str = isinstance(label_value, str)
        if not prompt_is_str or not label_is_str:
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
    """Print basic statistics for token lengths."""
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
    """Entrypoint for computing token statistics."""
    args = parse_args()

    _login_to_hf()

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

    print("\nToken statistics for guard_og.py datasets")
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
