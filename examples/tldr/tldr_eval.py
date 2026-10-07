r"""Evaluate a Reddit TL;DR summarizer on the ``trl-lib/tldr`` test split.

Each row is a Reddit post (subreddit, title, body) and the one- or
two-sentence TL;DR its author wrote. The model is asked for a TL;DR in a
user turn, so a base chat model is scored with the same prompt as the
fine-tuned one. Thinking is turned off at generation time: a summary is
a direct answer.

This module also defines the task for ``tldr_train.py``: the dataset,
the prompt (``build_messages``) and the scoring (``compute_metrics``:
ROUGE-1/2/L against the author's TL;DR, length in words, and the share
of invalid replies, which are empty ones or ones carrying think tags).

Usage:
    # Base model
    uv run python -m examples.tldr.tldr_eval --model_path Qwen/Qwen3.5-4B

    # Adapters saved by tldr_train.py
    uv run python -m examples.tldr.tldr_eval \\
        --model_path ./models/qwen3.5-4b-lora-tldr

Dataset: ``trl-lib/tldr``, TRL's prompt-completion version of
``webis/tldr-17`` (CC-BY-4.0).

See ``examples/tldr/README.md`` for results, hardware needs and how to
adapt the example to your own data.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, cast

import torch
import unsloth  # noqa: F401  # patches transformers; must come first
from datasets import load_dataset
from rouge_score import rouge_scorer
from unsloth import FastLanguageModel


if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import Dataset, DatasetDict
    from transformers import (
        BatchEncoding,
        GenerationMixin,
        PreTrainedTokenizerBase,
    )


DATASET_NAME = "trl-lib/tldr"
DEFAULT_MODEL_PATH = "./models/qwen3.5-4b-lora-tldr"
# Posts run up to about 500 tokens plus the instruction and the summary.
MAX_SEQ_LENGTH = 1024
INSTRUCTION = (
    "Write a TL;DR of the Reddit post below in one or two sentences. "
    "Answer with the TL;DR only."
)
# The dataset's prompts end with the cue the author's TL;DR followed.
PROMPT_SUFFIX = "TL;DR:"
THINK_TAGS = ("<think>", "</think>")
ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")

type Message = dict[str, str]


def load_tldr() -> DatasetDict:
    """Load ``trl-lib/tldr`` with task-neutral column names.

    The ``prompt`` and ``completion`` columns are renamed to ``post`` and
    ``summary``: TRL trains a prompt-completion dataset as is and ignores
    the formatting function.

    Returns:
        The ``train``, ``validation`` and ``test`` splits (columns
        ``post`` and ``summary``).
    """
    dataset = cast("DatasetDict", load_dataset(DATASET_NAME))
    return dataset.rename_columns({"prompt": "post", "completion": "summary"})


def build_messages(post: str, summary: str | None = None) -> list[Message]:
    """Build the chat for one post.

    Args:
        post: The post as the dataset gives it (``SUBREDDIT:``, ``TITLE:``,
            ``POST:`` sections ending with ``TL;DR:``); the cue is dropped
            since the instruction asks for the TL;DR.
        summary: The author's TL;DR. When given it is added as the
            assistant turn (training); when ``None`` the chat ends at the
            user turn (generation).

    Returns:
        A user message, followed by the assistant answer if ``summary`` is
        given.
    """
    post = post.strip().removesuffix(PROMPT_SUFFIX).strip()
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{post}"}]
    if summary is not None:
        messages.append({"role": "assistant", "content": summary.strip()})
    return messages


def is_valid_summary(generated: str) -> bool:
    """Tell whether a reply is a usable summary.

    Args:
        generated: Decoded model output without the prompt.

    Returns:
        False for an empty reply or one containing a think tag (the model
        reasoned instead of answering, or echoed the tag).
    """
    text = generated.strip()
    return bool(text) and not any(tag in text for tag in THINK_TAGS)


def compute_metrics(
    references: Sequence[str], predictions: Sequence[str]
) -> dict[str, float]:
    """Score summaries against the authors' TL;DRs.

    Args:
        references: The authors' TL;DRs.
        predictions: Generated replies, in the same order.

    Returns:
        Mean ROUGE-1, ROUGE-2 and ROUGE-L F-measures (stemmed; an invalid
        reply scores 0), ``mean_words`` of the valid replies and
        ``invalid_rate`` (see ``is_valid_summary``).
    """
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
    totals = dict.fromkeys(ROUGE_TYPES, 0.0)
    words = 0
    invalid = 0
    for reference, prediction in zip(references, predictions, strict=True):
        if not is_valid_summary(prediction):
            invalid += 1
            continue
        scores = scorer.score(reference.strip(), prediction.strip())
        for rouge_type in ROUGE_TYPES:
            totals[rouge_type] += scores[rouge_type].fmeasure
        words += len(prediction.split())
    valid = len(predictions) - invalid
    metrics = {
        rouge_type: totals[rouge_type] / len(predictions)
        for rouge_type in ROUGE_TYPES
    }
    metrics["mean_words"] = words / valid if valid else 0.0
    metrics["invalid_rate"] = invalid / len(predictions)
    return metrics


def load_for_generation(
    model_path: str, *, load_in_4bit: bool
) -> tuple[GenerationMixin, PreTrainedTokenizerBase]:
    """Load a base model or saved LoRA adapters for generation.

    Unsloth loads an adapter directory as its base model with the adapters
    attached.

    Args:
        model_path: Hub id, merged checkpoint or adapter directory.
        load_in_4bit: Quantize the base weights to 4 bits.

    Returns:
        The model switched to inference mode and its tokenizer, set to pad
        on the left as batched generation requires.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=load_in_4bit,
        attn_implementation="sdpa",
    )
    FastLanguageModel.for_inference(model)
    # Multimodal checkpoints (Qwen 3.5, Gemma 4) come with a processor; the
    # text tokenizer inside it renders and pads the chats.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    tokenizer.padding_side = "left"
    return model, tokenizer


def stop_token_ids(
    model: GenerationMixin, tokenizer: PreTrainedTokenizerBase
) -> list[int]:
    """Token ids that end generation: the model's and the tokenizer's.

    Some checkpoints (Qwen 3.5) declare only the end-of-text token as
    end of sequence, while the chat template closes a turn with the
    tokenizer's EOS; a fine-tuned adapter emits only the latter and
    would otherwise run on into another turn.

    Args:
        model: The generating model.
        tokenizer: Its tokenizer.

    Returns:
        Sorted, de-duplicated ids: the generation config's
        ``eos_token_id`` (an id, a list or None) plus the tokenizer's.
    """
    configured = model.generation_config.eos_token_id
    ids = set(configured if isinstance(configured, list) else [configured])
    ids.add(tokenizer.eos_token_id)
    return sorted(i for i in ids if i is not None)


def generate_replies(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    chats: Sequence[list[Message]],
    *,
    max_new_tokens: int,
) -> list[str]:
    """Greedily generate one reply per chat, as a single batch.

    Thinking is off (``enable_thinking=False``): Qwen 3.5 then renders
    the empty think block its training texts carry; templates without
    the switch ignore it.

    Args:
        model: Model from ``load_for_generation``.
        tokenizer: Its tokenizer; must have a chat template.
        chats: Conversations ending with a user turn.
        max_new_tokens: Generation budget per reply.

    Returns:
        The decoded replies without the prompt or special tokens.
    """
    encoded = tokenizer.apply_chat_template(
        list(chats),
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = cast("BatchEncoding", encoded).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids(model, tokenizer),
        )
    # Left padding: every prompt ends at the same position.
    replies = outputs[:, inputs["input_ids"].shape[1] :]
    return list(tokenizer.batch_decode(replies, skip_special_tokens=True))


def predict_summaries(
    model_path: str, test_split: Dataset, args: argparse.Namespace
) -> tuple[list[str], list[str]]:
    """Summarize every post of ``test_split``.

    Args:
        model_path: Model to evaluate (see ``load_for_generation``).
        test_split: Rows with ``post`` and ``summary``.
        args: Parsed CLI arguments (``batch_size``, ``max_new_tokens``,
            ``load_in_4bit``).

    Returns:
        The authors' TL;DRs and the generated replies.
    """
    model, tokenizer = load_for_generation(
        model_path, load_in_4bit=args.load_in_4bit
    )
    posts: list[str] = test_split["post"]
    references: list[str] = test_split["summary"]
    predictions: list[str] = []
    for start in range(0, len(posts), args.batch_size):
        chats = [
            build_messages(post)
            for post in posts[start : start + args.batch_size]
        ]
        predictions.extend(
            generate_replies(
                model, tokenizer, chats, max_new_tokens=args.max_new_tokens
            )
        )
        print(f"Summarized {len(predictions)}/{len(posts)} posts")
    return references, predictions


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the model path, sample limit and generation settings.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a TL;DR summarizer on the trl-lib/tldr test set"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Hub id, merged model or adapter directory "
            f"(default: {DEFAULT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Evaluate only the first N test rows (-1: all 6,553)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Posts generated per batch (default: 32)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model in 4-bit (default: bf16)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
        help="Maximum tokens generated per summary (default: 96)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help=(
            "Print the first N posts' reference and reply, and the first "
            "N invalid replies (default: 3)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate ``--model_path`` on the TL;DR test split and print metrics.

    Returns:
        None. Example replies, metrics and peak GPU memory are printed.
    """
    args = parse_args()
    test_split = load_tldr()["test"]
    if args.max_samples > 0:
        test_split = test_split.select(range(args.max_samples))

    references, predictions = predict_summaries(
        args.model_path, test_split, args
    )

    shown = zip(references[: args.show], predictions[: args.show], strict=True)
    for reference, prediction in shown:
        print(f"\nreference: {reference.strip()}")
        print(f"reply:     {prediction.strip()}")
    invalid = [p for p in predictions if not is_valid_summary(p)]
    for prediction in invalid[: args.show]:
        print(f"\ninvalid reply: {prediction!r}")
    print(f"\nModel: {args.model_path} ({len(references)} test posts)")
    for name, value in compute_metrics(references, predictions).items():
        print(f"{name}: {value:.4f}")
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak_gpu_memory_gb: {peak_gb:.2f}")


if __name__ == "__main__":
    main()
