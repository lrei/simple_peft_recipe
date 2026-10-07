r"""Evaluate a banking intent classifier on the Banking77 test split.

Customer-support messages are classified into one of 77 banking intents
(``card_arrival``, ``lost_or_stolen_card``, ...). The model sees the list of
intent names in the prompt and must answer with exactly one of them, so a
base model can be scored with the same prompt as the fine-tuned one.

This module also defines the task for ``intent_train.py``: the dataset, the
prompt (``build_messages``) and the answer parsing (``extract_intent``).

Usage:
    # Base model
    uv run python -m examples.intent.intent_eval \\
        --model_path ibm-granite/granite-3.3-2b-instruct

    # Adapters saved by intent_train.py
    uv run python -m examples.intent.intent_eval \\
        --model_path ./models/granite-3.3-2b-lora-banking77

Dataset: ``legacy-datasets/banking77`` (CC-BY-4.0) is the Parquet copy of
``PolyAI/banking77``; the original repo only has a loading script, which
``datasets`` does not run.

See ``examples/intent/README.md`` for results, hardware needs and how to
adapt the example to your own data.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, cast

import torch
import unsloth  # noqa: F401  # patches transformers; must come first
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from unsloth import FastLanguageModel


if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import Dataset, DatasetDict
    from transformers import (
        BatchEncoding,
        GenerationMixin,
        PreTrainedTokenizerBase,
    )


DATASET_NAME = "legacy-datasets/banking77"
DEFAULT_MODEL_PATH = "./models/granite-3.3-2b-lora-banking77"
# The longest prompt (77 intent names + message) is about 600 tokens.
MAX_SEQ_LENGTH = 1024
INVALID_PREDICTION = "INVALID_PREDICTION"
INSTRUCTION = (
    "Classify the customer message into exactly one banking intent. "
    "Answer with the intent name only."
)

type Message = dict[str, str]


def load_banking77() -> tuple[DatasetDict, list[str]]:
    """Load Banking77 and its intent names.

    The ``text`` column is renamed to ``message``: given a ``text`` column,
    the SFT trainer trains on it as is and ignores the formatting function.

    Returns:
        The ``train`` and ``test`` splits (columns ``message`` and
        ``label``, an integer index) and the intent names in label-index
        order.
    """
    dataset = cast("DatasetDict", load_dataset(DATASET_NAME))
    label_names: list[str] = dataset["train"].features["label"].names
    return dataset.rename_column("text", "message"), label_names


def build_messages(
    text: str, label_names: Sequence[str], intent: str | None = None
) -> list[Message]:
    """Build the chat for one customer message.

    Args:
        text: Customer message to classify.
        label_names: Every intent the model may answer with; listed in the
            prompt.
        intent: Gold intent name. When given it is added as the assistant
            turn (training); when ``None`` the chat ends at the user turn
            (generation).

    Returns:
        A user message, followed by the assistant answer if ``intent`` is
        given.
    """
    prompt = (
        f"{INSTRUCTION}\n\nIntents: {', '.join(label_names)}\n\n"
        f"Message: {text.strip()}"
    )
    messages = [{"role": "user", "content": prompt}]
    if intent is not None:
        messages.append({"role": "assistant", "content": intent})
    return messages


def extract_intent(generated: str, label_names: Sequence[str]) -> str | None:
    """Map a generated answer to an intent name by exact match.

    Only the first line is read and compared case-insensitively, so a
    trailing explanation does not count but capitalisation does not matter.

    Args:
        generated: Decoded model output without the prompt.
        label_names: Valid intent names.

    Returns:
        The matching intent name, or ``None`` if the answer is not one.
    """
    lines = generated.strip().splitlines()
    answer = lines[0].strip().lower() if lines else ""
    by_lowercase = {name.lower(): name for name in label_names}
    return by_lowercase.get(answer)


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], label_names: Sequence[str]
) -> dict[str, float]:
    """Compute accuracy and macro-F1 over the 77 intents.

    Args:
        y_true: Gold intent names.
        y_pred: Predicted intent names; ``INVALID_PREDICTION`` for answers
            that matched no intent.
        label_names: All intent names.

    Returns:
        ``accuracy``, ``f1_macro`` (averaged over ``label_names`` only, so an
        invalid answer counts as an error, not as an extra class) and
        ``invalid_rate``.
    """
    invalid = sum(prediction == INVALID_PREDICTION for prediction in y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(label_names),
                average="macro",
                zero_division=0,
            )
        ),
        "invalid_rate": invalid / len(y_pred),
    }


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
    **template_kwargs: Any,
) -> list[str]:
    """Greedily generate one reply per chat, as a single batch.

    Args:
        model: Model from ``load_for_generation``.
        tokenizer: Its tokenizer; must have a chat template.
        chats: Conversations ending with a user turn.
        max_new_tokens: Generation budget per reply.
        **template_kwargs: Extra chat-template variables, e.g.
            ``enable_thinking=False``.

    Returns:
        The decoded replies without the prompt or special tokens.
    """
    encoded = tokenizer.apply_chat_template(
        list(chats),
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
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


def predict_intents(
    model_path: str,
    test_split: Dataset,
    label_names: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    """Classify every message of ``test_split``.

    Args:
        model_path: Model to evaluate (see ``load_for_generation``).
        test_split: Rows with ``message`` and ``label``.
        label_names: Intent names in label-index order.
        args: Parsed CLI arguments (``batch_size``, ``max_new_tokens``,
            ``load_in_4bit``).

    Returns:
        Gold intent names and predicted ones (``INVALID_PREDICTION`` when the
        answer is not an intent name).
    """
    model, tokenizer = load_for_generation(
        model_path, load_in_4bit=args.load_in_4bit
    )
    y_true = [label_names[label] for label in test_split["label"]]
    y_pred: list[str] = []
    texts = test_split["message"]
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        chats = [build_messages(text, label_names) for text in batch]
        # Qwen 3.5 renders an empty think block in the generation prompt,
        # as in its training texts; other templates ignore the flag.
        replies = generate_replies(
            model,
            tokenizer,
            chats,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=False,
        )
        y_pred.extend(
            extract_intent(reply, label_names) or INVALID_PREDICTION
            for reply in replies
        )
        print(f"Classified {len(y_pred)}/{len(texts)} messages")
    return y_true, y_pred


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the model path, sample limit and generation settings.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an intent classifier on the Banking77 test set"
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
        default=-1,
        help="Evaluate only the first N test rows (-1: all 3080)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Messages generated per batch (default: 32)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model in 4-bit (default: bf16)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Maximum tokens generated per answer (default: 16)",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate ``--model_path`` on Banking77 and print the metrics.

    Returns:
        None. Metrics and peak GPU memory are printed to stdout.
    """
    args = parse_args()
    dataset, label_names = load_banking77()
    test_split = dataset["test"]
    if args.max_samples > 0:
        test_split = test_split.select(range(args.max_samples))

    y_true, y_pred = predict_intents(
        args.model_path, test_split, label_names, args
    )

    print(f"\nModel: {args.model_path} ({len(y_true)} test messages)")
    for name, value in compute_metrics(y_true, y_pred, label_names).items():
        print(f"{name}: {value:.4f}")
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak_gpu_memory_gb: {peak_gb:.2f}")


if __name__ == "__main__":
    main()
