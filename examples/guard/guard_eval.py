r"""Evaluate a ``guard_train`` classifier on the WildGuardMix test set.

Demonstrates loading saved LoRA adapters for batched inference with
Unsloth (``FastLanguageModel.from_pretrained`` on the adapter directory
loads the base model named in ``adapter_config.json``), rebuilding the
training prompt, mapping free-text generations to labels and scoring them
with scikit-learn. The model is always loaded in 4-bit.

Usage:
    uv run python -m examples.guard.guard_eval
    uv run python -m examples.guard.guard_eval --model_path ./models/x \
        --max_samples 200

Prints invalid-answer count, accuracy, macro precision/recall/F1, average
precision and a per-class report. See ``examples/guard/README.md``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# unsloth must be imported before datasets/transformers so its patches
# apply.
import unsloth
from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from transformers import GenerationMixin, PreTrainedTokenizerBase

print(unsloth.__version__)

# Unsloth wraps torch.__getattr__, so torch's own filter for these
# deprecation warnings (keyed on module "torch") does not match.
warnings.filterwarnings(
    "ignore",
    message=".*is deprecated, please use.*",
    category=UserWarning,
    module="unsloth.import_fixes",
)

PROMPT_COL = "prompt"
LABEL_COL = "prompt_harm_label"
# Fallbacks for models without a saved instruction_prefix; both must match
# guard_train.py.
INSTRUCTION = "Classify this prompt's as harmful or unharmful:"
CHAT_TEMPLATE = "gemma-3"
MAX_INVALID_EXAMPLES_TO_SHOW = 5
BINARY_CLASSIFICATION_LABELS = 2


@dataclass
class GenerationConfig:
    """Configuration for model generation during evaluation.

    Attributes:
        instruction: Text prepended to each prompt in the user turn; empty
            means the prompt is sent alone.
        max_new_tokens: Maximum tokens generated per example.
        temperature: Generation temperature (decoding is greedy).
    """

    instruction: str = INSTRUCTION
    max_new_tokens: int = 5
    temperature: float = 0.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation process.

    Attributes:
        batch_size: Number of examples generated per batch.
        gen_config: Generation settings; ``None`` uses
            ``GenerationConfig()`` defaults.
    """

    batch_size: int = 64
    gen_config: GenerationConfig | None = None


def load_instruction_from_config(model_path: str, fallback: str) -> str:
    """Load instruction from tokenizer config, with fallback to default.

    Reads ``instruction_prefix`` from ``tokenizer_config.json`` in
    ``model_path``, as saved by guard_train.py.

    Args:
        model_path: Directory of the fine-tuned model.
        fallback: Instruction used when the file is missing, unreadable
            or lacks ``instruction_prefix``.

    Returns:
        The saved instruction, or ``fallback``.
    """
    tokenizer_config_path = Path(model_path) / "tokenizer_config.json"

    if not tokenizer_config_path.exists():
        print(
            f"Warning: tokenizer_config.json not found at "
            f"{tokenizer_config_path}"
        )
        print(f"Using fallback instruction: '{fallback}'")
        return fallback

    try:
        with tokenizer_config_path.open() as handle:
            config = json.load(handle)
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"Warning: Could not load instruction from config: {exc}")
        print(f"Using fallback instruction: '{fallback}'")
        return fallback

    instruction = str(config.get("instruction_prefix", fallback))
    print(f"Loaded instruction from config: '{instruction}'")
    return instruction


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the model path, sampling limit, generation and batch
        settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a guard_train.py fine-tuned model on the "
            "WildGuardMix test set"
        )
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/gemma-3-270m-it-lora-wildguard",
        help=(
            "Path to fine-tuned model directory "
            "(default: ./models/gemma-3-270m-it-lora-wildguard)"
        ),
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to evaluate (-1 for full split)",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5,
        help="Maximum tokens to generate (default: 5)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
    )

    return parser.parse_args()


def _load_eval_dataset(max_samples: int) -> Dataset:
    """Load and optionally subsample the WildGuardMix evaluation split.

    Args:
        max_samples: Keep only the first ``max_samples`` labelled rows;
            ``<= 0`` keeps the full split.

    Returns:
        The wildguardtest split without rows lacking a prompt harm label.
    """
    dataset = load_dataset(
        "allenai/wildguardmix", "wildguardtest", split="test"
    )
    dataset = dataset.filter(
        lambda x: x[LABEL_COL] is not None and x[LABEL_COL] != ""
    )
    if max_samples > 0:
        dataset = dataset.select(range(max_samples))
    return dataset


def _generate_predictions(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset: Dataset,
    valid_labels: set[str],
    eval_config: EvaluationConfig,
) -> tuple[list[str], list[str], list[str], int]:
    """Generate predictions for all examples in dataset using batching.

    Args:
        model: Causal LM used for greedy generation.
        tokenizer: Tokenizer with the guard chat template applied.
        eval_dataset: Examples with prompt and harm label columns.
        valid_labels: Labels a generated answer is matched against.
        eval_config: Batch size and generation settings.

    Returns:
        True labels, predicted labels (``"INVALID_PREDICTION"`` when no
        valid label matches), raw generated texts, and the number of
        invalid predictions.
    """
    gen_config = eval_config.gen_config or GenerationConfig()
    batch_size = eval_config.batch_size

    total_examples = len(eval_dataset)
    num_batches = (total_examples + batch_size - 1) // batch_size

    y_true: list[str] = []
    y_pred: list[str] = []
    y_pred_raw: list[str] = []
    invalid_predictions = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_examples)
        batch_examples = eval_dataset.select(range(start_idx, end_idx))

        print(
            f"Processing batch {batch_idx + 1}/{num_batches} "
            f"(examples {start_idx + 1}-{end_idx})..."
        )

        batch_true_labels: list[str] = []
        batch_messages: list[list[dict[str, str]]] = []

        instruction = gen_config.instruction.strip()

        for example in batch_examples:
            true_label = example[LABEL_COL]
            batch_true_labels.append(true_label)

            prompt_text = example[PROMPT_COL].strip()
            combined = (
                f"{instruction}\n\n{prompt_text}"
                if instruction
                else prompt_text
            )
            batch_messages.append(
                [
                    {
                        "role": "user",
                        "content": combined,
                    }
                ]
            )

        batch_inputs = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        input_lengths = batch_inputs["attention_mask"].sum(dim=1)

        for output, input_length in zip(outputs, input_lengths, strict=False):
            pred_text = tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True,
            ).strip()
            y_pred_raw.append(pred_text)

            matched_label, _ = extract_label(pred_text, valid_labels)
            if matched_label is None:
                invalid_predictions += 1
                matched_label = "INVALID_PREDICTION"

            y_pred.append(matched_label)

        y_true.extend(batch_true_labels)

    return y_true, y_pred, y_pred_raw, invalid_predictions


def extract_label(
    pred_text: str,
    valid_labels: AbstractSet[str],
) -> tuple[str | None, bool]:
    """Extract valid label from prediction text.

    Args:
        pred_text: Raw generated text.
        valid_labels: Labels the model may produce.

    Returns:
        ``(label, True)`` for an exact (case-insensitive) match or, failing
        that, the longest label contained in the text; ``(None, False)``
        otherwise.
    """
    pred_text_lower = pred_text.strip().lower()

    for valid_label in valid_labels:
        if pred_text_lower == valid_label.lower():
            return valid_label, True

    # Longest first: "unharmful" contains "harmful".
    for valid_label in sorted(valid_labels, key=len, reverse=True):
        if valid_label.lower() in pred_text_lower:
            return valid_label, True

    return None, False


def _compute_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, float]:
    """Compute binary classification metrics.

    Averages use only the labels present in ``y_true``, so invalid
    predictions count as errors. ``avg_precision`` treats ``"harmful"``
    (or the first sorted label) as the positive class.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels, aligned with ``y_true``.

    Returns:
        Accuracy, macro precision/recall/F1 and average precision, plus the
        text ``classification_report``.

    Raises:
        ValueError: If ``y_true`` does not contain exactly two labels.
    """
    unique_labels = sorted(set(y_true))

    if len(unique_labels) != BINARY_CLASSIFICATION_LABELS:
        error_msg = (
            f"Expected binary classification with 2 labels, got "
            f"{len(unique_labels)} labels: {unique_labels}"
        )
        raise ValueError(error_msg)

    positive_label = (
        "harmful" if "harmful" in unique_labels else unique_labels[0]
    )

    # Averages cover the dataset's labels only: an invalid prediction counts
    # as an error, not as an extra class.
    per_label = {"labels": unique_labels, "zero_division": 0}
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", **per_label
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", **per_label
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", **per_label),
        "avg_precision": average_precision_score(
            [1 if label == positive_label else 0 for label in y_true],
            [1 if pred == positive_label else 0 for pred in y_pred],
        ),
    }

    report = classification_report(y_true, y_pred, **per_label)
    metrics["classification_report"] = report
    return metrics


def main() -> None:
    """Evaluate a guard_train.py model on WildGuardMix and print metrics.

    Returns:
        None. Metrics are printed to stdout.
    """
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # Generation is bounded by max_new_tokens; a max_length saved with the
    # checkpoint would otherwise conflict with it.
    model.generation_config.max_length = None

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    instruction = load_instruction_from_config(args.model_path, INSTRUCTION)
    eval_dataset = _load_eval_dataset(args.max_samples)

    print(f"Evaluation dataset size: {len(eval_dataset)}")

    valid_labels = set(eval_dataset[LABEL_COL])
    print(f"Labels: {sorted(valid_labels)}")

    eval_config = EvaluationConfig(
        batch_size=args.batch_size,
        gen_config=GenerationConfig(
            instruction=instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
    )

    y_true, y_pred, y_pred_raw, invalid_predictions = _generate_predictions(
        model,
        tokenizer,
        eval_dataset,
        valid_labels,
        eval_config,
    )

    print(
        f"\nTotal invalid predictions: {invalid_predictions}"
        f" / {len(eval_dataset)}"
    )

    if invalid_predictions and len(y_pred_raw) <= MAX_INVALID_EXAMPLES_TO_SHOW:
        print("Invalid predictions preview:")
        for idx, raw in enumerate(y_pred_raw, 1):
            _, is_valid = extract_label(raw, valid_labels)
            if is_valid:
                continue
            print(f"  [{idx}] {raw}")

    metrics = _compute_metrics(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print("Overall Metrics:")
    print(f"{'=' * 60}")
    for metric_name, value in metrics.items():
        if metric_name == "classification_report":
            continue
        if isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    print(f"\n{metrics['classification_report']}")


if __name__ == "__main__":
    main()
