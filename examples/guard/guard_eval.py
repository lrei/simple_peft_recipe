"""Evaluate WildGuardMix classification models fine-tuned with guard_og.py."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import unsloth
from datasets import Dataset, load_dataset
from huggingface_hub import login
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
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

print(unsloth.__version__)

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

PROMPT_COL = "prompt"
LABEL_COL = "prompt_harm_label"
INSTRUCTION = "Classify this prompt's as harmful or unharmful:"
CHAT_TEMPLATE = "gemma-3"
MAX_INVALID_EXAMPLES_TO_SHOW = 5
BINARY_CLASSIFICATION_LABELS = 2


def _parse_max_seq_length(value: str) -> int | None:
    """Parse max_seq_length allowing the keyword 'none'."""
    if value.lower() == "none":
        return None
    return int(value)


@dataclass
class GenerationConfig:
    """Configuration for model generation during evaluation."""

    instruction: str = INSTRUCTION
    max_new_tokens: int = 5
    temperature: float = 0.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation process."""

    batch_size: int = 64
    gen_config: GenerationConfig | None = None


def load_instruction_from_config(model_path: str, fallback: str) -> str:
    """Load instruction from tokenizer config, with fallback to default."""
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

    instruction = config.get("instruction_prefix", fallback)
    print(f"✓ Loaded instruction from config: '{instruction}'")
    return instruction


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a guard_og.py fine-tuned model on the "
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
        type=_parse_max_seq_length,
        default=2048,
        help="Maximum sequence length or 'none' for unlimited",
    )

    return parser.parse_args()


def _load_eval_dataset(max_samples: int) -> Dataset:
    """Load and optionally subsample the WildGuardMix evaluation split."""
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
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset: Dataset,
    valid_labels: set[str],
    eval_config: EvaluationConfig,
) -> tuple[list[str], list[str], list[str], int]:
    """Generate predictions for all examples in dataset using batching."""
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

            matched_label, is_valid = extract_label(pred_text, valid_labels)
            if not is_valid:
                invalid_predictions += 1
                matched_label = "INVALID_PREDICTION"

            y_pred.append(matched_label)

        y_true.extend(batch_true_labels)

    return y_true, y_pred, y_pred_raw, invalid_predictions


def extract_label(
    pred_text: str,
    valid_labels: set[str],
) -> tuple[str | None, bool]:
    """Extract valid label from prediction text."""
    pred_text_lower = pred_text.strip().lower()

    for valid_label in valid_labels:
        if pred_text_lower == valid_label.lower():
            return valid_label, True

    for valid_label in valid_labels:
        if valid_label.lower() in pred_text_lower:
            return valid_label, True

    return None, False


def _compute_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, float]:
    """Compute binary classification metrics."""
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

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "avg_precision": average_precision_score(
            [1 if label == positive_label else 0 for label in y_true],
            [1 if pred == positive_label else 0 for pred in y_pred],
        ),
    }

    report = classification_report(y_true, y_pred, zero_division="warn")
    metrics["classification_report"] = report
    return metrics


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

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
