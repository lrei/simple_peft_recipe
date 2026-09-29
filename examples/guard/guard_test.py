"""Classify prompts typed at the console with a ``guard_train`` model.

Demonstrates single-prompt inference with saved LoRA adapters: load them
with Unsloth, apply the training chat template and instruction, and
generate the label greedily. Loads in 16-bit unless ``--load_in_4bit``.

Usage:
    uv run python -m examples.guard.guard_test
    uv run python -m examples.guard.guard_test --model_path ./models/x

Type a prompt, read ``Model label: harmful`` or ``unharmful``; ``exit``,
``quit``, Ctrl+C or Ctrl+D stops. See ``examples/guard/README.md``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

# unsloth must be imported before transformers so its patches apply.
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


print(unsloth.__version__)

# Unsloth wraps torch.__getattr__, so torch's own filter for these
# deprecation warnings (keyed on module "torch") does not match.
warnings.filterwarnings(
    "ignore",
    message=".*is deprecated, please use.*",
    category=UserWarning,
    module="unsloth.import_fixes",
)

# Fallbacks for models without a saved instruction_prefix; both must match
# guard_train.py.
INSTRUCTION = "Classify this prompt's as harmful or unharmful:"
CHAT_TEMPLATE = "gemma-3"


def load_instruction_from_config(model_path: str, fallback: str) -> str:
    """Load instruction from tokenizer config or fall back to a default.

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

    if tokenizer_config_path.exists():
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

    print(
        f"Warning: tokenizer_config.json not found at {tokenizer_config_path}"
    )
    print(f"Using fallback instruction: '{fallback}'")
    return fallback


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the model path, generation and loading options.
    """
    parser = argparse.ArgumentParser(
        description="Interactively classify prompts with a fine-tuned model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/gemma-3-270m-it-lora-wildguard",
        help=(
            "Path to trained model directory "
            "(default: ./models/gemma-3-270m-it-lora-wildguard)"
        ),
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate (default: 10)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit loading (default: disabled)",
    )

    return parser.parse_args()


def main() -> None:
    """Load the model and classify prompts read from stdin until exit.

    Returns:
        None. Predictions are printed to stdout.
    """
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    instruction = load_instruction_from_config(args.model_path, INSTRUCTION)
    FastLanguageModel.for_inference(model)
    # Generation is bounded by max_new_tokens; a max_length saved with the
    # checkpoint would otherwise conflict with it.
    model.generation_config.max_length = None

    prompt_message = (
        "\nInteractive safety classification. Enter prompts to classify."
        "\nType 'exit' or 'quit' to stop, or press Ctrl+C/Ctrl+D."
    )
    print(prompt_message)

    try:
        while True:
            try:
                prompt = input("\nUser prompt: ")
            except EOFError:
                print("\nReceived EOF. Exiting.")
                break

            prompt = prompt.strip()

            if not prompt:
                print("Empty prompt; type a prompt or 'exit'.")
                continue

            if prompt.lower() in {"exit", "quit"}:
                print("Exiting.")
                break

            user_message = (
                f"{instruction}\n\n{prompt}" if instruction else prompt
            )
            messages = [{"role": "user", "content": user_message}]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=False,
            )

            response = tokenizer.decode(
                outputs[0][inputs.shape[1] :], skip_special_tokens=True
            ).strip()

            print(f"Model label: {response}")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
