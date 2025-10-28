"""Interactive console tester for guard_og.py fine-tuned models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import unsloth
from huggingface_hub import login
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


print(unsloth.__version__)

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

INSTRUCTION = "Classify this prompt's as harmful or unharmful:"
CHAT_TEMPLATE = "gemma-3"


def load_instruction_from_config(model_path: str, fallback: str) -> str:
    """Load instruction from tokenizer config or fall back to a default."""
    tokenizer_config_path = Path(model_path) / "tokenizer_config.json"

    if tokenizer_config_path.exists():
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

    print(
        f"Warning: tokenizer_config.json not found at {tokenizer_config_path}"
    )
    print(f"Using fallback instruction: '{fallback}'")
    return fallback


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
    """Main interactive loop."""
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=None,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    instruction = load_instruction_from_config(args.model_path, INSTRUCTION)
    FastLanguageModel.for_inference(model)

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
                print("Please enter a non-empty prompt or type 'exit'.")
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
