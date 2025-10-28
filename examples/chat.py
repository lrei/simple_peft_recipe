"""Interactive chat interface for instruction-tuned models."""

from __future__ import annotations

import argparse
import os
from typing import Final, Protocol, runtime_checkable


# Force protobuf to use the pure-Python implementation so Unsloth's helpers
# remain compatible with newer protobuf wheels.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


print(unsloth.__version__)

DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a pirate, always respond in pirate speech."
)
DEFAULT_MAX_SEQ_LENGTH: Final[int] = 4096


@runtime_checkable
class SupportsGenerationConfig(Protocol):
    """Protocol for models exposing a generation_config attribute."""

    generation_config: object


@runtime_checkable
class SupportsThinkingToggle(Protocol):
    """Protocol for configs exposing an enable_thinking flag."""

    enable_thinking: bool


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with a PESFT instruction-tuned model"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help=(
            "Model identifier or local path to the trained checkpoint "
            "(default: ./models/speftr-instruction)"
        ),
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512).",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7).",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p (default: 0.9).",
    )

    parser.add_argument(
        "--chat_template",
        type=str,
        default="chatml",
        help="Chat template to use (default: chatml).",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit loading (default: disabled).",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=(
            "Maximum sequence length to allocate for the model "
            f"(default: {DEFAULT_MAX_SEQ_LENGTH})."
        ),
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help=(
            "System prompt to prepend to every conversation "
            "(default: pirate instructions)."
        ),
    )

    return parser.parse_args()


def _disable_thinking_mode(model: object) -> None:
    """Disable thinking tokens if the model supports them."""
    if not isinstance(model, SupportsGenerationConfig):
        return
    generation_config = model.generation_config
    if isinstance(generation_config, SupportsThinkingToggle):
        generation_config.enable_thinking = False


def main() -> None:
    """Main interactive chat loop."""
    args = parse_args()
    system_prompt = args.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT

    print(f"Loading model from {args.model_name_or_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=args.chat_template,
    )

    if hasattr(tokenizer, "default_system_prompt"):
        tokenizer.default_system_prompt = system_prompt

    FastLanguageModel.for_inference(model)
    _disable_thinking_mode(model)

    print("\n" + "=" * 60)
    print("Interactive Chat")
    print("=" * 60)
    print(
        "Type your messages and press Enter. "
        "Type 'exit' or 'quit' to stop, or press Ctrl+C/Ctrl+D."
    )
    print("=" * 60 + "\n")

    conversation_history: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    try:
        while True:
            try:
                user_input = input("\nYou: ")
            except EOFError:
                print("\n\nReceived EOF. Exiting.")
                break

            user_input = user_input.strip()

            if not user_input:
                print("Please enter a non-empty message or type 'exit'.")
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting chat.")
                break

            conversation_history.append(
                {"role": "user", "content": user_input}
            )

            prompt_text = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=False,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_tokens = outputs[0][input_ids.shape[1] :]
            response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            print(f"\033[1;34mAssistant:\033[0m {response}")

            conversation_history.append(
                {"role": "assistant", "content": response}
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
