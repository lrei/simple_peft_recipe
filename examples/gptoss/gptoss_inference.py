r"""Chat with the GPT-OSS reasoning-language adapter without speftr.

Loads the 4-bit base recorded in the adapter's ``adapter_config.json``,
attaches the LoRA adapter with peft and answers each prompt twice, with
the adapter and with it disabled (the base model), printing the analysis
channel (the reasoning) and the final channel (the answer) separately.

Unsloth is used only to load the base: ``unsloth/gpt-oss-*-unsloth-bnb-4bit``
stores every expert as its own 4-bit linear layer, which the adapter
targets and only Unsloth's gpt-oss model code builds (transformers'
``GptOssForCausalLM`` expects fused expert tensors). Unsloth's own gpt-oss
attention is turned off: in eval mode it gives the sliding-window layers
the full-attention mask, which is wrong past 128 tokens. Generation is
plain transformers + peft.

Only this 4-bit base runs the adapter. vLLM LoRA needs a 16-bit base, and
merging into OpenAI's MXFP4 or a 16-bit gpt-oss is not covered here.

Usage:
    uv run python -m examples.gptoss.gptoss_inference
    uv run python -m examples.gptoss.gptoss_inference \\
        --reasoning_language French --prompt "Why is the sky blue?"
    CUDA_VISIBLE_DEVICES=0,1 uv run python \\
        -m examples.gptoss.gptoss_inference \\
        --adapter_dir ./models/speftr-gptoss-120b --device_map auto

See "Use the trained model without speftr" in ``examples/gptoss/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch


if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import BatchEncoding, PreTrainedTokenizerBase


type Message = dict[str, str]

DEFAULT_ADAPTER_DIR = "./models/speftr-gptoss-20b"
DEFAULT_PROMPTS = (
    "What is the capital of Australia?",
    "Why do cats purr? Answer in two sentences.",
)
DEFAULT_LANGUAGE = "German"
DEFAULT_MAX_NEW_TOKENS = 512
MAX_SEQ_LENGTH = 2048
# One harmony message in generated text: its channel and its content, up
# to the end-of-message token or the end of the (possibly cut) text.
CHANNEL_PATTERN = re.compile(
    r"<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|\Z)",
    re.DOTALL,
)


def build_chats(
    prompts: list[str], reasoning_language: str, system_prompt: str = ""
) -> list[list[Message]]:
    """Wrap each prompt in the system + user turns used in training.

    Args:
        prompts: User messages.
        reasoning_language: Language the analysis channel should use, as
            the dataset names it (``German``, ``French``, ...).
        system_prompt: Optional further instructions after the language
            line, like the dataset's persona prompts.

    Returns:
        One chat per prompt. The system turn is ``reasoning language: X``
        (plus the instructions); gpt-oss's template renders it as the
        developer message.
    """
    system = f"reasoning language: {reasoning_language}"
    if system_prompt.strip():
        system = f"{system}\n\n{system_prompt.strip()}"
    return [
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt.strip()},
        ]
        for prompt in prompts
    ]


def parse_channels(generated: str) -> dict[str, str]:
    """Split generated harmony text into its channels.

    Args:
        generated: Decoded completion **with** special tokens, starting
            after the prompt's ``<|start|>assistant``.

    Returns:
        Channel name (``analysis``, ``final``, ...) to content. A message
        cut by the token budget keeps what was generated; for a repeated
        channel the first message wins.
    """
    channels: dict[str, str] = {}
    for channel, content in CHANNEL_PATTERN.findall(generated):
        channels.setdefault(channel, content.strip())
    return channels


def read_base_model(adapter_dir: str) -> str:
    """Return the base model recorded in ``adapter_config.json``.

    Args:
        adapter_dir: Directory written by ``save_model("lora")``.

    Returns:
        The 4-bit base the adapter was trained on.
    """
    config_path = Path(adapter_dir) / "adapter_config.json"
    config = cast("dict[str, Any]", json.loads(config_path.read_text()))
    return cast("str", config["base_model_name_or_path"])


def load_adapter_model(
    adapter_dir: str, device_map: str | None
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the 4-bit base with Unsloth and attach the adapter with peft.

    Args:
        adapter_dir: Directory written by ``save_model("lora")``.
        device_map: ``None`` for one GPU, or e.g. ``"auto"`` to spread a
            model that does not fit one GPU over all visible GPUs.

    Returns:
        The adapted model in eval mode and the tokenizer saved with the
        adapter (harmony template), padding on the left.
    """
    # Read when the model is loaded; must be set before from_pretrained.
    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
    # unsloth must be imported before peft/transformers.
    from unsloth import FastLanguageModel  # noqa: PLC0415, I001
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    options = {} if device_map is None else {"device_map": device_map}
    base, _ = FastLanguageModel.from_pretrained(
        model_name=read_base_model(adapter_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        **options,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)  # nosec B615
    tokenizer.padding_side = "left"
    return model, tokenizer


def generate(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    chats: list[list[Message]],
    max_new_tokens: int,
) -> list[str]:
    """Answer a batch of chats with greedy decoding.

    Args:
        model: The adapted model (adapter enabled or disabled).
        tokenizer: Its tokenizer, padding on the left.
        chats: Conversations ending with a user turn.
        max_new_tokens: Generation budget per reply, reasoning included.

    Returns:
        The completions with harmony tokens, for ``parse_channels``.
    """
    encoded = tokenizer.apply_chat_template(
        chats,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = cast("BatchEncoding", encoded).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Left padding: every prompt ends at the same position.
    completions = outputs[:, inputs["input_ids"].shape[1] :]
    return list(tokenizer.batch_decode(completions))


def print_reply(label: str, completion: str) -> None:
    """Print the analysis and final channels of one completion.

    Args:
        label: ``adapter`` or ``base``.
        completion: Output of ``generate``.

    Returns:
        None. A missing channel prints as ``(none)``; the final channel is
        missing when the token budget ran out during the reasoning.
    """
    channels = parse_channels(completion)
    print(f"[{label}] analysis: {channels.get('analysis', '(none)')}")
    print(f"[{label}] final:    {channels.get('final', '(none)')}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the adapter, device map, prompts, reasoning
        language, system prompt and generation budget.
    """
    parser = argparse.ArgumentParser(
        description="Chat with the GPT-OSS adapter without speftr."
    )
    parser.add_argument("--adapter_dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--device_map",
        default=None,
        help="'auto' spreads the model over all visible GPUs "
        "(default: one GPU)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="User message; repeat for a batch (default: two questions).",
    )
    parser.add_argument(
        "--reasoning_language",
        default=DEFAULT_LANGUAGE,
        help=f"Language of the reasoning (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--system_prompt",
        default="",
        help="Instructions added after the reasoning-language line",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    return parser.parse_args()


def main() -> None:
    """Answer the prompts with the adapter and the base model.

    Returns:
        None. Both replies are printed per prompt.
    """
    args = parse_args()
    prompts: list[str] = args.prompt or list(DEFAULT_PROMPTS)
    chats = build_chats(prompts, args.reasoning_language, args.system_prompt)
    model, tokenizer = load_adapter_model(args.adapter_dir, args.device_map)

    adapter_replies = generate(model, tokenizer, chats, args.max_new_tokens)
    with model.disable_adapter():
        base_replies = generate(model, tokenizer, chats, args.max_new_tokens)

    for prompt, adapter, base in zip(
        prompts, adapter_replies, base_replies, strict=True
    ):
        print(
            f"\n>>> {prompt} (reasoning language: {args.reasoning_language})"
        )
        print_reply("adapter", adapter)
        print_reply("base", base)


if __name__ == "__main__":
    main()
