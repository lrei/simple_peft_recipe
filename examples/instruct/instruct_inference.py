r"""Chat with the pirate instruct adapter without speftr or Unsloth.

Runs the adapter saved by ``instruct`` two ways and prints both replies
for every message, which should agree:

1. Adapter route (``run_adapter_route``): the base model plus the LoRA
   adapter, attached with peft (``--engine transformers``) or passed to
   vLLM as a ``LoRARequest`` (``--engine vllm``).
2. Merged route (``merge_adapter``, then ``run_merged_route``): the
   adapter merged into the 16-bit base (``merge_and_unload``) and saved
   with the tokenizer to ``--merged_dir``, a standalone model that loads
   like any Hub checkpoint.

Only torch, transformers, peft and, for ``--engine vllm``, vllm are needed.

Usage:
    uv run python -m examples.instruct.instruct_inference \\
        --base_model Qwen/Qwen3-0.6B
    uv run python -m examples.instruct.instruct_inference \\
        --base_model Qwen/Qwen3-0.6B --engine vllm \\
        --prompt "What is the capital of France?"

The example trains on the pre-quantized
``unsloth/Qwen3-0.6B-unsloth-bnb-4bit``, so ``--base_model`` names its
16-bit original.

See "Use the trained model without speftr" in ``examples/instruct/README.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch


if TYPE_CHECKING:
    from collections.abc import Sequence

    from peft import PeftModel
    from transformers import (
        BatchEncoding,
        GenerationMixin,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )


type Message = dict[str, str]

DEFAULT_ADAPTER_DIR = "./models/speftr-instruction"
# Default --system_prompt of instruct.py.
SYSTEM_PROMPT = "You are a pirate, always respond in pirate speech."
DEFAULT_PROMPTS = (
    "If I have more pieces at the time of stalemate, have I won?",
    "What is the capital of France?",
)
DEFAULT_MAX_NEW_TOKENS = 128
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {}
MAX_MODEL_LEN = 2048
# Most padded tokens per transformers batch: rows times the longest
# prompt plus max_new_tokens, the length the KV cache grows to.
TOKEN_BUDGET = 16_384
# vLLM reserves this share of GPU memory whatever the model size.
GPU_MEMORY_UTILIZATION = 0.8
# The values vLLM accepts for max_lora_rank.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


@dataclass(frozen=True)
class GenerationSettings:
    """Generation options shared by every route.

    Attributes:
        engine: ``"transformers"`` (peft) or ``"vllm"``.
        max_new_tokens: Generation budget per reply.
        max_model_len: Context length in tokens, prompt plus
            ``max_new_tokens``. Both engines cut a longer prompt from
            the start, with a warning (``truncate_prompts``).
        token_budget: Transformers engine: most padded tokens per batch,
            rows times the longest prompt plus ``max_new_tokens``.
    """

    engine: str = "transformers"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_model_len: int = MAX_MODEL_LEN
    token_budget: int = TOKEN_BUDGET


def build_chats(
    prompts: list[str], system_prompt: str = SYSTEM_PROMPT
) -> list[list[Message]]:
    """Wrap each message in the system + user turns used in training.

    Args:
        prompts: User messages.
        system_prompt: System message; the one the adapter was trained
            with.

    Returns:
        One chat per message: the system prompt, then the message.
    """
    return [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.strip()},
        ]
        for prompt in prompts
    ]


def read_adapter_config(adapter_dir: str) -> dict[str, Any]:
    """Read ``adapter_config.json`` (base model id, rank, targets).

    Args:
        adapter_dir: Directory written by ``save_model("lora")``.

    Returns:
        The parsed config.
    """
    return cast(
        "dict[str, Any]",
        json.loads((Path(adapter_dir) / "adapter_config.json").read_text()),
    )


def model_architecture(model_id: str) -> str:
    """Return the model class named in a checkpoint's config.

    Args:
        model_id: Hub id or local directory.

    Returns:
        The first entry of ``architectures``, e.g. ``Gemma3ForCausalLM``.
    """
    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(model_id)  # nosec B615
    return cast("list[str]", config.architectures)[0]


def resolve_base_model(adapter_dir: str, base_model: str | None) -> str:
    """Pick the 16-bit base model the adapter is applied to.

    Args:
        adapter_dir: Directory written by ``save_model("lora")``.
        base_model: Explicit base, or ``None`` for the one recorded in
            ``adapter_config.json``.

    Returns:
        The base model id or path.

    Raises:
        ValueError: If the base is a quantized checkpoint, which neither
            merging nor vLLM LoRA supports.
    """
    from transformers import AutoConfig  # noqa: PLC0415

    model_id = (
        base_model
        or read_adapter_config(adapter_dir)["base_model_name_or_path"]
    )
    config = AutoConfig.from_pretrained(model_id)  # nosec B615
    if getattr(config, "quantization_config", None) is not None:
        msg = (
            f"{model_id} is quantized; pass its 16-bit original as "
            "--base_model."
        )
        raise ValueError(msg)
    return cast("str", model_id)


def load_model(model_id: str) -> PreTrainedModel:
    """Load a checkpoint in bf16 with the class named in its config.

    Multimodal checkpoints (Qwen 3.5, Gemma 4) need their
    ``...ForConditionalGeneration`` class: ``AutoModelForCausalLM`` builds
    a text-only class whose module names match no adapter weight.

    Args:
        model_id: Hub id or local directory.

    Returns:
        The model in eval mode, on the GPU when one is available.
    """
    import transformers  # noqa: PLC0415

    model_class = getattr(transformers, model_architecture(model_id))
    # User-supplied id or local path: no Hub revision to pin (B615).
    model = model_class.from_pretrained(  # nosec B615
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    return cast("PreTrainedModel", model.eval())


def load_tokenizer(model_dir: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer saved with the adapter or merged model.

    It carries the training chat template, which can differ from the base
    model's.

    Args:
        model_dir: Adapter or merged model directory.

    Returns:
        The tokenizer, padding on the left for batched generation.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_dir)  # nosec B615
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_adapter_model(adapter_dir: str, base_id: str) -> PeftModel:
    """Attach the LoRA adapter to its base model.

    Args:
        adapter_dir: Directory written by ``save_model("lora")``.
        base_id: 16-bit base model.

    Returns:
        The adapted model in eval mode.
    """
    from peft import PeftModel  # noqa: PLC0415

    model = PeftModel.from_pretrained(load_model(base_id), adapter_dir)
    model.eval()
    return model


def length_batches(
    lengths: Sequence[int], token_budget: int
) -> list[list[int]]:
    """Group sequences into batches of similar length.

    Sequences are taken longest first and a batch grows while its padded
    size (rows times the longest row) fits ``token_budget``; a sequence
    longer than the budget gets a batch of its own.

    Args:
        lengths: Token count of each sequence.
        token_budget: Maximum padded tokens per batch.

    Returns:
        Batches of indices into ``lengths``; every index appears once.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    for index in order:
        # The first row of each batch is its longest.
        if (
            batches
            and (len(batches[-1]) + 1) * lengths[batches[-1][0]]
            <= token_budget
        ):
            batches[-1].append(index)
        else:
            batches.append([index])
    return batches


def prompt_token_limit(settings: GenerationSettings) -> int:
    """Return the most prompt tokens that leave room for the reply.

    Args:
        settings: Context length and reply length.

    Returns:
        ``max_model_len`` minus ``max_new_tokens``.

    Raises:
        ValueError: If ``max_new_tokens`` leaves no room for a prompt.
    """
    limit = settings.max_model_len - settings.max_new_tokens
    if limit < 1:
        msg = (
            f"--max_new_tokens {settings.max_new_tokens} leaves no room "
            f"for a prompt in --max_model_len {settings.max_model_len}; "
            "lower --max_new_tokens or raise --max_model_len."
        )
        raise ValueError(msg)
    return limit


def warn_long_prompts(
    lengths: Sequence[int], settings: GenerationSettings
) -> int:
    """Warn about prompts that lose their beginning to fit the context.

    Args:
        lengths: Prompt tokens of each input, chat template applied.
        settings: Context length and reply length.

    Returns:
        The number of prompts longer than ``prompt_token_limit``; a
        warning is printed when it is not 0.
    """
    limit = prompt_token_limit(settings)
    too_long = [length for length in lengths if length > limit]
    if too_long:
        print(
            f"\nWARNING: {len(too_long)} of {len(lengths)} prompts are "
            f"longer than {limit} tokens (context {settings.max_model_len} "
            f"minus max_new_tokens {settings.max_new_tokens}; longest: "
            f"{max(too_long)}). Their beginning is cut off, so the model "
            "never sees it.\n"
            "  Fix: raise --max_model_len (more memory), lower "
            "--max_new_tokens, or shorten those inputs."
        )
    return len(too_long)


def truncate_prompts(
    prompts: list[list[int]], settings: GenerationSettings
) -> list[list[int]]:
    """Cut prompts to ``prompt_token_limit`` tokens, keeping their end.

    Chat templates put the question and the assistant header at the end
    of a prompt, so the start is what goes. Prints the
    ``warn_long_prompts`` warning.

    Args:
        prompts: Token ids of each prompt, chat template applied.
        settings: Context length and reply length.

    Returns:
        The prompts, each at most ``prompt_token_limit`` tokens long.
    """
    warn_long_prompts([len(prompt) for prompt in prompts], settings)
    limit = prompt_token_limit(settings)
    return [prompt[-limit:] for prompt in prompts]


def vllm_truncation(settings: GenerationSettings) -> dict[str, Any]:
    """Return vLLM options that cut prompts like ``truncate_prompts``.

    Args:
        settings: Context length and reply length.

    Returns:
        ``tokenization_kwargs`` for ``LLM.chat``: keep the last
        ``prompt_token_limit`` tokens of each prompt.
    """
    return {
        "truncate_prompt_tokens": prompt_token_limit(settings),
        "truncation_side": "left",
    }


def encode_chats(
    tokenizer: PreTrainedTokenizerBase, chats: list[list[Message]]
) -> list[list[int]]:
    """Render and tokenize generation prompts, without padding.

    Args:
        tokenizer: Tokenizer with the training chat template.
        chats: Conversations ending with a user turn.

    Returns:
        The token ids of each chat, ending with the assistant header.
    """
    encoded = tokenizer.apply_chat_template(
        chats,
        add_generation_prompt=True,
        return_dict=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    return cast("list[list[int]]", cast("BatchEncoding", encoded)["input_ids"])


def generate_batch(
    model: GenerationMixin | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> list[str]:
    """Answer tokenized prompts greedily in one left-padded batch.

    Args:
        model: A causal LM, with or without an adapter.
        tokenizer: Its tokenizer, padding on the left.
        prompts: Token ids of each prompt, chat template applied.
        max_new_tokens: Generation budget per reply.

    Returns:
        The decoded replies without the prompt or special tokens, in the
        order of ``prompts``.
    """
    # The attention mask keeps the model from attending to the padding.
    padded = tokenizer.pad({"input_ids": prompts}, return_tensors="pt")
    inputs = padded.to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Left padding: every prompt ends at the same position.
    replies = outputs[:, inputs["input_ids"].shape[1] :]
    return list(tokenizer.batch_decode(replies, skip_special_tokens=True))


def generate_transformers(
    model: GenerationMixin | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    chats: list[list[Message]],
    settings: GenerationSettings,
) -> list[str]:
    """Answer chats with greedy decoding, in token-budget batches.

    Chats of similar length share a batch of at most
    ``settings.token_budget`` padded tokens, each row counting its prompt
    plus ``max_new_tokens`` (the length its KV cache grows to).

    Args:
        model: A causal LM, with or without an adapter.
        tokenizer: Its tokenizer, padding on the left.
        chats: Conversations ending with a user turn.
        settings: Reply length, context length and batch token budget.

    Returns:
        The decoded replies without the prompt or special tokens, in the
        order of ``chats``. A prompt longer than ``prompt_token_limit``
        loses its beginning (``truncate_prompts``).
    """
    prompts = truncate_prompts(encode_chats(tokenizer, chats), settings)
    lengths = [len(prompt) + settings.max_new_tokens for prompt in prompts]
    replies = [""] * len(prompts)
    for batch in length_batches(lengths, settings.token_budget):
        batch_replies = generate_batch(
            model,
            tokenizer,
            [prompts[index] for index in batch],
            settings.max_new_tokens,
        )
        for index, reply in zip(batch, batch_replies, strict=True):
            replies[index] = reply
    return replies


def free_gpu_memory() -> None:
    """Release the GPU memory of models that are no longer referenced.

    Returns:
        None.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vllm_max_lora_rank(adapter_dir: str) -> int:
    """Return the smallest vLLM ``max_lora_rank`` that fits the adapter.

    Args:
        adapter_dir: Directory holding ``adapter_config.json``.

    Returns:
        The smallest value in ``VLLM_LORA_RANKS`` at least the adapter's
        ``r``.
    """
    rank = read_adapter_config(adapter_dir)["r"]
    return next(limit for limit in VLLM_LORA_RANKS if limit >= rank)


def generate_vllm(
    model_id: str,
    chats: list[list[Message]],
    settings: GenerationSettings,
    adapter_dir: str | None = None,
) -> list[str]:
    """Answer a batch of chats greedily with an offline vLLM engine.

    The engine is started for this call and released at the end, so both
    routes can run in one process.

    Args:
        model_id: Base model (with ``adapter_dir``) or merged model.
        chats: Conversations ending with a user turn.
        settings: Reply length and context length
            (``max_model_len``).
        adapter_dir: LoRA adapter every request is routed through, or
            ``None`` to use ``model_id`` as is.

    Returns:
        The generated replies. A prompt longer than
        ``prompt_token_limit`` loses its beginning, as in
        ``generate_transformers``.
    """
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    # vLLM would otherwise apply the base model's own chat template.
    template = Path(adapter_dir or model_id) / "chat_template.jinja"
    # Tokenized only to count the prompts vLLM cuts (tokenization_kwargs).
    prompts = encode_chats(load_tokenizer(str(template.parent)), chats)
    warn_long_prompts([len(prompt) for prompt in prompts], settings)

    lora_options: dict[str, Any] = {}
    lora_request = None
    if adapter_dir is not None:
        lora_options = {
            "enable_lora": True,
            "max_lora_rank": vllm_max_lora_rank(adapter_dir),
        }
        lora_request = LoRARequest("adapter", 1, adapter_dir)
    multimodal = model_architecture(model_id).endswith(
        "ForConditionalGeneration"
    )
    llm = LLM(
        model=model_id,
        max_model_len=settings.max_model_len,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        # Skip the vision/audio encoders: the prompts are text only.
        language_model_only=multimodal,
        **lora_options,
    )
    outputs = llm.chat(
        chats,
        SamplingParams(temperature=0.0, max_tokens=settings.max_new_tokens),
        lora_request=lora_request,
        chat_template=template.read_text(),
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        tokenization_kwargs=vllm_truncation(settings),
    )
    del llm
    free_gpu_memory()
    return [output.outputs[0].text for output in outputs]


def merge_adapter(adapter_dir: str, base_id: str, merged_dir: str) -> None:
    """Merge the adapter into its base and save a standalone model.

    The merged model loads without peft, like any Hub checkpoint.

    Args:
        adapter_dir: Directory written by ``save_model("lora")``.
        base_id: 16-bit base model to merge into.
        merged_dir: Output directory; created or overwritten.

    Returns:
        None. Weights, config, tokenizer and, for multimodal bases, the
        processor are written to ``merged_dir``.
    """
    from transformers import AutoProcessor  # noqa: PLC0415

    model = load_adapter_model(adapter_dir, base_id)
    model.merge_and_unload().save_pretrained(merged_dir)
    tokenizer = load_tokenizer(adapter_dir)
    tokenizer.save_pretrained(merged_dir)
    if model_architecture(base_id).endswith("ForConditionalGeneration"):
        # The processor writes its own chat template; keep the adapter's.
        processor = AutoProcessor.from_pretrained(base_id)  # nosec B615
        processor.tokenizer = tokenizer
        processor.chat_template = tokenizer.chat_template
        processor.save_pretrained(merged_dir)


def run_adapter_route(
    adapter_dir: str,
    base_id: str,
    chats: list[list[Message]],
    settings: GenerationSettings,
) -> list[str]:
    """Generate with the base model and the separate LoRA adapter.

    Args:
        adapter_dir: Directory written by ``save_model("lora")``.
        base_id: 16-bit base model.
        chats: Conversations ending with a user turn.
        settings: Engine, reply length and limits.

    Returns:
        One reply per chat.
    """
    if settings.engine == "vllm":
        return generate_vllm(base_id, chats, settings, adapter_dir)
    model = load_adapter_model(adapter_dir, base_id)
    tokenizer = load_tokenizer(adapter_dir)
    return generate_transformers(model, tokenizer, chats, settings)


def run_merged_route(
    merged_dir: str,
    chats: list[list[Message]],
    settings: GenerationSettings,
) -> list[str]:
    """Generate with the merged model saved by ``merge_adapter``.

    Args:
        merged_dir: Directory written by ``merge_adapter``.
        chats: Conversations ending with a user turn.
        settings: Engine, reply length and limits.

    Returns:
        One reply per chat.
    """
    if settings.engine == "vllm":
        return generate_vllm(merged_dir, chats, settings)
    # Causal LM classes subclass GenerationMixin.
    model = cast("GenerationMixin", load_model(merged_dir))
    tokenizer = load_tokenizer(merged_dir)
    return generate_transformers(model, tokenizer, chats, settings)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the adapter, base model, merged directory, engine,
        prompts, generation budget, context length and batch token
        budget.
    """
    parser = argparse.ArgumentParser(
        description="Chat with the instruct adapter, separately and "
        "merged, without speftr."
    )
    parser.add_argument("--adapter_dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--base_model",
        default=None,
        help="16-bit base model (default: the one in adapter_config.json).",
    )
    parser.add_argument(
        "--merged_dir",
        default=None,
        help="Where the merged model is saved "
        "(default: <adapter_dir>-merged).",
    )
    parser.add_argument(
        "--engine", choices=["transformers", "vllm"], default="transformers"
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="User message; repeat for a batch (default: two questions).",
    )
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=MAX_MODEL_LEN,
        help="Context length: prompt plus --max_new_tokens tokens "
        f"(default: {MAX_MODEL_LEN}). A longer prompt loses its beginning, "
        "with a warning.",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=TOKEN_BUDGET,
        help="Transformers engine: most padded tokens per batch, rows "
        "times the longest prompt plus --max_new_tokens (default: "
        f"{TOKEN_BUDGET}). A longer input runs alone.",
    )
    return parser.parse_args()


def main() -> None:
    """Answer the messages through both routes and print the replies.

    Returns:
        None. Replies are printed to stdout; the merged model is written to
        ``--merged_dir``.
    """
    args = parse_args()
    prompts: list[str] = args.prompt or list(DEFAULT_PROMPTS)
    chats = build_chats(prompts, args.system_prompt)
    merged_dir = args.merged_dir or f"{args.adapter_dir.rstrip('/')}-merged"
    base_id = resolve_base_model(args.adapter_dir, args.base_model)
    settings = GenerationSettings(
        args.engine, args.max_new_tokens, args.max_model_len, args.token_budget
    )

    adapter_replies = run_adapter_route(
        args.adapter_dir, base_id, chats, settings
    )
    free_gpu_memory()
    merge_adapter(args.adapter_dir, base_id, merged_dir)
    free_gpu_memory()
    merged_replies = run_merged_route(merged_dir, chats, settings)

    print(f"\nEngine: {args.engine}. Merged model saved to {merged_dir}")
    for prompt, adapter, merged in zip(
        prompts, adapter_replies, merged_replies, strict=True
    ):
        print(f"\n>>> {prompt}")
        print(f"adapter: {adapter.strip()}")
        print(f"merged:  {merged.strip()}")


if __name__ == "__main__":
    main()
