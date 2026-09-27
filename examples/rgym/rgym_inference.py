r"""Solve Reasoning Gym problems with the rgym adapter without speftr.

Runs the adapter saved by ``rgym`` two ways and prints both replies
(reasoning in ``<think>``, result in ``<answer>``) for every question,
which should agree:

1. Adapter route (``run_adapter_route``): the base model plus the LoRA
   adapter, attached with peft (``--engine transformers``) or passed to
   vLLM as a ``LoRARequest`` (``--engine vllm``).
2. Merged route (``merge_adapter``, then ``run_merged_route``): the
   adapter merged into the 16-bit base (``merge_and_unload``) and saved
   with the tokenizer to ``--merged_dir``, a standalone model that loads
   like any Hub checkpoint.

Only torch, transformers, peft and, for ``--engine vllm``, vllm are needed.

Usage:
    uv run python -m examples.rgym.rgym_inference \\
        --adapter_dir ./models/chainsum
    uv run python -m examples.rgym.rgym_inference \\
        --adapter_dir ./models/chainsum --engine vllm

See "Use the trained model without speftr" in ``examples/rgym/README.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch


if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import (
        BatchEncoding,
        GenerationMixin,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )


type Message = dict[str, str]

DEFAULT_ADAPTER_DIR = "./models/sudoku-perl"
# reasoning_gym.utils.SYSTEM_PROMPTS["DeepSeekZero"], rgym.py's default
# --developer_prompt.
DEVELOPER_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it.\nThe assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within <think> "
    "</think> and <answer> </answer> tags, respectively, i.e., <think> "
    "reasoning process here </think>\n<answer>answer here</answer>\nDo not "
    "explain your reasoning inside the answer tags, provide only the final "
    "answer. When an example is provided, you should strictly follow the "
    "format of the output/answer in that example.\n"
)
# Questions from Reasoning Gym's chain_sum task.
DEFAULT_PROMPTS = (
    "State the final answer to the following arithmetic problem: 4 + 3 =",
    "State the final answer to the following arithmetic problem: 812 + 880 =",
)
DEFAULT_MAX_NEW_TOKENS = 512
# As in training: Qwen3-style templates otherwise insert an empty
# <think></think> block.
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": True}
MAX_MODEL_LEN = 2048
# vLLM reserves this share of GPU memory whatever the model size.
GPU_MEMORY_UTILIZATION = 0.8
# The values vLLM accepts for max_lora_rank.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def build_chats(questions: list[str]) -> list[list[Message]]:
    """Wrap each question in the system + user turns used in training.

    Args:
        questions: Reasoning Gym questions.

    Returns:
        One chat per question: ``DEVELOPER_PROMPT`` as the system message,
        then the question.
    """
    return [
        [
            {"role": "system", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": question},
        ]
        for question in questions
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


def generate_transformers(
    model: GenerationMixin | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    chats: list[list[Message]],
    max_new_tokens: int,
) -> list[str]:
    """Answer a batch of chats with greedy decoding.

    Args:
        model: A causal LM, with or without an adapter.
        tokenizer: Its tokenizer, padding on the left.
        chats: Conversations ending with a user turn.
        max_new_tokens: Generation budget per reply.

    Returns:
        The decoded replies without the prompt or special tokens.
    """
    encoded = tokenizer.apply_chat_template(
        chats,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
        **CHAT_TEMPLATE_KWARGS,
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
    replies = outputs[:, inputs["input_ids"].shape[1] :]
    return list(tokenizer.batch_decode(replies, skip_special_tokens=True))


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
    max_new_tokens: int,
    adapter_dir: str | None = None,
) -> list[str]:
    """Answer a batch of chats greedily with an offline vLLM engine.

    The engine is started for this call and released at the end, so both
    routes can run in one process.

    Args:
        model_id: Base model (with ``adapter_dir``) or merged model.
        chats: Conversations ending with a user turn.
        max_new_tokens: Generation budget per reply.
        adapter_dir: LoRA adapter every request is routed through, or
            ``None`` to use ``model_id`` as is.

    Returns:
        The generated replies.
    """
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

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
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        # Skip the vision/audio encoders: the prompts are text only.
        language_model_only=multimodal,
        **lora_options,
    )
    # vLLM would otherwise apply the base model's own chat template.
    template = Path(adapter_dir or model_id) / "chat_template.jinja"
    outputs = llm.chat(
        chats,
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
        lora_request=lora_request,
        chat_template=template.read_text(),
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
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
    engine: str,
    max_new_tokens: int,
) -> list[str]:
    """Generate with the base model and the separate LoRA adapter.

    Args:
        adapter_dir: Directory written by ``save_model("lora")``.
        base_id: 16-bit base model.
        chats: Conversations ending with a user turn.
        engine: ``"transformers"`` (peft) or ``"vllm"`` (``LoRARequest``).
        max_new_tokens: Generation budget per reply.

    Returns:
        One reply per chat.
    """
    if engine == "vllm":
        return generate_vllm(base_id, chats, max_new_tokens, adapter_dir)
    model = load_adapter_model(adapter_dir, base_id)
    tokenizer = load_tokenizer(adapter_dir)
    return generate_transformers(model, tokenizer, chats, max_new_tokens)


def run_merged_route(
    merged_dir: str,
    chats: list[list[Message]],
    engine: str,
    max_new_tokens: int,
) -> list[str]:
    """Generate with the merged model saved by ``merge_adapter``.

    Args:
        merged_dir: Directory written by ``merge_adapter``.
        chats: Conversations ending with a user turn.
        engine: ``"transformers"`` or ``"vllm"``.
        max_new_tokens: Generation budget per reply.

    Returns:
        One reply per chat.
    """
    if engine == "vllm":
        return generate_vllm(merged_dir, chats, max_new_tokens)
    # Causal LM classes subclass GenerationMixin.
    model = cast("GenerationMixin", load_model(merged_dir))
    tokenizer = load_tokenizer(merged_dir)
    return generate_transformers(model, tokenizer, chats, max_new_tokens)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the adapter, base model, merged directory, engine,
        prompts and generation budget.
    """
    parser = argparse.ArgumentParser(
        description="Solve Reasoning Gym questions with the rgym adapter, "
        "separately and merged, without speftr."
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
        help="Question; repeat for a batch (default: two chain_sum "
        "questions).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    return parser.parse_args()


def main() -> None:
    """Answer the questions through both routes and print the replies.

    Returns:
        None. Replies are printed to stdout; the merged model is written to
        ``--merged_dir``.
    """
    args = parse_args()
    prompts: list[str] = args.prompt or list(DEFAULT_PROMPTS)
    chats = build_chats(prompts)
    merged_dir = args.merged_dir or f"{args.adapter_dir.rstrip('/')}-merged"
    base_id = resolve_base_model(args.adapter_dir, args.base_model)

    adapter_replies = run_adapter_route(
        args.adapter_dir, base_id, chats, args.engine, args.max_new_tokens
    )
    free_gpu_memory()
    merge_adapter(args.adapter_dir, base_id, merged_dir)
    free_gpu_memory()
    merged_replies = run_merged_route(
        merged_dir, chats, args.engine, args.max_new_tokens
    )

    print(f"\nEngine: {args.engine}. Merged model saved to {merged_dir}")
    for prompt, adapter, merged in zip(
        prompts, adapter_replies, merged_replies, strict=True
    ):
        print(f"\n>>> {prompt}")
        print(f"adapter: {adapter.strip()}")
        print(f"merged:  {merged.strip()}")


if __name__ == "__main__":
    main()
