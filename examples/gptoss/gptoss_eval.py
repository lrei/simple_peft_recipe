r"""Evaluate the GPT-OSS reasoning-language adapter against its base model.

Every row of ``HuggingFaceH4/Multilingual-Thinking`` asks for a
"reasoning language" (English, French, German, Spanish or Italian) in its
system turn. gpt-oss reasons in English whatever it is told; the adapter
should make the analysis channel follow the requested language. On the
held-out rows (``load_splits``) this script reports, for the base model
and for the adapter on the same 4-bit base:

- **eval loss**: mean token loss on the assistant turn (analysis and
  final channels), masked exactly as in training;
- **reasoning-language compliance**: the share of generated analysis
  channels whose language (``py3langid``) is the requested one, per
  language and overall. Only the start of the reasoning is generated
  (``--max_new_tokens``).

This module also defines the task for ``gptoss.py``: the dataset and its
fixed train/eval split, the response-only markers and the channel parsing.

Usage (``py3langid`` comes with the ``gptoss`` extra):
    uv run --extra gptoss python -m examples.gptoss.gptoss_eval
    uv run --extra gptoss python -m examples.gptoss.gptoss_eval \\
        --adapter_dir ./models/speftr-gptoss-120b \\
        --device_map unsloth_balanced --batch_size 4 --max_new_tokens 128 \\
        --num_rows 40

Results and hardware: ``examples/gptoss/README.md``.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import torch
import unsloth  # noqa: F401  # patches transformers; must come first
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from datasets import Dataset
    from peft import PeftModel
    from transformers import BatchEncoding, PreTrainedTokenizerBase


type Message = dict[str, str]
# Unsloth's masking function: a batch of input_ids in, labels out.
type LabelMask = Callable[
    [Mapping[str, list[list[int]]]], dict[str, list[list[int]]]
]

DATASET_NAME = "HuggingFaceH4/Multilingual-Thinking"
DEFAULT_ADAPTER_DIR = "./models/speftr-gptoss-20b"
# The dataset has only a train split; these rows are held out for the
# eval loss during training and for this script.
EVAL_SIZE = 100
SPLIT_SEED = 0
MAX_SEQ_LENGTH = 2048
# Harmony markers: tokens after "<|start|>assistant" up to the next
# message are trained, i.e. both the analysis and the final channel.
INSTRUCTION_PART = "<|start|>user<|message|>"
RESPONSE_PART = "<|start|>assistant"
LANGUAGE_CODES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
}
# One harmony message in generated text: its channel and its content, up
# to the end-of-message token or the end of the (possibly cut) text.
CHANNEL_PATTERN = re.compile(
    r"<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|\Z)",
    re.DOTALL,
)


def load_splits(
    eval_size: int = EVAL_SIZE, seed: int = SPLIT_SEED
) -> tuple[Dataset, Dataset]:
    """Load Multilingual-Thinking and hold out ``eval_size`` random rows.

    Args:
        eval_size: Number of held-out rows.
        seed: Seed of the split; the defaults give the split used for
            training and evaluation.

    Returns:
        ``(train_dataset, eval_dataset)`` with the columns ``messages``
        (system, user, assistant with ``thinking``), ``reasoning_language``,
        ``developer``, ``user``, ``analysis`` and ``final``.
    """
    dataset = cast("Dataset", load_dataset(DATASET_NAME, split="train"))
    split = dataset.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]


def prompt_messages(messages: Sequence[Message]) -> list[Message]:
    """Drop the assistant turns of a conversation, keeping the prompt.

    Args:
        messages: A ``messages`` value of the dataset.

    Returns:
        The system and user messages, ready for a generation prompt.
    """
    return [message for message in messages if message["role"] != "assistant"]


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


def detect_language(text: str) -> str:
    """Identify the language of ``text`` with ``py3langid`` (offline).

    Args:
        text: Reasoning text, ideally a sentence or more.

    Returns:
        An ISO 639-1 code such as ``"fr"``, chosen among the 97 languages
        py3langid knows, not only the five of the dataset.
    """
    import py3langid  # noqa: PLC0415  # optional extra "gptoss"

    return cast("str", py3langid.classify(text)[0])


def is_compliant(requested_language: str, analysis: str | None) -> bool:
    """Tell whether an analysis channel is in the requested language.

    Args:
        requested_language: Dataset name of the language, e.g. ``German``.
        analysis: Analysis channel content, or ``None`` if there was none.

    Returns:
        ``True`` when there is a non-empty analysis channel detected as
        the requested language.
    """
    if not analysis:
        return False
    return detect_language(analysis) == LANGUAGE_CODES[requested_language]


def compliance_rates(
    requested: Sequence[str], compliant: Sequence[bool]
) -> dict[str, float]:
    """Share of compliant analyses per requested language and overall.

    Args:
        requested: Requested language of every row.
        compliant: Whether each row's analysis was in that language.

    Returns:
        Language name (sorted) to rate in [0, 1], then ``"overall"``.
    """
    totals = Counter(requested)
    hits = Counter(
        language
        for language, ok in zip(requested, compliant, strict=True)
        if ok
    )
    rates = {
        language: hits[language] / totals[language] for language in totals
    }
    rates = dict(sorted(rates.items()))
    rates["overall"] = sum(hits.values()) / len(requested)
    return rates


@contextmanager
def experts_in_training_mode(model: torch.nn.Module) -> Iterator[None]:
    """Run Unsloth's 4-bit gpt-oss experts on their routed tokens only.

    In eval mode they run every token through every expert and zero-weight
    the unselected ones: ``num_experts / top_k`` times the expert memory
    of training mode (8x on 20b, 32x on 120b), too much for long rows.
    Training mode gives the same output (the experts have no dropout) but
    loops over the experts in Python with a host sync per layer, which is
    slow for the one-token steps of generation; use it for long
    teacher-forced passes only. Unsloth names its 4-bit expert class
    ``GptOssExperts``. Training-mode experts return float32, so run the
    model under autocast.

    Args:
        model: A loaded gpt-oss model in eval mode, adapter attached or
            not. Do not call ``eval()`` or ``generate`` inside the block.

    Yields:
        None. On exit the experts are back in eval mode.
    """
    experts = [
        module
        for module in model.modules()
        if type(module).__name__ == "GptOssExperts"
    ]
    for module in experts:
        module.training = True
    try:
        yield
    finally:
        for module in experts:
            module.training = False


def load_adapter(
    adapter_dir: str, device_map: str | None
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the adapter on the 4-bit base recorded with it.

    ``model.disable_adapter()`` then gives the base model without a second
    copy of the weights. Unsloth's own gpt-oss attention is turned off for
    this process: in eval mode it gives the sliding-window layers the
    full-attention mask, which is wrong past 128 tokens; transformers'
    attention applies each layer's mask.

    Args:
        adapter_dir: Directory written by ``PESFT.save_model("lora")``.
        device_map: ``None`` for one GPU, or e.g. ``"unsloth_balanced"``
            to split a model that does not fit one GPU.

    Returns:
        The adapted model in inference mode and its tokenizer (harmony
        template), padding on the left for batched generation.
    """
    # Read when the model is loaded, so gptoss.py importing this module
    # still trains with Unsloth's attention.
    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
    options = {} if device_map is None else {"device_map": device_map}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        **options,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    return cast("PeftModel", model), tokenizer


def generate_completions(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    chats: Sequence[list[Message]],
    max_new_tokens: int,
) -> list[str]:
    """Greedily complete a batch of chats, keeping special tokens.

    Args:
        model: Model from ``load_adapter`` (adapter enabled or not).
        tokenizer: Its tokenizer.
        chats: Conversations ending with a user turn.
        max_new_tokens: Generation budget per completion.

    Returns:
        Decoded completions with harmony tokens, for ``parse_channels``.
    """
    encoded = tokenizer.apply_chat_template(
        list(chats),
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


def score_compliance(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    rows: Dataset,
    args: argparse.Namespace,
) -> list[bool]:
    """Generate the start of every row's reasoning and check its language.

    Args:
        model: Model from ``load_adapter`` (adapter enabled or not).
        tokenizer: Its tokenizer.
        rows: Held-out rows with ``messages`` and ``reasoning_language``.
        args: Parsed CLI arguments (``batch_size``, ``max_new_tokens``).

    Returns:
        Per row, whether the analysis channel was in the requested
        language.
    """
    compliant: list[bool] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        chats = [prompt_messages(messages) for messages in batch["messages"]]
        completions = generate_completions(
            model, tokenizer, chats, args.max_new_tokens
        )
        for language, completion in zip(
            batch["reasoning_language"], completions, strict=True
        ):
            analysis = parse_channels(completion).get("analysis")
            compliant.append(is_compliant(language, analysis))
        print(f"Generated {len(compliant)}/{len(rows)}")
    return compliant


def build_label_mask(
    tokenizer: PreTrainedTokenizerBase,
) -> LabelMask:
    """Return the training mask: ``-100`` outside the assistant turn.

    Args:
        tokenizer: Tokenizer with the harmony template.

    Returns:
        Unsloth's ``train_on_responses_only`` function for the example's
        markers: batch of ``input_ids`` in, ``labels`` out.
    """
    return cast(
        "LabelMask",
        train_on_responses_only(
            None,
            instruction_part=INSTRUCTION_PART,
            response_part=RESPONSE_PART,
            tokenizer=tokenizer,
            return_function=True,
        ),
    )


def eval_loss(
    model: PeftModel, tokenizer: PreTrainedTokenizerBase, rows: Dataset
) -> float:
    """Mean assistant-turn loss over the rows, one row at a time.

    Rows are rendered and truncated to ``MAX_SEQ_LENGTH`` tokens and run
    under autocast in the model dtype as in training, so the value is
    comparable to the trainer's ``eval_loss``.

    Args:
        model: Model from ``load_adapter`` (adapter enabled or not).
        tokenizer: Its tokenizer.
        rows: Held-out rows with ``messages``.

    Returns:
        The average over rows of each row's mean token loss.
    """
    label_mask = build_label_mask(tokenizer)
    losses: list[float] = []
    for messages in rows["messages"]:
        text = cast(
            "str", tokenizer.apply_chat_template(messages, tokenize=False)
        )
        input_ids = tokenizer(text, add_special_tokens=False).input_ids
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        labels = label_mask({"input_ids": [input_ids]})["labels"][0]
        # Training-mode experts return float32; autocast runs the layers
        # after them in the model dtype, as in training.
        with (
            torch.inference_mode(),
            torch.autocast(model.device.type, dtype=model.dtype),
            experts_in_training_mode(model),
        ):
            output = model(
                input_ids=torch.tensor([input_ids], device=model.device),
                labels=torch.tensor([labels], device=model.device),
            )
        losses.append(float(output.loss))
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the adapter directory, device map and generation
        settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Eval loss and reasoning-language compliance of the GPT-OSS "
            "adapter vs its base model on the held-out rows"
        )
    )
    parser.add_argument(
        "--adapter_dir",
        default=DEFAULT_ADAPTER_DIR,
        help=f"Adapter saved by gptoss.py (default: {DEFAULT_ADAPTER_DIR})",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="e.g. 'unsloth_balanced' to split the model over all visible "
        "GPUs (default: one GPU)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Prompts generated per batch; decoding is bound by per-step "
        "overhead, so larger batches are faster until memory runs out "
        "(default: 32, ~21.6 GiB peak with gpt-oss-20b)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=320,
        help="Tokens generated per prompt; the start of the reasoning is "
        "enough to identify its language (default: 320)",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=None,
        help="Evaluate only the first N held-out rows (a random sample, "
        "the split is shuffled); generation on gpt-oss-120b is slow "
        "(default: all 100)",
    )
    return parser.parse_args()


def main() -> None:
    """Score the base model and the adapter and print a comparison.

    Returns:
        None. Eval losses, compliance per language and peak GPU memory are
        printed to stdout.
    """
    args = parse_args()
    _, eval_rows = load_splits()
    if args.num_rows is not None:
        eval_rows = eval_rows.select(range(args.num_rows))
    model, tokenizer = load_adapter(args.adapter_dir, args.device_map)

    with model.disable_adapter():
        base_loss = eval_loss(model, tokenizer, eval_rows)
        base_compliant = score_compliance(model, tokenizer, eval_rows, args)
    # Printed before the adapter pass so a run cut short keeps the base
    # numbers.
    requested = eval_rows["reasoning_language"]
    print(f"\nbase eval_loss {base_loss:.4f}")
    print(f"base compliance {compliance_rates(requested, base_compliant)}")
    adapter_loss = eval_loss(model, tokenizer, eval_rows)
    adapter_compliant = score_compliance(model, tokenizer, eval_rows, args)

    base_rates = compliance_rates(requested, base_compliant)
    adapter_rates = compliance_rates(requested, adapter_compliant)
    counts = Counter(requested)
    counts["overall"] = len(requested)

    print(f"\nAdapter: {args.adapter_dir} ({len(eval_rows)} held-out rows)")
    print(f"eval_loss  base {base_loss:.4f}  adapter {adapter_loss:.4f}")
    print("\nReasoning-language compliance (analysis channel)")
    print(f"{'language':<10} {'rows':>4} {'base':>6} {'adapter':>8}")
    for language, base_rate in base_rates.items():
        print(
            f"{language:<10} {counts[language]:>4} {base_rate:>6.0%} "
            f"{adapter_rates[language]:>8.0%}"
        )
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\npeak_gpu_memory_gb: {peak_gb:.2f}")


if __name__ == "__main__":
    main()
