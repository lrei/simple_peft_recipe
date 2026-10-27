r"""Evaluate the GPT-OSS reasoning-language adapter against its base model.

Every row of ``HuggingFaceH4/Multilingual-Thinking`` asks for a
"reasoning language" (English, French, German, Spanish or Italian) in its
system turn. gpt-oss reasons in English whatever it is told; the adapter
should make the analysis channel follow the requested language. On the
held-out rows (``load_splits``) this script reports, for the base model
and for the adapter on the same 4-bit base:

- eval loss: mean token loss on the assistant turn (analysis and
  final channels), masked exactly as in training;
- reasoning-language compliance: the share of generated analysis
  channels whose language (``py3langid``) is the requested one, per
  language and overall. Only the start of the reasoning is generated
  (``--max_new_tokens``).

This module also defines the task for ``gptoss.py``: the dataset and its
fixed train/eval split, the response-only markers and the channel parsing.

Usage (``py3langid`` comes with the ``gptoss`` extra):
    uv run --extra gptoss python -m examples.gptoss.gptoss_eval
    uv run --extra gptoss python -m examples.gptoss.gptoss_eval \\
        --adapter_dir ./models/speftr-gptoss-120b \\
        --device_map unsloth_balanced --token_budget 1312 \\
        --max_new_tokens 128 --num_rows 40

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

# Label of tokens the loss skips (padding, prompt), as in transformers.
IGNORE_INDEX = -100
DATASET_NAME = "HuggingFaceH4/Multilingual-Thinking"
DEFAULT_ADAPTER_DIR = "./models/speftr-gptoss-20b"
# The dataset has only a train split; these rows are held out for the
# eval loss during training and for this script.
EVAL_SIZE = 100
SPLIT_SEED = 0
# Training length (gptoss.py); longer rows are cut in training.
MAX_SEQ_LENGTH = 2048
# This script's limit: the longest held-out row has about 2,500 tokens.
EVAL_MAX_SEQ_LENGTH = 4096
# Padded tokens (prompt plus --max_new_tokens per row) per generation
# batch.
TOKEN_BUDGET = 16_384
# Padded tokens per eval-loss batch. The forward pass keeps the logits of
# every token over the 201,088-token vocabulary; 6144 is the trainer's
# eval batch in gptoss.py (3 rows of up to 2048 tokens).
LOSS_TOKEN_BUDGET = 6144
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


def truncate_prompts(
    prompts: list[list[int]], max_seq_length: int, max_new_tokens: int
) -> tuple[list[list[int]], int]:
    """Cut prompts so that each fits ``max_seq_length`` with its reply.

    Chat templates put the question and the assistant header at the end
    of a prompt, so the start is what goes. A warning is printed when a
    prompt is cut.

    Args:
        prompts: Token ids of each prompt, chat template applied.
        max_seq_length: The limit passed to ``from_pretrained``.
        max_new_tokens: Tokens each reply may add.

    Returns:
        The prompts, each at most ``max_seq_length - max_new_tokens``
        tokens long (the last ones kept), and how many were cut.

    Raises:
        ValueError: If ``max_new_tokens`` leaves no room for a prompt.
    """
    limit = max_seq_length - max_new_tokens
    if limit < 1:
        message = (
            f"--max_new_tokens {max_new_tokens} leaves no room for a "
            f"prompt in --max_seq_length {max_seq_length}; lower "
            "--max_new_tokens or raise --max_seq_length."
        )
        raise ValueError(message)
    too_long = [len(prompt) for prompt in prompts if len(prompt) > limit]
    if too_long:
        print(
            f"\nWARNING: {len(too_long)} of {len(prompts)} prompts are "
            f"longer than {limit} tokens (context {max_seq_length} minus "
            f"max_new_tokens {max_new_tokens}; longest: {max(too_long)}). "
            "Their beginning is cut off, so the model never sees it.\n"
            "  Fix: raise --max_seq_length (more memory), lower "
            "--max_new_tokens, or shorten those inputs."
        )
    return [prompt[-limit:] for prompt in prompts], len(too_long)


def truncate_rows(
    sequences: list[list[int]], max_seq_length: int
) -> tuple[list[list[int]], int]:
    """Cut whole conversations to ``max_seq_length``, keeping their start.

    Training also keeps the start of a long row: the loss covers a whole
    conversation, and its later tokens depend on the earlier ones. A
    warning is printed when a row is cut.

    Args:
        sequences: Token ids of each conversation.
        max_seq_length: The limit passed to ``from_pretrained``.

    Returns:
        The sequences, each at most ``max_seq_length`` tokens long (the
        first ones kept), and how many were cut.
    """
    too_long = [len(ids) for ids in sequences if len(ids) > max_seq_length]
    if too_long:
        print(
            f"\nWARNING: {len(too_long)} of {len(sequences)} rows are "
            f"longer than --max_seq_length={max_seq_length} tokens "
            f"(longest: {max(too_long)}). Their end is cut off, so it is "
            "not scored.\n"
            "  Fix: raise --max_seq_length (more memory), or shorten those "
            "rows."
        )
    return [ids[:max_seq_length] for ids in sequences], len(too_long)


def load_adapter(
    adapter_dir: str, device_map: str | None, max_seq_length: int
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
        max_seq_length: Longest sequence the model must handle.

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
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        **options,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    return cast("PeftModel", model), tokenizer


def encode_prompts(
    tokenizer: PreTrainedTokenizerBase, rows: Dataset
) -> list[list[int]]:
    """Render and tokenize the rows' prompts, without padding.

    Args:
        tokenizer: Tokenizer with the harmony template.
        rows: Rows with ``messages``.

    Returns:
        The token ids of each row's system and user turns, ending with
        the assistant header.
    """
    chats = [prompt_messages(messages) for messages in rows["messages"]]
    encoded = tokenizer.apply_chat_template(
        chats, add_generation_prompt=True, return_dict=True
    )
    input_ids = cast("BatchEncoding", encoded)["input_ids"]
    return cast("list[list[int]]", input_ids)


def generate_completions(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[list[int]],
    max_new_tokens: int,
) -> list[str]:
    """Greedily complete a batch of prompts, keeping special tokens.

    Args:
        model: Model from ``load_adapter`` (adapter enabled or not).
        tokenizer: Its tokenizer.
        prompts: Token ids from ``encode_prompts``.
        max_new_tokens: Generation budget per completion.

    Returns:
        Decoded completions with harmony tokens, for ``parse_channels``.
    """
    # Left padding with the attention mask, so every prompt ends where
    # generation starts and the padding is not attended to.
    padded = tokenizer.pad(
        {"input_ids": list(prompts)},
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    inputs = padded.to(model.device)
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
) -> tuple[list[bool], int]:
    """Generate the start of every row's reasoning and check its language.

    Rows are generated in batches of similar length that fit
    ``--token_budget`` (prompt plus ``--max_new_tokens`` per row). A
    prompt longer than ``--max_seq_length`` minus ``--max_new_tokens``
    loses its beginning (``truncate_prompts``).

    Args:
        model: Model from ``load_adapter`` (adapter enabled or not).
        tokenizer: Its tokenizer.
        rows: Held-out rows with ``messages`` and ``reasoning_language``.
        args: Parsed CLI arguments (``token_budget``, ``max_new_tokens``,
            ``max_seq_length``).

    Returns:
        Per row, in row order, whether the analysis channel was in the
        requested language, and the number of prompts that were cut.
    """
    prompts, truncated = truncate_prompts(
        encode_prompts(tokenizer, rows),
        args.max_seq_length,
        args.max_new_tokens,
    )
    lengths = [len(prompt) + args.max_new_tokens for prompt in prompts]
    languages: list[str] = rows["reasoning_language"]
    compliant = [False] * len(prompts)
    done = 0
    for batch in length_batches(lengths, args.token_budget):
        completions = generate_completions(
            model,
            tokenizer,
            [prompts[index] for index in batch],
            args.max_new_tokens,
        )
        for index, completion in zip(batch, completions, strict=True):
            analysis = parse_channels(completion).get("analysis")
            compliant[index] = is_compliant(languages[index], analysis)
        done += len(batch)
        print(f"Generated {done}/{len(rows)}")
    return compliant, truncated


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


def encode_rows(
    tokenizer: PreTrainedTokenizerBase, rows: Dataset
) -> list[list[int]]:
    """Render and tokenize whole conversations, as in training.

    Args:
        tokenizer: Tokenizer with the harmony template.
        rows: Rows with ``messages``.

    Returns:
        The token ids of each conversation, assistant turn included.
    """
    sequences = []
    for messages in rows["messages"]:
        text = cast(
            "str", tokenizer.apply_chat_template(messages, tokenize=False)
        )
        sequences.append(tokenizer(text, add_special_tokens=False).input_ids)
    return sequences


def batch_loss(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    sequences: Sequence[list[int]],
    labels: Sequence[list[int]],
) -> tuple[float, int]:
    """Mean token loss of one batch and the number of tokens it covers.

    Runs under autocast in the model dtype with the experts in training
    mode, as in training.

    Args:
        model: Model from ``load_adapter`` (adapter enabled or not).
        tokenizer: Its tokenizer; pads the batch on the right.
        sequences: Token ids of each row.
        labels: Each row's labels, ``IGNORE_INDEX`` where no loss is taken.

    Returns:
        The mean loss over the predicted label tokens and their count.
    """
    padded = tokenizer.pad(
        {"input_ids": list(sequences)},
        padding=True,
        padding_side="right",
        return_tensors="pt",
    )
    inputs = padded.to(model.device)
    width = inputs["input_ids"].shape[1]
    padded_labels = [
        row + [IGNORE_INDEX] * (width - len(row)) for row in labels
    ]
    # The loss predicts token t + 1 from token t: the first label of a row
    # is never a target.
    targets = sum(label != IGNORE_INDEX for row in labels for label in row[1:])
    # Training-mode experts return float32; autocast runs the layers after
    # them in the model dtype, as in training.
    with (
        torch.inference_mode(),
        torch.autocast(model.device.type, dtype=model.dtype),
        experts_in_training_mode(model),
    ):
        output = model(
            **inputs,
            labels=torch.tensor(padded_labels, device=model.device),
        )
    return float(output.loss), targets


def eval_loss(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    rows: Dataset,
    args: argparse.Namespace,
) -> tuple[float, int]:
    """Mean assistant-turn token loss over the rows.

    Rows are rendered in full, cut to ``--max_seq_length`` tokens
    (``truncate_rows``), masked as in training and run in batches of
    similar length that fit ``--loss_token_budget``. The mean is taken
    over all assistant tokens of all rows, so it does not depend on the
    batching.

    Args:
        model: Model from ``load_adapter`` (adapter enabled or not).
        tokenizer: Its tokenizer.
        rows: Held-out rows with ``messages``.
        args: Parsed CLI arguments (``loss_token_budget``,
            ``max_seq_length``).

    Returns:
        The token-weighted mean loss and the number of rows that were
        cut.
    """
    label_mask = build_label_mask(tokenizer)
    sequences, truncated = truncate_rows(
        encode_rows(tokenizer, rows), args.max_seq_length
    )
    lengths = [len(sequence) for sequence in sequences]
    loss_sum = 0.0
    token_count = 0
    for batch in length_batches(lengths, args.loss_token_budget):
        batch_ids = [sequences[index] for index in batch]
        labels = label_mask({"input_ids": batch_ids})["labels"]
        loss, targets = batch_loss(model, tokenizer, batch_ids, labels)
        loss_sum += loss * targets
        token_count += targets
    return loss_sum / token_count, truncated


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
        "--token_budget",
        type=int,
        default=TOKEN_BUDGET,
        help="Padded tokens (prompt plus --max_new_tokens per row) per "
        "generation batch; a longer row runs alone "
        f"(default: {TOKEN_BUDGET})",
    )
    parser.add_argument(
        "--loss_token_budget",
        type=int,
        default=LOSS_TOKEN_BUDGET,
        help="Padded tokens per eval-loss batch; a longer row runs alone "
        f"(default: {LOSS_TOKEN_BUDGET})",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=EVAL_MAX_SEQ_LENGTH,
        help="Longest row (eval loss) or prompt plus generated tokens the "
        "model is loaded for; a longer row loses its end and a longer "
        "prompt its beginning, with a warning "
        f"(default: {EVAL_MAX_SEQ_LENGTH})",
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
        None. Eval losses, compliance per language, the number of cut rows
        and prompts and peak GPU memory are printed to stdout.
    """
    args = parse_args()
    _, eval_rows = load_splits()
    if args.num_rows is not None:
        eval_rows = eval_rows.select(range(args.num_rows))
    model, tokenizer = load_adapter(
        args.adapter_dir, args.device_map, args.max_seq_length
    )

    with model.disable_adapter():
        base_loss, cut_rows = eval_loss(model, tokenizer, eval_rows, args)
        base_compliant, cut_prompts = score_compliance(
            model, tokenizer, eval_rows, args
        )
    # Printed before the adapter pass so a run cut short keeps the base
    # numbers.
    requested = eval_rows["reasoning_language"]
    print(f"\nbase eval_loss {base_loss:.4f}")
    print(f"base compliance {compliance_rates(requested, base_compliant)}")
    adapter_loss, _ = eval_loss(model, tokenizer, eval_rows, args)
    adapter_compliant, _ = score_compliance(model, tokenizer, eval_rows, args)

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
    # Scores of cut inputs are not comparable with uncut ones.
    print(f"\ntruncated_rows (eval loss): {cut_rows}")
    print(f"truncated_prompts (compliance): {cut_prompts}")
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak_gpu_memory_gb: {peak_gb:.2f}")


if __name__ == "__main__":
    main()
