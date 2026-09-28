# SPDX-FileCopyrightText: 2026 Luis Rei
# SPDX-License-Identifier: BSD-2-Clause
r"""Check whether a LoRA rank has enough capacity for a dataset.

Question answered: *is my LoRA adapter large enough to absorb this
dataset, and what is the smallest rank that is?* Two numbers are
computed and compared:

1. **Adapter size**: the exact count of trainable LoRA parameters for a
   model, rank and set of target modules. The model is built from its
   config on the ``meta`` device (no weights are downloaded or allocated)
   and wrapped with ``peft.get_peft_model``. On multimodal checkpoints
   (Gemma 4, Qwen 3.5) only the language-model decoder is counted, as in
   ``speftr.perl``. LoRA adds ``r * (in + out)`` parameters per targeted
   linear layer, so the count is linear in the rank.

   Mixture-of-Experts models (gpt-oss, Qwen3 MoE, Gemma 4 26B-A4B, ...)
   keep their experts as 3-D ``(num_experts, in, out)`` parameters of an
   ``experts`` module rather than as linear layers. PESFT (Unsloth)
   adapts them when the MLP projections are targeted: the fused
   ``gate_up_proj`` for ``gate_proj``/``up_proj`` and ``down_proj`` for
   ``down_proj``, each adding ``r * num_experts * (in + out)``. These
   expert parameters are counted in sft mode and not in rl mode, where
   PERL adapts linear layers only; ``include_experts`` overrides that.
2. **Required parameters**: an estimate of how many parameters the
   dataset needs, from the capacity argument in "LoRA Without Regret"
   (Schulman and Thinking Machines Lab, 2025,
   https://thinkingmachines.ai/blog/lora/).

Formula
-------
The post gives no explicit formula; this module derives one from three of
its claims:

- "neural networks can store 2 bits per parameter" in the long-training
  limit, citing Allen-Zhu and Li (2024), "Physics of Language Models:
  Part 3.3, Knowledge Capacity Scaling Laws" (``BITS_PER_PARAMETER``);
- "LLM datasets usually have a loss of around 1 bit (0.69 nats) per
  token", an upper bound on the bits needed to memorize SFT data
  (``SFT_BITS_PER_TOKEN``, overridable with ``--bits_per_token``);
- policy-gradient RL gets "only O(1) bits per episode"; its MATH example
  assumes one bit per completion (~10,000 problems x 32 samples = 320,000
  bits) (``RL_BITS_PER_EPISODE``).

So::

    sft: bits = trained_tokens * bits_per_token
    rl:  bits = episodes * 1,  episodes = prompts * num_generations * epochs
                               (or max_steps * completions_per_step)
    required_parameters = bits / 2
    minimum_rank = ceil(required_parameters / parameters_per_rank)

SFT epochs are ignored: repeating the same tokens adds no information. RL
episodes are fresh samples, so they count every epoch.

Dataset formats (SFT)
---------------------
Without a column option the format is detected from the column names, in
this order (TRL's standard dataset formats):

- conversational, a ``messages`` or ``conversations`` column::

      {"messages": [{"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"}]}

  ShareGPT messages (``{"from": "human" | "gpt" | "system",
  "value": ...}``) are mapped to ``role``/``content``. A ``content`` given
  as a list of parts (``[{"type": "text", "text": ...}, {"type":
  "image"}]``) counts only its text parts, joined by newlines.
- prompt-completion, ``prompt`` and ``completion`` columns, either both
  strings (standard) or both message lists (conversational)::

      {"prompt": "The sky is", "completion": " blue."}
      {"prompt": [{"role": "user", "content": "Sky colour?"}],
       "completion": [{"role": "assistant", "content": "Blue."}]}

- language modeling, a ``text`` column: ``{"text": "The sky is blue."}``.

Other layouts are named explicitly:

- ``--messages_column``: a message list under another name;
- ``--prompt_column`` (one or more, joined by a blank line in the order
  given) with ``--response_column`` and optionally ``--system_column``:
  rendered as a system/user/assistant chat, e.g. Dolly's
  ``instruction`` + ``context`` -> ``response``;
- ``--text_column``: plain text under another name;
- ``--formatting_func module.path:function``: any layout, rendered by your
  own function (below).

Conversations are rendered with the tokenizer's chat template; a base
model without one cannot render them, so use ``--formatting_func`` or
``--text_column`` there.

Trained tokens with ``--responses_only``
----------------------------------------
- conversational: every assistant turn counts the tokens it adds after the
  rendered history plus assistant header (TRL ``assistant_only_loss``);
- prompt-completion: the completion tokens, i.e. tokens of prompt +
  completion (+ EOS for strings) minus tokens of the prompt (with the
  generation header for conversations), as TRL's ``completion_only_loss``;
- prompt/response columns: the assistant turn, as for conversational data;
- ``--formatting_func``: the text after each ``--response_part`` up to the
  next ``--instruction_part``, as unsloth's ``train_on_responses_only``
  (``PESFTConfig.train_on_responses``). Both markers are required and must
  appear in every formatted text;
- language modeling: not supported (no response to separate).

Formatting function contract
----------------------------
The same batched function you pass to ``PESFT.train``: it receives a batch
of rows as a mapping from column name to a list of values and returns one
training text per row::

    def format_batch(batch):
        return [
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>\n"
            for q, a in zip(batch["question"], batch["answer"])
        ]

On the CLI it must be importable as ``module.path:function`` from the
working directory.

Large datasets
--------------
``--sample_size N`` tokenizes N random rows (seed 0) and scales the count
to the whole set; the report marks the result as extrapolated.
``--max_samples N`` instead limits the training set to its first N rows.
RL mode only counts rows (any columns); with ``--max_steps`` no dataset is
needed at all.

Limitations
-----------
The result is an order-of-magnitude estimate, not a guarantee: the bits
per token of your data may be far below 1 (use your model's measured loss
in bits with ``--bits_per_token``), and the post finds that LoRA past its
capacity trains less efficiently rather than hitting a hard floor. The
README recipe deliberately keeps more headroom (1 parameter per SFT token,
minimum rank 8). Only the model config and tokenizer are downloaded; the
dataset is loaded in full. Unsloth is not imported.

Examples:
    Command line::

        uv run python -m speftr.lora_budget \
            --model_name_or_path Qwen/Qwen3-0.6B --dataset trl-lib/Capybara
        uv run python -m speftr.lora_budget \
            --model_name_or_path Qwen/Qwen3-0.6B --dataset trl-lib/tldr \
            --responses_only --sample_size 500
        uv run python -m speftr.lora_budget \
            --model_name_or_path Qwen/Qwen3-0.6B --dataset data/dolly.csv \
            --prompt_column instruction context --response_column response
        uv run python -m speftr.lora_budget \
            --model_name_or_path Qwen/Qwen3-0.6B --dataset data/train.jsonl \
            --formatting_func my_project.data:format_batch --responses_only \
            --instruction_part $'<|im_start|>user\n' \
            --response_part $'<|im_start|>assistant\n'
        uv run python -m speftr.lora_budget --mode rl \
            --model_name_or_path Qwen/Qwen3-0.6B --max_steps 500

    Python::

        from speftr.lora_budget import LoraBudgetConfig, estimate_lora_budget

        budget = estimate_lora_budget(
            LoraBudgetConfig(
                model_name_or_path="Qwen/Qwen3-0.6B",
                dataset="trl-lib/Capybara",
                responses_only=True,
            )
        )
        print(budget.minimum_rank, budget.sufficient)

    An already loaded ``datasets.Dataset`` can be passed as the second
    argument of ``estimate_lora_budget`` instead of ``dataset=``.
"""

from __future__ import annotations

import argparse
import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import torch

from speftr.perl import PERLConfig, _language_model_targets


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


BITS_PER_PARAMETER: Final[float] = 2.0
SFT_BITS_PER_TOKEN: Final[float] = 1.0
RL_BITS_PER_EPISODE: Final[float] = 1.0
SFT_DEFAULT_RANK: Final[int] = 8
LOCAL_BUILDERS: Final[dict[str, str]] = {
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".arrow": "arrow",
    ".txt": "text",
}
SHAREGPT_ROLES: Final[dict[str, str]] = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}
FORMAT_BATCH_SIZE: Final[int] = 1000
EXPERT_WEIGHT_NDIM: Final[int] = 3

CONVERSATIONAL: Final[str] = "conversational"
PROMPT_COMPLETION: Final[str] = "prompt-completion"
PROMPT_RESPONSE: Final[str] = "prompt-response chat"
LANGUAGE_MODELING: Final[str] = "language modeling"
FORMATTING_FUNC: Final[str] = "formatting_func"

type Message = dict[str, str]
type Conversation = list[Message]
type Sample = (
    str | Conversation | tuple[str, str] | tuple[Conversation, Conversation]
)
type FormattingFunc = Callable[[Mapping[str, list[object]]], list[str]]

EPILOG: Final[str] = r"""examples:
  %(prog)s --model_name_or_path Qwen/Qwen3-0.6B --dataset trl-lib/Capybara
  %(prog)s --model_name_or_path Qwen/Qwen3-0.6B --dataset trl-lib/tldr \
      --responses_only --sample_size 500
  %(prog)s --model_name_or_path Qwen/Qwen3-0.6B --dataset dolly.csv \
      --prompt_column instruction context --response_column response
  %(prog)s --model_name_or_path Qwen/Qwen3-0.6B --dataset train.jsonl \
      --formatting_func my_project.data:format_batch --responses_only \
      --instruction_part $'<|im_start|>user\n' \
      --response_part $'<|im_start|>assistant\n'
  %(prog)s --mode rl --model_name_or_path Qwen/Qwen3-0.6B --max_steps 500

Without a column option the dataset format is detected: a messages or
conversations column (conversational), prompt + completion columns
(prompt-completion) or a text column (language modeling). Local files:
.json .jsonl .parquet .csv .tsv .arrow .txt or a save_to_disk directory.
"""


class LoraBudgetError(ValueError):
    """The options or the dataset do not allow an estimate."""


@dataclass
class LoraBudgetConfig:
    """Inputs of a LoRA capacity estimate; mirrors the CLI flags.

    Attributes:
        model_name_or_path: Model id or local path (config and tokenizer).
        dataset: HF dataset id, local file or ``save_to_disk`` directory.
            Optional only in rl mode with ``max_steps``.
        dataset_config: HF dataset configuration name.
        split: Split to load (also selects from a saved ``DatasetDict``).
        data_files: Files or globs forwarded to ``datasets.load_dataset``.
        mode: ``"sft"`` counts tokens, ``"rl"`` counts episodes.
        lora_r: LoRA rank; ``None`` uses 8 for sft and PERL's for rl.
        target_modules: Linear layer names to adapt; the MLP names also
            select the matching MoE expert parameters.
        include_experts: Count LoRA on fused MoE expert parameters;
            ``None`` counts them for sft (PESFT adapts them) and not for
            rl (PERL does not).
        text_column: Plain-text column; every token is trained.
        messages_column: Message-list column.
        prompt_column: Columns joined into the user turn.
        response_column: Column with the assistant turn.
        system_column: Optional column with a system message.
        formatting_func: Batched formatting function, or its
            ``module.path:function`` import path.
        responses_only: Count only response tokens.
        instruction_part: Marker starting a user turn in formatted text.
        response_part: Marker starting a response in formatted text.
        max_samples: Use only the first N rows as the training set.
        sample_size: Tokenize N random rows and extrapolate.
        bits_per_token: Information per trained token (sft).
        num_generations: Completions per prompt (rl).
        num_train_epochs: Epochs (rl).
        max_steps: Optimizer steps; overrides epochs (rl).
        completions_per_step: Completions per optimizer step (rl).
    """

    model_name_or_path: str = field(
        metadata={
            "help": (
                "Model id or local path; only config and tokenizer are read."
            )
        }
    )
    dataset: str | None = field(
        default=None,
        metadata={
            "help": (
                "HF dataset id, local .json/.jsonl/.parquet/.csv/.tsv/"
                ".arrow/.txt file or save_to_disk directory. Optional in rl "
                "mode with --max_steps."
            )
        },
    )
    dataset_config: str | None = field(
        default=None,
        metadata={
            "help": "HF dataset configuration name (e.g. 'main' for gsm8k)."
        },
    )
    split: str = field(
        default="train",
        metadata={
            "help": (
                "Split of a hub dataset or saved DatasetDict (default: train)."
            )
        },
    )
    data_files: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Files or globs passed to datasets.load_dataset, e.g. "
                "--dataset json --data_files 'data/*.jsonl'."
            )
        },
    )
    mode: Literal["sft", "rl"] = field(
        default="sft",
        metadata={
            "help": (
                "sft counts trained tokens, rl counts episodes (default: sft)."
            )
        },
    )
    lora_r: int | None = field(
        default=None,
        metadata={
            "help": (
                f"LoRA rank (default: {SFT_DEFAULT_RANK} for sft, "
                f"{PERLConfig.lora_r} for rl, as PESFT/PERL)."
            )
        },
    )
    target_modules: list[str] = field(
        default_factory=lambda: PERLConfig().target_modules,
        metadata={
            "help": (
                "Linear layer names to adapt (default: all attention and MLP)."
            )
        },
    )
    include_experts: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Count LoRA on fused MoE expert weights (gate_up_proj, "
                "down_proj of an experts module) selected by the MLP "
                "targets (default: true for sft, as PESFT/Unsloth adapts "
                "them; false for rl, as PERL does not)."
            )
        },
    )
    text_column: str | None = field(
        default=None,
        metadata={"help": "Plain-text column; every token is trained."},
    )
    messages_column: str | None = field(
        default=None,
        metadata={
            "help": "Column with a list of chat messages (role/content)."
        },
    )
    prompt_column: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Column(s) forming the user turn, joined by a blank line in "
                "the given order; needs --response_column."
            )
        },
    )
    response_column: str | None = field(
        default=None,
        metadata={
            "help": (
                "Column with the assistant turn, paired with --prompt_column."
            )
        },
    )
    system_column: str | None = field(
        default=None,
        metadata={
            "help": "Column with a system message, used with --prompt_column."
        },
    )
    # HfArgumentParser types a flag by the second union member, so the CLI
    # takes the import path while Python callers may pass the function.
    formatting_func: FormattingFunc | str | None = field(
        default=None,
        metadata={
            "help": (
                "module.path:function of a batched formatting function as "
                "passed to PESFT.train (batch of columns in, list of texts "
                "out); its texts are counted."
            )
        },
    )
    responses_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Count only response tokens: assistant turns, completions, "
                "or the text after --response_part with --formatting_func."
            )
        },
    )
    instruction_part: str | None = field(
        default=None,
        metadata={
            "help": (
                "Marker starting a user turn in formatting_func texts, "
                "e.g. $'<|im_start|>user\\n' in bash."
            )
        },
    )
    response_part: str | None = field(
        default=None,
        metadata={
            "help": (
                "Marker starting a response in formatting_func texts, "
                "e.g. $'<|im_start|>assistant\\n' in bash."
            )
        },
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "Use only the first N rows as the training set."},
    )
    sample_size: int | None = field(
        default=None,
        metadata={
            "help": "sft: tokenize N random rows and extrapolate to all rows."
        },
    )
    bits_per_token: float = field(
        default=SFT_BITS_PER_TOKEN,
        metadata={
            "help": (
                "Information per trained token; use your model's loss in "
                f"bits (default: {SFT_BITS_PER_TOKEN})."
            )
        },
    )
    num_generations: int = field(
        default=PERLConfig.num_generations,
        metadata={
            "help": (
                "Completions per prompt (default: "
                f"{PERLConfig.num_generations})."
            )
        },
    )
    num_train_epochs: int = field(
        default=PERLConfig.num_train_epochs,
        metadata={"help": f"Epochs (default: {PERLConfig.num_train_epochs})."},
    )
    max_steps: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optimizer steps; overrides epochs and makes --dataset "
                "optional."
            )
        },
    )
    completions_per_step: int = field(
        default=PERLConfig.per_device_train_batch_size,
        metadata={
            "help": (
                "Completions per optimizer step, used with --max_steps "
                f"(default: {PERLConfig.per_device_train_batch_size})."
            )
        },
    )

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> LoraBudgetConfig:
        """Create a config from command line arguments.

        Args:
            argv: Arguments to parse; ``None`` parses ``sys.argv``.

        Returns:
            The config holding every parsed flag.
        """
        args = cls.get_argument_parser().parse_args(argv)
        return cls(**vars(args))

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """Build the command line parser from the dataclass fields.

        Each field becomes a ``--<name>`` flag (also ``--<name-with-dashes>``)
        with the field's default and ``help`` metadata.

        Returns:
            Parser with one flag per ``LoraBudgetConfig`` field.
        """
        from transformers.hf_argparser import (  # noqa: PLC0415
            DataClassType,
            HfArgumentParser,
        )

        parser: argparse.ArgumentParser = HfArgumentParser(
            DataClassType(LoraBudgetConfig),
            prog="python -m speftr.lora_budget",
            description=(
                "Estimate whether a LoRA rank has the capacity for a "
                "dataset. required_parameters = bits / 2 (2 bits per "
                "parameter); sft bits = trained tokens x bits_per_token, "
                "rl bits = episodes x 1. Constants from 'LoRA Without "
                "Regret' (Schulman et al., 2025); the result is an "
                "order-of-magnitude estimate."
            ),
            epilog=EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        return parser


@dataclass(frozen=True)
class LoraBudget:
    """Result of a LoRA capacity estimate.

    Attributes:
        rank: LoRA rank the adapter was counted at.
        adapter_parameters: Trainable LoRA parameters at ``rank``.
        expert_parameters: Part of ``adapter_parameters`` on fused MoE
            expert weights; 0 when not counted or the model has none.
        parameters_per_rank: Adapter parameters per unit of rank.
        rows: Training rows, ``None`` for rl without a dataset.
        data_format: Detected or chosen dataset format (sft only).
        trained_tokens: Tokens SFT trains on (sft only).
        episodes: Sampled completions over the run (rl only).
        extrapolated: Whether ``trained_tokens`` was scaled up from a
            random sample.
        bits: Information the training run must absorb.
        required_parameters: ``bits / 2``.
        minimum_rank: Smallest rank holding ``required_parameters``.
    """

    rank: int
    adapter_parameters: int
    expert_parameters: int
    parameters_per_rank: int
    rows: int | None
    data_format: str | None
    trained_tokens: int | None
    episodes: int | None
    extrapolated: bool
    bits: float
    required_parameters: float
    minimum_rank: int

    @property
    def sufficient(self) -> bool:
        """Whether the adapter holds the required parameters.

        Returns:
            ``adapter_parameters >= required_parameters``.
        """
        return self.adapter_parameters >= self.required_parameters


@dataclass(frozen=True)
class DataFormat:
    """Where a row's training data lives and how it is rendered.

    Attributes:
        kind: One of the format constants (``CONVERSATIONAL``, ...).
        columns: Columns read; for ``PROMPT_RESPONSE`` the prompt columns
            followed by the response column.
        system_column: System-message column (``PROMPT_RESPONSE`` only).
    """

    kind: str
    columns: tuple[str, ...] = ()
    system_column: str | None = None

    def __str__(self) -> str:
        """Describe the format for the report.

        Returns:
            The kind followed by the columns it reads, if any.
        """
        columns = [*self.columns]
        if self.system_column:
            columns.append(self.system_column)
        return f"{self.kind} ({', '.join(columns)})" if columns else self.kind


def _validate(config: LoraBudgetConfig, *, has_rows: bool) -> None:
    """Reject option combinations that cannot produce an estimate.

    Args:
        config: Estimate inputs.
        has_rows: Whether a dataset was named or passed in.

    Returns:
        None when the options are consistent.

    Raises:
        LoraBudgetError: If a required option is missing or two options
            conflict.
    """
    needs_rows = config.mode == "sft" or config.max_steps is None
    markers = (config.instruction_part, config.response_part)
    problems = {
        "--dataset is required (optional only in rl with --max_steps)": (
            needs_rows and not has_rows
        ),
        "--prompt_column and --response_column go together": (
            bool(config.prompt_column) != bool(config.response_column)
        ),
        "--system_column needs --prompt_column and --response_column": (
            bool(config.system_column) and not config.prompt_column
        ),
        "--responses_only needs data with a response part": (
            config.responses_only and bool(config.text_column)
        ),
        "--responses_only with --formatting_func needs --instruction_part "
        "and --response_part": (
            bool(config.formatting_func)
            and config.responses_only
            and not all(markers)
        ),
    }
    for message, failed in problems.items():
        if failed:
            raise LoraBudgetError(message)


def count_lora_parameters(
    model_name_or_path: str,
    rank: int,
    target_modules: list[str],
    *,
    include_experts: bool = True,
) -> int:
    """Count trainable LoRA parameters without loading any weights.

    Only the config is fetched; the model is instantiated on the ``meta``
    device. Multimodal checkpoints are restricted to the language model.

    Args:
        model_name_or_path: Model id or local path.
        rank: LoRA rank.
        target_modules: Leaf names of the linear layers to adapt.
        include_experts: Also count fused MoE expert parameters selected
            by the MLP names in ``target_modules`` (as PESFT adapts them).

    Returns:
        Number of trainable adapter parameters.

    Example:
        >>> count_lora_parameters(
        ...     "openai/gpt-oss-20b", 1, PERLConfig().target_modules
        ... )
        11556864
    """
    linear, experts = _adapter_parameters(
        model_name_or_path, rank, target_modules
    )
    return linear + experts if include_experts else linear


def _adapter_parameters(
    model_name_or_path: str, rank: int, target_modules: list[str]
) -> tuple[int, int]:
    """Count LoRA parameters on linear layers and on MoE expert weights.

    Args:
        model_name_or_path: Model id or local path.
        rank: LoRA rank.
        target_modules: Leaf names of the linear layers to adapt.

    Returns:
        Parameters on targeted linear layers (as peft adapts them) and on
        the fused expert weights those targets select.

    Raises:
        NoMatchingPeftModuleError: If the targets select neither a linear
            layer nor an expert weight.
    """
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from peft.utils import NoMatchingPeftModuleError  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForCausalLM,
    )

    config = AutoConfig.from_pretrained(model_name_or_path)  # nosec B615
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    experts = _expert_lora_parameters(model, rank, target_modules)
    lora_config = LoraConfig(
        r=rank,
        target_modules=_language_model_targets(model, target_modules),
    )
    try:
        peft_model = get_peft_model(model, lora_config)
    except NoMatchingPeftModuleError:
        # Expert-only targets (e.g. ``down_proj`` on gpt-oss) match no
        # linear layer.
        if not experts:
            raise
        return 0, experts
    trainable, _ = peft_model.get_nb_trainable_parameters()
    return int(trainable), experts


def _expert_targeted(projection: str, target_modules: list[str]) -> bool:
    """Tell whether ``target_modules`` selects an expert projection.

    Args:
        projection: Expert parameter name (``gate_up_proj``,
            ``down_proj``, ``gate_proj`` or ``up_proj``).
        target_modules: Leaf names of the linear layers to adapt.

    Returns:
        True if the projection is named, or for the fused
        ``gate_up_proj`` if ``gate_proj`` or ``up_proj`` is.
    """
    if projection in target_modules:
        return True
    fused_parts = {"gate_proj", "up_proj"}
    return projection == "gate_up_proj" and bool(
        fused_parts.intersection(target_modules)
    )


def _expert_lora_parameters(
    model: torch.nn.Module, rank: int, target_modules: list[str]
) -> int:
    """Count LoRA parameters on fused MoE expert weights.

    An expert weight is a 3-D ``(num_experts, in, out)`` parameter
    (either inner order) of a module whose name ends in ``experts``.
    LoRA gives every expert its own ``r * (in + out)`` adapter.

    Args:
        model: Model, typically on the ``meta`` device.
        rank: LoRA rank.
        target_modules: Leaf names of the linear layers to adapt.

    Returns:
        Adapter parameters over all selected expert weights; 0 for dense
        models and for experts stored as linear layers.
    """
    total = 0
    for name, parameter in model.named_parameters():
        module_name, _, projection = name.rpartition(".")
        if (
            parameter.ndim == EXPERT_WEIGHT_NDIM
            and module_name.endswith("experts")
            and _expert_targeted(projection, target_modules)
        ):
            num_experts, rows, columns = parameter.shape
            total += rank * num_experts * (rows + columns)
    return total


def load_rows(config: LoraBudgetConfig) -> Dataset:
    """Load the training rows named by ``config.dataset``.

    A local file is loaded by its suffix, a ``save_to_disk`` directory
    with ``load_from_disk`` and anything else with ``load_dataset``
    (hub id, local dataset directory or builder name with
    ``data_files``). The result is truncated to ``max_samples``.

    Args:
        config: Estimate inputs with ``dataset``, ``dataset_config``,
            ``split``, ``data_files`` and ``max_samples``.

    Returns:
        The rows that make up the training set.
    """
    from datasets import load_dataset  # noqa: PLC0415

    path = Path(cast("str", config.dataset))
    saved_files = ("state.json", "dataset_dict.json")
    if path.is_file():
        dataset = _load_local_file(path)
    elif any((path / name).is_file() for name in saved_files):
        dataset = _load_saved_dataset(path, config.split)
    else:
        loaded = load_dataset(  # nosec B615
            str(config.dataset),
            config.dataset_config,
            split=config.split,
            data_files=config.data_files,
        )
        dataset = cast("Dataset", loaded)
    return _first_rows(dataset, config.max_samples)


def _load_local_file(path: Path) -> Dataset:
    """Load a local data file with the ``datasets`` builder for its suffix.

    Args:
        path: File ending in one of ``LOCAL_BUILDERS``.

    Returns:
        All rows of the file; ``.txt`` gives one ``text`` row per line.

    Raises:
        LoraBudgetError: If the suffix is not supported.
    """
    from datasets import load_dataset  # noqa: PLC0415

    suffix = path.suffix.lower()
    if suffix not in LOCAL_BUILDERS:
        msg = (
            f"unsupported file type {suffix!r} ({path}); use "
            f"{', '.join(LOCAL_BUILDERS)}, a save_to_disk directory or a "
            "hub dataset id"
        )
        raise LoraBudgetError(msg)
    options = {"delimiter": "\t"} if suffix == ".tsv" else {}
    loaded = load_dataset(  # nosec B615
        LOCAL_BUILDERS[suffix], data_files=str(path), split="train", **options
    )
    return cast("Dataset", loaded)


def _load_saved_dataset(path: Path, split: str) -> Dataset:
    """Load a ``Dataset.save_to_disk`` or ``DatasetDict`` directory.

    Args:
        path: Directory written by ``save_to_disk``.
        split: Split to take from a ``DatasetDict``.

    Returns:
        The saved dataset, or its ``split`` for a ``DatasetDict``.

    Raises:
        LoraBudgetError: If a ``DatasetDict`` lacks ``split``.
    """
    from datasets import DatasetDict, load_from_disk  # noqa: PLC0415

    saved = load_from_disk(str(path))
    if not isinstance(saved, DatasetDict):
        return cast("Dataset", saved)
    if split not in saved:
        msg = f"split {split!r} not in {path}; splits: {', '.join(saved)}"
        raise LoraBudgetError(msg)
    return saved[split]


def _first_rows(dataset: Dataset, max_samples: int | None) -> Dataset:
    """Keep the first ``max_samples`` rows, or all when ``None``.

    Args:
        dataset: Rows to truncate.
        max_samples: Row limit.

    Returns:
        The truncated dataset.
    """
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def detect_format(
    column_names: list[str], config: LoraBudgetConfig
) -> DataFormat:
    """Pick the dataset format from the column options or column names.

    Args:
        column_names: Columns of the dataset.
        config: Estimate inputs with the column options.

    Returns:
        The explicit format if a column option is set, otherwise the
        first TRL standard format the columns match.

    Raises:
        LoraBudgetError: If a named column is missing, no format matches
            or ``responses_only`` is asked of plain text.
    """
    if config.formatting_func is not None:
        return DataFormat(FORMATTING_FUNC)
    data_format = _explicit_format(config) or _standard_format(column_names)
    named = [*data_format.columns, data_format.system_column]
    missing = [name for name in named if name and name not in column_names]
    if missing:
        msg = (
            f"columns {missing} not in dataset; its columns are: "
            f"{', '.join(column_names)}"
        )
        raise LoraBudgetError(msg)
    if config.responses_only and data_format.kind == LANGUAGE_MODELING:
        msg = (
            f"--responses_only needs a response part, but {data_format} "
            "is plain text; use --formatting_func with --response_part"
        )
        raise LoraBudgetError(msg)
    return data_format


def _explicit_format(config: LoraBudgetConfig) -> DataFormat | None:
    """Build the format named by a column option.

    Args:
        config: Estimate inputs with the column options.

    Returns:
        The format, or ``None`` when no column option is set.
    """
    if config.text_column:
        return DataFormat(LANGUAGE_MODELING, (config.text_column,))
    if config.messages_column:
        return DataFormat(CONVERSATIONAL, (config.messages_column,))
    if config.prompt_column and config.response_column:
        return DataFormat(
            PROMPT_RESPONSE,
            (*config.prompt_column, config.response_column),
            config.system_column,
        )
    return None


def _standard_format(column_names: list[str]) -> DataFormat:
    """Detect a TRL standard dataset format from the column names.

    Args:
        column_names: Columns of the dataset.

    Returns:
        Conversational (``messages``/``conversations``),
        prompt-completion or language modeling (``text``), in that order.

    Raises:
        LoraBudgetError: If no standard format matches.
    """
    for name in ("messages", "conversations"):
        if name in column_names:
            return DataFormat(CONVERSATIONAL, (name,))
    if "prompt" in column_names and "completion" in column_names:
        return DataFormat(PROMPT_COMPLETION, ("prompt", "completion"))
    if "text" in column_names:
        return DataFormat(LANGUAGE_MODELING, ("text",))
    msg = (
        "no standard dataset format (messages, conversations, "
        "prompt + completion, text) among the columns: "
        f"{', '.join(column_names)}. Name the data with --messages_column, "
        "--prompt_column with --response_column, --text_column or "
        "--formatting_func"
    )
    raise LoraBudgetError(msg)


def row_to_sample(
    row: Mapping[str, object], data_format: DataFormat
) -> Sample:
    """Turn a dataset row into text, a conversation or a prompt pair.

    Args:
        row: One dataset row.
        data_format: Format from ``detect_format`` (not formatting_func).

    Returns:
        Plain text, a message list, or a ``(prompt, completion)`` tuple of
        strings or of message lists.
    """
    first_column = data_format.columns[0]
    if data_format.kind == LANGUAGE_MODELING:
        return str(row[first_column])
    if data_format.kind == CONVERSATIONAL:
        return normalize_messages(row[first_column])
    if data_format.kind == PROMPT_COMPLETION:
        prompt, completion = row["prompt"], row["completion"]
        if isinstance(prompt, str):
            return prompt, str(completion)
        return normalize_messages(prompt), normalize_messages(completion)
    return _prompt_response_messages(row, data_format)


def _prompt_response_messages(
    row: Mapping[str, object], data_format: DataFormat
) -> Conversation:
    """Build a system/user/assistant chat from prompt and response columns.

    Empty prompt columns (e.g. Dolly's optional ``context``) are skipped.

    Args:
        row: One dataset row.
        data_format: ``PROMPT_RESPONSE`` format.

    Returns:
        The conversation, with a leading system message if configured.
    """
    *prompt_columns, response_column = data_format.columns
    prompt = "\n\n".join(
        str(row[name]) for name in prompt_columns if row[name]
    )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(row[response_column])},
    ]
    if data_format.system_column:
        system = str(row[data_format.system_column])
        messages.insert(0, {"role": "system", "content": system})
    return messages


def normalize_messages(messages: object) -> Conversation:
    """Convert role/content or ShareGPT messages to text-only messages.

    Args:
        messages: List of ``{"role", "content"}`` or ``{"from",
            "value"}`` mappings; content may be a list of typed parts.

    Returns:
        Messages with string ``role`` and ``content``; ShareGPT roles are
        mapped (``human`` -> ``user``, ``gpt`` -> ``assistant``).
    """
    normalized = []
    for message in cast("list[Mapping[str, object]]", messages):
        if "role" in message:
            role, content = message["role"], message["content"]
        else:
            role, content = message["from"], message["value"]
        normalized.append(
            {
                "role": SHAREGPT_ROLES.get(str(role), str(role)),
                "content": _text_content(content),
            }
        )
    return normalized


def _text_content(content: object) -> str:
    """Return the text of a message content, dropping non-text parts.

    Args:
        content: A string, ``None`` or a list of ``{"type", "text"}``
            parts.

    Returns:
        The text; text parts are joined by newlines.
    """
    if isinstance(content, list):
        parts = cast("list[Mapping[str, object]]", content)
        return "\n".join(
            str(part["text"]) for part in parts if part["type"] == "text"
        )
    return "" if content is None else str(content)


def _text_length(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> int:
    """Count the tokens of ``text``.

    Args:
        tokenizer: Tokenizer matching the model.
        text: Text to tokenize.
        add_special_tokens: Add the tokenizer's BOS/EOS as in training.

    Returns:
        Token count.
    """
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return len(encoded["input_ids"])


def _chat_length(
    tokenizer: PreTrainedTokenizerBase,
    messages: Conversation,
    *,
    add_generation_prompt: bool = False,
) -> int:
    """Count the tokens of ``messages`` rendered with the chat template.

    Args:
        tokenizer: Tokenizer with a chat template.
        messages: Conversation to render.
        add_generation_prompt: Append the assistant-turn header.

    Returns:
        Token count of the rendered conversation.

    Raises:
        LoraBudgetError: If the tokenizer has no chat template.
    """
    if not tokenizer.chat_template:
        msg = (
            f"{tokenizer.name_or_path} has no chat template, so "
            "conversational data cannot be rendered; use --formatting_func "
            "or --text_column"
        )
        raise LoraBudgetError(msg)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return _text_length(tokenizer, str(text), add_special_tokens=False)


def count_trained_tokens(
    tokenizer: PreTrainedTokenizerBase,
    sample: Sample,
    *,
    responses_only: bool,
) -> int:
    """Count the tokens SFT would train on for one sample.

    Args:
        tokenizer: Tokenizer (with a chat template for conversations).
        sample: Output of ``row_to_sample``.
        responses_only: Count only assistant or completion tokens.

    Returns:
        Number of trained tokens.
    """
    if isinstance(sample, tuple):
        return _prompt_completion_tokens(
            tokenizer, sample, responses_only=responses_only
        )
    if isinstance(sample, str):
        return _text_length(tokenizer, sample)
    if not responses_only:
        return _chat_length(tokenizer, sample)
    return _assistant_tokens(tokenizer, sample)


def _assistant_tokens(
    tokenizer: PreTrainedTokenizerBase, conversation: Conversation
) -> int:
    """Count the tokens of every assistant turn in a conversation.

    Each assistant turn contributes the tokens it adds after the rendered
    history plus assistant header.

    Args:
        tokenizer: Tokenizer with a chat template.
        conversation: Messages to render.

    Returns:
        Number of assistant tokens.
    """
    total = 0
    for index, message in enumerate(conversation):
        if message["role"] != "assistant":
            continue
        with_response = _chat_length(tokenizer, conversation[: index + 1])
        prompt = _chat_length(
            tokenizer, conversation[:index], add_generation_prompt=True
        )
        total += with_response - prompt
    return total


def _prompt_completion_tokens(
    tokenizer: PreTrainedTokenizerBase,
    sample: tuple[str, str] | tuple[Conversation, Conversation],
    *,
    responses_only: bool,
) -> int:
    """Count prompt-completion tokens as TRL's SFTTrainer builds them.

    Strings are concatenated with an EOS appended to the completion;
    conversations are rendered with the chat template.

    Args:
        tokenizer: Tokenizer matching the model.
        sample: ``(prompt, completion)`` strings or message lists.
        responses_only: Count only completion tokens.

    Returns:
        Completion tokens with ``responses_only``, else all tokens.
    """
    prompt, completion = sample
    if isinstance(prompt, str):
        eos = str(tokenizer.eos_token or "")
        text = cast("str", completion)
        if not text.endswith(eos):
            text += eos
        total = _text_length(tokenizer, prompt + text)
        prompt_length = _text_length(tokenizer, prompt)
    else:
        messages = cast("Conversation", completion)
        total = _chat_length(tokenizer, prompt + messages)
        prompt_length = _chat_length(
            tokenizer, prompt, add_generation_prompt=True
        )
    return total - prompt_length if responses_only else total


def count_marked_response_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    instruction_part: str,
    response_part: str,
) -> int:
    """Count tokens after each ``response_part`` up to ``instruction_part``.

    Mirrors unsloth's ``train_on_responses_only`` on formatted text.

    Args:
        tokenizer: Tokenizer matching the model.
        text: One formatted training text.
        instruction_part: Marker starting a user turn.
        response_part: Marker starting a response.

    Returns:
        Number of response tokens.

    Raises:
        LoraBudgetError: If a marker does not occur in ``text``.
    """
    for flag, marker in (
        ("--instruction_part", instruction_part),
        ("--response_part", response_part),
    ):
        if marker not in text:
            msg = f"{flag} {marker!r} not found in formatted text {text!r}"
            raise LoraBudgetError(msg)
    total = 0
    for chunk in text.split(response_part)[1:]:
        response = chunk.split(instruction_part, 1)[0]
        total += _text_length(tokenizer, response, add_special_tokens=False)
    return total


def _resolve_formatting_func(spec: str | FormattingFunc) -> FormattingFunc:
    """Import a ``module.path:function`` formatting function.

    Args:
        spec: Import path, or the function itself.

    Returns:
        The formatting function.

    Raises:
        LoraBudgetError: If ``spec`` has no ``:function`` part.
    """
    if not isinstance(spec, str):
        return spec
    module_name, _, function_name = spec.partition(":")
    if not function_name:
        msg = f"--formatting_func {spec!r} must be module.path:function"
        raise LoraBudgetError(msg)
    module = importlib.import_module(module_name)
    return cast("FormattingFunc", getattr(module, function_name))


def _formatted_token_counts(
    rows: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    config: LoraBudgetConfig,
) -> Iterator[int]:
    """Yield trained tokens of each text of ``config.formatting_func``.

    Args:
        rows: Rows to format, passed in batches.
        tokenizer: Tokenizer matching the model.
        config: Estimate inputs with ``formatting_func``,
            ``responses_only`` and the markers.

    Yields:
        Trained tokens per formatted text.
    """
    formatting_func = _resolve_formatting_func(
        cast("str | FormattingFunc", config.formatting_func)
    )
    for batch in rows.iter(batch_size=FORMAT_BATCH_SIZE):
        columns = cast("Mapping[str, list[object]]", batch)
        for text in formatting_func(columns):
            if config.responses_only:
                yield count_marked_response_tokens(
                    tokenizer,
                    text,
                    cast("str", config.instruction_part),
                    cast("str", config.response_part),
                )
            else:
                yield _text_length(tokenizer, text)


def _row_token_counts(
    rows: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    data_format: DataFormat,
    config: LoraBudgetConfig,
) -> Iterator[int]:
    """Yield trained tokens per row.

    Args:
        rows: Rows to count.
        tokenizer: Tokenizer matching the model.
        data_format: Format from ``detect_format``.
        config: Estimate inputs with ``responses_only``.

    Returns:
        Iterator of trained tokens per row (or per formatted text).
    """
    if data_format.kind == FORMATTING_FUNC:
        return _formatted_token_counts(rows, tokenizer, config)
    return (
        count_trained_tokens(
            tokenizer,
            row_to_sample(cast("Mapping[str, object]", row), data_format),
            responses_only=config.responses_only,
        )
        for row in rows
    )


def estimate_sft_tokens(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    data_format: DataFormat,
    config: LoraBudgetConfig,
) -> tuple[int, bool]:
    """Count trained tokens, extrapolating from a sample if requested.

    Args:
        dataset: Training rows.
        tokenizer: Tokenizer matching the model.
        data_format: Format from ``detect_format``.
        config: Estimate inputs with ``sample_size`` and
            ``responses_only``.

    Returns:
        The token count and whether it was extrapolated from a sample.
    """
    sample_size = config.sample_size or len(dataset)
    extrapolated = sample_size < len(dataset)
    rows = dataset
    if extrapolated:
        rows = dataset.shuffle(seed=0).select(range(sample_size))
    tokens = sum(_row_token_counts(rows, tokenizer, data_format, config))
    if extrapolated:
        tokens = round(tokens * len(dataset) / len(rows))
    return tokens, extrapolated


def rl_episodes(num_prompts: int, config: LoraBudgetConfig) -> int:
    """Count RL episodes (sampled completions) over the whole run.

    Args:
        num_prompts: Number of prompts in the training set.
        config: Estimate inputs with ``max_steps``,
            ``completions_per_step``, ``num_generations`` and
            ``num_train_epochs``.

    Returns:
        ``max_steps * completions_per_step`` when ``max_steps`` is set,
        otherwise ``num_prompts * num_generations * num_train_epochs``.
    """
    if config.max_steps is not None:
        return int(config.max_steps * config.completions_per_step)
    return int(num_prompts * config.num_generations * config.num_train_epochs)


def required_parameters(bits: float) -> float:
    """Parameters needed to store ``bits`` at 2 bits per parameter.

    Args:
        bits: Information the training run must absorb.

    Returns:
        ``bits / BITS_PER_PARAMETER``.
    """
    return bits / BITS_PER_PARAMETER


def minimum_rank(required: float, parameters_per_rank: int) -> int:
    """Smallest LoRA rank whose adapter holds ``required`` parameters.

    Args:
        required: Required parameter count.
        parameters_per_rank: Adapter parameters at rank 1.

    Returns:
        The minimum rank, at least 1.
    """
    return max(1, math.ceil(required / parameters_per_rank))


def _training_rows(
    config: LoraBudgetConfig, dataset: Dataset | None
) -> Dataset | None:
    """Validate ``config`` and return the training rows.

    Args:
        config: Estimate inputs.
        dataset: Already loaded rows, preferred over ``config.dataset``.

    Returns:
        ``dataset`` truncated to ``max_samples``, else the loaded
        ``config.dataset``, else ``None`` (rl with ``max_steps``).
    """
    has_rows = dataset is not None or config.dataset is not None
    _validate(config, has_rows=has_rows)
    if dataset is not None:
        return _first_rows(dataset, config.max_samples)
    return load_rows(config) if config.dataset is not None else None


def _sft_measure(
    rows: Dataset, config: LoraBudgetConfig
) -> tuple[DataFormat, int, bool]:
    """Detect the format and count the trained tokens of ``rows``.

    Args:
        rows: Training rows.
        config: Estimate inputs.

    Returns:
        The format, the trained tokens and whether they were
        extrapolated.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    data_format = detect_format(rows.column_names, config)
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        config.model_name_or_path
    )
    tokens, extrapolated = estimate_sft_tokens(
        rows, tokenizer, data_format, config
    )
    return data_format, tokens, extrapolated


def estimate_lora_budget(
    config: LoraBudgetConfig, dataset: Dataset | None = None
) -> LoraBudget:
    """Estimate whether a LoRA adapter has the capacity for a dataset.

    Args:
        config: Estimate inputs (model, rank, dataset, format, mode).
        dataset: Already loaded rows; used instead of ``config.dataset``
            (``max_samples`` still applies).

    Returns:
        Adapter size, trained tokens or episodes, required parameters and
        minimum rank.
    """
    rows = _training_rows(config, dataset)
    rank = config.lora_r
    if rank is None:
        rank = SFT_DEFAULT_RANK if config.mode == "sft" else PERLConfig.lora_r
    include_experts = config.include_experts
    if include_experts is None:
        include_experts = config.mode == "sft"
    linear, experts = _adapter_parameters(
        config.model_name_or_path, rank, config.target_modules
    )
    if not include_experts:
        experts = 0
    adapter = linear + experts
    data_format, tokens, episodes, extrapolated = None, None, None, False
    if config.mode == "rl":
        episodes = rl_episodes(0 if rows is None else len(rows), config)
        bits = episodes * RL_BITS_PER_EPISODE
    else:
        sft_rows = cast("Dataset", rows)
        data_format, tokens, extrapolated = _sft_measure(sft_rows, config)
        bits = tokens * config.bits_per_token
    required = required_parameters(bits)
    return LoraBudget(
        rank=rank,
        adapter_parameters=adapter,
        expert_parameters=experts,
        parameters_per_rank=adapter // rank,
        rows=None if rows is None else len(rows),
        data_format=None if data_format is None else str(data_format),
        trained_tokens=tokens,
        episodes=episodes,
        extrapolated=extrapolated,
        bits=bits,
        required_parameters=required,
        minimum_rank=minimum_rank(required, adapter // rank),
    )


def _print_report(config: LoraBudgetConfig, budget: LoraBudget) -> None:
    """Print the estimate as aligned ``label: value`` lines.

    Args:
        config: Estimate inputs.
        budget: Estimate result.

    Returns:
        None. The report is printed to stdout.
    """
    print(f"Model:           {config.model_name_or_path}")
    print(f"Targets:         {', '.join(config.target_modules)}")
    print(
        f"Adapter params:  {budget.adapter_parameters:,} (rank {budget.rank})"
    )
    if budget.expert_parameters:
        print(
            f"  MoE experts:   {budget.expert_parameters:,} of these "
            "(fused expert weights; --include_experts false drops them)"
        )
    print(f"Per unit rank:   {budget.parameters_per_rank:,}")
    rows = "none" if budget.rows is None else f"{budget.rows:,}"
    print(f"Rows:            {rows} ({config.mode})")
    if budget.episodes is not None:
        print(f"Episodes:        {budget.episodes:,} (1 bit per episode)")
    else:
        note = (
            f" (extrapolated from {config.sample_size:,} sampled rows)"
            if budget.extrapolated
            else ""
        )
        label = (
            "Response tokens:" if config.responses_only else "Trained tokens:"
        )
        print(f"Format:          {budget.data_format}")
        print(f"{label:<17}{budget.trained_tokens:,}{note}")
        print(f"Bits per token:  {config.bits_per_token}")
    verdict = "suffices" if budget.sufficient else "is too small"
    print(
        f"Required params: {budget.required_parameters:,.0f} "
        "(estimate: bits / 2)"
    )
    print(
        f"Minimum rank:    {budget.minimum_rank} (estimate); "
        f"rank {budget.rank} {verdict}"
    )


def main() -> None:
    """Print the adapter size, the required size and the minimum rank.

    Returns:
        None. The report is printed to stdout.

    Raises:
        SystemExit: With the error message if the options or the dataset
            do not allow an estimate.
    """
    config = LoraBudgetConfig.from_args()
    try:
        budget = estimate_lora_budget(config)
    except LoraBudgetError as error:
        msg = f"error: {error}"
        raise SystemExit(msg) from None
    _print_report(config, budget)


if __name__ == "__main__":
    main()
