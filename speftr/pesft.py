# SPDX-FileCopyrightText: 2025-2026 Luis Rei
# SPDX-License-Identifier: BSD-2-Clause
"""PESFT: Parameter-Efficient Supervised Fine-Tuning with LoRA.

Trains LoRA adapters on a causal language model with Unsloth and TRL's
SFTTrainer. ``PESFTConfig`` holds the settings; ``PESFT`` loads, trains and
saves.

Example:
    Basic usage with default configuration:

    >>> config = PESFTConfig()
    >>> trainer = PESFT(config)
    >>> trainer.train(train_dataset, eval_dataset, formatting_func)

    Command-line usage:

    >>> config = PESFTConfig.from_args()
    >>> trainer = PESFT(config)
    >>> trainer.train(train_dataset, eval_dataset, formatting_func)

Multi-GPU:
    Data parallel (DDP), one process per GPU, each holding a full model
    copy; Unsloth places each process on its own GPU::

        torchrun --nproc_per_node N your_script.py ...

    ``accelerate launch`` works the same way. The effective batch size is
    ``per_device_train_batch_size * gradient_accumulation_steps * N``.
    Only the main process prints the parameters and writes files.

    A model that does not fit on one GPU can instead be split across all
    visible GPUs in a single process (no torchrun) with
    ``device_map="unsloth_balanced"``: Unsloth's planner reserves room
    for the output head and logits on the head's GPU, which transformers'
    ``"balanced"`` does not. Layers then run one GPU at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch


if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase
    from trl import SFTConfig, SFTTrainer


@dataclass
class PESFTConfig:
    r"""Configuration for Parameter-Efficient Supervised Fine-Tuning.

    Hyperparameters for LoRA fine-tuning with TRL's SFTTrainer. Defaults
    come from the guard example (``examples/guard/guard_train.py``).

    Fields are grouped as model (``model_name_or_path`` to
    ``response_part``), LoRA (``lora_r`` to ``target_modules``), training
    (``output_dir`` to ``packing``) and other settings.

    With ``train_on_responses=True`` the instruction tokens are masked and
    the loss covers only the assistant's responses, so the model is not
    trained to reproduce the instruction format.

    The instruction_part and response_part define the chat template
    markers that separate user instructions from model responses.
    Different model families use different formats:

    - Gemma models (2, 3, 3n): instruction_part
      ``"<start_of_turn>user\n"``, response_part
      ``"<start_of_turn>model\n"``
    - Llama models (3, 3.1, 3.2, 3.3, 4): instruction_part
      ``"<|start_header_id|>user<|end_header_id|>\n\n"``, response_part
      ``"<|start_header_id|>assistant<|end_header_id|>\n\n"``
    - Qwen models (2.5, 3): instruction_part ``"<|im_start|>user\n"``,
      response_part ``"<|im_start|>assistant\n"``

    With ``train_on_responses`` set, the markers must match the model's
    chat template.

    Attributes:
        model_name_or_path: HuggingFace model name or local path
        max_seq_length: Maximum sequence length in tokens
        load_in_4bit: Use 4-bit quantization (QLoRA)
        load_in_8bit: Load the base model in 8-bit (bitsandbytes
            LLM.int8). Slower than 4-bit and 16-bit; cannot be combined
            with ``load_in_4bit``.
        device_map: Passed to the model loader when set, e.g.
            ``"unsloth_balanced"`` to split one model's layers across all
            visible GPUs in a single process, for models that do not fit
            on one GPU. Not for torchrun/DDP, where Unsloth puts each
            process on its own GPU (``cuda:LOCAL_RANK``). None keeps
            Unsloth's placement.
        attn_implementation: Attention backend passed to the model loader.
            ``"sdpa"`` is fastest on a 3090; Unsloth's own default (flex
            attention) recompiles its block masks for every new sequence
            length there.
        chat_template: Unsloth chat template name, or None to keep the
            tokenizer's own template (needed for models Unsloth has no
            named template for)
        train_on_responses: Only train on response tokens, not instructions
        instruction_part: Chat template marker for instruction/user turn
        response_part: Chat template marker for response/assistant turn
        lora_r: LoRA rank (adapter dimension)
        lora_alpha: LoRA scaling factor
        lora_layers: Layer preset used to derive target_modules on the CLI
        use_gradient_checkpointing: Gradient checkpointing mode:
            "unsloth" (Unsloth's checkpointing, which offloads activations
            to CPU), "true" (same) or "false" (off: faster, but activation
            memory grows with batch size and sequence length)
        target_modules: List of module names to apply LoRA
        output_dir: Directory for saving checkpoints and final model
        num_train_epochs: Number of training epochs
        max_steps: Maximum training steps (-1 = use epochs)
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient
        scheduler: Learning rate scheduler type
        warmup_steps: Number of warmup steps
        warmup_ratio: Fraction of training steps for warmup (0.0 = none)
        logging_steps: Log metrics every N steps
        eval_strategy: Evaluation strategy (epoch/steps/no). With
            ``epoch``/``steps`` the checkpoint with the best ``eval_loss``
            is loaded at the end, so ``save_strategy`` must match. With
            ``no`` there is no evaluation during training and the final
            model is kept; ``train`` still evaluates it once at the end
            when given an eval dataset, and ``save_strategy`` is free
            (e.g. ``steps`` for resumable checkpoints).
        eval_steps: Evaluate every N steps (when eval_strategy='steps')
        save_strategy: Checkpoint saving strategy (epoch/steps/no)
        save_steps: Save every N steps (when save_strategy='steps')
        save_total_limit: Maximum number of checkpoints to keep
        optim: Optimizer name; any transformers ``optim`` value
        max_grad_norm: Gradient clipping norm
        report_to: Experiment tracking backend
        packing: Pack multiple short examples into one sequence
        padding_free: Concatenate each batch into one unpadded sequence.
            Off by default: without FlashAttention (e.g. a 3090 with SDPA)
            it is several times slower than padded batches.
        router_aux_loss_coef: Weight of the Mixture-of-Experts router
            load-balancing loss; no effect on dense models. 0 disables it
            and stops the model from returning router logits. The router
            itself is not a LoRA target (frozen); the loss can only nudge
            the adapters through the hidden states that feed it. Some MoE
            implementations (Unsloth's 4-bit gpt-oss) return no router
            logits and fail when it is enabled.
        eval_in_train_mode: Module class names (``type(module).__name__``)
            that stay in training mode during evaluation, both during and
            after training. For models whose eval-mode kernels are wrong or
            much heavier than their training-mode ones. Only safe for
            modules with no dropout or other training-only behaviour of
            their own; their children (e.g. LoRA layers) still follow
            ``eval()``. Evaluation runs without gradients either way.
            Empty (the default): plain evaluation.
        validate_save: Validate model files after saving
        save_method: Strategy used when saving the trained model
        random_state: Random seed for reproducibility
    """

    # Model configuration
    model_name_or_path: str = "unsloth/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: str | None = None
    attn_implementation: str = "sdpa"
    chat_template: str | None = "qwen2.5"
    train_on_responses: bool = False
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"

    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 32
    lora_layers: str = "all"
    use_gradient_checkpointing: str = "unsloth"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )

    # Training configuration
    output_dir: str = "./models/speftr-sft"
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    scheduler: str = "constant"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    logging_steps: int = 100
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    save_strategy: str = "epoch"
    save_steps: int | None = None
    save_total_limit: int = 1
    optim: str = "adamw_8bit"
    max_grad_norm: float = 1.0
    report_to: str = "none"
    packing: bool = False
    padding_free: bool = False
    router_aux_loss_coef: float = 0.0
    eval_in_train_mode: list[str] = field(default_factory=list)

    # Other configuration
    validate_save: bool = True
    save_method: str = "lora"
    random_state: int = 42

    def __post_init__(self) -> None:
        """Reject option combinations that cannot load a model.

        Returns:
            None. Validation only.

        Raises:
            ValueError: If ``load_in_4bit`` and ``load_in_8bit`` are both
                set.
        """
        if self.load_in_4bit and self.load_in_8bit:
            msg = "load_in_4bit and load_in_8bit are mutually exclusive"
            raise ValueError(msg)

    @classmethod
    def from_args(cls, args: argparse.Namespace | None = None) -> PESFTConfig:
        """Create config from command-line arguments.

        Args:
            args: Parsed arguments (if None, will parse from sys.argv)

        Returns:
            PESFTConfig instance with values from command line
        """
        if args is None:
            parser = cls.get_argument_parser()
            args = parser.parse_args()

        # Extract only the fields that exist in PESFTConfig
        config_dict = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                config_dict[field_name] = getattr(args, field_name)

        # Handle negated boolean flags
        if hasattr(args, "no_validate_save"):
            config_dict["validate_save"] = not args.no_validate_save

        # Convert lora_layers to target_modules
        if hasattr(args, "lora_layers"):
            if args.lora_layers == "mlp":
                config_dict["target_modules"] = [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif args.lora_layers == "attention":
                config_dict["target_modules"] = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ]
            else:  # "all"
                config_dict["target_modules"] = [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ]

        return cls(**config_dict)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """Get argument parser with all PESFT configuration options.

        Returns:
            ArgumentParser configured with PESFT parameters
        """
        parser = argparse.ArgumentParser(
            description="Parameter-Efficient Supervised Fine-Tuning with LoRA"
        )

        # Model arguments
        model_group = parser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--model_name_or_path",
            type=str,
            default="unsloth/Qwen2.5-0.5B-Instruct",
            help=(
                "Model name or path (default: unsloth/Qwen2.5-0.5B-Instruct)"
            ),
        )
        model_group.add_argument(
            "--max_seq_length",
            type=int,
            default=2048,
            help="Maximum sequence length in tokens (default: 2048)",
        )
        model_group.add_argument(
            "--load_in_4bit",
            action="store_true",
            dest="load_in_4bit",
            help="Use 4-bit quantization (QLoRA)",
            default=False,
        )
        model_group.add_argument(
            "--load_in_8bit",
            action="store_true",
            help=(
                "Load the base model in 8-bit (bitsandbytes); slower than "
                "4-bit; not with --load_in_4bit"
            ),
        )
        model_group.add_argument(
            "--device_map",
            type=str,
            default=None,
            help=(
                "Model placement, e.g. 'unsloth_balanced' to split a model "
                "too large for one GPU across all visible GPUs in one "
                "process. Not for torchrun (default: Unsloth's placement)"
            ),
        )
        model_group.add_argument(
            "--chat_template",
            type=str,
            default="qwen2.5",
            help="Chat template name (default: qwen2.5)",
        )
        model_group.add_argument(
            "--train_on_responses",
            action="store_true",
            help=(
                "Enable train_on_responses_only to mask instruction tokens "
                "and train only on assistant responses (default: disabled)"
            ),
        )
        model_group.add_argument(
            "--instruction_part",
            type=str,
            default="<|im_start|>user\n",
            help=(
                "Chat template marker for instruction/user turn. "
                "For Qwen ChatML models: '<|im_start|>user\\n'. "
                "For Gemma models: '<start_of_turn>user\\n'. "
                "For Llama models: "
                "'<|start_header_id|>user<|end_header_id|>\\n\\n'. "
                "For other models, copy the marker from the model's chat "
                "template. "
                "(default: <|im_start|>user\\n for Qwen ChatML)"
            ),
        )
        model_group.add_argument(
            "--response_part",
            type=str,
            default="<|im_start|>assistant\n",
            help=(
                "Chat template marker for response/assistant turn. "
                "For Qwen ChatML models: '<|im_start|>assistant\\n'. "
                "For Gemma models: '<start_of_turn>model\\n'. "
                "For Llama models: "
                "'<|start_header_id|>assistant<|end_header_id|>\\n\\n'. "
                "For other models, copy the marker from the model's chat "
                "template. "
                "(default: <|im_start|>assistant\\n for Qwen ChatML)"
            ),
        )

        # LoRA arguments
        lora_group = parser.add_argument_group("LoRA Configuration")
        lora_group.add_argument(
            "--lora_r",
            type=int,
            default=8,
            help="LoRA rank (default: 8)",
        )
        lora_group.add_argument(
            "--lora_alpha",
            type=int,
            default=32,
            help="LoRA alpha (default: 32)",
        )
        lora_group.add_argument(
            "--lora_layers",
            type=str,
            choices=["all", "attention", "mlp"],
            default="all",
            help=(
                "LoRA target layers: 'all' (attention + MLP), "
                "'attention' (attention layers only), "
                "'mlp' (MLP layers only) (default: all)"
            ),
        )
        lora_group.add_argument(
            "--use_gradient_checkpointing",
            type=str,
            choices=["unsloth", "True", "False"],
            default="unsloth",
            help=(
                "Gradient checkpointing: 'unsloth' (Unsloth's "
                "checkpointing, offloads activations to CPU), 'True' "
                "(standard checkpointing), 'False' (off) "
                "(default: unsloth)"
            ),
        )

        # Training arguments
        train_group = parser.add_argument_group("Training Configuration")
        train_group.add_argument(
            "--output_dir",
            type=str,
            default="./models/speftr-sft",
            help=(
                "Output directory for saving model "
                "(default: ./models/speftr-sft)"
            ),
        )
        train_group.add_argument(
            "--num_train_epochs",
            type=int,
            default=3,
            help="Number of training epochs (default: 3)",
        )
        train_group.add_argument(
            "--max_steps",
            type=int,
            default=-1,
            help=(
                "Stop after this many optimizer steps; -1 trains "
                "num_train_epochs (default: -1)"
            ),
        )
        train_group.add_argument(
            "--per_device_train_batch_size",
            type=int,
            default=32,
            help="Per device train batch size (default: 32)",
        )
        train_group.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=1,
            help="Per device eval batch size (default: 1)",
        )
        train_group.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Gradient accumulation steps (default: 1)",
        )
        train_group.add_argument(
            "--learning_rate",
            type=float,
            default=2e-4,
            help="Learning rate (default: 2e-4)",
        )
        train_group.add_argument(
            "--warmup_ratio",
            type=float,
            default=0.0,
            help="Fraction of steps used for warmup (default: 0.0)",
        )
        train_group.add_argument(
            "--optim",
            type=str,
            default="adamw_8bit",
            help=(
                "Optimizer; any transformers optim name, e.g. adamw_8bit, "
                "adamw_torch, paged_adamw_8bit (pages optimizer state to "
                "CPU under memory spikes), adamw_torch_4bit "
                "(default: adamw_8bit)"
            ),
        )
        train_group.add_argument(
            "--max_grad_norm",
            type=float,
            default=1.0,
            help="Gradient clipping norm (default: 1.0)",
        )
        train_group.add_argument(
            "--packing",
            action="store_true",
            help="Enable sequence packing during training (default: disabled)",
        )
        train_group.add_argument(
            "--router_aux_loss_coef",
            type=float,
            default=0.0,
            help=(
                "MoE router load-balancing loss weight; 0 disables it "
                "(default: 0.0)"
            ),
        )
        train_group.add_argument(
            "--eval_in_train_mode",
            nargs="*",
            default=[],
            metavar="CLASS_NAME",
            help=(
                "Module class names kept in training mode during "
                "evaluation, for eval-mode kernels that are wrong or too "
                "heavy; only for modules without dropout (default: none)"
            ),
        )
        train_group.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="Weight decay (default: 0.0)",
        )
        train_group.add_argument(
            "--scheduler",
            type=str,
            choices=["constant", "constant_with_warmup", "linear", "cosine"],
            default="constant",
            help=(
                "LR scheduler: 'constant' (flat), 'constant_with_warmup' "
                "(warmup then flat), 'linear' (decay), or 'cosine' "
                "(default: constant)"
            ),
        )
        train_group.add_argument(
            "--eval_strategy",
            type=str,
            choices=["no", "steps", "epoch"],
            default="epoch",
            help=(
                "Evaluation strategy: no, steps, or epoch. 'steps'/'epoch' "
                "reload the best checkpoint at the end (save_strategy must "
                "match); 'no' keeps the final model and, with an eval "
                "dataset, evaluates it once after training "
                "(default: epoch)"
            ),
        )
        train_group.add_argument(
            "--random_state",
            type=int,
            default=42,
            help="Random state for reproducibility (default: 42)",
        )
        train_group.add_argument(
            "--eval_steps",
            type=int,
            default=None,
            help=(
                "Evaluate every N steps (when eval_strategy='steps'). "
                "When load_best_model_at_end=True, should be a multiple "
                "of save_steps for synchronization (default: None)"
            ),
        )
        train_group.add_argument(
            "--save_steps",
            type=int,
            default=None,
            help=(
                "Save every N steps (when save_strategy='steps'). "
                "When load_best_model_at_end=True, should be a multiple "
                "of eval_steps for synchronization (default: None)"
            ),
        )
        train_group.add_argument(
            "--save_strategy",
            type=str,
            choices=["no", "steps", "epoch"],
            default="epoch",
            help=(
                "Checkpoint save strategy: 'no', 'steps', or 'epoch' "
                "(default: epoch)"
            ),
        )

        # Other arguments
        other_group = parser.add_argument_group("Other Configuration")
        other_group.add_argument(
            "--save_method",
            type=str,
            default="lora",
            help=(
                "Model save strategy: 'lora' (adapters) or Unsloth merged "
                "options such as 'merged_16bit'."
            ),
        )
        other_group.add_argument(
            "--no_validate_save",
            action="store_true",
            help="Skip model save validation",
        )

        return parser


def _is_main_process() -> bool:
    """Tell whether this process should print summaries and write files.

    torchrun and ``accelerate launch`` set ``RANK`` for every process;
    without a launcher it is unset and the single process is the main one.

    Returns:
        True for global rank 0 or when not launched distributed.
    """
    return os.environ.get("RANK", "0") == "0"


def _stay_in_training_mode(
    module: torch.nn.Module,
    mode: bool = True,  # noqa: FBT001, FBT002
) -> torch.nn.Module:
    """Replacement ``train`` that keeps ``module`` itself in training mode.

    The children still get the requested mode, so a LoRA dropout inside
    ``module`` follows ``train()``/``eval()`` as usual.

    Args:
        module: The module whose ``train`` this replaces.
        mode: Requested mode, passed on to the children only; the
            signature is ``nn.Module.train``'s.

    Returns:
        ``module``, with ``training`` set to True.
    """
    for child in module.children():
        child.train(mode)
    module.training = True
    return module


@contextmanager
def _modules_in_training_mode(
    model: torch.nn.Module, class_names: Collection[str]
) -> Iterator[None]:
    """Keep the modules of the named classes in training mode.

    For modules whose forward picks a kernel from ``self.training`` and
    whose eval-mode kernel is wrong or much heavier than the training-mode
    one. Only the matching modules themselves stay in training mode; their
    children still follow ``train()``/``eval()``, so e.g. a LoRA dropout
    inside them is off during evaluation. Does nothing when
    ``class_names`` is empty.

    Args:
        model: The model being trained.
        class_names: Class names matched against ``type(module).__name__``;
            the modules must have no dropout or other training-only
            behaviour of their own.

    Yields:
        None. On exit the modules follow ``train()``/``eval()`` again.
    """
    modules = [
        module
        for module in model.modules()
        if type(module).__name__ in class_names
    ]
    for module in modules:
        # An instance attribute shadows nn.Module.train, which eval() calls
        # recursively on every submodule.
        module.train = partial(_stay_in_training_mode, module)
        module.training = True
    try:
        yield
    finally:
        for module in modules:
            del module.train


type ParametersInput = Mapping[str, object] | object
type SerializedParameters = dict[str, object]


def _normalize_parameters(parameters: ParametersInput) -> SerializedParameters:
    """Return a serializable mapping of PESFT parameters.

    Args:
        parameters: Dataclass instance or mapping containing configuration
            values that should be saved alongside training artifacts.

    Returns:
        A dictionary that mirrors ``parameters`` and can be JSON serialized.

    Raises:
        TypeError: If ``parameters`` is neither a dataclass instance nor a
            mapping.
    """
    if is_dataclass(parameters):
        if isinstance(parameters, type):
            msg = "parameters dataclass must be an instance, not a class"
            raise TypeError(msg)
        return cast("SerializedParameters", asdict(parameters))
    if isinstance(parameters, Mapping):
        return dict(parameters)
    msg = "parameters must be a dataclass instance or mapping to be serialized"
    raise TypeError(msg)


def display_parameters(parameters: ParametersInput) -> SerializedParameters:
    """Print parameters and return them as a dictionary.

    Args:
        parameters: Dataclass instance or mapping describing the current
            configuration.

    Returns:
        Copy of ``parameters`` as a standard dictionary for downstream use.
    """
    data = _normalize_parameters(parameters)

    print("\nPESFT parameters:")
    for key in sorted(data):
        print(f"  {key}: {data[key]}")

    return data


def save_parameters_to_json(
    parameters: ParametersInput, output_dir: str
) -> Path:
    """Persist parameters to ``speftr.json`` inside ``output_dir``.

    Args:
        parameters: Dataclass instance or mapping describing the run.
        output_dir: Directory where ``speftr.json`` should be written.

    Returns:
        Path to the written JSON file.
    """
    data = _normalize_parameters(parameters)

    output_path = Path(output_dir) / "speftr.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)

    print(f"\nParameters saved to {output_path}")

    return output_path


class PESFT:
    """Parameter-Efficient Supervised Fine-Tuning trainer.

    Loads a model with Unsloth, attaches LoRA adapters, trains and
    evaluates with TRL's SFTTrainer, and saves the adapters or a merged
    model. Lifecycle: ``PESFT(config)``, ``load_model()`` (``train`` calls
    it when needed), ``train(...)``, ``save_model()``.

    Instances expose ``config`` (training configuration), ``model``
    (language model with LoRA adapters), ``tokenizer`` (with the chat
    template applied), ``trainer`` (TRL SFTTrainer instance, set once
    train() is called) and ``training_args``.

    Example:
        >>> config = PESFTConfig(
        ...     model_name_or_path="unsloth/Qwen2.5-0.5B-Instruct",
        ...     output_dir="/path/to/output",
        ...     num_train_epochs=3,
        ... )
        >>> trainer = PESFT(config)
        >>> trainer.train(train_dataset, eval_dataset, formatting_func)
        >>> trainer.save_model()
    """

    def __init__(self, config: PESFTConfig) -> None:
        """Initialize the trainer and persist the supplied configuration.

        Args:
            config: Fully populated training configuration.
        """
        from huggingface_hub import constants as hub_constants  # noqa: PLC0415

        # Dataset preparation forks workers; a fork while huggingface_hub's
        # telemetry thread holds its HTTP client lock deadlocks the child.
        hub_constants.HF_HUB_DISABLE_TELEMETRY = True

        # Unsloth patches transformers, trl and peft internals on import, so
        # it must be imported before them.
        import unsloth  # noqa: PLC0415

        # Unsloth wraps torch.__getattr__, so torch's own filter for these
        # deprecation warnings (keyed on module "torch") does not match.
        warnings.filterwarnings(
            "ignore",
            message=".*is deprecated, please use.*",
            category=UserWarning,
            module="unsloth.import_fixes",
        )
        from peft import PeftModel  # noqa: PLC0415, F401

        # Import other libraries after unsloth so its patches apply
        from transformers import PreTrainedTokenizerBase  # noqa: PLC0415, F401
        from trl import SFTConfig, SFTTrainer  # noqa: PLC0415, F401

        self.config = config
        self.model: PeftModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.trainer: SFTTrainer | None = None
        self.training_args: SFTConfig | None = None

        self._unsloth_version = unsloth.__version__

        if _is_main_process():
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            params_dict = display_parameters(self.config)
            save_parameters_to_json(params_dict, self.config.output_dir)

    def _validate_step_synchronization(self) -> None:
        """Validate eval/save cadence when both operate on steps.

        The Hugging Face Trainer expects ``save_steps`` to be a multiple of
        ``eval_steps`` whenever ``load_best_model_at_end`` is enabled. When
        that relationship is broken the "best" checkpoint can lag behind
        evaluation, so a warning is printed.

        Returns:
            None. The result of the check is printed to stdout.
        """
        if (
            self.config.eval_strategy != "steps"
            or self.config.save_strategy != "steps"
        ):
            return

        if self.config.eval_steps is None or self.config.save_steps is None:
            return

        eval_steps = self.config.eval_steps
        save_steps = self.config.save_steps

        # Check if save_steps is a multiple of eval_steps
        if save_steps % eval_steps != 0:
            print(
                f"Warning: save_steps ({save_steps}) is not a multiple of "
                f"eval_steps ({eval_steps})."
            )
            print(
                "  load_best_model_at_end=True can then pick a best "
                "evaluation step that has no saved checkpoint."
            )
            suggested = eval_steps * (save_steps // eval_steps + 1)
            print(
                "  Set save_steps to a multiple of eval_steps, "
                f"e.g. {suggested}."
            )
        else:
            print(
                f"Step synchronization validated: save_steps ({save_steps}) "
                f"is a multiple of eval_steps ({eval_steps})"
            )

    def _model_load_kwargs(self) -> dict[str, object]:
        """Build keyword arguments for ``FastLanguageModel.from_pretrained``.

        Returns:
            Loader arguments from ``self.config``. ``device_map`` is only
            included when set, so Unsloth keeps its own placement
            (including one GPU per process under torchrun) otherwise.
        """
        kwargs: dict[str, object] = {
            "model_name": self.config.model_name_or_path,
            "max_seq_length": self.config.max_seq_length,
            "dtype": None,
            "load_in_4bit": self.config.load_in_4bit,
            "load_in_8bit": self.config.load_in_8bit,
            "attn_implementation": self.config.attn_implementation,
        }
        if self.config.device_map is not None:
            kwargs["device_map"] = self.config.device_map
        return kwargs

    def load_model(
        self,
    ) -> tuple[PeftModel, PreTrainedTokenizerBase]:
        """Load a base model with Unsloth and attach LoRA adapters.

        Returns:
            Tuple containing the PEFT-wrapped model and tokenizer. The
            tokenizer carries ``config.chat_template`` when one is set.
        """
        # Unsloth before peft/transformers so its patches apply.
        from unsloth import FastLanguageModel  # noqa: PLC0415, I001
        from unsloth.chat_templates import get_chat_template  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415, F401
        from transformers import PreTrainedTokenizerBase  # noqa: PLC0415, F401

        print(f"Loading model: {self.config.model_name_or_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            **self._model_load_kwargs()
        )

        if self.config.chat_template:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=self.config.chat_template,
            )

        # Add LoRA adapters
        # Convert string gradient checkpointing option to appropriate value.
        # Unsloth supports its own checkpointing mode, in addition to the
        # canonical True/False flags exposed by transformers.
        gradient_checkpointing: str | bool
        if self.config.use_gradient_checkpointing.lower() == "unsloth":
            gradient_checkpointing = "unsloth"
        elif self.config.use_gradient_checkpointing.lower() == "true":
            gradient_checkpointing = True
        elif self.config.use_gradient_checkpointing.lower() == "false":
            gradient_checkpointing = False
        else:
            gradient_checkpointing = "unsloth"  # fallback to default

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            bias="none",
            use_gradient_checkpointing=gradient_checkpointing,
            random_state=self.config.random_state,
        )

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def _build_training_arguments(self) -> SFTConfig:
        """Construct the ``trl.SFTConfig`` for this run.

        Returns:
            SFTConfig populated from ``self.config``. Precision follows the
            GPU: bf16 where supported, otherwise fp16; both off without
            CUDA.
        """
        from trl import SFTConfig  # noqa: PLC0415

        # Determine precision based on GPU capability
        # This applies to activations regardless of weight quantization
        # (weights can be 4-bit with load_in_4bit=True)
        use_bf16 = (
            torch.cuda.is_bf16_supported()
            if torch.cuda.is_available()
            else False
        )
        use_fp16 = not use_bf16 if torch.cuda.is_available() else False

        # SFTConfig enables checkpointing by default, which would override
        # use_gradient_checkpointing="false" set on the model.
        gradient_checkpointing = (
            self.config.use_gradient_checkpointing.lower() != "false"
        )

        return SFTConfig(
            output_dir=self.config.output_dir,
            max_length=self.config.max_seq_length,
            packing=self.config.packing,
            padding_free=self.config.padding_free,
            router_aux_loss_coef=self.config.router_aux_loss_coef,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=(
                self.config.per_device_train_batch_size
            ),
            per_device_eval_batch_size=(
                self.config.per_device_eval_batch_size
            ),
            gradient_accumulation_steps=(
                self.config.gradient_accumulation_steps
            ),
            max_grad_norm=self.config.max_grad_norm,
            gradient_checkpointing=gradient_checkpointing,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.scheduler,
            # ``warmup_steps`` takes either a step count or, as a float in
            # [0, 1), a fraction of total steps.
            warmup_steps=self.config.warmup_ratio or self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            # None is only read when save_strategy="steps", where HF
            # validates it; otherwise it passes through untouched.
            save_steps=self.config.save_steps,  # pyright: ignore[reportArgumentType]
            save_total_limit=self.config.save_total_limit,
            bf16=use_bf16,
            fp16=use_fp16,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            train_sampling_strategy="group_by_length",
            report_to=self.config.report_to,
            # The best checkpoint needs evaluations during training; with
            # eval_strategy="no" the final model is kept and evaluated once.
            load_best_model_at_end=self.config.eval_strategy != "no",
            metric_for_best_model="eval_loss",
            prediction_loss_only=True,
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        formatting_func: Callable[[Mapping[str, object]], list[str]],
        *,
        resume_from_checkpoint: str | bool | None = None,
    ) -> None:
        """Train the model with ``trl.SFTTrainer``.

        Args:
            train_dataset: Dataset used for supervised fine-tuning.
            eval_dataset: Optional evaluation dataset, evaluated once after
                training and, per ``eval_strategy``, during it. Pass
                ``None`` to skip evaluation entirely.
            formatting_func: Callable that converts raw dataset rows into
                chat-formatted strings understood by the tokenizer.
            resume_from_checkpoint: Either a path to a checkpoint directory,
                ``True`` to auto-detect the latest checkpoint, or ``None`` to
                start fresh.

        Returns:
            None. Training metrics are logged through TRL and the model is
            updated in-place.
        """
        if self.model is None or self.tokenizer is None:
            model, tokenizer = self.load_model()
        else:
            model, tokenizer = self.model, self.tokenizer

        # Validate step synchronization for eval/save strategies
        self._validate_step_synchronization()

        print(f"\nTrain dataset size: {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"Eval dataset size: {len(eval_dataset)}")
        else:
            print("Eval dataset: None (evaluation disabled)")

        # Create training arguments
        training_args = self._build_training_arguments()
        self.training_args = training_args

        print("\nEvaluation settings:")
        print(
            "  per_device_eval_batch_size: "
            f"{training_args.per_device_eval_batch_size}"
        )
        print(f"  bf16: {training_args.bf16}")
        print(f"  fp16: {training_args.fp16}")

        # ``__init__`` imported unsloth, so trl is already patched.
        from trl import SFTTrainer  # noqa: PLC0415, I001
        from unsloth.chat_templates import train_on_responses_only  # noqa: PLC0415

        # ``formatting_func`` keeps the dataset lightweight: rows are turned
        # into chat-formatted text while the dataset is prepared, so the
        # raw columns never need a text field of their own.
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # Multimodal checkpoints load a processor; text SFT must hand the
            # trainer its tokenizer, which is what sequence lengths (for
            # length-grouped batching) are derived from.
            processing_class=getattr(tokenizer, "tokenizer", tokenizer),
            # Unsloth's SFTTrainer also accepts batched formatting functions
            # (columns in, list of texts out); TRL's annotation does not.
            formatting_func=formatting_func,  # pyright: ignore[reportArgumentType]
        )
        # Unsloth's trainer turns on bf16/fp16 "full eval", which makes
        # evaluate() outside training cast the whole model to 16-bit. Its
        # compiled kernels then return NaN for models whose norm weights it
        # keeps in float32 (Gemma 4 in 4-bit). Evaluation during training
        # never casts, so turning it off makes both evaluations agree.
        trainer.args.bf16_full_eval = False
        trainer.args.fp16_full_eval = False
        self.trainer = trainer

        # Apply train_on_responses_only if enabled
        if self.config.train_on_responses:
            print(
                "\nApplying train_on_responses_only "
                "(masking instruction tokens)..."
            )
            trainer = cast(
                "SFTTrainer",
                train_on_responses_only(
                    trainer,
                    instruction_part=self.config.instruction_part,
                    response_part=self.config.response_part,
                ),
            )
            self.trainer = trainer
        else:
            print("\nTraining on full sequences, instruction tokens included.")
            print(
                "  Pass --train_on_responses with matching template markers "
                "to mask instruction tokens during training."
            )

        with _modules_in_training_mode(model, self.config.eval_in_train_mode):
            self._run_training(
                trainer, resume_from_checkpoint=resume_from_checkpoint
            )
            if eval_dataset is not None:
                self._run_final_evaluation(trainer)

    @staticmethod
    def _run_training(
        trainer: SFTTrainer, *, resume_from_checkpoint: str | bool | None
    ) -> None:
        """Run ``trainer.train``, resuming when asked.

        Args:
            trainer: The configured trainer.
            resume_from_checkpoint: See ``train``.

        Returns:
            None. The model is trained in place.
        """
        print("\nStarting training...")
        if resume_from_checkpoint:
            if isinstance(resume_from_checkpoint, bool):
                print("Resuming from latest checkpoint (auto-detect)...")
                trainer.train(resume_from_checkpoint=True)
            else:
                print(f"Resuming from checkpoint: {resume_from_checkpoint}")
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()

    @staticmethod
    def _run_final_evaluation(trainer: SFTTrainer) -> None:
        """Evaluate the trained model and print the metrics.

        Args:
            trainer: The trainer, after ``train``.

        Returns:
            None. The metrics, or the out-of-memory error, are printed.
        """
        print("\nEvaluating on test set...")
        try:
            eval_results = trainer.evaluate()
            print(f"Evaluation results: {eval_results}")
        except torch.cuda.OutOfMemoryError as e:
            # The trained model is intact; free the evaluation's memory so
            # save_model() can still write the adapter.
            torch.cuda.empty_cache()
            print(f"Warning: evaluation ran out of GPU memory: {e}")
            print(
                "  The model is trained; save it, then evaluate with a "
                "smaller per_device_eval_batch_size."
            )

    def _save_training_metadata(self) -> None:
        """Persist configuration and training arguments alongside artifacts.

        Writes ``speftr.json`` (the PESFT config) and ``training_args.json``
        (the SFTConfig) next to the saved model, to reproduce or inspect the
        run.

        Returns:
            None. Files are written to ``config.output_dir``.
        """
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_parameters_to_json(self.config, self.config.output_dir)

        args = self.training_args
        if args is None and self.trainer is not None:
            # If training already ran, prefer the trainer's view of the
            # arguments because it contains any defaults injected by TRL.
            args = self.trainer.args
        if args is None:
            args = self._build_training_arguments()
            self.training_args = args

        args_path = output_path / "training_args.json"
        args_data = args.to_dict()
        serializable_args: dict[str, object] = dict(args_data)
        with args_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable_args, handle, indent=2, sort_keys=True)
        print(f"Training arguments saved to {args_path}")

    def save_model(
        self,
        save_method: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        """Save the fine-tuned adapters (or a merged model) to disk.

        Args:
            save_method: Strategy to use. ``None`` defers to the configuration.
                Common values: ``"lora"`` for adapters only or Unsloth's
                merged variants such as ``"merged_16bit"``.
            output_dir: Target directory. Defaults to
                ``self.config.output_dir`` when omitted.

        Under a distributed launch only the main process writes; the
        others return immediately.

        Returns:
            None. Files are written to the chosen output directory.

        Raises:
            ValueError: If the model/tokenizer have not been loaded yet.
        """
        if self.model is None or self.tokenizer is None:
            msg = "Model and tokenizer must be loaded before saving"
            raise ValueError(msg)
        if not _is_main_process():
            return

        if save_method is None:
            save_method = self.config.save_method
        save_method = save_method.strip() or "lora"
        normalized_method = save_method.lower()

        # Use provided output_dir or fall back to config
        save_dir = (
            output_dir if output_dir is not None else self.config.output_dir
        )

        if normalized_method == "lora":
            print(f"\nSaving adapters to {save_dir}...")
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
        else:
            print(
                "\nSaving merged model "
                f"(save_method={normalized_method}) "
                f"to {save_dir}..."
            )
            self.model.save_pretrained_merged(
                save_dir,
                self.tokenizer,
                save_method=save_method,
            )
            print("Merged model saved.")

        self._save_training_metadata()
