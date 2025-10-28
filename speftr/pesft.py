"""PESFT: Parameter-Efficient Supervised Fine-Tuning with LoRA.

This module provides a reusable class-based interface for training language
models with LoRA adapters using TRL's SFTTrainer. The design follows
HuggingFace's pattern of separating configuration from training logic.

Example:
    Basic usage with default configuration:

    >>> config = PESFTConfig()
    >>> trainer = PESFT(config)
    >>> trainer.train(train_dataset, eval_dataset, formatting_func)

    Command-line usage:

    >>> config = PESFTConfig.from_args()
    >>> trainer = PESFT(config)
    >>> trainer.train(train_dataset, eval_dataset, formatting_func)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch


if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase, TrainingArguments
    from trl import SFTTrainer


@dataclass
class PESFTConfig:
    r"""Configuration for Parameter-Efficient Supervised Fine-Tuning.

    This dataclass contains all hyperparameters needed for LoRA fine-tuning
    with TRL's SFTTrainer. Defaults are based on the working configuration
    from guard_og.py.

    Attributes:
        Model Configuration:
            model_name_or_path: HuggingFace model name or local path
            max_seq_length: Maximum sequence length (None = no limit)
            load_in_4bit: Use 4-bit quantization (QLoRA)
            chat_template: Chat template name for formatting
            train_on_responses: Only train on response tokens, not instructions
            instruction_part: Chat template marker for instruction/user turn
            response_part: Chat template marker for response/assistant turn

        Train on Responses Only Configuration:
            When train_on_responses=True, the model only learns from the
            assistant's responses, not from the user's instructions. This is
            useful for:
            - Reducing overfitting to specific instruction formats
            - Focusing learning on response generation quality
            - Following best practices for instruction-following models

            The instruction_part and response_part define the chat template
            markers that separate user instructions from model responses.
            Different model families use different formats:

            Gemma models (2, 3, 3n):
                instruction_part: "<start_of_turn>user\n"
                response_part: "<start_of_turn>model\n"

            Llama models (3, 3.1, 3.2, 3.3, 4):
                instruction_part: (
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                )
                response_part: (
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )

            Qwen models (2.5, 3):
                instruction_part: "<|im_start|>user\n"
                response_part: "<|im_start|>assistant\n"

            Users must ensure these markers align with their model's chat
            template format when enabling train_on_responses_only.

        LoRA Configuration:
            lora_r: LoRA rank (adapter dimension)
            lora_alpha: LoRA scaling factor
            target_modules: List of module names to apply LoRA

        Training Configuration:
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
            eval_strategy: Evaluation strategy (epoch/steps/no)
            eval_steps: Evaluate every N steps (when eval_strategy='steps')
            save_strategy: Checkpoint saving strategy (epoch/steps/no)
            save_steps: Save every N steps (when save_strategy='steps')
            save_total_limit: Maximum number of checkpoints to keep
            optim: Optimizer name
            max_grad_norm: Gradient clipping norm
            report_to: Experiment tracking backend

        Other Configuration:
            validate_save: Validate model files after saving
            save_method: Strategy used when saving the trained model
            random_state: Random seed for reproducibility
    """

    # Model configuration
    model_name_or_path: str = "unsloth/Qwen2.5-0.5B-Instruct"
    max_seq_length: int | None = 2048
    load_in_4bit: bool = False
    chat_template: str = "qwen2.5"
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
    output_dir: str = "/data2/peft/qwen25-lora-wildguard"
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

    # Other configuration
    validate_save: bool = True
    save_method: str = "lora"
    random_state: int = 42

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
            type=_parse_max_seq_length,
            default=4096,
            help="Maximum sequence length or 'none' for unlimited",
        )
        model_group.add_argument(
            "--load_in_4bit",
            action="store_true",
            dest="load_in_4bit",
            help="Use 4-bit quantization (QLoRA)",
            default=False,
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
                "For other models, specify manually based on your model's "
                "chat template format. "
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
                "For other models, specify manually based on your model's "
                "chat template format. "
                "(default: <|im_start|>assistant\\n for Qwen ChatML)"
            ),
        )

        # LoRA arguments
        lora_group = parser.add_argument_group("LoRA Configuration")
        lora_group.add_argument(
            "--lora_r",
            type=int,
            default=32,
            help="LoRA rank (default: 32)",
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
                "Gradient checkpointing option: 'unsloth' (use Unsloth's "
                "optimized checkpointing), 'True' (enable standard "
                "checkpointing), 'False' (disable checkpointing) "
                "(default: unsloth)"
            ),
        )

        # Training arguments
        train_group = parser.add_argument_group("Training Configuration")
        train_group.add_argument(
            "--output_dir",
            type=str,
            default="/data2/speftr/model",
            help=(
                "Output directory for saving model "
                "(default: /data2/speftr/model)"
            ),
        )
        train_group.add_argument(
            "--num_train_epochs",
            type=int,
            default=3,
            help="Number of training epochs (default: 3)",
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
            help="Evaluation strategy: no, steps, or epoch (default: epoch)",
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

    print(f"\n✓ Parameters saved to {output_path}")

    return output_path


def _parse_max_seq_length(value: str) -> int | None:
    """Parse ``--max_seq_length`` CLI input into an integer or ``None``.

    Args:
        value: Raw string supplied via the command line.

    Returns:
        Parsed integer length, or ``None`` if the user passed ``"none"``.
    """
    if value.lower() == "none":
        return None
    return int(value)


class PESFT:
    """Parameter-Efficient Supervised Fine-Tuning trainer.

    This class encapsulates the complete training pipeline for fine-tuning
    language models with LoRA adapters using TRL's SFTTrainer. It handles
    model loading, LoRA setup, training, evaluation, and model saving.

    The design follows HuggingFace's pattern of separating configuration
    (PESFTConfig) from training logic (PESFT class).

    Attributes:
        config: Training configuration
        model: Language model with LoRA adapters
        tokenizer: Tokenizer with chat template applied
        trainer: TRL SFTTrainer instance (after train() is called)

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
        # Unsloth patches several Transformer internals during import. We do
        # it first so later imports (transformers, trl, peft) see the patched
        # implementations and automatically benefit from the speedups.
        import unsloth  # noqa: PLC0415, I001
        from peft import PeftModel  # noqa: PLC0415, F401

        # Now import other libraries after unsloth
        from transformers import PreTrainedTokenizerBase, TrainingArguments  # noqa: PLC0415, F401
        from trl import SFTTrainer  # noqa: PLC0415, F401

        self.config = config
        self.model: PeftModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.trainer: SFTTrainer | None = None
        self.training_args: TrainingArguments | None = None

        self._unsloth_version = unsloth.__version__

        # Prepare output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        params_dict = display_parameters(self.config)
        save_parameters_to_json(params_dict, self.config.output_dir)

    def _validate_step_synchronization(self) -> None:
        """Validate eval/save cadence when both operate on steps.

        The Hugging Face Trainer expects ``save_steps`` to be a multiple of
        ``eval_steps`` whenever ``load_best_model_at_end`` is enabled. When
        that relationship is broken the "best" checkpoint can lag behind
        evaluation, so we emit an explicit warning for the user.
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
                f"⚠️  Warning: save_steps ({save_steps}) is not a multiple of "
                f"eval_steps ({eval_steps})."
            )
            print(
                "   This may cause issues with load_best_model_at_end=True "
                "since checkpoints won't be saved at evaluation points."
            )
            suggested = eval_steps * (save_steps // eval_steps + 1)
            print(
                "   Consider adjusting save_steps to be a multiple of "
                f"eval_steps (e.g., {suggested})."
            )
        else:
            print(
                f"✓ Step synchronization validated: save_steps ({save_steps}) "
                f"is a multiple of eval_steps ({eval_steps})"
            )

    def load_model(
        self,
    ) -> tuple[PeftModel, PreTrainedTokenizerBase]:
        """Load a base model with Unsloth and attach LoRA adapters.

        Returns:
            Tuple containing the PEFT-wrapped model and tokenizer. The
            tokenizer is augmented with the requested chat template so
            downstream formatting functions can rely on it.
        """
        # Import unsloth first to ensure optimizations are applied
        from unsloth import FastLanguageModel  # noqa: PLC0415, I001
        from unsloth.chat_templates import get_chat_template  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415, F401
        from transformers import PreTrainedTokenizerBase  # noqa: PLC0415, F401

        print(f"Loading model: {self.config.model_name_or_path}")

        # Load base model with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name_or_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
        )

        # Apply chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.config.chat_template,
        )

        # Add LoRA adapters
        # Convert string gradient checkpointing option to appropriate value.
        # Unsloth supports its own checkpointing mode, in addition to the
        # canonical True/False flags exposed by transformers.
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

    def _build_training_arguments(self) -> TrainingArguments:
        """Construct ``transformers.TrainingArguments`` for SFT.

        Returns:
            TrainingArguments populated from ``self.config``. Precision flags
            (bf16/fp16) are inferred from the local hardware so users do not
            have to remember the correct combination.
        """
        # Import unsloth first to ensure optimizations are applied
        from transformers import TrainingArguments  # noqa: PLC0415

        # Determine precision based on GPU capability
        # This applies to activations regardless of weight quantization
        # (weights can be 4-bit with load_in_4bit=True)
        use_bf16 = (
            torch.cuda.is_bf16_supported()
            if torch.cuda.is_available()
            else False
        )
        use_fp16 = not use_bf16 if torch.cuda.is_available() else False

        return TrainingArguments(
            output_dir=self.config.output_dir,
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
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.scheduler,
            warmup_steps=self.config.warmup_steps,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            bf16=use_bf16,
            fp16=use_fp16,
            bf16_full_eval=use_bf16,
            fp16_full_eval=use_fp16,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            group_by_length=True,
            report_to=self.config.report_to,
            load_best_model_at_end=True,
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
            eval_dataset: Optional evaluation dataset. Pass ``None`` to skip
                evaluation entirely.
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
            self.load_model()

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

        # Log critical eval settings to verify configuration
        print("\n=== Evaluation Configuration ===")
        print(
            "per_device_eval_batch_size: "
            f"{training_args.per_device_eval_batch_size}"
        )
        print(f"bf16: {training_args.bf16}")
        print(f"fp16: {training_args.fp16}")
        print(f"bf16_full_eval: {training_args.bf16_full_eval}")
        print(f"fp16_full_eval: {training_args.fp16_full_eval}")
        print("=" * 35)

        # Import unsloth first to ensure optimizations are applied
        from trl import SFTTrainer  # noqa: PLC0415, I001
        from unsloth.chat_templates import train_on_responses_only  # noqa: PLC0415

        # Initialize trainer
        # ``formatting_func`` keeps the dataset lightweight. TRL calls it
        # lazily so prompts are formatted only when a batch is assembled.
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=self.config.packing,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=self.config.max_seq_length,
            formatting_func=formatting_func,
        )
        self.trainer = trainer

        # Apply train_on_responses_only if enabled
        if self.config.train_on_responses:
            print(
                "\n🦥 Applying train_on_responses_only "
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
            print("   ✓ train_on_responses_only applied successfully")
        else:
            print(
                "\n📝 Training on full sequence "
                "(including instruction tokens)."
            )
            print(
                "   Pass --train_on_responses with matching template markers "
                "to mask instruction tokens during training."
            )

        # Train
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

        # Evaluate (only if eval_dataset provided)
        if eval_dataset is not None:
            print("\nEvaluating on test set...")
            try:
                eval_results = trainer.evaluate()
                print(f"Evaluation results: {eval_results}")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"⚠ Evaluation failed with OOM: {e}")
                print(
                    "  Model is already trained. "
                    "Evaluate separately if needed."
                )

    def _save_training_metadata(self) -> None:
        """Persist configuration and training arguments alongside artifacts.

        Saving these files directly next to the adapters makes it trivial to
        reproduce a run or understand its hyperparameters long after the
        training job finished.
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
        print(f"✓ Training arguments saved to {args_path}")

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

        Raises:
            ValueError: If the model/tokenizer have not been loaded yet.
        """
        if self.model is None or self.tokenizer is None:
            msg = "Model and tokenizer must be loaded before saving"
            raise ValueError(msg)

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
            print("✓ Merged model saved successfully")

        self._save_training_metadata()
