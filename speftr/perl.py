"""PERL: Parameter-Efficient Reinforcement Learning with LoRA.

This module provides a reusable class-based interface for training language
models with LoRA adapters using TRL's GRPOTrainer. The design follows
HuggingFace's pattern of separating configuration from training logic.

Based on "LoRA without Regret" (Schulman et al., 2025) which found:
- Very low ranks (even rank 1) work effectively for RL
- Optimal learning rate is ~10x higher than full fine-tuning
- Effective batch sizes should remain < 32
- Train for 1 epoch (RL requires minimal capacity)

Reference: https://huggingface.co/docs/trl/main/en/lora_without_regret

Example:
    Basic usage with default configuration:

    >>> config = PERLConfig()
    >>> trainer = PERL(config)
    >>> trainer.train(dataset, reward_funcs)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch


if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase
    from trl import GRPOConfig, GRPOTrainer


@dataclass
class PERLConfig:
    r"""Configuration for Parameter-Efficient Reinforcement Learning.

    This dataclass contains all hyperparameters needed for GRPO training
    with LoRA. Defaults based on "LoRA without Regret" (Schulman et al., 2025)
    which recommends: rank 1, learning rate 10x higher than full fine-tuning,
    effective batch sizes < 32, and 1 epoch of training.

    Attributes:
        Model Configuration:
            model_name_or_path: HuggingFace model name or local path
            max_seq_length: Maximum sequence length (None = no limit)
            load_in_4bit: Use 4-bit quantization
            fast_inference: Enable vLLM fast inference
            gpu_memory_utilization: GPU memory usage (0.0-1.0)

        LoRA Configuration (RL-optimized):
            lora_r: LoRA rank (default: 1, per "LoRA without Regret")
            lora_alpha: LoRA scaling factor (default: 32)
            target_modules: List of module names to apply LoRA
            use_gradient_checkpointing: Gradient checkpointing mode

        GRPO Training Configuration:
            output_dir: Directory for saving checkpoints
            num_train_epochs: Number of training epochs (default: 1)
            max_steps: Maximum training steps (-1 = use epochs)
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            gradient_accumulation_steps: Gradient accumulation (default: 1)
            learning_rate: Learning rate (default: 1e-5, 10x full FT)
            weight_decay: Weight decay coefficient
            scheduler: Learning rate scheduler (default: cosine)
            warmup_ratio: Fraction of training steps for warmup
            logging_steps: Log metrics every N steps
            save_strategy: When to save checkpoints ("epoch" or "steps")
            save_steps: Save checkpoint every N steps (if strategy="steps")
            optim: Optimizer name
            report_to: Experiment tracking backend

        Generation Configuration:
            temperature: Sampling temperature
            num_generations: Number of generations per prompt (default: 8)
            max_prompt_length: Maximum prompt length in tokens
            max_completion_length: Maximum completion length
            vllm_min_p: Min-p sampling parameter
            vllm_top_p: Top-p nucleus sampling
            vllm_top_k: Top-k sampling (-1 = disabled)

        Other Configuration:
            random_state: Random seed for reproducibility
    """

    # Model configuration
    model_name_or_path: str = "unsloth/Qwen3-4B-Base"
    max_seq_length: int | None = 2048
    load_in_4bit: bool = False
    use_vllm: bool = True  # Use vLLM for generation (faster)
    vllm_mode: str = "colocate"  # "colocate" or "server"
    vllm_gpu_memory_utilization: float = 0.2  # Low memory for vLLM
    vllm_enable_sleep_mode: bool = True  # Sleep vLLM during optimization

    # LoRA configuration (RL-optimized per "LoRA without Regret")
    lora_r: int = 1  # Very low rank works for RL!
    lora_alpha: int = 32  # Standard alpha
    use_gradient_checkpointing: str = "true"  # TRL-only mode
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # GRPO training configuration
    output_dir: str = "./models/perl-reasoning"
    num_train_epochs: int = 2  # Default: 2 epochs
    max_steps: int = -1  # -1 = use num_train_epochs
    per_device_train_batch_size: int = 1  # 1 prompt at a time
    per_device_eval_batch_size: int = 8  # Evaluation batch size
    gradient_accumulation_steps: int = 1  # Default single prompt per update
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_ratio: float = 0.0
    logging_steps: int = 1
    save_strategy: str = "epoch"  # Save at epoch boundaries
    save_steps: int | None = None  # Only used if save_strategy="steps"
    optim: str = "adamw_8bit"
    report_to: str = "none"

    # Generation configuration
    # 8 generations for GRPO quality, split into mini-batches for memory
    temperature: float = 1.0
    num_generations: int = 8  # Paper recommends 8-16
    max_prompt_length: int | None = None
    max_completion_length: int | None = None
    min_p: float | None = 0.0  # Minimum probability for sampling
    top_p: float = 0.95  # Nucleus sampling
    top_k: int | None = 20  # Top-k sampling (-1 or None = disabled)
    stop_sequences: list[str] | None = None
    include_stop_str_in_output: bool = True

    # Resume configuration
    resume_from_checkpoint: str | None = None

    # Other configuration
    random_state: int = 3407

    @classmethod
    def from_args(cls, args: argparse.Namespace | None = None) -> PERLConfig:
        """Create config from command-line arguments.

        Args:
            args: Parsed arguments (if None, will parse from sys.argv)

        Returns:
            PERLConfig instance with values from command line
        """
        if args is None:
            parser = cls.get_argument_parser()
            args = parser.parse_args()

        config_dict = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                config_dict[field_name] = getattr(args, field_name)

        return cls(**config_dict)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """Get argument parser with all PERL configuration options.

        Returns:
            ArgumentParser configured with PERL parameters
        """
        parser = argparse.ArgumentParser(
            description=(
                "Parameter-Efficient Reinforcement Learning with GRPO"
            )
        )

        # Model arguments
        model_group = parser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--model_name_or_path",
            type=str,
            default="unsloth/Qwen3-4B-Base",
            help="Model name or path (default: unsloth/Qwen3-4B-Base)",
        )
        model_group.add_argument(
            "--max_seq_length",
            type=int,
            default=2048,
            help="Maximum sequence length (default: 2048)",
        )

        # Training arguments
        train_group = parser.add_argument_group("Training Configuration")
        train_group.add_argument(
            "--output_dir",
            type=str,
            default="./models/perl-reasoning",
            help="Output directory (default: ./models/perl-reasoning)",
        )
        train_group.add_argument(
            "--max_steps",
            type=int,
            default=100,
            help="Maximum training steps (default: 100)",
        )
        train_group.add_argument(
            "--learning_rate",
            type=float,
            default=1e-5,
            help="Learning rate (default: 1e-5)",
        )

        return parser


class PERL:
    """Parameter-Efficient Reinforcement Learning trainer.

    This class encapsulates the complete GRPO training pipeline for
    fine-tuning language models with LoRA adapters. It handles model
    loading, LoRA setup, GRPO training, and model saving.

    Attributes:
        config: Training configuration
        model: Language model with LoRA adapters
        tokenizer: Tokenizer
        trainer: TRL GRPOTrainer instance (after train() is called)

    Example:
        >>> config = PERLConfig(
        ...     model_name_or_path="unsloth/Qwen3-4B-Base",
        ...     max_steps=100,
        ... )
        >>> trainer = PERL(config)
        >>> trainer.load_model()
        >>> trainer.train(dataset, reward_funcs)
        >>> trainer.save_model()
    """

    def __init__(self, config: PERLConfig) -> None:
        """Initialize the PERL trainer and echo the configuration.

        Args:
            config: Fully specified reinforcement-learning configuration.
        """
        # Import required modules
        from peft import LoraConfig, get_peft_model, PeftModel  # noqa: PLC0415, F401, I001
        from transformers import (  # noqa: PLC0415, F401
            AutoModelForCausalLM,
            AutoTokenizer,
            PreTrainedTokenizerBase,
        )
        from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415, F401

        self.config = config
        self.model: PeftModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.trainer: GRPOTrainer | None = None

        # Prepare output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        print("\nPERL Configuration:")
        for key, value in sorted(asdict(self.config).items()):
            print(f"  {key}: {value}")

    def set_pretrained_model(
        self, model: PeftModel, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """Set pre-trained model to continue training.

        Use this to continue training LoRA adapters from a previous stage
        (e.g., after supervised fine-tuning with PESFT) instead of loading
        a new model. This allows seamless continuation of the same LoRA
        adapters across different training stages.

        Args:
            model: Pre-trained model with LoRA adapters
            tokenizer: Tokenizer from the pre-training stage

        Example:
            >>> sft_trainer = PESFT(sft_config)
            >>> sft_trainer.train(...)
            >>> rl_trainer = PERL(rl_config)
            >>> rl_trainer.set_pretrained_model(
            ...     sft_trainer.model, sft_trainer.tokenizer
            ... )
            >>> rl_trainer.train(...)
        """
        self.model = model
        self.tokenizer = tokenizer
        print("\n✓ Using pre-trained model from previous training stage")

    def load_model(
        self,
    ) -> tuple[PeftModel, PreTrainedTokenizerBase]:
        """Load a merged model and attach RL-specific LoRA adapters.

        Returns:
            Tuple containing the PEFT-wrapped model and tokenizer.
        """
        # Import required modules
        from peft import LoraConfig, get_peft_model, PeftModel  # noqa: PLC0415, F401, I001
        from transformers import (  # noqa: PLC0415, F401
            AutoModelForCausalLM,
            AutoTokenizer,
            PreTrainedTokenizerBase,
        )

        print(f"\nLoading model: {self.config.model_name_or_path}")
        print("Using TRL-only mode (no Unsloth dependencies)")

        # Load merged model from disk
        print("  Loading merged model with transformers...")
        # Try flash_attention_2, fall back to sdpa if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            print("  Using Flash Attention 2")
        except (ImportError, ValueError):
            print("  Flash Attention 2 not available, using SDPA")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path
        )

        # Add LoRA adapters for RL training
        print(f"  Adding LoRA adapters (rank={self.config.lora_r})...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.0,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

        # Enable gradient checkpointing if requested (TRL-only)
        use_grad_ckpt = self.config.use_gradient_checkpointing.lower()
        if use_grad_ckpt == "true":
            # For standard HF models we must call both helpers to keep the
            # input graph differentiable while checkpoints are recomputed.
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")

        self.model = model
        self.tokenizer = tokenizer

        if (
            self.config.stop_sequences is None
            and tokenizer.eos_token is not None
        ):
            # Default to stopping on EOS so generations align with chat format
            # even when the user does not supply explicit stop strings.
            self.config.stop_sequences = [tokenizer.eos_token]

        print("✓ Model loaded successfully")
        return model, tokenizer

    def _calculate_max_lengths(self) -> tuple[int | None, int | None]:
        """Derive sensible prompt/completion limits when unspecified.

        Returns:
            Pair of integers (or ``None``) representing ``max_prompt_length``
            and ``max_completion_length`` to feed into GRPO.
        """
        max_prompt_length = self.config.max_prompt_length
        if max_prompt_length is None and self.config.max_seq_length:
            max_prompt_length = self.config.max_seq_length // 2

        max_completion_length = self.config.max_completion_length
        if (
            max_completion_length is None
            and self.config.max_seq_length
            and max_prompt_length
        ):
            max_completion_length = (
                self.config.max_seq_length - max_prompt_length
            )

        return max_prompt_length, max_completion_length

    def _print_grpo_config(self) -> None:
        """Log the GRPO hyperparameters relevant to sampling throughput."""
        grad_accum = self.config.gradient_accumulation_steps
        print("\nGRPO Configuration:")
        print(f"  num_generations: {self.config.num_generations}")
        print(f"  gradient_accumulation_steps: {grad_accum}")

    def _create_grpo_config(
        self, max_prompt_length: int | None, max_completion_length: int | None
    ) -> GRPOConfig:
        """Build a GRPOConfig with optional vLLM-specific arguments.

        Args:
            max_prompt_length: Maximum number of tokens to sample from the
                prompt portion of each example.
            max_completion_length: Maximum number of new tokens to generate
                per prompt.

        Returns:
            Initialised ``GRPOConfig`` ready to hand to ``GRPOTrainer``.
        """
        # Import required modules
        from trl import GRPOConfig  # noqa: PLC0415

        grad_accum = self.config.gradient_accumulation_steps

        grpo_kwargs = {
            "temperature": self.config.temperature,
            "min_p": self.config.min_p,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "lr_scheduler_type": self.config.scheduler,
            "optim": self.config.optim,
            "logging_steps": self.config.logging_steps,
            "per_device_train_batch_size": (
                self.config.per_device_train_batch_size
            ),
            "per_device_eval_batch_size": (
                self.config.per_device_eval_batch_size
            ),
            "gradient_accumulation_steps": grad_accum,
            "num_generations": self.config.num_generations,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "num_train_epochs": self.config.num_train_epochs,
            "max_steps": self.config.max_steps,
            "save_strategy": self.config.save_strategy,
            "save_steps": self.config.save_steps,
            "report_to": self.config.report_to,
            "output_dir": self.config.output_dir,
            "use_vllm": self.config.use_vllm,
            "resume_from_checkpoint": self.config.resume_from_checkpoint,
        }

        if self.config.use_vllm:
            print("\nvLLM Generation Enabled:")
            print(f"  mode: {self.config.vllm_mode}")
            print(f"  min_p: {self.config.min_p}")
            print(f"  top_p: {self.config.top_p}")
            print(f"  top_k: {self.config.top_k}")
            print(
                f"  gpu_memory_utilization: "
                f"{self.config.vllm_gpu_memory_utilization}"
            )
            print(f"  sleep_mode: {self.config.vllm_enable_sleep_mode}")
            print(f"  max_completion_length: {max_completion_length}")

            grpo_kwargs["vllm_mode"] = self.config.vllm_mode
            grpo_kwargs["vllm_gpu_memory_utilization"] = (
                self.config.vllm_gpu_memory_utilization
            )
            grpo_kwargs["vllm_enable_sleep_mode"] = (
                self.config.vllm_enable_sleep_mode
            )

            # CRITICAL: Set max_tokens for vLLM to respect
            # max_completion_length
            generation_kwargs: dict[str, Any] = {}
            if max_completion_length is not None:
                generation_kwargs["max_tokens"] = max_completion_length
            if self.config.stop_sequences is not None:
                generation_kwargs["stop"] = self.config.stop_sequences
                print(f"  stop_sequences: {self.config.stop_sequences}")
                if self.tokenizer is not None:
                    stop_token_ids: list[int] = []
                    for sequence in self.config.stop_sequences:
                        token_id = self.tokenizer.convert_tokens_to_ids(
                            sequence
                        )
                        if isinstance(token_id, int) and token_id >= 0:
                            stop_token_ids.append(token_id)
                    if stop_token_ids:
                        generation_kwargs["stop_token_ids"] = stop_token_ids
                        print(f"  stop_token_ids: {stop_token_ids}")
            # These kwargs are passed straight to vLLM so it mirrors HF's
            # sampling contract. Keeping them centralized avoids mismatches
            # between offline evaluation and online RL.
            generation_kwargs["include_stop_str_in_output"] = (
                self.config.include_stop_str_in_output
            )
            if generation_kwargs:
                grpo_kwargs["generation_kwargs"] = generation_kwargs
                print(f"  generation_kwargs: {generation_kwargs}")

        return GRPOConfig(**grpo_kwargs)

    def train(
        self,
        dataset: Dataset,
        reward_funcs: list[Callable[..., list[float]]],
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Run GRPO training with the provided reward functions.

        Args:
            dataset: Dataset containing prompts and any metadata needed by the
                reward functions.
            reward_funcs: Reward callables that accept the sampled completions
                and return per-sample scores.
            eval_dataset: Optional evaluation dataset for periodic validation.

        Returns:
            None. Progress is logged to stdout and the underlying model is
            updated in-place.
        """
        # Import required modules
        from trl import GRPOTrainer  # noqa: PLC0415

        if self.model is None or self.tokenizer is None:
            self.load_model()

        print(f"\nTraining dataset size: {len(dataset)}")
        if eval_dataset:
            print(f"Eval dataset size: {len(eval_dataset)}")

        # Calculate max lengths if not specified
        max_prompt_length, max_completion_length = (
            self._calculate_max_lengths()
        )

        # Print GRPO configuration
        self._print_grpo_config()

        # Create GRPO training arguments
        training_args = self._create_grpo_config(
            max_prompt_length, max_completion_length
        )

        # Initialize GRPO trainer
        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            # ``processing_class`` tells TRL how to tokenize prompts during
            # sampling; handing it the tokenizer keeps the pipeline fully
            # deterministic across training and evaluation.
            "processing_class": self.tokenizer,
            "reward_funcs": reward_funcs,
            "args": training_args,
            "train_dataset": dataset,
        }

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        mem_before_trainer = torch.cuda.memory_allocated() / 1024**3
        # Logging VRAM usage helps users see how much headroom GRPOTrainer
        # consumes before the actual training loop begins.
        print(
            f"\nGPU memory before GRPOTrainer init: "
            f"{mem_before_trainer:.2f} GB"
        )

        trainer = GRPOTrainer(**trainer_kwargs)
        self.trainer = trainer

        mem_after_trainer = torch.cuda.memory_allocated() / 1024**3
        print(
            f"GPU memory after GRPOTrainer init: "
            f"{mem_after_trainer:.2f} GB "
            f"(+{mem_after_trainer - mem_before_trainer:.2f} GB)"
        )

        # Train
        print("\nStarting GRPO training...")
        trainer.train()

        print("\nGRPO training complete!")

    def save_model(self, save_method: str = "lora") -> None:
        """Persist the policy (adapters or merged weights) to disk.

        Args:
            save_method: Strategy identifier such as ``"lora"`` or
                ``"merged_16bit"``.

        Raises:
            ValueError: If the model/tokenizer have not been initialised.
        """
        if self.model is None or self.tokenizer is None:
            msg = "Model and tokenizer must be loaded before saving"
            raise ValueError(msg)

        if save_method == "lora":
            print(f"\nSaving LoRA adapters to {self.config.output_dir}...")
            self.model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
        else:
            print(
                f"\nSaving merged model ({save_method}) "
                f"to {self.config.output_dir}..."
            )
            self.model.save_pretrained_merged(
                self.config.output_dir,
                self.tokenizer,
                save_method=save_method,
            )

        print("✓ Model saved successfully")
