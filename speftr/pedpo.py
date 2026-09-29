# SPDX-FileCopyrightText: 2026 Luis Rei
# SPDX-License-Identifier: BSD-2-Clause
"""PEDPO: Parameter-Efficient Direct Preference Optimization with LoRA.

Trains LoRA adapters on a causal language model with transformers, peft and
TRL's DPOTrainer (no Unsloth). ``PEDPOConfig`` holds the settings;
``PEDPO`` loads, trains and saves. Callers pass the preference dataset to
``PEDPO.train``.

Data: a ``datasets.Dataset`` with ``chosen`` and ``rejected`` columns and
optionally ``prompt``. Without ``prompt``, ``chosen`` and ``rejected`` are
full conversations and TRL takes their longest common prefix as the
prompt. Values are strings or lists of ``{"role", "content"}`` messages,
which TRL renders with the tokenizer's chat template.

Reference model: the policy as it is when training starts. TRL copies the
trainable adapter into a frozen adapter named ``"ref"`` and computes the
reference log-probabilities with it, so no second model is loaded. A fresh
adapter starts at zero, so the reference is then the base model.

Defaults: rank 1, since a preference pair carries about one bit, like an
RL episode ("LoRA Without Regret", Schulman et al., 2025); learning rate
5e-6, about 10x the ~5e-7 of full fine-tuning DPO, as in the Hugging Face
alignment-handbook QLoRA DPO recipe; ``beta`` 0.1 and the sigmoid loss of
the DPO paper (Rafailov et al., 2023).

Example:
    >>> trainer = PEDPO(PEDPOConfig(max_steps=100))
    >>> trainer.load_model()
    >>> trainer.train(preference_dataset)
    >>> trainer.save_model()
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch


if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    from trl import DPOConfig, DPOTrainer


def _checkpoint_model_class(model_name_or_path: str) -> type[Any]:
    """Return the model class a checkpoint declares in its config.

    ``AutoModelForCausalLM`` would pick a text-only class for some
    multimodal checkpoints (e.g. Qwen 3.5); the declared class keeps the
    parameter names of the checkpoint.

    Args:
        model_name_or_path: Hub id or local checkpoint path.

    Returns:
        The declared transformers class, or ``AutoModelForCausalLM`` when the
        config names none that transformers provides.
    """
    import transformers  # noqa: PLC0415

    # User-supplied id, often a local path: no Hub revision to pin (B615).
    config = transformers.AutoConfig.from_pretrained(  # nosec B615
        model_name_or_path
    )
    architectures = config.architectures or []
    model_class: type[Any] = transformers.AutoModelForCausalLM
    if architectures and hasattr(transformers, architectures[0]):
        model_class = getattr(transformers, architectures[0])
    return model_class


def _device_map() -> str | dict[str, int]:
    """Choose where ``from_pretrained`` places the base model.

    Returns:
        The whole model on this process's GPU (``{"": LOCAL_RANK}``) when
        ``WORLD_SIZE`` is above 1 (DDP via torchrun or ``accelerate
        launch``), since every rank trains a full copy; otherwise
        ``"auto"``, which uses one GPU or splits a model too large for one
        across all visible GPUs.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return {"": int(os.environ.get("LOCAL_RANK", "0"))}
    return "auto"


def _is_main_process() -> bool:
    """Tell whether this process should write files.

    torchrun and ``accelerate launch`` set ``RANK`` for every process;
    without a launcher it is unset and the single process is the main one.

    Returns:
        True for global rank 0 or when not launched distributed.
    """
    return os.environ.get("RANK", "0") == "0"


def _quantization_config(
    *, load_in_4bit: bool, load_in_8bit: bool
) -> BitsAndBytesConfig | None:
    """Return the bitsandbytes config, or ``None`` for bf16 loading.

    Args:
        load_in_4bit: Quantize the base weights to 4 bits (NF4, QLoRA).
        load_in_8bit: Quantize the base weights to 8 bits (LLM.int8).

    Returns:
        An 8-bit config; an NF4 double-quantized config computing in bf16;
        or ``None`` when neither quantization is requested.
    """
    if not (load_in_4bit or load_in_8bit):
        return None
    from transformers import BitsAndBytesConfig  # noqa: PLC0415

    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _language_model_targets(
    model: PreTrainedModel, module_names: list[str]
) -> list[str] | str:
    """Restrict LoRA targets to the text decoder of multimodal checkpoints.

    Multimodal checkpoints (e.g. Gemma 4) also use names like ``q_proj`` in
    their vision and audio towers, where LoRA is unwanted and peft may not
    support the layer types. Those models keep their decoder under
    ``language_model``.

    Args:
        model: Loaded base model.
        module_names: Leaf module names to adapt (``q_proj``, ...).

    Returns:
        ``module_names`` unchanged for text-only models, otherwise a peft
        regex matching those names inside ``language_model`` only.
    """
    has_language_model = any(
        ".language_model." in f".{name}." for name, _ in model.named_modules()
    )
    if not has_language_model:
        return module_names
    return rf".*\.language_model\..*\.({'|'.join(module_names)})"


@dataclass
class PEDPOConfig:
    """Configuration for Parameter-Efficient Direct Preference Optimization.

    Fields are grouped as model loading (``model_name_or_path`` to
    ``load_in_8bit``), LoRA (``lora_r`` to ``target_modules``), DPO
    (``beta`` to ``precompute_ref_batch_size``), training (``output_dir``
    to ``report_to``) and other settings.

    Attributes:
        model_name_or_path: HuggingFace model name or local path. DPO
            starts from an instruction-tuned (SFT) model with a chat
            template.
        max_length: Maximum tokens of prompt plus completion; longer
            sequences are truncated at the end. ``None`` disables
            truncation.
        load_in_4bit: Load the base model in 4-bit (bitsandbytes NF4,
            QLoRA).
        load_in_8bit: Load the base model in 8-bit (bitsandbytes
            LLM.int8). Slower than 4-bit; excludes ``load_in_4bit``.
        lora_r: LoRA rank (default: 1). A preference pair carries about
            one bit, like an RL episode, so rank 1 has the capacity for
            large preference sets; check with ``python -m
            speftr.lora_budget --mode dpo``.
        lora_alpha: LoRA scaling factor (default: 32).
        lora_dropout: Dropout on the LoRA input (default: 0.0).
        target_modules: Module names to apply LoRA to (default: all
            attention and MLP projections).
        beta: DPO temperature (default: 0.1). Higher keeps the policy
            closer to the reference.
        loss_type: TRL DPO loss names (default: ``["sigmoid"]``, the DPO
            paper's loss). Several names are combined with equal weights.
        precompute_ref_log_probs: Run the reference over the whole train
            (and eval) dataset once before training and cache its
            log-probabilities, so training steps run only the policy.
            With LoRA the reference is the same base weights with the
            ``"ref"`` adapter, so this saves the per-step reference
            forward pass, not a second model's memory; for one epoch the
            total reference compute is the same. Incompatible with
            ``use_liger_kernel`` and with streaming (``IterableDataset``)
            datasets.
        precompute_ref_batch_size: Batch size of the precompute pass
            (``None``: the train or eval batch size). It needs no
            gradients, so it can be larger than the training batch.
        use_gradient_checkpointing: ``"true"`` recomputes activations in
            the backward pass instead of storing them (less memory,
            slower); ``"false"`` stores them. Forwarded to TRL's
            ``gradient_checkpointing``.
        output_dir: Directory for checkpoints and the saved model.
        num_train_epochs: Number of training epochs (default: 1).
        max_steps: Maximum optimizer steps (-1 = use epochs).
        per_device_train_batch_size: Preference pairs per device per
            micro-batch (default: 4). Each pair runs two sequences.
        per_device_eval_batch_size: Pairs per device per eval batch.
        gradient_accumulation_steps: Micro-batches per optimizer step
            (default: 4, so 16 pairs per step on one GPU).
        learning_rate: Learning rate (default: 5e-6, about 10x full
            fine-tuning DPO).
        weight_decay: Weight decay coefficient.
        scheduler: Learning rate scheduler (default: constant).
        warmup_ratio: Fraction of training steps for warmup.
        logging_steps: Log metrics every N steps.
        eval_strategy: When to evaluate during training (``"no"``,
            ``"steps"``, ``"epoch"``); needs an eval dataset unless
            ``"no"``. ``train`` evaluates once after training whenever an
            eval dataset is given.
        eval_steps: Evaluate every N steps (when ``eval_strategy="steps"``).
        save_strategy: When to save checkpoints (``"no"``, ``"steps"``,
            ``"epoch"``).
        save_steps: Save a checkpoint every N steps (when
            ``save_strategy="steps"``).
        optim: Optimizer name (any transformers ``optim``).
        use_liger_kernel: Compute the DPO loss with Liger's fused kernel
            from the last hidden state, so the full logits are never
            materialized. Needs the ``liger`` extra. Incompatible with
            ``precompute_ref_log_probs`` and with LoRA on ``lm_head``.
        report_to: Experiment tracking backend.
        random_state: Random seed for reproducibility.
    """

    # Model configuration
    model_name_or_path: str = "allenai/OLMo-2-0425-1B-SFT"
    max_length: int | None = 1024
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # LoRA configuration
    lora_r: int = 1
    lora_alpha: int = 32
    lora_dropout: float = 0.0
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

    # DPO configuration
    beta: float = 0.1
    loss_type: list[str] = field(default_factory=lambda: ["sigmoid"])
    precompute_ref_log_probs: bool = False
    precompute_ref_batch_size: int | None = None

    # Training configuration
    use_gradient_checkpointing: str = "true"
    output_dir: str = "./models/pedpo"
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    scheduler: str = "constant"
    warmup_ratio: float = 0.0
    logging_steps: int = 1
    eval_strategy: str = "no"
    eval_steps: int | None = None
    save_strategy: str = "epoch"
    save_steps: int | None = None
    optim: str = "adamw_8bit"
    use_liger_kernel: bool = False
    report_to: str = "none"

    # Other configuration
    random_state: int = 3407

    def __post_init__(self) -> None:
        """Reject option combinations that cannot train.

        Returns:
            None. Validation only.

        Raises:
            ValueError: If both ``load_in_4bit`` and ``load_in_8bit`` are
                set, or ``use_liger_kernel`` is combined with
                ``precompute_ref_log_probs``.
        """
        if self.load_in_4bit and self.load_in_8bit:
            msg = "load_in_4bit and load_in_8bit are mutually exclusive"
            raise ValueError(msg)
        if self.use_liger_kernel and self.precompute_ref_log_probs:
            msg = (
                "use_liger_kernel and precompute_ref_log_probs are "
                "mutually exclusive"
            )
            raise ValueError(msg)

    @classmethod
    def from_args(cls, args: argparse.Namespace | None = None) -> PEDPOConfig:
        """Create a config from command line arguments.

        Args:
            args: Parsed arguments; ``None`` parses ``sys.argv``.

        Returns:
            Config with the parsed values; fields without a flag keep
            their defaults.
        """
        if args is None:
            args = cls.get_argument_parser().parse_args()
        config_dict = {
            name: getattr(args, name)
            for name in cls.__dataclass_fields__
            if hasattr(args, name)
        }
        return cls(**config_dict)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """Get an argument parser with the PEDPO options.

        Returns:
            Parser whose defaults are the ``PEDPOConfig`` defaults.
        """
        defaults = PEDPOConfig()
        parser = argparse.ArgumentParser(
            description="Parameter-Efficient Direct Preference Optimization"
        )
        group = parser.add_argument_group("Model and Training Configuration")
        group.add_argument(
            "--model_name_or_path",
            type=str,
            default=defaults.model_name_or_path,
            help=(
                f"Model name or path (default: {defaults.model_name_or_path})"
            ),
        )
        group.add_argument(
            "--output_dir",
            type=str,
            default=defaults.output_dir,
            help=f"Output directory (default: {defaults.output_dir})",
        )
        group.add_argument(
            "--num_train_epochs",
            type=int,
            default=defaults.num_train_epochs,
            help=f"Training epochs (default: {defaults.num_train_epochs})",
        )
        group.add_argument(
            "--max_steps",
            type=int,
            default=defaults.max_steps,
            help="Maximum training steps; -1 trains num_train_epochs",
        )
        group.add_argument(
            "--learning_rate",
            type=float,
            default=defaults.learning_rate,
            help=f"Learning rate (default: {defaults.learning_rate})",
        )
        group.add_argument(
            "--lora_r",
            type=int,
            default=defaults.lora_r,
            help=f"LoRA rank (default: {defaults.lora_r})",
        )
        PEDPOConfig._add_dpo_arguments(parser, defaults)
        PEDPOConfig._add_memory_arguments(parser, defaults)
        return parser

    @staticmethod
    def _add_dpo_arguments(
        parser: argparse.ArgumentParser, defaults: PEDPOConfig
    ) -> None:
        """Add the DPO loss, length and reference options.

        Args:
            parser: Parser to extend in place.
            defaults: Default config supplying the flag defaults.

        Returns:
            None. The options are added to ``parser``.
        """
        group = parser.add_argument_group("DPO Configuration")
        group.add_argument(
            "--beta",
            type=float,
            default=defaults.beta,
            help=f"DPO temperature (default: {defaults.beta})",
        )
        group.add_argument(
            "--loss_type",
            nargs="+",
            default=defaults.loss_type,
            help=f"TRL DPO loss name(s) (default: {defaults.loss_type})",
        )
        group.add_argument(
            "--max_length",
            type=int,
            default=defaults.max_length,
            help=(
                "Maximum prompt + completion tokens (default: "
                f"{defaults.max_length})"
            ),
        )
        group.add_argument(
            "--precompute_ref_log_probs",
            action="store_true",
            help="Compute reference log-probs once before training",
        )
        group.add_argument(
            "--precompute_ref_batch_size",
            type=int,
            default=defaults.precompute_ref_batch_size,
            help="Batch size of the precompute pass (default: train batch)",
        )

    @staticmethod
    def _add_memory_arguments(
        parser: argparse.ArgumentParser, defaults: PEDPOConfig
    ) -> None:
        """Add the memory and batch options.

        Args:
            parser: Parser to extend in place.
            defaults: Default config supplying the flag defaults.

        Returns:
            None. The options are added to ``parser``.
        """
        group = parser.add_argument_group("Memory and Batch Configuration")
        group.add_argument(
            "--load_in_4bit",
            action="store_true",
            help="Load the base model in 4-bit (QLoRA)",
        )
        group.add_argument(
            "--load_in_8bit",
            action="store_true",
            help="Load the base model in 8-bit",
        )
        group.add_argument(
            "--optim",
            type=str,
            default=defaults.optim,
            help=f"Optimizer (default: {defaults.optim})",
        )
        group.add_argument(
            "--per_device_train_batch_size",
            type=int,
            default=defaults.per_device_train_batch_size,
            help=(
                "Preference pairs per device per micro-batch (default: "
                f"{defaults.per_device_train_batch_size})"
            ),
        )
        group.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=defaults.gradient_accumulation_steps,
            help=(
                "Micro-batches per optimizer step (default: "
                f"{defaults.gradient_accumulation_steps})"
            ),
        )
        group.add_argument(
            "--use_gradient_checkpointing",
            choices=["true", "false"],
            default=defaults.use_gradient_checkpointing,
            help=(
                "Recompute activations to save memory (default: "
                f"{defaults.use_gradient_checkpointing})"
            ),
        )
        group.add_argument(
            "--use_liger_kernel",
            action="store_true",
            help="Fused Liger DPO loss; needs the liger extra",
        )


class PEDPO:
    """Train LoRA adapters on a causal LM with DPO.

    Loads a model with transformers and peft and attaches LoRA adapters
    (or continues from ``set_pretrained_model``), trains with TRL's
    DPOTrainer, and saves the adapters or a merged model.

    ``model`` and ``tokenizer`` are ``None`` until ``load_model`` or
    ``set_pretrained_model`` runs; ``trainer`` is ``None`` until ``train``
    runs. After ``train`` the model also holds TRL's frozen ``"ref"``
    adapter; ``save_model`` writes only the trained one.

    Example:
        >>> trainer = PEDPO(PEDPOConfig(max_steps=100))
        >>> trainer.load_model()
        >>> trainer.train(train_dataset, eval_dataset)
        >>> trainer.save_model()
    """

    def __init__(self, config: PEDPOConfig) -> None:
        """Create the output directory and echo the configuration.

        Args:
            config: Fully specified DPO configuration.
        """
        self.config = config
        self.model: PeftModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.trainer: DPOTrainer | None = None

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        print("\nPEDPO Configuration:")
        for key, value in sorted(asdict(self.config).items()):
            print(f"  {key}: {value}")

    def set_pretrained_model(
        self, model: PeftModel, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """Continue training the LoRA adapter of an existing model.

        Use this after supervised fine-tuning (e.g. with PESFT) instead of
        ``load_model``. DPO then updates the same adapter, and the
        reference is the model as passed here: TRL copies the adapter
        (which must be named ``"default"``, as peft names the first one)
        into a frozen ``"ref"`` adapter. ``lora_r`` and
        ``target_modules`` are ignored.

        Args:
            model: Model with a trainable LoRA adapter named ``"default"``.
            tokenizer: Tokenizer of that model, with a chat template for
                conversational data.

        Returns:
            None. The model and tokenizer are stored on the instance.
        """
        self.model = model
        self.tokenizer = tokenizer
        print("\nContinuing from the model passed to set_pretrained_model")

    def load_model(self) -> tuple[PeftModel, PreTrainedTokenizerBase]:
        """Load the checkpoint and attach new LoRA adapters.

        TRL sets the tokenizer's pad token to EOS if it has none and pads
        the preference batches itself, so the tokenizer is used as loaded.

        Returns:
            The PEFT-wrapped model and the tokenizer.
        """
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        print(f"\nLoading model: {self.config.model_name_or_path}")
        model_class = _checkpoint_model_class(self.config.model_name_or_path)
        load_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": _device_map(),
            "quantization_config": _quantization_config(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
            ),
        }
        # User-supplied id, often a local path: no Hub revision to pin.
        try:
            model = model_class.from_pretrained(  # nosec B615
                self.config.model_name_or_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
            print("  Using Flash Attention 2")
        except (ImportError, ValueError):
            print("  Flash Attention 2 not available, using SDPA")
            model = model_class.from_pretrained(  # nosec B615
                self.config.model_name_or_path,
                attn_implementation="sdpa",
                **load_kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            self.config.model_name_or_path
        )

        print(f"  Adding LoRA adapters (rank={self.config.lora_r})...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=_language_model_targets(
                model, self.config.target_modules
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora_config)
        self.tokenizer = tokenizer
        print("Model loaded.")
        return self.model, tokenizer

    def _create_dpo_config(self) -> DPOConfig:
        """Build the TRL ``DPOConfig`` from this configuration.

        Returns:
            Training arguments for ``DPOTrainer``.
        """
        from trl import DPOConfig  # noqa: PLC0415

        config = self.config
        # A dict, as TRL types the optional step counts as plain floats.
        dpo_kwargs: dict[str, Any] = {
            "output_dir": config.output_dir,
            "max_length": config.max_length,
            "beta": config.beta,
            "loss_type": config.loss_type,
            "precompute_ref_log_probs": config.precompute_ref_log_probs,
            "precompute_ref_batch_size": config.precompute_ref_batch_size,
            # DPOTrainer enables checkpointing (and the input gradients a
            # PEFT model needs for it) on the model when this is set.
            "gradient_checkpointing": (
                config.use_gradient_checkpointing.lower() == "true"
            ),
            "num_train_epochs": config.num_train_epochs,
            "max_steps": config.max_steps,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "per_device_eval_batch_size": config.per_device_eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "lr_scheduler_type": config.scheduler,
            # A float in [0, 1) is read as a fraction of total steps.
            "warmup_steps": config.warmup_ratio,
            "logging_steps": config.logging_steps,
            "eval_strategy": config.eval_strategy,
            "eval_steps": config.eval_steps,
            "save_strategy": config.save_strategy,
            "save_steps": config.save_steps,
            "optim": config.optim,
            "use_liger_kernel": config.use_liger_kernel,
            "report_to": config.report_to,
            "seed": config.random_state,
        }
        return DPOConfig(**dpo_kwargs)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        *,
        resume_from_checkpoint: str | bool | None = None,
    ) -> None:
        """Run DPO training, then evaluate once if an eval set is given.

        Calls ``load_model`` first if no model is set.

        Args:
            train_dataset: Preference pairs: ``chosen`` and ``rejected``
                columns, plus ``prompt`` unless both hold the full
                conversation (see the module docstring).
            eval_dataset: Optional preference pairs in the same format,
                evaluated after training and, per ``eval_strategy``,
                during it.
            resume_from_checkpoint: Checkpoint directory, ``True`` for the
                latest checkpoint in ``output_dir``, or ``None`` to start
                fresh.

        Returns:
            None. The model is updated in place.
        """
        from trl import DPOTrainer  # noqa: PLC0415

        if self.model is None or self.tokenizer is None:
            model, tokenizer = self.load_model()
        else:
            model, tokenizer = self.model, self.tokenizer

        print(f"\nTraining dataset size: {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"Eval dataset size: {len(eval_dataset)}")

        # ref_model=None: TRL computes reference log-probs with a frozen
        # copy of the adapter. peft_config=None: the model is already a
        # PeftModel.
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=self._create_dpo_config(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        self.trainer = trainer

        print("\nStarting DPO training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("\nDPO training complete.")

        if eval_dataset is not None:
            self._run_final_evaluation(trainer)

    @staticmethod
    def _run_final_evaluation(trainer: DPOTrainer) -> None:
        """Evaluate the trained model and print the metrics.

        Args:
            trainer: The trainer, after ``train``.

        Returns:
            None. The metrics, or the out-of-memory error, are printed.
        """
        print("\nEvaluating...")
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

    def save_model(self, save_method: str = "lora") -> None:
        """Write the trained adapter or the merged model and the tokenizer.

        ``"lora"`` saves only the active (trained) adapter, not TRL's
        ``"ref"`` copy. ``"merged_16bit"`` folds the active adapter into
        the bf16 base weights with ``merge_and_unload``, which replaces
        ``self.model`` with the plain merged model, so call it last. Under
        a distributed launch only the main process writes.

        Args:
            save_method: ``"lora"`` (adapter only) or ``"merged_16bit"``.

        Returns:
            None. Files are written to ``config.output_dir``.

        Raises:
            ValueError: If no model is set or ``save_method`` is not
                supported.
        """
        if self.model is None or self.tokenizer is None:
            msg = "Model and tokenizer must be loaded before saving"
            raise ValueError(msg)
        if save_method not in {"lora", "merged_16bit"}:
            msg = (
                f"Unsupported save_method {save_method!r}; "
                "use 'lora' or 'merged_16bit'"
            )
            raise ValueError(msg)
        if not _is_main_process():
            return

        output_dir = self.config.output_dir
        if save_method == "lora":
            print(f"\nSaving LoRA adapter to {output_dir}...")
            self.model.save_pretrained(
                output_dir,
                selected_adapters=[cast("str", self.model.active_adapter)],
            )
        else:
            print(f"\nSaving merged 16-bit model to {output_dir}...")
            merged = self.model.merge_and_unload()
            merged.save_pretrained(output_dir)
            self.model = merged
        self.tokenizer.save_pretrained(output_dir)
        print("Model saved.")
