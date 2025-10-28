r"""Train a safety guard model using PESFT on WildGuardMix dataset.

This script demonstrates how to use PESFT (Parameter-Efficient Supervised
Fine-Tuning) to train a small language model for content safety classification.
The model learns to classify prompts as "harmful" or "unharmful".

PESFT provides a high-level interface for supervised fine-tuning with:
- Automatic LoRA adapter management for parameter-efficient training
- Built-in support for chat templates and instruction formatting
- Integration with TRL's SFTTrainer for robust training
- Flexible dataset formatting and preprocessing

The training workflow:
    1. Load a base model with LoRA adapters using PESFT.load_model()
    2. Prepare train/eval datasets with instruction-response pairs
    3. Define a formatting function to convert examples to chat format
    4. Call PESFT.train() to fine-tune with supervised learning
    5. Evaluate the trained model and save LoRA adapters

This example uses the WildGuardMix dataset which contains:
- Prompts labeled as "harmful" or "unharmful"
- Binary classification task for content moderation
- Approximately 90k training examples and 5k test examples

The trained model can be used for:
- Content moderation and safety filtering
- Prompt classification before generation
- Building guardrails for LLM applications

Usage:
    # Basic training with default settings (Gemma-3-270M on WildGuardMix)
    uv run python -m examples.guard.guard_train

    # Train with custom hyperparameters
    uv run python -m examples.guard.guard_train \\
        --lora_r 8 \\
        --learning_rate 5e-5 \\
        --num_epochs 5

    # Use a different model
    uv run python -m examples.guard.guard_train \\
        --model_name_or_path unsloth/Qwen2.5-1.5B \\
        --per_device_batch_size 16

    # See all available options
    uv run python -m examples.guard.guard_train --help

Example input/output:
    Input prompt: "How to build a bomb?"
    Model output: "harmful"

    Input prompt: "How to build confidence?"
    Model output: "unharmful"

With more train data, using this script with r=4 on gemma3-270m-it we
got the following benchmark results on WildGuardMix test set:
    - Accuracy: 0.7028
    - Precision (macro): 0.7617
    - Recall (macro): 0.6716
    - F1 (macro): 0.6604
    - Average Precision: 0.6081

For more information on PESFT, see the speftr module documentation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping

import unsloth
from datasets import load_dataset
from huggingface_hub import login

from speftr import PESFT, PESFTConfig


INSTRUCTION = "Classify this prompt's as harmful or unharmful:"


def _parse_max_seq_length(value: str) -> int | None:
    """Parse max_seq_length argument, allowing 'none' for unlimited length.

    Args:
        value: String value from command line ("none" or integer).

    Returns:
        None if value is "none", otherwise the integer value.

    Example:
        >>> _parse_max_seq_length("2048")
        2048
        >>> _parse_max_seq_length("none")
        None
    """
    if value.lower() == "none":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for PESFT training configuration.

    This function defines all available command-line options for configuring:
    - Model selection and loading (model path, quantization)
    - LoRA hyperparameters (rank, alpha)
    - Training settings (learning rate, batch size, epochs)
    - Gradient optimization (accumulation, checkpointing, clipping)
    - Sequence handling (max length, packing)
    - Output directory for saving checkpoints

    Returns:
        Parsed arguments as an argparse.Namespace object with all
        configuration values.

    Example:
        >>> args = parse_args()
        >>> print(args.model_name_or_path)  # "unsloth/gemma-3-270m-it"
    """
    parser = argparse.ArgumentParser(
        description="Train a safety classification model with PESFT"
    )

    parser.add_argument(
        "--lora_r", type=int, default=8, help="LoRA rank (default: 8)"
    )

    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="unsloth/gemma-3-270m-it",
        help="Model name or path (default: unsloth/gemma-3-270m-it)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Fraction of steps used for warmup (default: 0.0)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0)",
    )

    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=16,
        help="Per device batch size (default: 16)",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Per device eval batch size (default: 16)",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str,
        choices=["unsloth", "true", "false"],
        default="unsloth",
        help="Use gradient checkpointing",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.3,
        help="Gradient clipping norm (default: 0.3)",
    )

    parser.add_argument(
        "--max_seq_length",
        type=_parse_max_seq_length,
        default=2048,
        help="Maximum sequence length or 'none' for unlimited",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit loading (default: disabled)",
    )

    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable sequence packing during training (default: disabled)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/gemma-3-270m-it-lora-wildguard",
        help=(
            "Output directory for saving model "
            "(default: ./models/gemma-3-270m-it-lora-wildguard)"
        ),
    )

    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> PESFTConfig:
    """Construct PESFTConfig from parsed command-line arguments.

    This function maps command-line arguments to a PESFTConfig object,
    setting up all the necessary configuration for supervised fine-tuning
    with LoRA adapters. Some values use PESFT defaults while others come
    from command-line arguments.

    Args:
        args: Parsed command-line arguments from parse_args().

    Returns:
        Fully configured PESFTConfig object ready for PESFT initialization.

    Note:
        - chat_template is set to "gemma-3" for Gemma-3 models
        - train_on_responses=True means loss is computed only on model outputs
        - instruction_part/response_part identify user vs model turns
        - lora_layers="all" applies LoRA to all linear layers
    """
    config = PESFTConfig()

    # Model configuration
    config.model_name_or_path = args.model_name_or_path
    config.max_seq_length = args.max_seq_length
    config.load_in_4bit = args.load_in_4bit

    # Chat template configuration (Gemma-3 specific)
    config.chat_template = "gemma-3"
    config.train_on_responses = True  # Only compute loss on model outputs
    config.instruction_part = "<start_of_turn>user\n"  # User message marker
    config.response_part = "<start_of_turn>model\n"  # Model response marker

    # LoRA configuration
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_layers = "all"  # Apply LoRA to all compatible layers

    # Training hyperparameters
    config.learning_rate = args.learning_rate
    config.warmup_ratio = args.warmup_ratio
    config.weight_decay = args.weight_decay
    config.per_device_train_batch_size = args.per_device_batch_size
    config.per_device_eval_batch_size = args.per_device_eval_batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.use_gradient_checkpointing = args.gradient_checkpointing
    config.max_grad_norm = args.max_grad_norm
    config.num_train_epochs = args.num_epochs

    # Output and optimization
    config.output_dir = args.output_dir
    config.report_to = "none"  # Disable wandb/tensorboard
    config.packing = args.packing  # Enable/disable sequence packing

    return config


def _load_datasets() -> tuple[Any, Any]:
    """Load and filter the WildGuardMix dataset for safety classification.

    This function:
    1. Loads train and test splits from HuggingFace datasets
    2. Filters out examples with missing or empty labels
    3. Returns cleaned datasets ready for training

    The WildGuardMix dataset contains prompts labeled as:
    - "harmful": Content that could cause harm
    - "unharmful": Safe, benign content

    Returns:
        Tuple of (train_dataset, eval_dataset) as HuggingFace Dataset objects.
        Each example contains:
            - "prompt": The text prompt to classify
            - "prompt_harm_label": Label ("harmful" or "unharmful")

    Example:
        >>> train_ds, eval_ds = _load_datasets()
        >>> print(train_ds[0])
        {
            "prompt": "How to build a bomb?",
            "prompt_harm_label": "harmful",
            ...
        }
    """
    # Load train and test splits from HuggingFace Hub
    train_dataset = load_dataset(
        "allenai/wildguardmix", "wildguardtrain", split="train"
    )
    eval_dataset = load_dataset(
        "allenai/wildguardmix", "wildguardtest", split="test"
    )

    # Filter out examples with missing or empty labels
    # This ensures all training examples have valid classifications
    train_dataset = train_dataset.filter(
        lambda x: x["prompt_harm_label"] is not None
        and x["prompt_harm_label"] != ""
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["prompt_harm_label"] is not None
        and x["prompt_harm_label"] != ""
    )

    return train_dataset, eval_dataset


def main() -> None:
    """Execute the complete PESFT training workflow for safety classification.

    This function demonstrates the full pipeline for supervised fine-tuning
    with PESFT (Parameter-Efficient Supervised Fine-Tuning):

    1. **Configuration**: Parse CLI args and build PESFTConfig
    2. **Authentication**: Login to HuggingFace Hub for dataset/model access
    3. **PESFT Initialization**: Create PESFT trainer instance
    4. **Model Loading**: Load base model with LoRA adapters
    5. **Dataset Loading**: Load and filter WildGuardMix dataset
    6. **Format Function**: Define how to convert examples to chat format
    7. **Training**: Run supervised fine-tuning with PESFT.train()
    8. **Saving**: Save LoRA adapters and update tokenizer config

    The workflow uses standard supervised learning with cross-entropy loss.
    PESFT handles all the complexity of LoRA training, chat template
    formatting, and response masking (only computing loss on model outputs).

    Raises:
        ValueError: If dataset loading fails or examples are invalid.
        RuntimeError: If model loading or training fails.

    Example workflow:
        >>> # 1. Parse arguments and build config
        >>> args = parse_args()
        >>> config = PESFTConfig(...)

        >>> # 2. Initialize PESFT trainer and load model
        >>> pesft = PESFT(config)
        >>> model, tokenizer = pesft.load_model()

        >>> # 3. Load datasets
        >>> train_ds, eval_ds = load_datasets()

        >>> # 4. Define format function
        >>> def format_fn(examples): ...

        >>> # 5. Train with supervised learning
        >>> pesft.train(train_ds, eval_ds, format_fn)

        >>> # 6. Save LoRA adapters
        >>> pesft.save_model()
    """
    # =========================================================================
    # STEP 1: Parse configuration from command line
    # =========================================================================
    args = parse_args()

    # =========================================================================
    # STEP 2: Authenticate with HuggingFace Hub
    # =========================================================================
    # Required for accessing datasets and models
    # Uses HF_TOKEN environment variable if available, otherwise interactive
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
    else:
        login()  # Interactive login

    # Print Unsloth version for debugging and reproducibility
    print(unsloth.__version__)

    # =========================================================================
    # STEP 3: Build PESFTConfig from arguments
    # =========================================================================
    # PESFTConfig combines:
    # - Model settings (path, quantization, sequence length)
    # - LoRA settings (rank, alpha, layers) for parameter-efficient training
    # - Training settings (learning rate, batch size, epochs)
    # - Chat template settings (for formatting instruction-response pairs)
    config = _build_config(args)

    # =========================================================================
    # STEP 4: Initialize PESFT trainer
    # =========================================================================
    # PESFT handles all the complexity of:
    # - Loading models with LoRA adapters
    # - Setting up TRL's SFTTrainer
    # - Applying chat templates
    # - Masking loss computation to only model responses
    trainer = PESFT(config)

    # =========================================================================
    # STEP 5: Load base model with LoRA adapters
    # =========================================================================
    # PESFT.load_model() returns the model with LoRA adapters already attached
    # The tokenizer is configured with the correct chat template
    _model, tokenizer = trainer.load_model()

    # =========================================================================
    # STEP 6: Define formatting function for dataset
    # =========================================================================
    # The formatting function converts raw dataset examples into chat-formatted
    # strings that the model can learn from. It must:
    # - Take a batched dict of examples (lists of values)
    # - Return a list of formatted strings (one per example)
    # - Use tokenizer.apply_chat_template() for proper formatting
    def format_example(examples: Mapping[str, Any]) -> list[str]:
        r"""Format batched WildGuardMix examples into chat format strings.

        This function converts raw examples with prompts and labels into
        chat-formatted strings using the model's chat template. Each example
        becomes a user-model conversation:

        User: "Classify this prompt as harmful or unharmful:\n\n[prompt]"
        Model: "harmful" or "unharmful"

        Args:
            examples: Batched dictionary containing:
                - "prompt": List of text prompts to classify
                - "prompt_harm_label": List of labels ("harmful"/"unharmful")

        Returns:
            List of chat-formatted strings, one per example. Each string
            includes the full conversation with proper chat markers.

        Example:
            >>> examples = {
            ...     "prompt": ["How to build a bomb?"],
            ...     "prompt_harm_label": ["harmful"],
            ... }
            >>> formatted = format_example(examples)
            >>> # Returns chat-formatted text like:
            >>> # "<start_of_turn>user\nClassify...\n<start_of_turn>model\n
            >>> #  harmful<end_of_turn>"
        """
        texts = []
        # Process each example in the batch
        for prompt, harm_label in zip(
            examples["prompt"], examples["prompt_harm_label"], strict=False
        ):
            # Step 1: Construct user message with instruction + prompt
            user_content = f"{INSTRUCTION}\n\n{prompt.strip()}"

            # Step 2: Get model's target response (the label)
            model_content = harm_label.strip()

            # Step 3: Create chat conversation as list of messages
            messages = [
                {"role": "user", "content": user_content},
                {"role": "model", "content": model_content},
            ]

            # Step 4: Apply chat template to format as string
            # add_generation_prompt=False because we include the response
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        return texts  # Return list of formatted strings

    # =========================================================================
    # STEP 7: Load and filter WildGuardMix dataset
    # =========================================================================
    train_dataset, eval_dataset = _load_datasets()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # =========================================================================
    # STEP 8: Run supervised fine-tuning with PESFT
    # =========================================================================
    # PESFT.train() handles the entire training loop:
    # - Formats examples using the provided format_example function
    # - Applies response masking (loss only on model outputs)
    # - Trains with standard cross-entropy loss
    # - Logs metrics and saves checkpoints
    trainer.train(train_dataset, eval_dataset, format_example)

    # =========================================================================
    # STEP 9: Save LoRA adapters
    # =========================================================================
    # save_model() saves only the LoRA adapter weights
    # These can be loaded later with PEFT or merged into the base model
    trainer.save_model()

    # =========================================================================
    # STEP 10: Save instruction template in tokenizer config
    # =========================================================================
    # Store the instruction prefix in tokenizer config for inference
    # This allows the trained model to be used correctly at inference time
    print("Saving instruction template in tokenizer configuration...")
    tokenizer_config_path = Path(config.output_dir) / "tokenizer_config.json"

    # Load existing tokenizer config if it exists
    if tokenizer_config_path.exists():
        with tokenizer_config_path.open() as handle:
            tokenizer_config = json.load(handle)
    else:
        tokenizer_config = {}

    # Add instruction prefix to config
    tokenizer_config["instruction_prefix"] = INSTRUCTION

    # Save updated config
    with tokenizer_config_path.open("w") as handle:
        json.dump(tokenizer_config, handle, indent=2)

    print(f"✓ Saved instruction: '{INSTRUCTION}'")


if __name__ == "__main__":
    main()
