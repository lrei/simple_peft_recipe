r"""Train an instruction-following model using PESFT on Dolly Pirate dataset.

This script demonstrates how to use PESFT (Parameter-Efficient Supervised
Fine-Tuning) to train a language model for general instruction-following tasks.
The model learns to respond helpfully to diverse user instructions.

PESFT provides a high-level interface for supervised fine-tuning with:
- Automatic LoRA adapter management for parameter-efficient training
- Built-in support for chat templates and multi-turn conversations
- Integration with TRL's SFTTrainer for robust training
- Flexible dataset formatting and preprocessing
- Optional sequence packing for efficient training

The training workflow:
    1. Load a base model with LoRA adapters using PESFT.load_model()
    2. Prepare train/eval datasets with conversation examples
    3. Define a formatting function to convert examples to chat format
    4. Call PESFT.train() to fine-tune with supervised learning
    5. Evaluate the trained model and save LoRA adapters

This example uses the TeeZee/dolly-15k-pirate-speech dataset which contains:
- Instruction-following examples rewritten in a playful pirate dialect
- Multi-turn conversations grounded in the Dolly 15k prompts
- High-quality human-curated instructions with stylized responses
- Approximately 15k training examples with consistent persona voice
  (this script automatically reserves 5% for evaluation).

The trained model can be used for:
- General instruction following and task completion
- Conversational assistants and chatbots
- Domain-specific question answering
- Multi-turn dialogue systems

Usage:
    # Basic training with default settings (Qwen3-0.6B on Dolly Pirate)
    uv run python -m examples.instruct.instruct

    # Train with custom hyperparameters
    uv run python -m examples.instruct.instruct \\
        --lora_r 8 \\
        --learning_rate 5e-5 \\
        --num_epochs 2

    # Override the default pirate system prompt
    uv run python -m examples.instruct.instruct \\
        --system_prompt "You are a courteous deckhand."

    # Use a different model with sequence packing
    uv run python -m examples.instruct.instruct \\
        --model_name_or_path meta-llama/Llama-3-8B \\
        --packing \\
        --per_device_batch_size 4

    # See all available options
    uv run python -m examples.instruct.instruct --help

Example input/output:
    User: "Explain quantum computing in simple terms"
    Model: "Quantum computing is a type of computing that uses quantum bits..."

    User: "Write a Python function to calculate factorial"
    Model: "Here's a Python function that calculates factorial."

For more information on PESFT, see the speftr module documentation.

Based on:
https://www.kaggle.com/code/viratchauhan/qwen-2-5-4-bit-q-3b-finetune-with-unsloth-w-b
"""

from __future__ import annotations

import argparse
import os
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    TypedDict,
    cast,
    runtime_checkable,
)


# protobuf >= 4 raises when transformers' generated descriptors are imported.
# Force the pure-Python fallback so Unsloth's chat-template helper works even
# when the environment ships the newer protobuf wheels.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import unsloth
from datasets import load_dataset
from huggingface_hub import login

from speftr import PESFT, PESFTConfig


DATASET_NAME = "TeeZee/dolly-15k-pirate-speech"
DEFAULT_MODEL = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
DEFAULT_OUTPUT_DIR = "./models/speftr-instruction"
DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a pirate, always respond in pirate speech."
)


@runtime_checkable
class SupportsGenerationConfig(Protocol):
    """Protocol for models exposing a generation_config attribute."""

    generation_config: object


@runtime_checkable
class SupportsThinkingToggle(Protocol):
    """Protocol for configs exposing an enable_thinking flag."""

    enable_thinking: bool


@runtime_checkable
class SupportsDefaultSystemPrompt(Protocol):
    """Protocol for tokenizers with a default_system_prompt attribute."""

    default_system_prompt: str


def _login_to_hf() -> None:
    """Authenticate with HuggingFace Hub using environment defaults."""
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
        return
    login()


def _normalize_system_prompt(raw_prompt: str) -> str:
    """Normalize system prompt input and fall back to the default."""
    prompt = raw_prompt.strip()
    if not prompt:
        return DEFAULT_SYSTEM_PROMPT
    return prompt


def _disable_thinking_mode(model: object) -> None:
    """Disable thinking mode when the model supports it."""
    if not isinstance(model, SupportsGenerationConfig):
        return
    generation_config = model.generation_config
    if isinstance(generation_config, SupportsThinkingToggle):
        generation_config.enable_thinking = False


def _set_default_system_prompt(tokenizer: object, system_prompt: str) -> None:
    """Assign the default system prompt when supported by the tokenizer."""
    if isinstance(tokenizer, SupportsDefaultSystemPrompt):
        tokenizer.default_system_prompt = system_prompt


def _normalize_column(value: object) -> list[str]:
    """Normalize a dataset column into a list of strings."""
    if isinstance(value, list):
        return [str(item) if item is not None else "" for item in value]
    if value is None:
        return []
    return [str(value)]


def _build_conversations_from_columns(
    batch: Mapping[str, object],
) -> list[list[dict[str, str]]]:
    """Construct conversations from instruction/context/response columns."""
    instructions = _normalize_column(batch.get("instruction"))
    contexts = _normalize_column(batch.get("context"))
    responses = _normalize_column(batch.get("response"))

    num_examples = max(
        len(instructions),
        len(contexts),
        len(responses),
    )
    if num_examples == 0:
        num_examples = 1

    conversations: list[list[dict[str, str]]] = []
    for index in range(num_examples):
        instruction = (
            instructions[index] if index < len(instructions) else ""
        ).strip()
        context = (
            contexts[index] if index < len(contexts) else ""
        ).strip()
        response = (
            responses[index] if index < len(responses) else ""
        ).strip()

        user_parts: list[str] = []
        if instruction:
            user_parts.append(instruction)
        if context:
            if instruction:
                user_parts.append(f"Context:\n{context}")
            else:
                user_parts.append(context)

        if user_parts:
            user_content = "\n\n".join(user_parts)
        else:
            user_content = "Ahoy matey, respond to this prompt."

        conversation: list[dict[str, str]] = [
            {"role": "user", "content": user_content},
        ]
        if response:
            conversation.append({"role": "assistant", "content": response})

        conversations.append(conversation)

    return conversations


def _extract_conversations(
    batch: Mapping[str, object],
) -> list[list[dict[str, str]]]:
    """Derive conversations from a batch, preferring prebuilt chat messages."""
    raw_messages = batch.get("messages", [])

    if isinstance(raw_messages, dict) or (
        isinstance(raw_messages, list)
        and raw_messages
        and isinstance(raw_messages[0], dict)
    ):
        single = cast("list[dict[str, str]]", raw_messages)
        return [single]

    if isinstance(raw_messages, list) and raw_messages:
        return cast("list[list[dict[str, str]]]", raw_messages)

    return _build_conversations_from_columns(batch)


def _inject_system_prompt(
    messages: list[dict[str, str]],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Insert or replace the leading system prompt in a conversation."""
    if messages and messages[0].get("role") == "system":
        return [
            {"role": "system", "content": system_prompt},
            *messages[1:],
        ]
    return [
        {"role": "system", "content": system_prompt},
        *messages,
    ]


def _format_conversations(
    conversations: list[list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
) -> list[str]:
    """Apply the chat template to prepared conversations."""
    texts: list[str] = []
    for messages in conversations:
        formatted_messages = _inject_system_prompt(messages, system_prompt)
        formatted_output = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(formatted_output, str):
            msg = "Chat template must produce string output."
            raise TypeError(msg)
        texts.append(formatted_output)
    return texts


def _build_format_batch(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
) -> Callable[[Mapping[str, object]], list[str]]:
    """Create a formatting function bound to tokenizer and system prompt."""

    def format_batch(batch: Mapping[str, object]) -> list[str]:
        conversations = _extract_conversations(batch)
        return _format_conversations(conversations, tokenizer, system_prompt)

    return format_batch


class ChatMessage(TypedDict):
    """Single conversational turn."""

    role: str
    content: str


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


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
    """Parse command line arguments for PESFT instruction-tuning configuration.

    This function defines all available command-line options for configuring:
    - Model selection and loading (model path, quantization)
    - Sequence handling (max length, chat template)
    - LoRA hyperparameters (rank, alpha)
    - Training settings (learning rate, batch size, epochs, scheduler)
    - Optimization (weight decay, gradient clipping, warmup)
    - Dataset configuration (eval split ratio, random seed)
    - Output options (save directory, save method, packing)

    Returns:
        Parsed arguments as an argparse.Namespace object with all
        configuration values.

    Example:
        >>> args = parse_args()
        >>> print(
        ...     args.model_name_or_path
        ... )  # "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an instruction-following model with PESFT on the "
            "TeeZee/dolly-15k-pirate-speech dataset."
        )
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL,
        help=(f"Base model to load. Defaults to {DEFAULT_MODEL}."),
    )
    parser.add_argument(
        "--max_seq_length",
        type=_parse_max_seq_length,
        default=2048,
        help="Maximum sequence length or 'none' for unlimited.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit loading (default: disabled).",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="chatml",
        help="Chat template to apply (default: chatml).",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help=(
            "System prompt inserted as the first conversation turn "
            "(default: pirate instructions)."
        ),
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (default: 8).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["constant", "constant_with_warmup", "linear", "cosine"],
        default="constant",
        help=(
            "Learning rate scheduler: 'constant', 'constant_with_warmup', "
            "'linear', or 'cosine' (default: constant)."
        ),
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Portion of training steps used for warmup (default: 0.0).",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Per-device train batch size (default: 8).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Per-device eval batch size (default: 8).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.3,
        help="Gradient clipping norm (default: 0.3).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for checkpoints and artifacts (default: "
            f"{DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--save_method",
        type=str,
        default="lora",
        help=(
            "Model save strategy: 'lora' adapters or merged variants such as "
            "'merged_16bit'and 'merged_4bit'."
        ),
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable sequence packing (default: disabled).",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.05,
        help="Fraction of examples reserved for evaluation (default: 0.05).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed for dataset splitting (default: 0).",
    )

    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> PESFTConfig:
    """Construct PESFTConfig from parsed command-line arguments.

    This function maps command-line arguments to a PESFTConfig object,
    setting up all the necessary configuration for instruction-following
    supervised fine-tuning with LoRA adapters.

    Args:
        args: Parsed command-line arguments from parse_args().

    Returns:
        Fully configured PESFTConfig object ready for PESFT initialization.

    Note:
        - chat_template="chatml" uses the ChatML format (OpenAI-style)
        - train_on_responses=True means loss is computed only on
          assistant outputs
        - instruction_part/response_part identify user vs assistant turns
        - save_method controls whether to save LoRA adapters or merged
          weights
    """
    config = PESFTConfig()

    # Model configuration
    config.model_name_or_path = args.model_name_or_path
    config.max_seq_length = args.max_seq_length
    config.load_in_4bit = args.load_in_4bit

    # Chat template configuration (ChatML format)
    config.chat_template = args.chat_template
    config.train_on_responses = True  # Only compute loss on assistant outputs
    config.instruction_part = "<|im_start|>user\n"  # User message marker
    config.response_part = (
        "<|im_start|>assistant\n"  # Assistant response marker
    )

    # LoRA configuration
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha

    # Training hyperparameters
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.scheduler = args.scheduler
    config.warmup_ratio = args.warmup_ratio
    config.per_device_train_batch_size = args.per_device_batch_size
    config.per_device_eval_batch_size = args.per_device_eval_batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.max_grad_norm = args.max_grad_norm
    config.num_train_epochs = args.num_epochs

    # Output and optimization
    config.output_dir = args.output_dir
    config.save_method = args.save_method  # "lora", "merged_16bit", etc.
    config.packing = args.packing  # Enable/disable sequence packing
    config.random_state = args.random_state
    config.report_to = "none"  # Disable wandb/tensorboard

    return config


def _load_datasets(eval_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    """Load Dolly Pirate dataset and create train/eval split.

    This function:
    1. Loads the TeeZee/dolly-15k-pirate-speech dataset from HuggingFace
    2. Randomly splits into train and eval sets using the specified ratio
    3. Returns both datasets ready for training

    The dataset remixes Dolly 15k instructions into pirate-flavored speech
    while keeping the original conversational structure for alignment tasks.

    Args:
        eval_ratio: Fraction of data to reserve for evaluation (0.0 to 1.0).
            For example, 0.05 means 5% eval, 95% train.
        seed: Random seed for reproducible train/eval splits.

    Returns:
        Tuple of (train_dataset, eval_dataset) as HuggingFace Dataset objects.
        Each example contains the columns "instruction", "context",
        and "response" which are later converted into chat-style messages.

    Raises:
        ValueError: If eval_ratio is not between 0 and 1 (exclusive).

    Example:
        >>> train_ds, eval_ds = _load_datasets(eval_ratio=0.05, seed=42)
        >>> print(train_ds[0]["messages"])
        [
            {"role": "user", "content": "Explain..."},
            {"role": "assistant", "content": "Sure..."}
        ]
    """
    # Validate eval_ratio is in valid range
    if not 0 < eval_ratio < 1:
        msg = "eval_ratio must be between 0 and 1 (exclusive)."
        raise ValueError(msg)

    # Load full dataset from HuggingFace Hub
    dataset = cast("Dataset", load_dataset(DATASET_NAME, split="train"))

    # Split into train and eval with specified ratio
    # test_size=eval_ratio means eval gets that fraction, train gets the rest
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def main() -> None:
    """Execute the complete PESFT training workflow for instruction-following.

    This function demonstrates the full pipeline for instruction-following
    supervised fine-tuning with PESFT (Parameter-Efficient Supervised
    Fine-Tuning):

    1. **Configuration**: Parse CLI args and build PESFTConfig
    2. **Authentication**: Login to HuggingFace Hub for dataset/model access
    3. **PESFT Initialization**: Create PESFT trainer instance
    4. **Model Loading**: Load base model with LoRA adapters
    5. **Format Function**: Define how to convert conversation examples to text
    6. **Dataset Loading**: Load and split Dolly Pirate dataset
    7. **Training**: Run supervised fine-tuning with PESFT.train()
    8. **Saving**: Save LoRA adapters to disk

    The workflow uses standard supervised learning with cross-entropy loss
    on multi-turn conversations. PESFT handles the complexity of LoRA
    training, chat template formatting, and response masking.

    Raises:
        ValueError: If dataset loading fails or eval_ratio is invalid.
        RuntimeError: If model loading or training fails.

    Example workflow:
        >>> # 1. Parse arguments and build config
        >>> args = parse_args()
        >>> config = PESFTConfig(...)

        >>> # 2. Initialize PESFT trainer and load model
        >>> pesft = PESFT(config)
        >>> model, tokenizer = pesft.load_model()

        >>> # 3. Define format function
        >>> def format_fn(batch): ...

        >>> # 4. Load datasets
        >>> train_ds, eval_ds = load_datasets()

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
    # Required for accessing gated models and datasets
    # Uses HF_TOKEN environment variable if available, otherwise
    # interactive login
    _login_to_hf()

    # Print Unsloth version for debugging and reproducibility
    print(unsloth.__version__)

    # =========================================================================
    # STEP 3: Build PESFTConfig from arguments
    # =========================================================================
    # PESFTConfig combines:
    # - Model settings (path, quantization, sequence length)
    # - LoRA settings (rank, alpha) for parameter-efficient training
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
    # - Masking loss computation to only assistant responses
    trainer = PESFT(config)

    # =========================================================================
    # STEP 5: Load base model with LoRA adapters
    # =========================================================================
    # PESFT.load_model() returns the model with LoRA adapters already attached
    # The tokenizer is configured with the correct chat template
    model, tokenizer = trainer.load_model()

    _disable_thinking_mode(model)
    system_prompt = _normalize_system_prompt(args.system_prompt)
    _set_default_system_prompt(tokenizer, system_prompt)

    # =========================================================================
    # STEP 6: Define formatting function for multi-turn conversations
    # =========================================================================
    # The formatting function converts raw dataset rows (either conversation
    # lists or instruction/response columns) into chat-formatted strings.
    format_batch = _build_format_batch(tokenizer, system_prompt)

    # =========================================================================
    # STEP 7: Load and split Dolly Pirate dataset
    # =========================================================================
    train_dataset, eval_dataset = _load_datasets(
        eval_ratio=args.eval_ratio,
        seed=config.random_state,
    )

    preview_data = train_dataset[:1]
    preview_texts = format_batch(preview_data)
    if preview_texts:
        preview_header = (
            "Preview: formatted conversation sent to the model"
        )
        print(preview_header)
        print("-" * len(preview_header))
        print(preview_texts[0])
        print("-" * len(preview_header))

    # =========================================================================
    # STEP 8: Run supervised fine-tuning with PESFT
    # =========================================================================
    # PESFT.train() handles the entire training loop:
    # - Formats examples using the provided format_batch function
    # - Applies response masking (loss only on assistant outputs)
    # - Trains with standard cross-entropy loss
    # - Logs metrics and saves checkpoints
    # - Optionally uses sequence packing for efficiency
    trainer.train(train_dataset, eval_dataset, format_batch)

    # =========================================================================
    # STEP 9: Save LoRA adapters
    # =========================================================================
    # save_model() saves LoRA adapters according to config.save_method:
    # - "lora": Save only adapter weights (most storage-efficient)
    # - "merged_16bit": Save merged model in FP16
    # - "merged_4bit": Save merged model quantized to 4-bit
    trainer.save_model()


if __name__ == "__main__":
    main()
