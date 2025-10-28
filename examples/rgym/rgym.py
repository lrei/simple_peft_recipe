r"""GRPO training for Reasoning Gym using PERL (Parameter-Efficient RL).

This script demonstrates how to use the PERL (Parameter-Efficient
Reinforcement Learning) class to fine-tune language models on reasoning
tasks using Group Relative Policy Optimization (GRPO) with LoRA adapters.

PERL provides a high-level interface for reinforcement learning with:
- Automatic LoRA adapter management for parameter-efficient training
- Integration with vLLM for fast generation during training
- Support for multiple reward functions (accuracy, format, etc.)
- Built-in evaluation and checkpointing

The training workflow:
    1. Load a base model with LoRA adapters using PERL.load_model()
    2. Prepare training/eval datasets with prompts and ground truth
    3. Define reward functions that score model completions (0.0 to 1.0)
    4. Call PERL.train() to optimize the model with GRPO
    5. Evaluate the trained model and save LoRA adapters

This example uses Reasoning Gym datasets for mathematical reasoning
tasks (e.g., chain_sum, spell_backward, sudoku) and trains models to:
- Generate structured reasoning with <think> and <answer> tags
- Produce correct answers that match expected solutions
- Use proper XML formatting for reasoning traces

Usage:
    # Basic training with default settings (Qwen3-1.7B on chain_sum)
    uv run python -m examples.rgym.rgym

    # Train on a different dataset with custom hyperparameters
    uv run python -m examples.rgym.rgym \\
        --dataset spell_backward \\
        --lora_r 8 \\
        --learning_rate 1e-5 \\
        --max_steps 1000

    # Use a different model with vLLM optimization
    uv run python -m examples.rgym.rgym \\
        --model_name_or_path Qwen/Qwen2.5-3B \\
        --vllm_gpu_memory_utilization 0.7 \\
        --vllm_sleep

    # See all available options
    uv run python -m examples.rgym.rgym --help

Example output format:
    The model learns to generate responses like:
        <think>
        Let me solve this step by step...
        [reasoning steps]
        </think>
        <answer>42</answer>

For more information on PERL and GRPO training, see the speftr.perl module.
"""

import argparse
import logging
import os
import re
import sys
from typing import Any

import reasoning_gym
import torch
import transformers
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, set_seed

from speftr.perl import PERL, PERLConfig


QUIET_TRANSFORMERS_LOGGERS = [
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "transformers.tokenization_utils",
]
# Disable vLLM verbose logging (sleep/wake messages, etc.)
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"


class ReasoningGymDataset(Dataset):
    """PyTorch Dataset wrapper for Reasoning Gym procedural datasets.

    This class adapts Reasoning Gym's ProceduralDataset interface to work
    with PERL training by:
    - Converting questions into chat-formatted prompts using tokenizer
    - Adding system prompts to guide reasoning behavior
    - Enabling thinking mode for structured reasoning traces
    - Preserving ground truth items for reward computation

    The dataset returns dictionaries with:
        - "prompt": Formatted string ready for model generation
        - "item": Original Reasoning Gym item with question/answer/metadata

    Attributes:
        tokenizer: HuggingFace tokenizer for chat template formatting
        data: Underlying ProceduralDataset from Reasoning Gym
        developer_prompt: System prompt text (e.g., instructions for reasoning)
        developer_role: Role name for system message (e.g., "system")
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        procedural_dataset: ProceduralDataset,
        developer_prompt: str,
        developer_role: str | None = None,
    ) -> None:
        """Initialize the Reasoning Gym dataset wrapper.

        Args:
            tokenizer: HuggingFace tokenizer with chat template support.
                Used to format conversations into model-specific prompts.
            procedural_dataset: Reasoning Gym ProceduralDataset instance.
                Provides questions, answers, and scoring logic.
            developer_prompt: System prompt instructing the model how to
                reason (e.g., "Think step by step before answering").
            developer_role: Chat role for system message (e.g., "system",
                "developer"). If None, no system message is added.
        """
        self.tokenizer = tokenizer
        self.data = procedural_dataset
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role

    def __len__(self) -> int:
        """Return the number of examples in the dataset.

        Returns:
            Total number of reasoning problems in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a formatted training example at the given index.

        This method:
        1. Retrieves the raw item from Reasoning Gym (question + metadata)
        2. Constructs a chat conversation with optional system prompt
        3. Formats the conversation using the tokenizer's chat template
        4. Enables thinking mode for structured reasoning generation

        Args:
            idx: Index of the example to retrieve.

        Returns:
            Dictionary containing:
                - "prompt" (str): Formatted prompt string ready for generation
                - "item" (dict): Original Reasoning Gym item for scoring
        """
        # Step 1: Get the raw question and metadata from Reasoning Gym
        item = self.data[idx]
        question = item["question"]

        # Step 2: Build chat conversation (system prompt + user question)
        chat = []
        if self.developer_role is not None:
            # Add system message to guide reasoning behavior
            chat.append(
                {"role": self.developer_role, "content": self.developer_prompt}
            )
        chat.append({"role": "user", "content": question})

        # Step 3: Format chat using model-specific template
        # enable_thinking=True adds special tokens for <think> tags
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return {"prompt": prompt, "item": item}


def accuracy_reward(
    completions: list[str],
    train_dataset: ReasoningGymDataset,
    **kwargs: Any,
) -> list[float]:
    """Compute accuracy reward by scoring model answers against ground truth.

    This reward function is the primary signal for GRPO training. It:
    1. Extracts the answer from each completion's <answer> tags
    2. Scores each extracted answer using Reasoning Gym's scoring logic
    3. Returns binary rewards (1.0 for correct, 0.0 for incorrect)

    The reward drives the model to generate correct solutions to reasoning
    problems. GRPO uses these rewards to update the policy by comparing
    rewards within each group of generations.

    Args:
        completions: List of model-generated completion strings. Each should
            contain reasoning in <think> tags and an answer in <answer> tags.
        train_dataset: ReasoningGymDataset instance providing ground truth
            and scoring logic via its internal ProceduralDataset.
        **kwargs: Additional arguments passed from PERL. Must contain:
            - "item": List of ground truth items from Reasoning Gym, one per
                completion. Each item is a dict with "question", "answer", etc.

    Returns:
        List of float rewards, one per completion. Each reward is:
            - 1.0 if the extracted answer matches the ground truth
            - 0.0 if the answer is incorrect or cannot be extracted

    Raises:
        ValueError: If "item" is missing from kwargs or length mismatch.
        TypeError: If items are not dictionaries.

    Example:
        >>> completions = ["<think>2+2=4</think><answer>4</answer>"]
        >>> items = [{"question": "2+2", "answer": "4"}]
        >>> rewards = accuracy_reward(completions, dataset, item=items)
        >>> # rewards = [1.0] because answer is correct
    """
    # Validate that ground truth items were provided
    if "item" not in kwargs:
        msg = (
            "The 'item' argument must be provided to compute accuracy reward."
        )
        raise ValueError(msg)
    if len(kwargs["item"]) != len(completions):
        msg = "Items and completions must have the same length."
        raise ValueError(msg)
    if not all(isinstance(item, dict) for item in kwargs["item"]):
        msg = "Each item must be a dictionary."
        raise TypeError(msg)

    # Extract answers from completions (looks for content in <answer> tags)
    answers = [extract_answer(c) for c in completions]

    # Score each extracted answer against its ground truth item
    # Reasoning Gym's score_answer returns 1.0 for correct, 0.0 otherwise
    return [
        train_dataset.data.score_answer(answer, item)
        for answer, item in zip(answers, kwargs["item"], strict=False)
    ]


def format_reward(completions: list[str], **_kwargs: Any) -> list[float]:
    """Reward proper use of structured reasoning tags.

    This auxiliary reward encourages the model to use the expected format:
        <think>reasoning steps here</think>
        <answer>final answer</answer>

    Each completion gets 0.25 points for each tag present (<think>,
    </think>, <answer>, </answer>), for a maximum of 1.0. This provides
    a shaping reward that guides the model toward proper formatting even
    when answers are incorrect.

    GRPO combines this with accuracy_reward by summing rewards from all
    reward functions. This helps the model learn both correctness AND
    proper structure.

    Args:
        completions: List of model-generated completion strings.
        **_kwargs: Unused keyword arguments (for compatibility with PERL).

    Returns:
        List of float rewards between 0.0 and 1.0, one per completion.
            - 1.0 = All four tags present (perfect formatting)
            - 0.75 = Three tags present
            - 0.5 = Two tags present
            - 0.25 = One tag present
            - 0.0 = No tags present

    Example:
        >>> completions = ["<think>reasoning</think><answer>42</answer>"]
        >>> rewards = format_reward(completions)
        >>> # rewards = [1.0] because all 4 tags are present
    """

    def count_tags(text: str) -> float:
        """Count reasoning tags in completion and return fractional reward."""
        count = 0.0
        # Check for opening <think> tag
        if re.search(r"\s*<think>\s*", text):
            count += 0.25
        # Check for closing </think> tag
        if re.search(r"\s*</think>\s*", text):
            count += 0.25
        # Check for opening <answer> tag
        if re.search(r"\s*<answer>\s*", text):
            count += 0.25
        # Check for closing </answer> tag
        if re.search(r"\s*</answer>\s*", text):
            count += 0.25
        return count

    return [count_tags(c) for c in completions]


class RewardLogger:
    """Wrapper for reward functions that logs example generations periodically.

    During GRPO training, the reward function is called many times (once per
    batch of generations). This class wraps the accuracy_reward function to
    periodically log example completions, making it easy to monitor training
    progress and debug reward computation.

    Every N reward calls, it logs:
    - The full completion text (reasoning + answer)
    - The extracted answer from <answer> tags
    - The expected/ground truth answer
    - Whether the answer was correct and the reward value

    This is useful for:
    - Verifying reward computation is working correctly
    - Monitoring model behavior during training
    - Debugging issues with answer extraction or scoring

    Attributes:
        train_dataset: ReasoningGymDataset for scoring answers
        logging_steps: Log an example every N reward function calls
        call_count: Number of times the reward function has been called
        logger: Python logger for outputting examples
    """

    def __init__(
        self, train_dataset: ReasoningGymDataset, logging_steps: int
    ) -> None:
        """Initialize the reward logger.

        Args:
            train_dataset: Training dataset with ground truth and scoring.
            logging_steps: Print an example every N reward computations.
                For example, logging_steps=50 logs on calls 50, 100, 150, etc.
        """
        self.train_dataset = train_dataset
        self.logging_steps = logging_steps
        self.call_count = 0
        self.logger = logging.getLogger(__name__)

    def accuracy_reward_with_logging(
        self, completions: list[str], **kwargs: Any
    ) -> list[float]:
        """Compute accuracy rewards and log the first example periodically.

        This method:
        1. Increments the call counter
        2. Computes rewards using the standard accuracy_reward function
        3. Every N calls, logs the first completion in the batch for inspection
        4. Returns the computed rewards unchanged

        Args:
            completions: List of model-generated completion strings.
            **kwargs: Additional arguments including "item" with ground truth.

        Returns:
            List of accuracy rewards (1.0 for correct, 0.0 for incorrect).
        """
        # Increment counter for tracking how many times rewards were computed
        self.call_count += 1

        # Compute rewards using the standard accuracy function
        rewards = accuracy_reward(completions, self.train_dataset, **kwargs)

        # Log the first example from this batch every N steps
        if self.call_count % self.logging_steps == 0 and len(completions) > 0:
            # Extract example data (first item in the batch)
            item = kwargs.get("item", [{}])[0] if "item" in kwargs else {}
            completion = completions[0]
            reward = rewards[0]
            expected = item.get("answer", "N/A")
            extracted = extract_answer(completion)

            # Determine if answer was correct (accuracy is binary 0 or 1)
            accuracy_threshold = 0.5
            is_correct = reward > accuracy_threshold

            # Log the example with clear formatting
            self.logger.info("=" * 80)
            self.logger.info("Example Generation (step %d):", self.call_count)
            self.logger.info("-" * 80)
            self.logger.info("Completion: %s", completion)
            self.logger.info("-" * 40)
            self.logger.info("Extracted Answer: %s", extracted)
            self.logger.info("Expected Answer: %s", expected)
            self.logger.info("Correct: %s (reward: %.2f)", is_correct, reward)
            self.logger.info("=" * 80)

        return rewards


def prepare_datasets(  # noqa: PLR0913
    dataset_specs: dict[str, dict],
    train_size: int,
    eval_size: int,
    tokenizer: PreTrainedTokenizerBase,
    developer_prompt: str,
    developer_role: str,
) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Create training and evaluation datasets from Reasoning Gym.

    This function:
    1. Converts dataset specifications into DatasetSpec objects
    2. Creates procedural datasets using Reasoning Gym (with different seeds)
    3. Wraps them in ReasoningGymDataset for PERL compatibility
    4. Returns both train and eval datasets ready for training

    The datasets use different random seeds to ensure train/eval separation.
    Each dataset generates problems procedurally, so every example is unique.

    Args:
        dataset_specs: Dictionary mapping dataset names to configuration dicts.
            Example: {"chain_sum": {"weight": 1.0, "config": {}}}
            Weights control sampling probability for composite datasets.
        train_size: Number of training examples to generate.
        eval_size: Number of evaluation examples to generate.
        tokenizer: HuggingFace tokenizer for formatting prompts.
        developer_prompt: System prompt text guiding reasoning behavior.
        developer_role: Chat role for system messages (e.g., "system").

    Returns:
        Tuple of (train_dataset, eval_dataset) as ReasoningGymDataset objects,
        ready to be passed to PERL.train().

    Example:
        >>> specs = {"chain_sum": {"weight": 1.0}}
        >>> train_ds, eval_ds = prepare_datasets(
        ...     dataset_specs=specs,
        ...     train_size=10000,
        ...     eval_size=256,
        ...     tokenizer=tokenizer,
        ...     developer_prompt="Think step by step",
        ...     developer_role="system",
        ... )
        >>> # Returns datasets with 10k training and 256 eval examples
    """
    # Step 1: Convert dict specs to DatasetSpec objects for Reasoning Gym
    specs = [
        DatasetSpec(
            name=name,
            weight=config.get("weight", 1.0),
            config=config.get("config", {}),
        )
        for name, config in dataset_specs.items()
    ]

    # Step 2: Create procedural datasets with different seeds for train/eval
    # Seed 1 = training data, Seed 2 = evaluation data
    train_data = reasoning_gym.create_dataset(
        "composite", seed=1, size=train_size, datasets=specs
    )
    eval_data = reasoning_gym.create_dataset(
        "composite", seed=2, size=eval_size, datasets=specs
    )

    # Step 3: Wrap in ReasoningGymDataset for prompt formatting and PERL compat
    train_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=train_data,
        developer_prompt=developer_prompt,
        developer_role=developer_role,
    )
    eval_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=eval_data,
        developer_prompt=developer_prompt,
        developer_role=developer_role,
    )

    return train_dataset, eval_dataset


def evaluate_model(  # noqa: PLR0913
    model,  # noqa: ANN001
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset: ReasoningGymDataset,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    batch_size: int = 16,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """Evaluate model on evaluation dataset with batched generation.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        min_p: Minimum token probability
        batch_size: Batch size for generation (default: 16)
        logger: Optional logger for progress

    Returns:
        Dictionary with evaluation metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    sample_count = len(eval_dataset)
    logger.info("=" * 80)
    logger.info(
        "Evaluating on %d samples (batch_size=%d)",
        sample_count,
        batch_size,
    )
    logger.info("-" * 80)

    model.eval()
    correct = 0
    total = 0
    accuracy_threshold = 0.5

    # Process in batches
    for batch_start in range(0, sample_count, batch_size):
        batch_end = min(batch_start + batch_size, sample_count)
        batch_prompts = []
        batch_items = []

        # Collect batch
        for i in range(batch_start, batch_end):
            example = eval_dataset[i]
            batch_prompts.append(example["prompt"])
            batch_items.append(example["item"])

        # Tokenize batch with left-padding for decoder-only models
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        # Ensure pad token is set for proper left-padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        tokenizer.padding_side = original_padding_side

        # Generate completions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and score
        for j, (output, item) in enumerate(
            zip(outputs, batch_items, strict=False)
        ):
            # Decode completion (skip input tokens)
            prompt_length = inputs.input_ids[j].shape[0]
            completion = tokenizer.decode(
                output[prompt_length:], skip_special_tokens=True
            )
            extracted_answer = extract_answer(completion)

            # Score answer
            score = eval_dataset.data.score_answer(extracted_answer, item)
            if score > accuracy_threshold:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("Evaluation Results:")
    logger.info("  Accuracy: %.2f%% (%d/%d)", accuracy * 100, correct, total)
    logger.info("=" * 80)

    return {"accuracy": accuracy, "correct": correct, "total": total}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for PERL training configuration.

    This function defines all available command-line options for configuring:
    - Model selection and loading (model path, quantization)
    - Dataset configuration (type, size, system prompts)
    - LoRA hyperparameters (rank, alpha)
    - Training settings (learning rate, batch size, steps)
    - GRPO parameters (num_generations, temperature, sampling)
    - vLLM configuration (GPU memory, sleep mode)
    - Logging and checkpointing options

    Returns:
        Parsed arguments as an `argparse.Namespace` with all config values.

    Example:
        >>> args = parse_args()  # Uses defaults if no CLI args provided
        >>> print(args.model_name_or_path)  # "Qwen/Qwen3-1.7B"
    """
    parser = argparse.ArgumentParser(
        description="GRPO training for Reasoning Gym with PERL"
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name or path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10000,
        help="Training dataset size (default: 10000)",
    )
    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        default=256,
        help="Evaluation dataset size (default: 256)",
    )
    parser.add_argument(
        "--developer_prompt",
        type=str,
        default="DeepSeekZero",
        help="System prompt key from SYSTEM_PROMPTS (default: DeepSeekZero)",
    )
    parser.add_argument(
        "--developer_role",
        type=str,
        default="system",
        help="Role for system prompt (default: system)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="chain_sum",
        help="Reasoning gym dataset to use (default: chain_sum)",
    )

    # LoRA configuration
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank for RL (default: 8)",
    )

    # Training configuration
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["constant", "constant_with_warmup", "linear", "cosine"],
        default="constant",
        help=(
            "Learning rate scheduler: constant, constant_with_warmup, "
            "linear, or cosine (default: constant)"
        ),
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum training steps (default: 100)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Per-device batch size (default: 8)",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Per-device evaluation batch size (default: 64)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,
        help="Number of generations per prompt (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (default: 0.95)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling cutoff (default: 20)",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Minimum token probability (default: 0.0)",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum prompt length (default: 512)",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=512,
        help="Maximum completion length (default: 512)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging steps (default: 50)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help=(
            "Save checkpoint every N steps when using TRL's step-based "
            "checkpointing (default: disabled)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/sudoku-perl",
        help="Output directory (default: ./models/sudoku-perl)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint path (default: None)",
    )

    # vLLM configuration
    parser.add_argument(
        "--no_vllm",
        action="store_false",
        dest="use_vllm",
        help="Disable vLLM for generation (vLLM enabled by default)",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization for vLLM (default: 0.5)",
    )
    parser.add_argument(
        "--vllm_sleep",
        action="store_true",
        dest="vllm_enable_sleep_mode",
        help="Enable vLLM sleep mode (default: disabled)",
    )
    parser.set_defaults(use_vllm=True, vllm_enable_sleep_mode=False)

    # Logging configuration
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Experiment tracking backend (default: none)",
    )

    # Other
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


def setup_logging() -> logging.Logger:
    """Configure logging for the training script.

    Sets up:
    - Formatted logging with timestamps to stdout
    - INFO level for main script and transformers library
    - ERROR level for noisy transformers sub-loggers (tokenization, config)

    This ensures clean, readable logs during training while suppressing
    unnecessary verbosity from library internals.

    Returns:
        Configured logger instance for this module.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    # Configure transformers logging
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Silence noisy sub-loggers that clutter output
    for noisy_logger in QUIET_TRANSFORMERS_LOGGERS:
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    return logger


def print_config(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Log the training configuration for reference.

    Prints all important hyperparameters and settings to help with:
    - Reproducibility (record exact settings used)
    - Debugging (verify parameters are as expected)
    - Comparison (track changes across runs)

    Args:
        args: Parsed command-line arguments with all configuration.
        logger: Logger instance for output.
    """
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("-" * 80)
    logger.info("Model: %s", args.model_name_or_path)
    logger.info("Dataset: %s (size=%d)", args.dataset, args.dataset_size)
    logger.info("Eval dataset size: %d", args.eval_dataset_size)
    logger.info("LoRA rank: %d", args.lora_r)
    logger.info("Learning rate: %s", args.learning_rate)
    logger.info("Scheduler: %s", args.scheduler)
    logger.info("Batch size: %d", args.per_device_train_batch_size)
    logger.info(
        "Eval batch size: %d",
        args.per_device_eval_batch_size,
    )
    logger.info(
        "Save steps: %s",
        "disabled" if args.save_steps is None else args.save_steps,
    )
    logger.info("Gradient accumulation: %d", args.gradient_accumulation_steps)
    logger.info(
        "Effective batch size: %d",
        args.per_device_train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("Num generations: %d", args.num_generations)
    logger.info("Temperature: %s", args.temperature)
    logger.info("Top-p: %s", args.top_p)
    logger.info("Top-k: %s", args.top_k)
    logger.info("Min-p: %s", args.min_p)
    logger.info("Max steps: %d", args.max_steps)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("=" * 80)


def print_system_prompt(prompt_key: str, logger: logging.Logger) -> str:
    """Look up and log the system prompt used for reasoning guidance.

    System prompts guide the model's reasoning behavior. Reasoning Gym
    provides several pre-defined prompts (e.g., "DeepSeekZero", "QwQ")
    that have been shown to improve reasoning performance.

    Args:
        prompt_key: Key for SYSTEM_PROMPTS dict (e.g., "DeepSeekZero").
        logger: Logger instance for output.

    Returns:
        The full system prompt text string.

    Raises:
        KeyError: If prompt_key is not found in SYSTEM_PROMPTS.
    """
    developer_prompt = SYSTEM_PROMPTS[prompt_key]
    logger.info("=" * 80)
    logger.info("System Prompt (%s):", prompt_key)
    logger.info("-" * 80)
    logger.info("%s", developer_prompt)
    logger.info("=" * 80)
    return developer_prompt


def main() -> None:
    """Execute the complete PERL training workflow for reasoning tasks.

    This function demonstrates the full pipeline for reinforcement learning
    with PERL (Parameter-Efficient Reinforcement Learning):

    1. **Configuration**: Parse CLI args and set random seed
    2. **Setup**: Configure logging for clean output
    3. **PERL Initialization**: Create PERLConfig and PERL trainer
    4. **Model Loading**: Load base model with LoRA adapters
    5. **Dataset Preparation**: Create Reasoning Gym datasets with prompts
    6. **Baseline Evaluation**: Measure base model performance
    7. **Reward Definition**: Set up reward functions for GRPO
    8. **Training**: Run GRPO training with PERL.train()
    9. **Final Evaluation**: Measure post-training performance
    10. **Saving**: Save LoRA adapters to disk

    The workflow uses GRPO (Group Relative Policy Optimization) to optimize
    the model policy based on reward signals from multiple generations per
    prompt. PERL handles all the complexity of LoRA training, vLLM integration,
    and GRPO optimization automatically.

    Raises:
        KeyError: If system prompt key is invalid.
        ValueError: If configuration parameters are incompatible.
        RuntimeError: If model loading or training fails.

    Example workflow:
        >>> # 1. Parse arguments (from CLI or defaults)
        >>> args = parse_args()

        >>> # 2. Create PERL configuration
        >>> config = PERLConfig(model_name_or_path="Qwen/Qwen3-1.7B", ...)

        >>> # 3. Initialize PERL trainer and load model
        >>> perl = PERL(config)
        >>> model, tokenizer = perl.load_model()

        >>> # 4. Prepare datasets with prompts
        >>> train_dataset, eval_dataset = prepare_datasets(...)

        >>> # 5. Define reward functions (accuracy + format)
        >>> reward_funcs = [accuracy_reward, format_reward]

        >>> # 6. Train with GRPO
        >>> perl.train(dataset=train_dataset, reward_funcs=reward_funcs)

        >>> # 7. Save LoRA adapters
        >>> perl.save_model(save_method="lora")
    """
    # =========================================================================
    # STEP 1: Parse configuration from command line arguments
    # =========================================================================
    args = parse_args()
    set_seed(args.random_state)  # Ensure reproducibility

    # =========================================================================
    # STEP 2: Set up clean logging (suppress verbose library messages)
    # =========================================================================
    logger = setup_logging()

    # Print configuration for reference and reproducibility
    print_config(args, logger)

    # Look up and display the system prompt that guides reasoning
    developer_prompt = print_system_prompt(args.developer_prompt, logger)

    # =========================================================================
    # STEP 3: Configure PERL for RL training
    # =========================================================================
    # PERL Configuration combines:
    # - Model settings (path, quantization, sequence length)
    # - LoRA settings (rank, alpha, target modules) for efficiency
    # - Training settings (learning rate, batch size, steps)
    # - GRPO settings (num_generations, temperature, sampling params)
    # - vLLM settings (GPU memory, sleep mode) for fast generation

    # Determine checkpoint strategy (save every N steps or once per epoch)
    save_strategy = "steps" if args.save_steps is not None else "epoch"

    perl_config = PERLConfig(
        # Model configuration
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        # LoRA configuration (alpha uses PERL defaults)
        lora_r=args.lora_r,
        # Training configuration
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        scheduler=args.scheduler,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # GRPO configuration (generation and sampling)
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # Output and logging
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        report_to=args.report_to,
        # vLLM configuration for fast generation during training
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        # Checkpoint resumption
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # =========================================================================
    # STEP 4: Initialize PERL trainer
    # =========================================================================
    # PERL handles all the complexity of:
    # - Loading models with LoRA adapters
    # - Setting up GRPO training
    # - Managing vLLM for generation
    # - Tracking training metrics
    perl = PERL(perl_config)

    # =========================================================================
    # STEP 5: Load base model with LoRA adapters
    # =========================================================================
    # PERL.load_model() returns the model with LoRA adapters already attached
    # The tokenizer is configured with the correct chat template
    logger.info("Loading model and tokenizer...")
    model, tokenizer = perl.load_model()

    # Enable thinking mode if the model supports it (e.g., Qwen3)
    # This adds special tokens for structured reasoning with <think> tags
    if getattr(model, "generation_config", None) is not None:
        try:
            model.generation_config.enable_thinking = True
        except AttributeError:
            logger.debug(
                "Model generation_config does not support enable_thinking."
            )

    # =========================================================================
    # STEP 6: Prepare training and evaluation datasets
    # =========================================================================
    # Datasets must provide:
    # - "prompt": Formatted string for model generation
    # - Any additional data needed by reward functions (e.g., "item")
    logger.info("Preparing datasets...")
    dataset_specs = {args.dataset: {"weight": 1.0}}
    train_dataset, eval_dataset = prepare_datasets(
        dataset_specs=dataset_specs,
        train_size=args.dataset_size,
        eval_size=args.eval_dataset_size,
        tokenizer=tokenizer,
        developer_prompt=developer_prompt,
        developer_role=args.developer_role,
    )

    # Show an example prompt to verify formatting is correct
    if len(train_dataset) > 0:
        example = train_dataset[0]
        logger.info("=" * 80)
        logger.info("Example Training Prompt:")
        logger.info("-" * 80)
        logger.info("%s", example["prompt"])
        logger.info("=" * 80)

    # =========================================================================
    # STEP 7: Evaluate base model before RL training
    # =========================================================================
    # This baseline measurement helps quantify the improvement from GRPO
    logger.info("Evaluating base model before RL training...")
    base_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        batch_size=args.per_device_eval_batch_size,
        logger=logger,
    )

    # =========================================================================
    # STEP 8: Define reward functions for GRPO
    # =========================================================================
    # Reward functions must have signature:
    #     def reward_func(completions: list[str], **kwargs) -> list[float]
    #
    # GRPO will:
    # 1. Generate multiple completions per prompt (num_generations)
    # 2. Call each reward function to score all completions
    # 3. Sum rewards across all reward functions
    # 4. Use relative rewards within each group to update the policy
    #
    # Here we use two reward functions:
    # - accuracy_reward: Primary signal (1.0 for correct, 0.0 for wrong)
    # - format_reward: Shaping signal (encourages proper tag structure)
    reward_logger = RewardLogger(train_dataset, args.logging_steps)
    reward_funcs = [
        reward_logger.accuracy_reward_with_logging,  # Logs examples
        format_reward,  # Encourages <think> and <answer> tags
    ]

    # =========================================================================
    # STEP 9: Run GRPO training with PERL
    # =========================================================================
    # PERL.train() handles the entire training loop:
    # - Generates completions using vLLM (if enabled)
    # - Computes rewards using provided reward functions
    # - Updates LoRA parameters with GRPO
    # - Logs metrics and saves checkpoints
    logger.info("Starting PERL training...")
    if args.resume_from_checkpoint:
        logger.info(
            "Resuming from checkpoint: %s", args.resume_from_checkpoint
        )

    perl.train(
        dataset=train_dataset,
        reward_funcs=reward_funcs,
        eval_dataset=eval_dataset,  # Optional: for TRL's built-in eval
    )

    # =========================================================================
    # STEP 10: Evaluate model after RL training
    # =========================================================================
    logger.info("Evaluating model after RL training...")
    final_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        batch_size=args.per_device_eval_batch_size,
        logger=logger,
    )

    # =========================================================================
    # STEP 11: Report training results
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Training Summary:")
    logger.info("-" * 80)
    logger.info("Base model accuracy: %.2f%%", base_results["accuracy"] * 100)
    logger.info(
        "Final model accuracy: %.2f%%", final_results["accuracy"] * 100
    )
    improvement = (final_results["accuracy"] - base_results["accuracy"]) * 100
    logger.info("Improvement: %+.2f%%", improvement)
    logger.info("=" * 80)

    # =========================================================================
    # STEP 12: Save LoRA adapters
    # =========================================================================
    # save_method="lora" saves only the adapter weights, not the full model
    # This is much more storage-efficient than saving merged weights
    # The adapters can be loaded later with PEFT or merged into the base model
    logger.info("Saving model...")
    perl.save_model(save_method="lora")

    # =========================================================================
    # STEP 13: Clean up resources
    # =========================================================================
    logger.info("Cleaning up...")
    del perl
    # Note: We skip torch.cuda.empty_cache() to avoid CUDA allocator conflicts

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
