r"""GRPO on a Reasoning Gym task with PERL, evaluated before and after.

Reasoning Gym generates verifiable problems (``chain_sum``,
``spell_backward``, ...) procedurally, so data is unlimited and every answer
can be checked. The script loads a model with fresh LoRA adapters
(``PERL.load_model``), measures its accuracy, trains it with GRPO
(``PERL.train``) on two summed rewards, measures it again and saves the
adapters (``PERL.save_model``).

- ``accuracy_reward``: Reasoning Gym's scorer on the text inside
  ``<answer>...</answer>``.
- ``format_reward``: 0.25 for each of ``<think>``, ``</think>``,
  ``<answer>``, ``</answer>`` present.

Prompts are rendered with the tokenizer's chat template (``enable_thinking``
on) and a Reasoning Gym system prompt asking for that tag format.

Usage:
    uv run python -m examples.rgym.rgym --output_dir ./models/chainsum \\
        --max_steps 100 --eval_dataset_size 100 --vllm_sleep
    uv run python -m examples.rgym.rgym --help

See ``examples/rgym/README.md`` for results, memory settings and how to
plug in your own task and reward.
"""

import argparse
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any

import datasets
import reasoning_gym
import torch
import transformers
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, set_seed

from speftr.perl import PERL, PERLConfig


if TYPE_CHECKING:
    from peft import PeftModel


QUIET_TRANSFORMERS_LOGGERS = [
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "transformers.tokenization_utils",
]
# Keeps vLLM from installing its own log handlers (sleep/wake chatter).
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"


class ReasoningGymDataset(Dataset):
    """Chat-formatted prompts over a Reasoning Gym procedural dataset.

    Each item is ``{"prompt": str, "item": dict}``: the rendered prompt
    (system message if ``developer_role`` is set, then the question, then
    the generation prompt) and the raw Reasoning Gym entry the scorer needs.

    Attributes:
        tokenizer: Tokenizer whose chat template renders the prompt.
        data: Underlying procedural dataset; ``data.score_answer`` scores
            answers.
        developer_prompt: System prompt text.
        developer_role: Role of the system message (``"system"``,
            ``"developer"``); ``None`` omits it.
    """

    tokenizer: PreTrainedTokenizerBase
    data: ProceduralDataset
    developer_prompt: str
    developer_role: str | None

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        procedural_dataset: ProceduralDataset,
        developer_prompt: str,
        developer_role: str | None = None,
    ) -> None:
        """Wrap a procedural dataset.

        Args:
            tokenizer: Tokenizer with a chat template.
            procedural_dataset: Reasoning Gym dataset (questions, answers,
                scorer).
            developer_prompt: System prompt text.
            developer_role: Role of the system message; ``None`` omits it.
        """
        self.tokenizer = tokenizer
        self.data = procedural_dataset
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role

    def __len__(self) -> int:
        """Return the number of problems.

        Returns:
            Size of the procedural dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Render problem ``idx`` as a prompt string.

        Args:
            idx: Problem index.

        Returns:
            ``{"prompt": str, "item": dict}``; ``item`` holds ``question``,
            ``answer`` and ``metadata``.
        """
        item = self.data[idx]
        chat = []
        if self.developer_role is not None:
            chat.append(
                {"role": self.developer_role, "content": self.developer_prompt}
            )
        chat.append({"role": "user", "content": item["question"]})
        # enable_thinking=True keeps Qwen3-style templates from inserting an
        # empty <think></think> block; other templates ignore it.
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return {"prompt": prompt, "item": item}


def to_hf_dataset(dataset: ReasoningGymDataset) -> datasets.Dataset:
    """Materialize a ``ReasoningGymDataset`` as a Hugging Face dataset.

    TRL's ``GRPOTrainer`` only accepts ``datasets.Dataset`` inputs. It
    passes the ``item`` column to the reward functions, which score it with
    the wrapper's ``data.score_answer``.

    Args:
        dataset: Wrapped reasoning-gym problems.

    Returns:
        Dataset with the same ``prompt`` and ``item`` rows.
    """
    return datasets.Dataset.from_list(
        [dataset[index] for index in range(len(dataset))]
    )


def accuracy_reward(
    completions: list[str],
    train_dataset: ReasoningGymDataset,
    **kwargs: Any,
) -> list[float]:
    """Score each completion's ``<answer>`` with Reasoning Gym's scorer.

    The last ``<answer>...</answer>`` block is extracted and passed to
    ``train_dataset.data.score_answer``, which returns 1.0 for a correct
    answer, 0.0 for a missing one and a task-specific value in between
    otherwise (partial credit for some tasks).

    Args:
        completions: Sampled completions (string prompts give strings).
        train_dataset: Dataset whose scorer is used.
        **kwargs: Must contain ``item``, the Reasoning Gym entry of each
            completion's prompt (TRL passes dataset columns this way); other
            keys are ignored.

    Returns:
        One score in [0.0, 1.0] per completion.

    Raises:
        ValueError: If ``item`` is missing or its length differs from
            ``completions``.
        TypeError: If an item is not a dict.
    """
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

    answers = [extract_answer(c) for c in completions]
    return [
        train_dataset.data.score_answer(answer, item)
        for answer, item in zip(answers, kwargs["item"], strict=False)
    ]


def format_reward(completions: list[str], **_kwargs: Any) -> list[float]:
    """Shaping reward for the ``<think>``/``<answer>`` tag format.

    Adds 0.25 for each of ``<think>``, ``</think>``, ``<answer>`` and
    ``</answer>`` found anywhere in the completion (presence only, not order
    or count). GRPO sums it with ``accuracy_reward``, so a well-formatted
    wrong answer still ranks above an unformatted one.

    Args:
        completions: Sampled completions.
        **_kwargs: Dataset columns and trainer state, unused.

    Returns:
        One reward in {0.0, 0.25, 0.5, 0.75, 1.0} per completion.
    """

    def count_tags(text: str) -> float:
        """Return 0.25 per reasoning tag present in ``text``.

        Args:
            text: Completion to inspect.

        Returns:
            Between 0.0 and 1.0.
        """
        count = 0.0
        if re.search(r"\s*<think>\s*", text):
            count += 0.25
        if re.search(r"\s*</think>\s*", text):
            count += 0.25
        if re.search(r"\s*<answer>\s*", text):
            count += 0.25
        if re.search(r"\s*</answer>\s*", text):
            count += 0.25
        return count

    return [count_tags(c) for c in completions]


class RewardLogger:
    """Wrap ``accuracy_reward`` to log a sample completion periodically.

    Every ``logging_steps`` reward calls it logs the first completion of the
    batch, its extracted answer, the expected answer and the reward, to
    check that answer extraction and scoring behave as intended.

    Attributes:
        train_dataset: Dataset whose scorer is used.
        logging_steps: Log on every N-th reward call.
        call_count: Reward calls so far.
        logger: Destination logger.
    """

    train_dataset: ReasoningGymDataset
    logging_steps: int
    call_count: int
    logger: logging.Logger

    def __init__(
        self, train_dataset: ReasoningGymDataset, logging_steps: int
    ) -> None:
        """Create the wrapper.

        Args:
            train_dataset: Dataset whose scorer is used.
            logging_steps: Log on every N-th reward call.
        """
        self.train_dataset = train_dataset
        self.logging_steps = logging_steps
        self.call_count = 0
        self.logger = logging.getLogger(__name__)

    def accuracy_reward_with_logging(
        self, completions: list[str], **kwargs: Any
    ) -> list[float]:
        """Compute ``accuracy_reward`` and log a sample every N calls.

        Args:
            completions: Sampled completions.
            **kwargs: Reward keyword arguments; must contain ``item``.

        Returns:
            The unchanged ``accuracy_reward`` scores.
        """
        self.call_count += 1
        rewards = accuracy_reward(completions, self.train_dataset, **kwargs)

        if self.call_count % self.logging_steps == 0 and len(completions) > 0:
            item = kwargs.get("item", [{}])[0] if "item" in kwargs else {}
            completion = completions[0]
            reward = rewards[0]
            expected = item.get("answer", "N/A")
            extracted = extract_answer(completion)
            accuracy_threshold = 0.5
            is_correct = reward > accuracy_threshold

            self.logger.info("=" * 80)
            self.logger.info(
                "Sample completion (reward call %d):", self.call_count
            )
            self.logger.info("-" * 80)
            self.logger.info("Completion: %s", completion)
            self.logger.info("-" * 40)
            self.logger.info("Extracted answer: %s", extracted)
            self.logger.info("Expected answer: %s", expected)
            self.logger.info("Correct: %s (reward: %.2f)", is_correct, reward)
            self.logger.info("=" * 80)

        return rewards


def prepare_datasets(  # noqa: PLR0913
    dataset_specs: dict[str, dict],
    train_size: int,
    eval_size: int,
    tokenizer: PreTrainedTokenizerBase,
    *,
    developer_prompt: str,
    developer_role: str,
) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Generate train and eval problems from Reasoning Gym.

    Both sets come from a ``composite`` dataset over ``dataset_specs``;
    seed 1 generates the training problems and seed 2 the evaluation ones.

    Args:
        dataset_specs: Task name to ``{"weight": float, "config": dict}``,
            e.g. ``{"chain_sum": {"weight": 1.0}}``. Weights set the
            sampling mix; ``config`` is the task's own options.
        train_size: Number of training problems.
        eval_size: Number of evaluation problems.
        tokenizer: Tokenizer with a chat template.
        developer_prompt: System prompt text.
        developer_role: Role of the system message.

    Returns:
        Train and eval datasets.
    """
    specs = [
        DatasetSpec(
            name=name,
            weight=config.get("weight", 1.0),
            config=config.get("config", {}),
        )
        for name, config in dataset_specs.items()
    ]
    train_data = reasoning_gym.create_dataset(
        "composite", seed=1, size=train_size, datasets=specs
    )
    eval_data = reasoning_gym.create_dataset(
        "composite", seed=2, size=eval_size, datasets=specs
    )
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
    model: "PeftModel",
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset: ReasoningGymDataset,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    batch_size: int = 16,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """Measure accuracy with transformers ``generate`` (not vLLM).

    Samples one completion per problem with the training sampling settings,
    so the result varies between runs. Prompts are truncated to 512 tokens
    and completions capped at 512 new tokens. A problem counts as correct
    when its ``accuracy_reward`` score exceeds 0.5. Puts ``model`` in eval
    mode.

    Args:
        model: Policy from ``PERL.load_model``.
        tokenizer: Its tokenizer; ``pad_token`` is set to ``eos_token`` if
            missing.
        eval_dataset: Problems to score.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff.
        min_p: Minimum token probability.
        batch_size: Prompts generated per batch.
        logger: Progress logger; defaults to this module's.

    Returns:
        ``accuracy`` (fraction), ``correct`` and ``total``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    sample_count = len(eval_dataset)
    logger.info("=" * 80)
    logger.info(
        "Evaluating on %d problems (batch size %d)",
        sample_count,
        batch_size,
    )
    logger.info("-" * 80)

    model.eval()
    correct = 0
    total = 0
    accuracy_threshold = 0.5

    for batch_start in range(0, sample_count, batch_size):
        batch_end = min(batch_start + batch_size, sample_count)
        batch_prompts = []
        batch_items = []
        for i in range(batch_start, batch_end):
            example = eval_dataset[i]
            batch_prompts.append(example["prompt"])
            batch_items.append(example["item"])

        # Left padding: every prompt ends at the same position, so the
        # completion starts right after the padded input.
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
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

        for j, (output, item) in enumerate(
            zip(outputs, batch_items, strict=False)
        ):
            prompt_length = inputs.input_ids[j].shape[0]
            completion = tokenizer.decode(
                output[prompt_length:], skip_special_tokens=True
            )
            extracted_answer = extract_answer(completion)
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
    """Parse command line arguments.

    Returns:
        Namespace with model, task, LoRA, GRPO, sampling, vLLM and output
        settings (see ``--help``).
    """
    parser = argparse.ArgumentParser(
        description="GRPO training for Reasoning Gym with PERL"
    )

    # Model
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

    # Task
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

    # LoRA and optimisation
    parser.add_argument(
        "--lora_r",
        type=int,
        default=1,
        help="LoRA rank for RL (default: 1)",
    )
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

    # GRPO sampling
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

    # Logging and checkpoints
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

    # vLLM and quantisation
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
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit (QLoRA); requires --no_vllm",
    )
    parser.set_defaults(use_vllm=True, vllm_enable_sleep_mode=False)

    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Experiment tracking backend (default: none)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


def setup_logging() -> logging.Logger:
    """Log to stdout at INFO, silencing noisy transformers sub-loggers.

    Returns:
        This module's logger.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    for noisy_logger in QUIET_TRANSFORMERS_LOGGERS:
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    return logger


def print_config(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Log the main run settings.

    Args:
        args: Parsed command line arguments.
        logger: Destination logger.

    Returns:
        None. The settings are logged at INFO level.
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
    """Look up a Reasoning Gym system prompt and log it.

    Args:
        prompt_key: Key of ``reasoning_gym.utils.SYSTEM_PROMPTS`` (e.g.
            ``"DeepSeekZero"``, ``"default"``). An unknown key raises
            ``KeyError``.
        logger: Destination logger.

    Returns:
        The system prompt text.
    """
    developer_prompt: str = SYSTEM_PROMPTS[prompt_key]
    logger.info("=" * 80)
    logger.info("System Prompt (%s):", prompt_key)
    logger.info("-" * 80)
    logger.info("%s", developer_prompt)
    logger.info("=" * 80)
    return developer_prompt


def build_perl_config(args: argparse.Namespace) -> PERLConfig:
    """Map parsed CLI arguments onto a PERLConfig.

    ``lora_alpha`` and ``target_modules`` keep PERL's defaults. Checkpoints
    are saved every ``--save_steps`` steps if given, else once per epoch.

    Args:
        args: Parsed command line arguments.

    Returns:
        GRPO config for the selected task.
    """
    save_strategy = "steps" if args.save_steps is not None else "epoch"
    return PERLConfig(
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        scheduler=args.scheduler,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        report_to=args.report_to,
        load_in_4bit=args.load_in_4bit,
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


def main() -> None:
    """Evaluate, train with GRPO, evaluate again and save the adapters.

    Returns:
        None. LoRA adapters and tokenizer are written to ``--output_dir``;
        accuracy before and after training is logged.
    """
    args = parse_args()
    set_seed(args.random_state)
    logger = setup_logging()
    print_config(args, logger)
    developer_prompt = print_system_prompt(args.developer_prompt, logger)

    perl = PERL(build_perl_config(args))
    logger.info("Loading model and tokenizer...")
    model, tokenizer = perl.load_model()

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
    if len(train_dataset) > 0:
        example = train_dataset[0]
        logger.info("=" * 80)
        logger.info("Example Training Prompt:")
        logger.info("-" * 80)
        logger.info("%s", example["prompt"])
        logger.info("=" * 80)

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

    # GRPO sums the rewards of all functions for each completion.
    reward_logger = RewardLogger(train_dataset, args.logging_steps)
    reward_funcs = [
        reward_logger.accuracy_reward_with_logging,
        format_reward,
    ]

    logger.info("Starting PERL training...")
    if args.resume_from_checkpoint:
        logger.info(
            "Resuming from checkpoint: %s", args.resume_from_checkpoint
        )
    perl.train(
        dataset=to_hf_dataset(train_dataset),
        reward_funcs=reward_funcs,
        eval_dataset=to_hf_dataset(eval_dataset),
    )

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

    logger.info("=" * 80)
    logger.info("Training Summary:")
    logger.info("-" * 80)
    logger.info("Base model accuracy: %.2f%%", base_results["accuracy"] * 100)
    logger.info(
        "Final model accuracy: %.2f%%", final_results["accuracy"] * 100
    )
    improvement = (final_results["accuracy"] - base_results["accuracy"]) * 100
    logger.info("Accuracy change: %+.2f points", improvement)
    logger.info("=" * 80)

    logger.info("Saving model...")
    perl.save_model(save_method="lora")

    logger.info("Cleaning up...")
    del perl
    # Colocated vLLM creates a torch.distributed process group; tear it down
    # explicitly or the interpreter can crash at exit while NCCL leaks it.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
