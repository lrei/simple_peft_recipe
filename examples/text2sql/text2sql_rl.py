r"""Refine the SFT text-to-SQL model with GRPO and an execution reward (PERL).

Stage 2 of the text-to-SQL example. The LoRA adapters saved by
``text2sql_sft.py`` are loaded on their base model with peft and handed to
PERL (``set_pretrained_model``), which keeps training them with GRPO. The
reward is ``score_sql`` from ``text2sql_eval.py``: 1.0 when
the query matches the gold query (text or SQLite result), 0.2 when it only
runs, 0.0 otherwise. Rows whose gold query does not run on its own schema
are dropped.

The policy is loaded in 4-bit, matching the SFT stage, and sampled with
transformers (``use_vllm=False``): PERL cannot sync LoRA updates into a
4-bit vLLM copy, and vLLM runs SmolLM3 through its Transformers backend,
which wraps the same transformers model class that is being trained, so the
training forward fails.

Usage:
    uv run python -m examples.text2sql.text2sql_rl --max_steps 150
    uv run python -m examples.text2sql.text2sql_eval \\
        --model_path ./models/smollm3-3b-sql-grpo

See ``examples/text2sql/README.md`` for the reward, results and how to
adapt it.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, cast

import torch

from examples.text2sql.text2sql_eval import (
    CHAT_TEMPLATE_KWARGS,
    DEFAULT_MODEL_PATH,
    build_messages,
    gold_executes,
    load_splits,
    score_sql,
)
from speftr import PERL, PERLConfig


if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase


DEFAULT_OUTPUT_DIR = "./models/smollm3-3b-sql-grpo"


def load_sft_adapters(
    adapter_dir: str, *, load_in_4bit: bool
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the SFT adapters, trainable, on their base model.

    Args:
        adapter_dir: Directory written by ``text2sql_sft.py``.
        load_in_4bit: Quantize the base weights to 4 bits (NF4, bf16
            compute), as PERL does for QLoRA.

    Returns:
        The model with trainable adapters and the saved tokenizer.
    """
    from peft import PeftConfig, PeftModel  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    base_name = cast("str", adapter_config.base_model_name_or_path)
    quantization = None
    if load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # Hub id recorded by the SFT run: no revision to pin (B615).
    base = AutoModelForCausalLM.from_pretrained(  # nosec B615
        base_name,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)  # nosec B615
    return model, tokenizer


def sql_reward(
    completions: list[str],
    context: list[str],
    answer: list[str],
    **_kwargs: Any,
) -> list[float]:
    """GRPO reward: ``score_sql`` of each completion against its gold query.

    TRL passes the dataset columns of each completion's prompt as keyword
    arguments, here ``context`` and ``answer``.

    Args:
        completions: Sampled replies.
        context: ``CREATE TABLE`` statements, one per completion.
        answer: Gold SQL, one per completion.
        **_kwargs: Other columns and trainer state, unused.

    Returns:
        One reward in {0.0, 0.2, 1.0} per completion.
    """
    return [
        score_sql(completion, schema, gold)
        for completion, schema, gold in zip(
            completions, context, answer, strict=True
        )
    ]


def build_prompt_dataset(
    rows: Dataset, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    """Render each row's chat prompt, keeping what the reward needs.

    Args:
        rows: sql-create-context rows whose gold query runs.
        tokenizer: Policy tokenizer with a chat template.

    Returns:
        Rows with a text ``prompt`` (ending with the assistant header) plus
        ``context`` and ``answer``.
    """

    def render(row: dict[str, str]) -> dict[str, str]:
        """Render one row's prompt.

        Args:
            row: Row with ``context`` and ``question``.

        Returns:
            The ``prompt`` column value.
        """
        chat = build_messages(row["context"], row["question"])
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        return {"prompt": cast("str", prompt)}

    return rows.map(render, remove_columns=["question"])


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with model, GRPO and generation settings.
    """
    parser = argparse.ArgumentParser(
        description="GRPO text-to-SQL training with an SQLite reward"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"SFT adapter directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where adapters are saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=150,
        help="Optimizer steps (default: 150)",
    )
    parser.add_argument(
        "--train_rows",
        type=int,
        default=2000,
        help="Training rows sampled before filtering (default: 2000)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Completions per prompt (default: 8)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PERLConfig:
    """Map CLI arguments onto a PERLConfig.

    A device batch holds one prompt's ``num_generations`` completions and
    each optimizer step accumulates 2 of them (16 completions by default).

    Args:
        args: Parsed arguments from ``parse_args``.

    Returns:
        GRPO config for a 4-bit policy sampled with transformers.
    """
    return PERLConfig(
        model_name_or_path=args.model_name_or_path,
        max_seq_length=1024,
        max_completion_length=128,
        load_in_4bit=True,
        use_vllm=False,
        learning_rate=args.learning_rate,
        scheduler="constant",
        max_steps=args.max_steps,
        num_train_epochs=1,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=2,
        logging_steps=10,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Continue training the SFT adapters with GRPO and save them.

    Returns:
        None. Adapters and tokenizer are written to ``--output_dir``.
    """
    args = parse_args()
    config = build_config(args)
    perl = PERL(config)
    model, tokenizer = load_sft_adapters(
        args.model_name_or_path, load_in_4bit=config.load_in_4bit
    )
    perl.set_pretrained_model(model, tokenizer)

    train_rows, _held_out = load_splits()
    rows = train_rows.shuffle(seed=1).select(range(args.train_rows))
    rows = rows.filter(
        lambda row: gold_executes(row["context"], row["answer"])
    )
    dataset = build_prompt_dataset(rows, tokenizer)
    print(dataset[0]["prompt"])

    perl.train(dataset, [sql_reward])
    perl.save_model()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak_gpu_memory_gb: {peak_gb:.2f}")


if __name__ == "__main__":
    main()
