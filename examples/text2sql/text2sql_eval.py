r"""Evaluate a text-to-SQL model by running its queries in SQLite.

An analyst asks a question about tables whose ``CREATE TABLE`` statements
are given; the model answers with one SQL query. This module defines that
task for ``text2sql_sft.py`` and ``text2sql_rl.py`` (dataset split, prompt,
SQL scoring) and scores a model on held-out rows. It uses plain
transformers + peft, not Unsloth, because the RL script imports it too.

Scoring (``score_sql``), also the GRPO reward:

- 1.0: the query equals the gold query after normalisation (case,
  whitespace, quote style, trailing ``;``), or it returns the same rows
  (in any order) as the gold query on a small database built from the
  schema (``build_database``) and the gold query returns at least one row;
- ``VALID_SQL_REWARD``: the query runs but its result differs;
- 0.0: no query, or SQLite rejects it.

The dataset has no rows, so ``build_database`` fills every table from the
literals of the gold query (plus two fillers), shifting them by one per
column. That makes the gold filters match some rows, so a wrong query
usually returns a different result. It is a heuristic, not a proof of
equivalence.

Usage:
    # Base model
    uv run python -m examples.text2sql.text2sql_eval \\
        --model_path HuggingFaceTB/SmolLM3-3B
    # After SFT or GRPO (adapter directories)
    uv run python -m examples.text2sql.text2sql_eval \\
        --model_path ./models/smollm3-3b-sql-sft

Dataset: ``b-mc2/sql-create-context`` (CC-BY-4.0, WikiSQL + Spider).
See ``examples/text2sql/README.md`` for results and how to adapt the task.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import Counter
from contextlib import closing
from typing import TYPE_CHECKING, Any, cast

import torch
from datasets import load_dataset


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        BatchEncoding,
        GenerationMixin,
        PreTrainedTokenizerBase,
    )


DATASET_NAME = "b-mc2/sql-create-context"
DEFAULT_MODEL_PATH = "./models/smollm3-3b-sql-sft"
HELD_OUT_ROWS = 500
SPLIT_SEED = 0
INSTRUCTION = (
    "Write one SQLite query that answers the question. "
    "Reply with the SQL query only."
)
# SmolLM3 answers directly instead of reasoning in <think> first.
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}
VALID_SQL_REWARD = 0.2
FILLER_VALUES = ("alpha", 7)
# SQLite VM steps per query before it is aborted (runaway joins, loops).
MAX_QUERY_STEPS = 1_000_000
LITERAL_PATTERN = re.compile(r"'([^']*)'|\"([^\"]*)\"|\b(\d+(?:\.\d+)?)\b")

type Message = dict[str, str]
type SqlValue = str | int | float


def load_splits(held_out_rows: int = HELD_OUT_ROWS) -> tuple[Dataset, Dataset]:
    """Split sql-create-context (one ``train`` split) into train/held-out.

    Args:
        held_out_rows: Rows reserved for evaluation; the split is seeded, so
            every script sees the same rows.

    Returns:
        Train and held-out rows with ``question``, ``context`` (``CREATE
        TABLE`` statements) and ``answer`` (gold SQL).
    """
    dataset = cast("Dataset", load_dataset(DATASET_NAME, split="train"))
    split = dataset.train_test_split(test_size=held_out_rows, seed=SPLIT_SEED)
    return split["train"], split["test"]


def build_messages(
    context: str, question: str, answer: str | None = None
) -> list[Message]:
    """Build the chat for one question.

    Args:
        context: ``CREATE TABLE`` statements of the tables involved.
        question: Question in English.
        answer: Gold SQL, added as the assistant turn when given (training);
            ``None`` ends the chat at the user turn (generation).

    Returns:
        A user message, followed by the answer if ``answer`` is given.
    """
    prompt = f"{INSTRUCTION}\n\nSchema:\n{context}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": prompt}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def extract_sql(completion: str) -> str:
    """Pull the SQL query out of a model reply.

    Drops a reasoning block ending in ``</think>``, keeps the inside of the
    first Markdown code fence if there is one, and strips a trailing ``;``.

    Args:
        completion: Generated reply.

    Returns:
        The query text, possibly empty.
    """
    text = completion.rsplit("</think>", 1)[-1]
    if "```" in text:
        text = text.split("```")[1].removeprefix("sql")
    return text.strip().rstrip(";").strip()


def normalize_sql(sql: str) -> str:
    """Normalise a query for exact-match comparison.

    Args:
        sql: Query text.

    Returns:
        The query case-folded, with single spaces, double quotes and no
        trailing ``;``.
    """
    collapsed = " ".join(sql.replace("'", '"').split())
    return collapsed.rstrip(";").strip().casefold()


def _literal_values(gold_sql: str) -> list[SqlValue]:
    """Collect the string and number literals of a query, plus fillers.

    Args:
        gold_sql: Gold query.

    Returns:
        Literals in query order, then ``FILLER_VALUES``.
    """
    values: list[SqlValue] = []
    for single, double, number in LITERAL_PATTERN.findall(gold_sql):
        if number:
            values.append(float(number) if "." in number else int(number))
        else:
            values.append(single or double)
    return [*values, *FILLER_VALUES]


def build_database(context: str, gold_sql: str) -> sqlite3.Connection:
    """Create the schema in memory and fill it from the gold query.

    Every table gets one row per value of ``_literal_values``; row ``i``,
    column ``j`` holds value ``(i + j) % n``. Queries on the connection are
    aborted after ``MAX_QUERY_STEPS`` SQLite VM steps. An invalid schema (a
    few dataset rows declare the same table twice) raises the
    ``sqlite3.Error`` of ``executescript``.

    Args:
        context: ``CREATE TABLE`` statements.
        gold_sql: Gold query whose literals seed the rows.

    Returns:
        An open in-memory connection; the caller closes it.
    """
    connection = sqlite3.connect(":memory:")
    connection.executescript(context)
    values = _literal_values(gold_sql)
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    for (table,) in tables:
        width = len(
            connection.execute(
                "SELECT name FROM pragma_table_info(?)", (table,)
            ).fetchall()
        )
        rows = [
            [values[(row + column) % len(values)] for column in range(width)]
            for row in range(len(values))
        ]
        placeholders = ", ".join("?" * width)
        # Table names come from the schema just created, not from input.
        insert = f'INSERT INTO "{table}" VALUES ({placeholders})'  # noqa: S608
        connection.executemany(insert, rows)
    connection.set_progress_handler(lambda: 1, MAX_QUERY_STEPS)
    return connection


def run_query(connection: sqlite3.Connection, sql: str) -> list[Any] | None:
    """Run one query and fetch all rows.

    Args:
        connection: Database from ``build_database``.
        sql: A single SQL statement.

    Returns:
        The result rows, or ``None`` if SQLite rejects or aborts the query
        (syntax error, unknown column, several statements, step limit).
    """
    try:
        return connection.execute(sql).fetchall()
    except sqlite3.Error:
        return None


def gold_executes(context: str, gold_sql: str) -> bool:
    """Tell whether a dataset row's gold query runs on its own schema.

    About 2.5% of sql-create-context rows reference columns missing from
    their ``CREATE TABLE`` statements; they cannot be scored by execution.

    Args:
        context: ``CREATE TABLE`` statements.
        gold_sql: Gold query.

    Returns:
        True if the schema builds and the gold query runs.
    """
    try:
        connection = build_database(context, gold_sql)
    except sqlite3.Error:
        return False
    with closing(connection):
        return run_query(connection, gold_sql) is not None


def score_sql(completion: str, context: str, gold_sql: str) -> float:
    """Score a reply against the gold query (see the module docstring).

    Args:
        completion: Generated reply.
        context: ``CREATE TABLE`` statements; must build (``gold_executes``).
        gold_sql: Gold query.

    Returns:
        1.0 for a match, ``VALID_SQL_REWARD`` for a query that runs, else
        0.0.
    """
    predicted = extract_sql(completion)
    if not predicted:
        return 0.0
    if normalize_sql(predicted) == normalize_sql(gold_sql):
        return 1.0
    with closing(build_database(context, gold_sql)) as connection:
        # Gold first: the prediction may modify the tables.
        gold_rows = run_query(connection, gold_sql)
        predicted_rows = run_query(connection, predicted)
    if predicted_rows is None:
        return 0.0
    if gold_rows and Counter(predicted_rows) == Counter(gold_rows):
        return 1.0
    return VALID_SQL_REWARD


def load_for_generation(
    model_path: str,
) -> tuple[GenerationMixin, PreTrainedTokenizerBase]:
    """Load a base model, merged checkpoint or LoRA adapters in bf16.

    transformers loads a directory holding ``adapter_config.json`` as its
    base model with the adapters attached (peft must be installed).

    Args:
        model_path: Hub id or local directory.

    Returns:
        The model in eval mode and its tokenizer, set to pad on the left as
        batched generation requires.
    """
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    # User-supplied id or local path: no Hub revision to pin (B615).
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # nosec B615
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def generate_sql(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    rows: Dataset,
    max_new_tokens: int,
) -> list[str]:
    """Greedily answer a batch of questions.

    Args:
        model: Model from ``load_for_generation``.
        tokenizer: Its tokenizer; must have a chat template.
        rows: Rows with ``context`` and ``question``.
        max_new_tokens: Generation budget per reply.

    Returns:
        The decoded replies without the prompt or special tokens.
    """
    chats = [
        build_messages(context, question)
        for context, question in zip(
            rows["context"], rows["question"], strict=True
        )
    ]
    encoded = tokenizer.apply_chat_template(
        chats,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
        **CHAT_TEMPLATE_KWARGS,
    )
    inputs = cast("BatchEncoding", encoded).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Left padding: every prompt ends at the same position.
    replies = outputs[:, inputs["input_ids"].shape[1] :]
    return list(tokenizer.batch_decode(replies, skip_special_tokens=True))


def evaluate(
    model_path: str, rows: Dataset, args: argparse.Namespace
) -> dict[str, float]:
    """Score ``model_path`` on ``rows``.

    Args:
        model_path: Model to evaluate (see ``load_for_generation``).
        rows: Held-out rows whose gold query executes.
        args: Parsed CLI arguments (``batch_size``, ``max_new_tokens``).

    Returns:
        ``exact_match`` (normalised string match), ``execution_accuracy``
        (score 1.0) and ``valid_sql`` (the query runs) as fractions.
    """
    model, tokenizer = load_for_generation(model_path)
    scores: list[float] = []
    exact_matches: list[bool] = []
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        batch = rows.select(range(start, end))
        replies = generate_sql(model, tokenizer, batch, args.max_new_tokens)
        for reply, row in zip(replies, batch, strict=True):
            gold = row["answer"]
            scores.append(score_sql(reply, row["context"], gold))
            predicted = normalize_sql(extract_sql(reply))
            exact_matches.append(predicted == normalize_sql(gold))
        print(f"Scored {len(scores)}/{len(rows)} questions")
    return {
        "exact_match": sum(exact_matches) / len(rows),
        "execution_accuracy": scores.count(1.0) / len(rows),
        "valid_sql": sum(score > 0.0 for score in scores) / len(rows),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the model path, sample limit and generation settings.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a text-to-SQL model on held-out rows"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Hub id, merged model or adapter directory "
            f"(default: {DEFAULT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Questions generated per batch (default: 32)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens generated per query (default: 128)",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate ``--model_path`` on the held-out rows and print metrics.

    Returns:
        None. Metrics and peak GPU memory are printed to stdout.
    """
    args = parse_args()
    _train_rows, held_out = load_splits()
    held_out = held_out.filter(
        lambda row: gold_executes(row["context"], row["answer"])
    )
    metrics = evaluate(args.model_path, held_out, args)

    print(f"\nModel: {args.model_path} ({len(held_out)} held-out questions)")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak_gpu_memory_gb: {peak_gb:.2f}")


if __name__ == "__main__":
    main()
