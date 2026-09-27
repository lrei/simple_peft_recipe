"""Tests for the text-to-SQL example's SQL extraction and SQLite reward.

Rows come from ``b-mc2/sql-create-context`` (train split; the ``row`` field
is the dataset index): Spider single- and multi-table questions, a WikiSQL
question, two rows whose gold query references a column missing from the
schema and one row declaring the same table twice.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from examples.text2sql.text2sql_eval import (
    VALID_SQL_REWARD,
    build_database,
    build_messages,
    extract_sql,
    gold_executes,
    normalize_sql,
    run_query,
    score_sql,
)
from examples.text2sql.text2sql_rl import build_prompt_dataset, sql_reward


@pytest.fixture(scope="module")
def rows(load_fixture) -> dict[int, dict]:
    return {
        row["row"]: row for row in load_fixture("sql_create_context_rows.json")
    }


@pytest.fixture(scope="module")
def valid_rows(rows) -> list[dict]:
    return [
        row for row in rows.values() if row["row"] not in (750, 1750, 1950)
    ]


def test_extract_sql_from_code_fence_and_reasoning():
    reply = "<think>\nneed MAX\n</think>\n```sql\nSELECT MAX(x) FROM t;\n```\n"
    assert extract_sql(reply) == "SELECT MAX(x) FROM t"


def test_extract_sql_plain_reply_strips_semicolon():
    assert extract_sql("  SELECT a FROM b ;\n") == "SELECT a FROM b"


def test_normalize_sql_ignores_case_whitespace_and_quote_style(rows):
    gold = rows[10000]["answer"]
    variant = (
        "select purpose  FROM table_name_71\n"
        "WHERE band = 'am' AND callsign = '6wh';"
    )
    assert normalize_sql(variant) == normalize_sql(gold)


def test_build_messages_includes_schema_and_question(rows):
    row = rows[6]
    prompt_only = build_messages(row["context"], row["question"])
    with_answer = build_messages(
        row["context"], row["question"], row["answer"]
    )
    assert len(prompt_only) == 1
    assert row["context"] in prompt_only[0]["content"]
    assert row["question"] in prompt_only[0]["content"]
    assert with_answer[-1] == {"role": "assistant", "content": row["answer"]}


def test_build_database_seeds_rows_the_gold_filter_matches(rows):
    row = rows[6]
    with closing(build_database(row["context"], row["answer"])) as connection:
        result = run_query(connection, row["answer"])
    assert result


def test_build_database_rejects_duplicate_table(rows):
    row = rows[1950]
    with pytest.raises(sqlite3.Error):
        build_database(row["context"], row["answer"])


def test_gold_executes_flags_broken_rows(rows, valid_rows):
    assert all(gold_executes(r["context"], r["answer"]) for r in valid_rows)
    for index in (750, 1750, 1950):
        assert not gold_executes(rows[index]["context"], rows[index]["answer"])


def test_score_gold_query_is_full_reward(valid_rows):
    for row in valid_rows:
        assert score_sql(row["answer"], row["context"], row["answer"]) == 1.0


def test_score_equivalent_query_by_execution(rows):
    row = rows[0]
    equivalent = "SELECT COUNT(age) FROM head WHERE 56 < age"
    assert score_sql(equivalent, row["context"], row["answer"]) == 1.0


def test_score_wrong_filter_value_is_partial(rows):
    row = rows[10000]
    wrong = 'SELECT purpose FROM table_name_71 WHERE band = "fm"'
    assert score_sql(wrong, row["context"], row["answer"]) == VALID_SQL_REWARD


@pytest.mark.parametrize(
    "reply",
    ["", "SELECT nope FROM head", "The answer is 3.", "SELECT 1; SELECT 2"],
)
def test_score_invalid_sql_is_zero(rows, reply):
    row = rows[0]
    assert score_sql(reply, row["context"], row["answer"]) == 0.0


def test_score_aborts_runaway_query(rows):
    row = rows[0]
    runaway = (
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
        "SELECT COUNT(*) FROM n"
    )
    assert score_sql(runaway, row["context"], row["answer"]) == 0.0


@settings(max_examples=200, deadline=None)
@given(reply=st.text(max_size=200))
def test_score_is_always_a_documented_reward(rows, reply):
    row = rows[6]
    score = score_sql(reply, row["context"], row["answer"])
    assert score in {0.0, VALID_SQL_REWARD, 1.0}


def test_sql_reward_scores_each_completion(rows):
    row = rows[3]
    completions = [row["answer"], "SELECT 1", "not sql"]
    rewards = sql_reward(
        completions,
        context=[row["context"]] * 3,
        answer=[row["answer"]] * 3,
        prompts=["unused"] * 3,
    )
    assert rewards == [1.0, VALID_SQL_REWARD, 0.0]


def test_prompt_dataset_ends_with_assistant_header(valid_rows, qwen_tokenizer):
    from datasets import Dataset

    dataset = build_prompt_dataset(
        Dataset.from_list(valid_rows), qwen_tokenizer
    )
    assert "question" not in dataset.column_names
    first = dataset[0]
    assert first["prompt"].endswith("<|im_start|>assistant\n")
    assert valid_rows[0]["question"] in first["prompt"]
    assert first["answer"] == valid_rows[0]["answer"]
