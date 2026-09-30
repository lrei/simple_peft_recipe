r"""Score a DPO adapter by log-likelihood, without generating or a judge.

Two measurements, each for the SFT model, the DPO adapter trained by
``prefs_train`` and AllenAI's released DPO model:

1. Held-out preference accuracy: on pairs of
   ``allenai/olmo-2-0425-1b-preference-mix`` that training never sees,
   the fraction where the model gives the chosen response a higher
   log-probability than the rejected one, summed over response tokens and
   per token (length-normalized). The same pairs also get the implicit
   reward accuracy below.
2. RewardBench implicit-reward accuracy: the DPO implicit reward is
   ``beta * (log pi(y|x) - log pi_ref(y|x))`` summed over the response
   tokens, with the SFT model as ``pi_ref``. A pair is correct when the
   chosen response gets the higher reward. Scores are reported per
   RewardBench section and overall, weighted as RewardBench does.

Both implicit-reward accuracies are also reported length-normalized:
the log-ratio divided by the number of response tokens, the reward of
the length-normalized DPO loss (TRL ``loss_type="sigmoid_norm"``).

The adapter's reference is the same model with the adapter disabled.
Prompts and responses are rendered with the chat template and split into
prompt and response tokens as TRL's ``DPOTrainer`` does in training.

This module also defines the data for ``prefs_train.py``
(``load_preference_splits``).

Usage:
    uv run python -m examples.prefs.prefs_eval
    uv run python -m examples.prefs.prefs_eval \\
        --adapter_dir ./models/olmo2-1b-lora-dpo --reference_model ""

Data: ``allenai/olmo-2-0425-1b-preference-mix`` and
``allenai/reward-bench`` (both ODC-BY). Models:
``allenai/OLMo-2-0425-1B-SFT`` and ``allenai/OLMo-2-0425-1B-DPO``
(Apache-2.0). Results: ``examples/prefs/README.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, cast

import torch
from datasets import load_dataset


if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import Dataset
    from transformers import (
        BatchEncoding,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )


type Message = dict[str, str]
# Token ids of prompt + response and the index of the first response token.
type Encoded = tuple[list[int], int]

DATASET_NAME = "allenai/olmo-2-0425-1b-preference-mix"
REWARDBENCH_NAME = "allenai/reward-bench"
SFT_MODEL = "allenai/OLMo-2-0425-1B-SFT"
DPO_MODEL = "allenai/OLMo-2-0425-1B-DPO"
DEFAULT_ADAPTER_DIR = "./models/olmo2-1b-lora-dpo"
SEED = 3407
EVAL_PAIRS = 1000
TRAIN_PAIRS = 10_000
BETA = 0.1
# OLMo 2's context length; longer sequences are cut at the end.
MAX_LENGTH = 4096
# Prompt + response tokens per forward pass, padding included.
TOKEN_BUDGET = 16_384
# RewardBench's section of each subset and its weight within the section
# (rewardbench/constants.py). math-prm has 447 rows but counts as much as
# the six hep-* code subsets together.
SECTIONS = {
    "Chat": (
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ),
    "Chat Hard": (
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ),
    "Safety": (
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ),
    "Reasoning": (
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ),
}
SUBSET_WEIGHTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}


def split_pairs(
    dataset: Dataset, eval_pairs: int, train_pairs: int, seed: int
) -> tuple[Dataset, Dataset]:
    """Draw disjoint train and held-out pairs with a fixed seed.

    Pairs whose chosen and rejected conversations are identical carry no
    preference and are dropped first.

    Args:
        dataset: Preference pairs with ``chosen`` and ``rejected``.
        eval_pairs: Number of held-out pairs.
        train_pairs: Number of training pairs.
        seed: Shuffle seed.

    Returns:
        The training pairs and the held-out pairs: the first
        ``eval_pairs`` rows of the shuffled dataset are held out and the
        next ``train_pairs`` rows are for training.
    """
    distinct = dataset.filter(
        lambda row: row["chosen"] != row["rejected"]
    ).shuffle(seed=seed)
    held_out = distinct.select(range(eval_pairs))
    train = distinct.select(range(eval_pairs, eval_pairs + train_pairs))
    return train, held_out


def load_preference_splits(
    eval_pairs: int = EVAL_PAIRS,
    train_pairs: int = TRAIN_PAIRS,
    seed: int = SEED,
) -> tuple[Dataset, Dataset]:
    """Load the OLMo 2 1B preference mix and split it (``split_pairs``).

    Args:
        eval_pairs: Number of held-out pairs.
        train_pairs: Number of training pairs.
        seed: Shuffle seed.

    Returns:
        Training and held-out pairs, columns ``chosen`` and ``rejected``
        (full conversations, the prompt turns first).
    """
    dataset = cast("Dataset", load_dataset(DATASET_NAME, split="train"))
    return split_pairs(
        dataset.select_columns(["chosen", "rejected"]),
        eval_pairs,
        train_pairs,
        seed,
    )


def split_prompt(
    chosen: list[Message], rejected: list[Message]
) -> tuple[list[Message], list[Message], list[Message]]:
    """Split two full conversations into the shared prompt and responses.

    Like TRL's ``extract_prompt``: the prompt is the longest run of
    leading turns the two conversations share.

    Args:
        chosen: Preferred conversation.
        rejected: Dispreferred conversation.

    Returns:
        The prompt turns, the chosen response turns and the rejected
        response turns.
    """
    shared = 0
    for chosen_turn, rejected_turn in zip(chosen, rejected, strict=False):
        if chosen_turn != rejected_turn:
            break
        shared += 1
    return chosen[:shared], chosen[shared:], rejected[shared:]


def chat_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[Message],
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    """Render messages with the chat template and tokenize them.

    Args:
        tokenizer: Tokenizer with a chat template.
        messages: Conversation to render.
        add_generation_prompt: Append the assistant header.

    Returns:
        The token ids.
    """
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    return cast("list[int]", cast("BatchEncoding", encoded)["input_ids"])


def encode_response(
    tokenizer: PreTrainedTokenizerBase,
    prompt: list[Message],
    response: list[Message],
    max_length: int,
) -> Encoded:
    """Tokenize prompt + response as TRL's ``DPOTrainer`` does.

    The prompt is rendered with the assistant header; the response tokens
    are what the full conversation adds after it, including the template's
    end-of-turn token.

    Args:
        tokenizer: Tokenizer with a chat template.
        prompt: Prompt turns.
        response: Response turns (one assistant message).
        max_length: Keep at most this many tokens from the start.

    Returns:
        The token ids and the index of the first response token.
    """
    prompt_ids = chat_token_ids(tokenizer, prompt, add_generation_prompt=True)
    full_ids = chat_token_ids(tokenizer, prompt + response)
    return full_ids[:max_length], len(prompt_ids)


def encode_conversation_pairs(
    tokenizer: PreTrainedTokenizerBase, pairs: Dataset, max_length: int
) -> list[Encoded]:
    """Encode preference pairs stored as full conversations.

    Args:
        tokenizer: Tokenizer with a chat template.
        pairs: Rows with ``chosen`` and ``rejected`` conversations.
        max_length: Token limit per sequence.

    Returns:
        Chosen and rejected sequences interleaved:
        ``[chosen_0, rejected_0, chosen_1, ...]``.
    """
    encoded = []
    for row in pairs:
        pair = cast("dict[str, Any]", row)
        prompt, chosen, rejected = split_prompt(
            pair["chosen"], pair["rejected"]
        )
        encoded.append(encode_response(tokenizer, prompt, chosen, max_length))
        encoded.append(
            encode_response(tokenizer, prompt, rejected, max_length)
        )
    return encoded


def encode_rewardbench(
    tokenizer: PreTrainedTokenizerBase, rows: Dataset, max_length: int
) -> list[Encoded]:
    """Encode RewardBench rows (prompt and responses as plain strings).

    Args:
        tokenizer: Tokenizer with a chat template.
        rows: Rows with ``prompt``, ``chosen`` and ``rejected`` strings.
        max_length: Token limit per sequence.

    Returns:
        Chosen and rejected sequences interleaved, as
        ``encode_conversation_pairs`` returns them.
    """
    encoded = []
    for row in rows:
        pair = cast("dict[str, str]", row)
        prompt = [{"role": "user", "content": pair["prompt"]}]
        for key in ("chosen", "rejected"):
            response = [{"role": "assistant", "content": pair[key]}]
            encoded.append(
                encode_response(tokenizer, prompt, response, max_length)
            )
    return encoded


def length_batches(
    lengths: Sequence[int], token_budget: int
) -> list[list[int]]:
    """Group sequences into batches of similar length.

    Sequences are taken longest first and a batch grows while its padded
    size (rows times the longest row) fits ``token_budget``; a sequence
    longer than the budget gets a batch of its own.

    Args:
        lengths: Token count of each sequence.
        token_budget: Maximum padded tokens per batch.

    Returns:
        Batches of indices into ``lengths``; every index appears once.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    for index in order:
        # The first row of each batch is its longest.
        if (
            batches
            and (len(batches[-1]) + 1) * lengths[batches[-1][0]]
            <= token_budget
        ):
            batches[-1].append(index)
        else:
            batches.append([index])
    return batches


def response_log_prob(
    logits: torch.Tensor, token_ids: list[int], start: int
) -> float:
    """Sum the log-probabilities of the response tokens of one sequence.

    Args:
        logits: Logits of this row, ``(padded_length, vocab)``; position
            ``t`` predicts token ``t + 1``.
        token_ids: The row's unpadded token ids.
        start: Index of the first response token.

    Returns:
        The summed log-probability, 0.0 when no response token is left.
    """
    targets = torch.tensor(token_ids[start:], device=logits.device)
    if targets.numel() == 0:
        return 0.0
    predicting = logits[start - 1 : len(token_ids) - 1].float()
    log_probs = predicting.log_softmax(dim=-1)
    return float(log_probs.gather(1, targets[:, None]).sum())


def response_log_probs(
    model: torch.nn.Module,
    sequences: list[Encoded],
    pad_token_id: int,
    token_budget: int = TOKEN_BUDGET,
) -> tuple[list[float], list[int]]:
    """Score the response tokens of many sequences in padded batches.

    Rows are padded on the right, which leaves the logits of the real
    tokens unchanged in a causal model.

    Args:
        model: Causal LM (with its adapter state as it should be scored).
        sequences: Token ids and response start per sequence.
        pad_token_id: Id used for padding.
        token_budget: Maximum padded tokens per forward pass.

    Returns:
        Per sequence, the summed response log-probability and the number
        of response tokens, in the order of ``sequences``.
    """
    device = next(model.parameters()).device
    sums = [0.0] * len(sequences)
    counts = [max(len(ids) - start, 0) for ids, start in sequences]
    lengths = [len(ids) for ids, _ in sequences]
    for batch in length_batches(lengths, token_budget):
        width = lengths[batch[0]]
        input_ids = torch.full((len(batch), width), pad_token_id)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, index in enumerate(batch):
            ids = sequences[index][0]
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
        for row, index in enumerate(batch):
            ids, start = sequences[index]
            sums[index] = response_log_prob(logits[row], ids, start)
    return sums, counts


def pair_accuracy(
    chosen: Sequence[float], rejected: Sequence[float]
) -> list[bool]:
    """Compare chosen and rejected scores pair by pair.

    Args:
        chosen: Score of each chosen response.
        rejected: Score of each rejected response, same order.

    Returns:
        True where the chosen score is strictly higher (ties are wrong).
    """
    return [c > r for c, r in zip(chosen, rejected, strict=True)]


def implicit_rewards(
    policy: Sequence[float], reference: Sequence[float], beta: float = BETA
) -> list[float]:
    """Compute DPO implicit rewards from summed log-probabilities.

    Args:
        policy: Response log-probabilities under the policy.
        reference: The same responses under the reference model.
        beta: DPO temperature.

    Returns:
        ``beta * (policy - reference)`` per response.
    """
    return [beta * (p - r) for p, r in zip(policy, reference, strict=True)]


def per_token(sums: Sequence[float], counts: Sequence[int]) -> list[float]:
    """Divide summed scores by their response token counts.

    Args:
        sums: Summed score of each response.
        counts: Response tokens of each response, same order.

    Returns:
        Mean score per token; a response without tokens keeps its sum.
    """
    return [total / max(n, 1) for total, n in zip(sums, counts, strict=True)]


def rewardbench_scores(
    subsets: Sequence[str], correct: Sequence[bool]
) -> dict[str, float]:
    """Aggregate per-pair results into RewardBench section scores.

    Each section is the mean of its subset accuracies weighted by
    ``SUBSET_WEIGHTS``; the overall score is the mean of the four
    sections.

    Args:
        subsets: RewardBench subset of each pair.
        correct: Whether each pair was ranked correctly.

    Returns:
        Accuracy per section and ``"Overall"``.
    """
    by_subset: dict[str, list[bool]] = {}
    for subset, is_correct in zip(subsets, correct, strict=True):
        by_subset.setdefault(subset, []).append(is_correct)
    scores = {}
    for section, members in SECTIONS.items():
        present = [name for name in members if name in by_subset]
        weighted = sum(
            mean(by_subset[name]) * SUBSET_WEIGHTS[name] for name in present
        )
        scores[section] = weighted / sum(SUBSET_WEIGHTS[n] for n in present)
    scores["Overall"] = mean(scores[section] for section in SECTIONS)
    return scores


def load_model(model_id: str) -> PreTrainedModel:
    """Load a causal LM in bf16 for scoring.

    Args:
        model_id: Hub id or local directory.

    Returns:
        The model in eval mode, on the GPU when one is available.
    """
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    # User-supplied id or local path: no Hub revision to pin (B615).
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    return cast("PreTrainedModel", model.eval())


def score_datasets(
    model: torch.nn.Module,
    datasets: dict[str, list[Encoded]],
    pad_token_id: int,
    token_budget: int,
) -> dict[str, tuple[list[float], list[int]]]:
    """Run ``response_log_probs`` on every encoded dataset.

    Args:
        model: Causal LM to score with.
        datasets: Encoded sequences by dataset name.
        pad_token_id: Padding id.
        token_budget: Maximum padded tokens per forward pass.

    Returns:
        Summed log-probabilities and response token counts by dataset.
    """
    return {
        name: response_log_probs(model, sequences, pad_token_id, token_budget)
        for name, sequences in datasets.items()
    }


def score_models(
    adapter_dir: str,
    reference_model: str,
    datasets: dict[str, list[Encoded]],
    args: argparse.Namespace,
) -> dict[str, dict[str, tuple[list[float], list[int]]]]:
    """Score every dataset with the SFT model, the adapter and the reference.

    Args:
        adapter_dir: LoRA adapter trained on the SFT model.
        reference_model: Released DPO model, or ``""`` to skip it.
        datasets: Encoded sequences by dataset name.
        args: Parsed arguments (``pad_token_id``, ``token_budget``).

    Returns:
        Scores by model name (``sft``, ``adapter``, ``allenai_dpo``) and
        dataset name.
    """
    from peft import PeftModel  # noqa: PLC0415

    base_id = json.loads(
        (Path(adapter_dir) / "adapter_config.json").read_text()
    )["base_model_name_or_path"]
    model = PeftModel.from_pretrained(load_model(base_id), adapter_dir)
    scores = {}
    with model.disable_adapter():
        scores["sft"] = score_datasets(
            model, datasets, args.pad_token_id, args.token_budget
        )
    scores["adapter"] = score_datasets(
        model, datasets, args.pad_token_id, args.token_budget
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if reference_model:
        scores["allenai_dpo"] = score_datasets(
            load_model(reference_model),
            datasets,
            args.pad_token_id,
            args.token_budget,
        )
    return scores


def kept_pair_accuracy(scores: Sequence[float], keep: list[int]) -> float:
    """Pairwise accuracy over the selected pairs of interleaved scores.

    Args:
        scores: ``[chosen_0, rejected_0, chosen_1, ...]``.
        keep: Indices of the chosen score of each pair to count.

    Returns:
        Fraction of kept pairs whose chosen score is higher.
    """
    return mean(
        pair_accuracy([scores[i] for i in keep], [scores[i + 1] for i in keep])
    )


def held_out_report(
    scores: dict[str, dict[str, tuple[list[float], list[int]]]],
) -> dict[str, dict[str, float]]:
    """Held-out preference accuracies per model.

    Pairs in which either response has no token left after truncation are
    skipped.

    Args:
        scores: Output of ``score_models``.

    Returns:
        Per model: ``summed`` and ``normalized`` log-likelihood accuracy,
        ``reward`` (implicit reward vs the SFT model; not for the SFT
        model itself), ``reward_normalized`` (the same reward divided by
        the response length) and ``pairs`` scored.
    """
    sft_sums = scores["sft"]["held_out"][0]
    report = {}
    for name, by_dataset in scores.items():
        sums, counts = by_dataset["held_out"]
        keep = [
            i for i in range(0, len(sums), 2) if counts[i] and counts[i + 1]
        ]
        report[name] = {
            "pairs": float(len(keep)),
            "summed": kept_pair_accuracy(sums, keep),
            "normalized": kept_pair_accuracy(per_token(sums, counts), keep),
        }
        if name != "sft":
            rewards = implicit_rewards(sums, sft_sums)
            report[name]["reward"] = kept_pair_accuracy(rewards, keep)
            report[name]["reward_normalized"] = kept_pair_accuracy(
                per_token(rewards, counts), keep
            )
    return report


def rewardbench_report(
    scores: dict[str, dict[str, tuple[list[float], list[int]]]],
    subsets: Sequence[str],
    *,
    normalize: bool = False,
) -> dict[str, dict[str, float]]:
    """RewardBench implicit-reward scores per policy, SFT as reference.

    Args:
        scores: Output of ``score_models``.
        subsets: RewardBench subset of each pair.
        normalize: Divide each reward by its response token count.

    Returns:
        Section and overall accuracy per policy (all models but ``sft``).
    """
    reference = scores["sft"]["rewardbench"][0]
    report = {}
    for name, by_dataset in scores.items():
        if name == "sft":
            continue
        sums, counts = by_dataset["rewardbench"]
        rewards = implicit_rewards(sums, reference)
        if normalize:
            rewards = per_token(rewards, counts)
        correct = pair_accuracy(rewards[0::2], rewards[1::2])
        report[name] = rewardbench_scores(subsets, correct)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with the adapter, reference model and scoring settings.
    """
    parser = argparse.ArgumentParser(
        description="Held-out preference and RewardBench accuracy of a DPO "
        "adapter, the SFT model and a released DPO model"
    )
    parser.add_argument(
        "--adapter_dir",
        default=DEFAULT_ADAPTER_DIR,
        help=f"LoRA adapter from prefs_train (default: {DEFAULT_ADAPTER_DIR})",
    )
    parser.add_argument(
        "--reference_model",
        default=DPO_MODEL,
        help=f'Released DPO model to compare, "" to skip (default: '
        f"{DPO_MODEL})",
    )
    parser.add_argument(
        "--eval_pairs",
        type=int,
        default=EVAL_PAIRS,
        help=f"Held-out pairs, as in training (default: {EVAL_PAIRS})",
    )
    parser.add_argument(
        "--rewardbench_rows",
        type=int,
        default=-1,
        help="Score a seeded sample of N RewardBench rows (default: -1, all)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH,
        help=f"Token limit per sequence (default: {MAX_LENGTH})",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=TOKEN_BUDGET,
        help=f"Padded tokens per forward pass (default: {TOKEN_BUDGET})",
    )
    return parser.parse_args()


def print_reports(
    held_out: dict[str, dict[str, float]],
    rewardbench: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Print the result tables.

    Args:
        held_out: Output of ``held_out_report``.
        rewardbench: ``rewardbench_report`` outputs by reward kind
            (``summed``, ``normalized``).

    Returns:
        None.
    """
    print("\nHeld-out preference accuracy")
    header = ["pairs", "summed", "per-tok", "reward", "rew/tok"]
    print(f"{'model':<12} " + " ".join(f"{c:>7}" for c in header))
    for name, entry in held_out.items():
        rewards = [
            f"{entry[key]:.3f}" if key in entry else "-"
            for key in ("reward", "reward_normalized")
        ]
        print(
            f"{name:<12} {entry['pairs']:>7.0f} {entry['summed']:>7.3f} "
            f"{entry['normalized']:>7.3f} {rewards[0]:>7} {rewards[1]:>7}"
        )
    columns = [*SECTIONS, "Overall"]
    for kind, report in rewardbench.items():
        print(
            f"\nRewardBench implicit-reward accuracy, {kind} "
            "(reference: SFT model)"
        )
        print(f"{'model':<12} " + " ".join(f"{c:>9}" for c in columns))
        for name, sections in report.items():
            values = " ".join(f"{sections[c]:>9.3f}" for c in columns)
            print(f"{name:<12} {values}")


def main() -> None:
    """Score the models and print and save the results.

    Returns:
        None. Tables are printed; ``prefs_eval.json`` is written to
        ``--adapter_dir``.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)  # nosec B615
    args.pad_token_id = tokenizer.pad_token_id
    _train, held_out = load_preference_splits(eval_pairs=args.eval_pairs)
    rewardbench = cast(
        "Dataset", load_dataset(REWARDBENCH_NAME, split="filtered")
    )
    if args.rewardbench_rows > 0:
        rewardbench = rewardbench.shuffle(seed=SEED).select(
            range(args.rewardbench_rows)
        )
    datasets = {
        "held_out": encode_conversation_pairs(
            tokenizer, held_out, args.max_length
        ),
        "rewardbench": encode_rewardbench(
            tokenizer, rewardbench, args.max_length
        ),
    }
    scores = score_models(
        args.adapter_dir, args.reference_model, datasets, args
    )
    results = {
        "held_out": held_out_report(scores),
        "rewardbench": rewardbench_report(scores, rewardbench["subset"]),
        "rewardbench_normalized": rewardbench_report(
            scores, rewardbench["subset"], normalize=True
        ),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print_reports(
        results["held_out"],
        {
            "summed": results["rewardbench"],
            "normalized": results["rewardbench_normalized"],
        },
    )
    print(f"\nPeak GPU memory (torch): {results['peak_gpu_gb']:.1f} GB")
    output = Path(args.adapter_dir) / "prefs_eval.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
