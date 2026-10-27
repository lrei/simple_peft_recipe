"""Tests for the per-example inference scripts (``<example>_inference.py``).

Every script carries the same loading, merging and generation code, so
those tests run once per script. Adapters are built on tiny real
checkpoints from the Hub (a text-only Qwen2 and the multimodal Qwen 3.5)
with random, non-zero LoRA weights: a loaded adapter changes the logits
measurably, one that fails to load leaves them equal to the base. Prompt
builders are checked against real dataset rows; token-budget batching and
prompt truncation use rows of ``dolly_pirate_rows.json``. vLLM's
truncation is checked on CPU with its tokenization parameters and a
stand-in engine; the GPU vLLM test uses Qwen2.5 0.5B.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
import types
from pathlib import Path

import pytest
import torch

from examples.big import big_inference
from examples.gptoss import gptoss_inference
from examples.guard import guard_inference
from examples.instruct import instruct_inference
from examples.intent import intent_inference
from examples.prefs import prefs_inference
from examples.rgym import rgym_inference
from examples.text2sql import text2sql_inference
from examples.tldr import tldr_inference


ROOT = Path(__file__).resolve().parents[1]
TINY_QWEN2 = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_QWEN35 = (
    "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink"
)
# vLLM needs head sizes of at least 16, which the tiny checkpoints lack.
VLLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QUANTIZED_MODEL = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
# Target pattern Unsloth writes for multimodal checkpoints (copied from a
# Qwen 3.5 adapter trained with speftr): language-model projections only.
MULTIMODAL_TARGETS = (
    r"(?:.*?(?:language|text).*?(?:self_attn|attention|attn|mixer|mlp"
    r"|feed_forward|ffn|dense|mixer).*?(?:gate_proj|up_proj|down_proj"
    r"|q_proj|k_proj|v_proj|o_proj))|(?:\bmodel\.layers\.[\d]{1,}\."
    r"(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer)"
    r"\.(?:(?:gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)))"
)
TARGETS = {
    TINY_QWEN2: ["q_proj", "v_proj", "down_proj"],
    VLLM_MODEL: ["q_proj", "v_proj", "down_proj"],
    TINY_QWEN35: MULTIMODAL_TARGETS,
}
SCRIPTS = [
    big_inference,
    guard_inference,
    instruct_inference,
    intent_inference,
    prefs_inference,
    text2sql_inference,
    tldr_inference,
    rgym_inference,
]
SCRIPT_IDS = [script.__name__.rsplit(".", 1)[-1] for script in SCRIPTS]
# Every script with token-budget generation: (module, generate function,
# name of its context-length setting).
GENERATORS = [
    (script, script.generate_transformers, "max_model_len")
    for script in SCRIPTS
] + [(gptoss_inference, gptoss_inference.generate, "max_seq_length")]
GENERATOR_IDS = [*SCRIPT_IDS, "gptoss_inference"]


def save_random_adapter(model_id: str, adapter_dir: Path) -> None:
    """Save a LoRA adapter with random weights and the base tokenizer.

    ``lora_B`` is randomised after peft's default zero init, so the saved
    config keeps ``init_lora_weights=True`` as in trained adapters: weights
    that fail to load stay zero.

    Args:
        model_id: Base checkpoint.
        adapter_dir: Output directory, laid out like ``save_model("lora")``.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    config = LoraConfig(
        r=4, target_modules=TARGETS[model_id], task_type="CAUSAL_LM"
    )
    base = guard_inference.load_model(model_id)
    model = get_peft_model(base, config)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.1)
    model.save_pretrained(str(adapter_dir))
    AutoTokenizer.from_pretrained(model_id).save_pretrained(adapter_dir)


def last_token_logits(model: torch.nn.Module) -> torch.Tensor:
    """Run a fixed token sequence and return the final position's logits.

    Args:
        model: Any causal LM.

    Returns:
        Logits for the last position, as float32 on the CPU.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([[1, 20, 300, 42, 7]], device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
    return logits[0, -1].float().cpu()


def count_nonzero_lora_b(model: torch.nn.Module) -> int:
    """Count ``lora_B`` matrices holding non-zero values.

    Args:
        model: A model with LoRA layers.

    Returns:
        The number of loaded (non-zero) ``lora_B`` weights.
    """
    return sum(
        bool(parameter.abs().sum() > 0)
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    )


@pytest.fixture(scope="module", params=[TINY_QWEN2, TINY_QWEN35])
def adapter(request, tmp_path_factory) -> tuple[str, Path]:
    adapter_dir = tmp_path_factory.mktemp("adapter")
    save_random_adapter(request.param, adapter_dir)
    return request.param, adapter_dir


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_IDS)
def test_adapter_route_applies_saved_weights(script, adapter):
    model_id, adapter_dir = adapter
    model = script.load_adapter_model(str(adapter_dir), model_id)

    assert type(model.get_base_model()).__name__ in model_id
    assert count_nonzero_lora_b(model) > 0
    adapted = last_token_logits(model)
    with model.disable_adapter():
        base = last_token_logits(model)
    assert not torch.allclose(adapted, base, atol=1e-3)


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_IDS)
def test_merged_route_matches_adapter_route(script, adapter, tmp_path):
    model_id, adapter_dir = adapter
    adapted = last_token_logits(
        script.load_adapter_model(str(adapter_dir), model_id)
    )
    chats = [[{"role": "user", "content": "Name three colours, please."}]]
    settings = script.GenerationSettings(max_new_tokens=4)
    adapter_replies = script.run_adapter_route(
        str(adapter_dir), model_id, chats, settings
    )

    script.merge_adapter(str(adapter_dir), model_id, str(tmp_path))
    merged = script.load_model(str(tmp_path))

    assert not any("lora" in name for name, _ in merged.named_parameters())
    assert (tmp_path / "tokenizer_config.json").is_file()
    assert (tmp_path / "chat_template.jinja").is_file()
    torch.testing.assert_close(
        last_token_logits(merged), adapted, atol=0.05, rtol=0.05
    )
    merged_replies = script.run_merged_route(str(tmp_path), chats, settings)
    assert merged_replies == adapter_replies


def test_merged_text_model_loads_with_auto_class(tmp_path):
    from transformers import AutoModelForCausalLM

    save_random_adapter(TINY_QWEN2, tmp_path / "adapter")
    guard_inference.merge_adapter(
        str(tmp_path / "adapter"), TINY_QWEN2, str(tmp_path / "merged")
    )

    model = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
    assert type(model).__name__ == "Qwen2ForCausalLM"


def test_generate_batches_with_left_padding(tmp_path):
    save_random_adapter(TINY_QWEN2, tmp_path)
    model = guard_inference.load_adapter_model(str(tmp_path), TINY_QWEN2)
    # float32: bf16 rounding differs between padded and unpadded rows.
    model = model.float()
    tokenizer = guard_inference.load_tokenizer(str(tmp_path))
    chats = guard_inference.build_chats(
        [
            "Hi",
            "Name three colours of the rainbow, please.",
            "What is the capital of France?",
        ]
    )
    encoded = tokenizer.apply_chat_template(
        chats, add_generation_prompt=True, return_dict=True
    )
    lengths = [len(ids) + 5 for ids in encoded["input_ids"]]
    # Room for two rows: the longest two share a padded batch.
    settings = guard_inference.GenerationSettings(
        max_new_tokens=5, token_budget=2 * max(lengths)
    )
    assert len(guard_inference.length_batches(lengths, 2 * max(lengths))) == 2

    batched = guard_inference.generate_transformers(
        model, tokenizer, chats, settings
    )
    single = [
        guard_inference.generate_transformers(
            model, tokenizer, [chat], settings
        )[0]
        for chat in chats
    ]
    assert batched == single


@pytest.mark.parametrize(
    "script", [entry[0] for entry in GENERATORS], ids=GENERATOR_IDS
)
def test_length_batches_fit_the_token_budget(
    script, load_fixture, qwen_tokenizer
):
    rows = load_fixture("dolly_pirate_rows.json")
    lengths = [
        len(qwen_tokenizer(row["response"])["input_ids"]) for row in rows
    ]
    budget = 3 * max(lengths)
    lengths.append(budget + 1)

    batches = script.length_batches(lengths, budget)

    indices = sorted(index for batch in batches for index in batch)
    assert indices == list(range(len(lengths)))
    # Longest first: the input above the budget runs alone.
    assert batches[0] == [len(lengths) - 1]
    padded = [
        len(batch) * max(lengths[index] for index in batch)
        for batch in batches[1:]
    ]
    assert max(padded) <= budget
    assert len(batches) < len(lengths)


@pytest.fixture(scope="module")
def tiny_model():
    return guard_inference.load_model(TINY_QWEN2)


@pytest.fixture(scope="module")
def dolly_chats(load_fixture) -> list[list[dict[str, str]]]:
    """Two short Dolly instructions around one long chat of responses."""
    rows = load_fixture("dolly_pirate_rows.json")
    long_text = "\n\n".join(row["response"] for row in rows * 3)
    return [
        [{"role": "user", "content": rows[0]["instruction"]}],
        [{"role": "user", "content": long_text}],
        [{"role": "user", "content": rows[1]["instruction"]}],
    ]


def prompt_ids(tokenizer, chats) -> list[list[int]]:
    """Generation-prompt token ids of each chat."""
    encoded = tokenizer.apply_chat_template(
        chats, add_generation_prompt=True, return_dict=True
    )
    return encoded["input_ids"]


@pytest.mark.parametrize("generator", GENERATORS, ids=GENERATOR_IDS)
def test_generate_truncates_input_longer_than_context(
    generator, tiny_model, dolly_chats, qwen_tokenizer, capsys
):
    script, generate, limit_name = generator
    prompts = prompt_ids(qwen_tokenizer, dolly_chats)
    longest = len(prompts[1])
    limit = longest - 5
    settings = script.GenerationSettings(
        max_new_tokens=8, **{limit_name: limit + 8}
    )

    truncated = script.truncate_prompts(prompts, settings)
    replies = generate(tiny_model, qwen_tokenizer, dolly_chats, settings)

    assert truncated == [prompts[0], prompts[1][-limit:], prompts[2]]
    assert len(replies) == len(dolly_chats)
    output = capsys.readouterr().out
    warning = (
        f"WARNING: 1 of 3 prompts are longer than {limit} tokens (context "
        f"{limit + 8} minus max_new_tokens 8; longest: {longest}). Their "
        "beginning is cut off, so the model never sees it."
    )
    assert output.count(warning) == 2
    assert f"Fix: raise --{limit_name} (more memory)" in output


@pytest.mark.parametrize("generator", GENERATORS, ids=GENERATOR_IDS)
def test_generate_rejects_max_new_tokens_filling_the_context(
    generator, tiny_model, dolly_chats, qwen_tokenizer
):
    script, generate, limit_name = generator
    settings = script.GenerationSettings(max_new_tokens=64, **{limit_name: 64})

    with pytest.raises(
        ValueError,
        match=rf"--max_new_tokens 64 leaves no room.*--{limit_name}",
    ):
        generate(tiny_model, qwen_tokenizer, dolly_chats, settings)


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_IDS)
def test_vllm_truncation_cuts_like_the_transformers_engine(
    script, dolly_chats, qwen_tokenizer
):
    from vllm.renderers import TokenizeParams

    prompts = prompt_ids(qwen_tokenizer, dolly_chats)
    limit = len(prompts[1]) - 5
    settings = script.GenerationSettings(
        max_new_tokens=8, max_model_len=limit + 8
    )
    options = script.vllm_truncation(settings)
    # The parameters LLM.chat builds from its tokenization_kwargs.
    params = TokenizeParams(
        max_total_tokens=settings.max_model_len, add_special_tokens=False
    ).with_kwargs(**options)

    cut = [
        params.apply_post_tokenization(None, {"prompt_token_ids": ids})[
            "prompt_token_ids"
        ]
        for ids in prompts
    ]

    assert options == {
        "truncate_prompt_tokens": limit,
        "truncation_side": "left",
    }
    assert cut == script.truncate_prompts(prompts, settings)


def test_generate_vllm_passes_truncation_and_warns(
    tmp_path, dolly_chats, qwen_tokenizer, monkeypatch, capsys
):
    import vllm

    calls = []

    class FakeLLM:
        """Stands in for ``vllm.LLM``: records ``chat`` calls."""

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def chat(self, chats, sampling_params, **kwargs):
            calls.append({"engine": self.kwargs, **kwargs})
            reply = types.SimpleNamespace(text="Arr.")
            return [types.SimpleNamespace(outputs=[reply]) for _ in chats]

    save_random_adapter(TINY_QWEN2, tmp_path)
    monkeypatch.setattr(vllm, "LLM", FakeLLM)
    prompts = prompt_ids(qwen_tokenizer, dolly_chats)
    limit = len(prompts[1]) - 5
    settings = guard_inference.GenerationSettings(
        engine="vllm", max_new_tokens=8, max_model_len=limit + 8
    )

    replies = guard_inference.generate_vllm(
        TINY_QWEN2, dolly_chats, settings, str(tmp_path)
    )

    assert replies == ["Arr."] * 3
    (call,) = calls
    assert call["engine"]["max_model_len"] == limit + 8
    assert call["tokenization_kwargs"] == {
        "truncate_prompt_tokens": limit,
        "truncation_side": "left",
    }
    assert call["lora_request"].lora_path == str(tmp_path)
    assert "WARNING: 1 of 3 prompts are longer than" in capsys.readouterr().out


def _write_adapter_config(adapter_dir: Path, base: str, rank: int) -> None:
    """Write a minimal ``adapter_config.json``.

    Args:
        adapter_dir: Directory to write into.
        base: ``base_model_name_or_path``.
        rank: LoRA ``r``.
    """
    config = {"base_model_name_or_path": base, "r": rank}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_IDS)
def test_resolve_base_model(script, tmp_path):
    _write_adapter_config(tmp_path, TINY_QWEN2, 8)

    assert script.resolve_base_model(str(tmp_path), None) == TINY_QWEN2
    assert script.resolve_base_model(str(tmp_path), VLLM_MODEL) == VLLM_MODEL
    _write_adapter_config(tmp_path, QUANTIZED_MODEL, 8)
    with pytest.raises(ValueError, match="16-bit original"):
        script.resolve_base_model(str(tmp_path), None)


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPT_IDS)
def test_vllm_max_lora_rank_rounds_up_to_vllm_rank(script, tmp_path):
    limits = []
    for rank in (1, 2, 8, 16, 20):
        _write_adapter_config(tmp_path, TINY_QWEN2, rank)
        limits.append(script.vllm_max_lora_rank(str(tmp_path)))
    assert limits == [1, 8, 8, 16, 32]


def test_guard_chats_use_training_instruction(load_fixture):
    rows = load_fixture("aegis2_test_rows.json")
    chats = guard_inference.build_chats([row["prompt"] for row in rows])

    assert chats[0] == [
        {
            "role": "user",
            "content": "Classify this prompt's as harmful or unharmful:\n\n"
            + rows[0]["prompt"],
        }
    ]
    assert len(chats) == len(rows)


def test_intent_chats_list_every_banking77_intent(load_fixture):
    fixture = load_fixture("banking77_test_rows.json")
    texts = [row["text"] for row in fixture["rows"]]
    chats = intent_inference.build_chats(texts)

    assert intent_inference.INTENTS.split(", ") == fixture["label_names"]
    content = chats[0][0]["content"]
    assert content.startswith(intent_inference.INSTRUCTION + "\n\nIntents: ")
    assert f"Intents: {', '.join(fixture['label_names'])}\n\n" in content
    assert content.endswith(f"\n\nMessage: {texts[0].strip()}")
    assert [chat[0]["role"] for chat in chats] == ["user"] * len(texts)


def test_text2sql_chats_match_training_prompt(load_fixture):
    from examples.text2sql.text2sql_eval import (
        CHAT_TEMPLATE_KWARGS,
        build_messages,
    )

    rows = load_fixture("sql_create_context_rows.json")
    chats = text2sql_inference.build_chats(
        [row["context"] for row in rows], [row["question"] for row in rows]
    )

    assert chats == [
        build_messages(row["context"], row["question"]) for row in rows
    ]
    assert text2sql_inference.CHAT_TEMPLATE_KWARGS == CHAT_TEMPLATE_KWARGS


def test_tldr_chats_drop_the_cue_and_use_training_instruction(load_fixture):
    rows = load_fixture("tldr_rows.json")
    posts = [row["prompt"] for row in rows]
    chats = tldr_inference.build_chats(posts)

    assert tuple(posts[:2]) == tldr_inference.DEFAULT_PROMPTS
    assert posts[0].endswith("TL;DR:")
    body = posts[0].removesuffix("TL;DR:").strip()
    assert chats[0] == [
        {
            "role": "user",
            "content": f"{tldr_inference.INSTRUCTION}\n\n{body}",
        }
    ]
    assert [chat[0]["role"] for chat in chats] == ["user"] * len(posts)
    assert tldr_inference.CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}


def test_instruct_chats_start_with_pirate_system_prompt(load_fixture):
    rows = load_fixture("dolly_pirate_rows.json")
    instructions = [row["instruction"] for row in rows if not row["context"]]
    chats = instruct_inference.build_chats(instructions)

    assert chats[0] == [
        {
            "role": "system",
            "content": "You are a pirate, always respond in pirate speech.",
        },
        {"role": "user", "content": instructions[0]},
    ]
    custom = instruct_inference.build_chats(instructions[:1], "Be brief.")
    assert custom[0][0] == {"role": "system", "content": "Be brief."}


def test_rgym_prompt_matches_training_prompt(qwen_tokenizer):
    import reasoning_gym
    from reasoning_gym.utils import SYSTEM_PROMPTS

    from examples.rgym.rgym import ReasoningGymDataset

    procedural = reasoning_gym.create_dataset("chain_sum", size=2, seed=2)
    training = ReasoningGymDataset(
        qwen_tokenizer,
        procedural,
        SYSTEM_PROMPTS["DeepSeekZero"],
        developer_role="system",
    )
    question = procedural[0]["question"]
    chat = rgym_inference.build_chats([question])[0]

    rendered = qwen_tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        **rgym_inference.CHAT_TEMPLATE_KWARGS,
    )
    assert rendered == training[0]["prompt"]


def _vllm_applies_adapter(adapter_dir: str, work_dir: str) -> None:
    """Assert vLLM applies the adapter in both routes.

    The random adapter makes near-tied tokens common, so the two routes
    are compared with the base model, not with each other.

    Runs in a subprocess: vLLM needs its own process to start its engine
    and release GPU memory reliably.

    Args:
        adapter_dir: Adapter directory built on ``VLLM_MODEL``.
        work_dir: Directory for the merged and base checkpoints.
    """
    from transformers import AutoTokenizer

    script = guard_inference
    chats = [[{"role": "user", "content": "Hello there, how are you today?"}]]
    base_dir = f"{work_dir}/base"
    script.load_model(VLLM_MODEL).save_pretrained(base_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(base_dir)
    script.merge_adapter(adapter_dir, VLLM_MODEL, f"{work_dir}/merged")
    script.free_gpu_memory()

    settings = script.GenerationSettings(max_new_tokens=8)
    adapted = script.generate_vllm(VLLM_MODEL, chats, settings, adapter_dir)
    merged = script.generate_vllm(f"{work_dir}/merged", chats, settings)
    base = script.generate_vllm(base_dir, chats, settings)
    assert adapted != base
    assert merged != base
    print("VLLM_ADAPTER_OK")


@pytest.mark.cuda
def test_vllm_applies_adapter_in_both_routes(tmp_path):
    save_random_adapter(VLLM_MODEL, tmp_path / "adapter")
    code = (
        "from tests.test_examples_inference import _vllm_applies_adapter; "
        f"_vllm_applies_adapter({str(tmp_path / 'adapter')!r}, "
        f"{str(tmp_path)!r})"
    )
    result = subprocess.run(  # noqa: S603  # nosec B603
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "VLLM_ADAPTER_OK" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-4000:]
    )


@pytest.mark.parametrize(
    "stop_script", [intent_inference, tldr_inference], ids=["intent", "tldr"]
)
@pytest.mark.parametrize(
    "configured", [None, 248044, [248044], [248046, 248044]]
)
def test_stop_token_ids_add_the_tokenizer_eos(stop_script, configured):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    model = types.SimpleNamespace(
        generation_config=types.SimpleNamespace(eos_token_id=configured)
    )
    ids = stop_script.stop_token_ids(model, tokenizer)
    assert tokenizer.eos_token_id in ids
    assert ids == sorted(set(ids))
    if isinstance(configured, int):
        assert configured in ids
    elif configured:
        assert set(configured) <= set(ids)
