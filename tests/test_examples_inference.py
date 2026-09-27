"""Tests for the per-example inference scripts (``<example>_inference.py``).

Every script carries the same loading, merging and generation code, so
those tests run once per script. Adapters are built on tiny real
checkpoints from the Hub (a text-only Qwen2 and the multimodal Qwen 3.5)
with random, non-zero LoRA weights: a loaded adapter changes the logits
measurably, one that fails to load leaves them equal to the base. Prompt
builders are checked against real dataset rows. The GPU vLLM test uses
Qwen2.5 0.5B.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import torch

from examples.guard import guard_inference
from examples.instruct import instruct_inference
from examples.intent import intent_inference
from examples.rgym import rgym_inference
from examples.text2sql import text2sql_inference


ROOT = Path(__file__).resolve().parents[1]
TINY_QWEN2 = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_QWEN35 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"
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
    guard_inference,
    instruct_inference,
    intent_inference,
    text2sql_inference,
    rgym_inference,
]
SCRIPT_IDS = [script.__name__.rsplit(".", 1)[-1] for script in SCRIPTS]


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
    adapter_replies = script.run_adapter_route(
        str(adapter_dir), model_id, chats, "transformers", 4
    )

    script.merge_adapter(str(adapter_dir), model_id, str(tmp_path))
    merged = script.load_model(str(tmp_path))

    assert not any("lora" in name for name, _ in merged.named_parameters())
    assert (tmp_path / "tokenizer_config.json").is_file()
    assert (tmp_path / "chat_template.jinja").is_file()
    torch.testing.assert_close(
        last_token_logits(merged), adapted, atol=0.05, rtol=0.05
    )
    merged_replies = script.run_merged_route(
        str(tmp_path), chats, "transformers", 4
    )
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
        ["Hi", "Name three colours of the rainbow, please."]
    )

    batched = guard_inference.generate_transformers(model, tokenizer, chats, 5)
    single = [
        guard_inference.generate_transformers(model, tokenizer, [chat], 5)[0]
        for chat in chats
    ]
    assert batched == single


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

    adapted = script.generate_vllm(VLLM_MODEL, chats, 8, adapter_dir)
    merged = script.generate_vllm(f"{work_dir}/merged", chats, 8)
    base = script.generate_vllm(base_dir, chats, 8)
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
