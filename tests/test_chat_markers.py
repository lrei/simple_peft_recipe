"""Chat marker inference against the real tokenizers of each family.

The tokenizers come from the Hugging Face hub (CPU only, a few files
each): Qwen 2.5 and 3.5 (ChatML), Gemma 4 and Gemma 3, Granite 3.3 and
gpt-oss (harmony).
"""

from __future__ import annotations

import pytest

from speftr import ChatMarkers, infer_chat_markers
from speftr.chat_markers import KNOWN_MARKERS


@pytest.mark.parametrize(
    ("model_name", "family"),
    [
        ("Qwen/Qwen2.5-0.5B-Instruct", "ChatML (Qwen 2.5 / 3 / 3.5 / 3.8)"),
        ("Qwen/Qwen3.5-4B", "ChatML (Qwen 2.5 / 3 / 3.5 / 3.8)"),
        ("google/gemma-4-E4B-it", "Gemma 4"),
        ("unsloth/gemma-3-270m-it", "Gemma 2 / 3 / 3n"),
        ("ibm-granite/granite-3.3-2b-instruct", "Granite 3.x"),
        ("openai/gpt-oss-20b", "gpt-oss (harmony)"),
    ],
)
def test_known_families_are_recognised(model_name, family):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    markers = infer_chat_markers(tokenizer)
    assert isinstance(markers, ChatMarkers)
    assert markers.family == family


def test_unknown_template_raises_with_the_rendered_text(
    qwen_tokenizer, monkeypatch
):
    monkeypatch.setattr(
        qwen_tokenizer,
        "chat_template",
        "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}",
    )
    with pytest.raises(ValueError, match=r"\[user\] Hello"):
        infer_chat_markers(qwen_tokenizer)


def test_known_marker_pairs_are_distinct():
    pairs = {(m.instruction_part, m.response_part) for m in KNOWN_MARKERS}
    assert len(pairs) == len(KNOWN_MARKERS)
