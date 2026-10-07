# SPDX-FileCopyrightText: 2026 Luis Rei
# SPDX-License-Identifier: BSD-2-Clause
r"""Chat template markers for response-only training.

Response-only training (``PESFTConfig.train_on_responses``) masks every
token before the response marker of each assistant turn, so the markers
must match the chat template of the model being trained. Each model
family uses its own markers. ``infer_chat_markers`` renders a short
conversation with the tokenizer and returns the marker pair of the
family whose markers appear in it::

    from transformers import AutoTokenizer
    from speftr.chat_markers import infer_chat_markers

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    markers = infer_chat_markers(tokenizer)
    markers.instruction_part  # "<|turn>user\n"
    markers.response_part  # "<|turn>model\n"

``PESFT.train`` calls it when ``instruction_part`` or ``response_part``
is empty. Templates outside ``KNOWN_MARKERS`` need explicit markers:
print ``tokenizer.apply_chat_template(messages, tokenize=False)`` and
copy the text that opens the user and assistant turns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class ChatMarkers(NamedTuple):
    """Markers that open the user and assistant turns of a chat template.

    Attributes:
        family: Model family the markers belong to, for messages.
        instruction_part: Text that opens a user turn.
        response_part: Text that opens an assistant turn.
    """

    family: str
    instruction_part: str
    response_part: str


KNOWN_MARKERS: tuple[ChatMarkers, ...] = (
    ChatMarkers(
        "ChatML (Qwen 2.5 / 3 / 3.5 / 3.8)",
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
    ),
    ChatMarkers("Gemma 4", "<|turn>user\n", "<|turn>model\n"),
    ChatMarkers(
        "Gemma 2 / 3 / 3n", "<start_of_turn>user\n", "<start_of_turn>model\n"
    ),
    ChatMarkers(
        "Llama 3.x",
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    ChatMarkers(
        "gpt-oss (harmony)", "<|start|>user<|message|>", "<|start|>assistant"
    ),
    ChatMarkers(
        "Granite 3.x",
        "<|start_of_role|>user<|end_of_role|>",
        "<|start_of_role|>assistant<|end_of_role|>",
    ),
)
"""Marker pairs of the supported chat template families."""

_PROBE_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]


def render_probe(tokenizer: PreTrainedTokenizerBase) -> str:
    """Render a one-exchange conversation with the tokenizer's template.

    Args:
        tokenizer: Tokenizer, or a processor wrapping one, with a chat
            template.

    Returns:
        The rendered text of a user turn followed by an assistant turn.
    """
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    rendered = tokenizer.apply_chat_template(_PROBE_MESSAGES, tokenize=False)
    return str(rendered)


def infer_chat_markers(tokenizer: PreTrainedTokenizerBase) -> ChatMarkers:
    """Find the known marker pair that the tokenizer's template renders.

    Args:
        tokenizer: Tokenizer, or a processor wrapping one, with a chat
            template.

    Returns:
        The first entry of ``KNOWN_MARKERS`` whose two markers both occur
        in the rendered conversation.

    Raises:
        ValueError: If no known pair matches. The message carries the
            rendered conversation so the markers can be copied from it.
    """
    rendered = render_probe(tokenizer)
    for markers in KNOWN_MARKERS:
        if (
            markers.instruction_part in rendered
            and markers.response_part in rendered
        ):
            return markers
    msg = (
        "No known chat markers match this template; set instruction_part "
        f"and response_part from the rendered conversation:\n{rendered}"
    )
    raise ValueError(msg)
