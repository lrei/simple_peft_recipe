# SPDX-FileCopyrightText: 2025-2026 Luis Rei
# SPDX-License-Identifier: BSD-2-Clause
"""speftr: Parameter-efficient training utilities with LoRA adapters.

Wrappers around TRL's trainers that train LoRA adapters on language models.

- PESFT: Supervised fine-tuning with SFTTrainer (uses Unsloth)
- PERL: Reinforcement learning with GRPOTrainer (TRL-only, no Unsloth)
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

# Lazy imports to avoid importing unsloth when only using PERL
from .perl import PERL, PERLConfig


try:  # pragma: no cover - metadata lookup is environment-dependent
    __version__ = version("speftr")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"


if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from .pesft import (
        PESFT,
        PESFTConfig,
        display_parameters,
        save_parameters_to_json,
    )


def __getattr__(name: str) -> object:
    """Lazily import PESFT symbols so PERL users never import unsloth.

    Args:
        name: Attribute requested from the ``speftr`` package.

    Returns:
        The requested object from ``speftr.pesft``.

    Raises:
        AttributeError: If ``name`` is not a lazily exported symbol.
    """
    if name in (
        "PESFT",
        "PESFTConfig",
        "display_parameters",
        "save_parameters_to_json",
    ):
        return getattr(import_module("speftr.pesft"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "PERL",
    "PESFT",
    "PERLConfig",
    "PESFTConfig",
    "__version__",
    "display_parameters",
    "save_parameters_to_json",
]
