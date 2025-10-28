"""speftr: Parameter-efficient training utilities with LoRA adapters.

This package provides reusable class-based interfaces for training language
models with LoRA adapters using TRL's trainers.

- PESFT: Supervised fine-tuning with SFTTrainer (uses Unsloth)
- PERL: Reinforcement learning with GRPOTrainer (TRL-only, no Unsloth)
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Protocol, cast

# Lazy imports to avoid importing unsloth when only using PERL
from .perl import PERL, PERLConfig


try:  # pragma: no cover - metadata lookup is environment-dependent
    __version__ = version("speftr")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"


class _PesftModule(Protocol):
    """Protocol describing the PESFT module's public API."""

    PESFT: type[object]
    PESFTConfig: type[object]

    @staticmethod
    def display_parameters(*args: object, **kwargs: object) -> object: ...

    @staticmethod
    def save_parameters_to_json(*args: object, **kwargs: object) -> object: ...


if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from .pesft import (
        PESFT,
        PESFTConfig,
        display_parameters,
        save_parameters_to_json,
    )


def __getattr__(name: str) -> object:
    """Lazy import for PESFT to avoid unsloth for PERL users."""
    if name in (
        "PESFT",
        "PESFTConfig",
        "display_parameters",
        "save_parameters_to_json",
    ):
        module = cast("_PesftModule", import_module("speftr.pesft"))

        if name == "PESFT":
            return module.PESFT
        if name == "PESFTConfig":
            return module.PESFTConfig
        if name == "display_parameters":
            return module.display_parameters
        if name == "save_parameters_to_json":
            return module.save_parameters_to_json
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
