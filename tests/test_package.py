"""Tests for the ``speftr`` package surface and its lazy PESFT exports."""

from __future__ import annotations

import subprocess
import sys

import pytest

import speftr
from speftr import perl, pesft


def test_all_contents():
    assert sorted(speftr.__all__) == sorted(
        [
            "PERL",
            "PESFT",
            "PERLConfig",
            "PESFTConfig",
            "__version__",
            "display_parameters",
            "save_parameters_to_json",
        ]
    )


def test_all_names_resolve():
    for name in speftr.__all__:
        assert getattr(speftr, name) is not None


def test_version_is_string():
    assert isinstance(speftr.__version__, str)
    assert speftr.__version__


def test_perl_exported_eagerly():
    assert speftr.PERL is perl.PERL
    assert speftr.PERLConfig is perl.PERLConfig


@pytest.mark.parametrize(
    "name",
    ["PESFT", "PESFTConfig", "display_parameters", "save_parameters_to_json"],
)
def test_lazy_pesft_exports(name):
    assert getattr(speftr, name) is getattr(pesft, name)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = speftr.nope


def test_lazy_exports_do_not_import_unsloth():
    # Fresh interpreter: this process may already have imported unsloth.
    code = (
        "import sys, speftr; "
        "assert 'speftr.pesft' not in sys.modules; "
        "speftr.PESFTConfig(); speftr.display_parameters({}); "
        "assert 'speftr.pesft' in sys.modules; "
        "assert 'unsloth' not in sys.modules, 'unsloth imported'"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
