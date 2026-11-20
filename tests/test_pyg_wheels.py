"""Tests for src/pyg_wheels.py (no torch and no network needed)."""

from __future__ import annotations

import pytest

from src.pyg_wheels import cuda_tag, install_command, parse_arguments, wheel_index_urls


def test_cuda_eleven_point_eight_is_called_cu118() -> None:
    assert cuda_tag("11.8") == "cu118"
    assert cuda_tag("12.4") == "cu124"


def test_a_torch_without_cuda_is_refused_with_a_hint_about_the_accelerator() -> None:
    with pytest.raises(RuntimeError, match="accelerator"):
        cuda_tag(None)


def test_a_patch_release_also_tries_the_x_y_zero_wheel_page() -> None:
    urls = wheel_index_urls("2.5.1+cu118", "11.8")
    assert urls == [
        "https://data.pyg.org/whl/torch-2.5.1+cu118.html",
        "https://data.pyg.org/whl/torch-2.5.0+cu118.html",
    ]


def test_an_x_y_zero_release_has_one_wheel_page() -> None:
    assert wheel_index_urls("2.4.0", "12.1") == [
        "https://data.pyg.org/whl/torch-2.4.0+cu121.html"
    ]


def test_pip_is_told_to_use_binary_wheels_only_and_every_wheel_page() -> None:
    command = install_command(["pyg_lib"], ["u1", "u2"])
    assert "--only-binary=:all:" in command
    assert command[-4:] == ["-f", "u1", "-f", "u2"]
    assert "pyg_lib" in command


def test_the_only_option_limits_the_install_to_the_named_add_ons() -> None:
    assert parse_arguments(["--only", "pyg_lib"]).only == ["pyg_lib"]
    assert parse_arguments([]).only is None
