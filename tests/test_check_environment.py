"""Tests for src/check_environment.py (the parts that need no torch)."""

from __future__ import annotations

from src.check_environment import (
    CheckResult,
    check_import,
    cuda_problem,
    failed,
    run_safely,
)


def test_a_cpu_only_torch_is_reported_as_having_no_cuda() -> None:
    assert "CPU-only" in cuda_problem("2.4.0", None, False)


def test_cuda_torch_without_a_usable_gpu_says_to_turn_the_accelerator_on() -> None:
    assert "GPU accelerator" in cuda_problem("2.4.0", "11.8", False)


def test_cuda_torch_with_a_gpu_has_no_problem() -> None:
    assert cuda_problem("2.4.0", "11.8", True) is None


def test_a_missing_module_fails_the_import_check_with_its_error() -> None:
    result = check_import("surely_not_an_installed_module")
    assert not result.passed and "surely_not_an_installed_module" in result.detail


def test_an_installed_module_passes_the_import_check() -> None:
    assert check_import("json").passed


def test_a_crashing_check_becomes_a_failed_result_instead_of_an_exception() -> None:
    def crash() -> CheckResult:
        raise OSError("libc10_cuda.so: cannot open shared object file")

    result = run_safely("scatter", crash)
    assert not result.passed and "libc10_cuda" in result.detail


def test_only_failed_checks_are_listed_as_problems() -> None:
    results = [CheckResult("a", True, ""), CheckResult("b", False, "broken")]
    assert [r.name for r in failed(results)] == ["b"]
