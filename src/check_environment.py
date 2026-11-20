"""Check that torch, CUDA and the graph libraries work together before any real run.

Prints the versions, then tries what Multi-GNN needs: a CUDA-capable torch, the
PyG extension libraries, `datatable` (IBM's formatter) and a neighbour-sampling
loader on a tiny graph. Exits with code 1 and a list of problems if anything is
wrong, so a notebook can stop instead of running on a broken setup.

Run inside the environment you want to check:
    python -m src.check_environment
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

# Imported by Multi-GNN, by IBM's formatter, or by PyG's neighbour sampler.
REQUIRED_MODULES = [
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "datatable",
    "wandb",
    "tqdm",
    "sklearn",
    "pandas",
]


@dataclass(frozen=True)
class CheckResult:
    """One check: its name, whether it passed, and a short explanation."""

    name: str
    passed: bool
    detail: str


def cuda_problem(
    torch_version: str, cuda_version: Optional[str], cuda_available: bool
) -> Optional[str]:
    """What is wrong with torch's CUDA setup, or None if it is fine."""
    if cuda_version is None:
        return "torch %s is a CPU-only build (no CUDA)" % torch_version
    if not cuda_available:
        return (
            "torch %s was built for CUDA %s but no GPU is usable: "
            "turn the GPU accelerator on" % (torch_version, cuda_version)
        )
    return None


def check_torch_cuda() -> CheckResult:
    """Is torch installed with a working CUDA?"""
    import torch

    detail = "torch %s, CUDA build %s, CUDA available: %s" % (
        torch.__version__,
        torch.version.cuda,
        torch.cuda.is_available(),
    )
    problem = cuda_problem(
        torch.__version__, torch.version.cuda, torch.cuda.is_available()
    )
    return CheckResult("torch CUDA", problem is None, problem or detail)


def check_import(module_name: str) -> CheckResult:
    """Can this module be imported?"""
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # noqa: BLE001 - any import failure is a finding
        return CheckResult("import " + module_name, False, str(error))
    version = getattr(module, "__version__", "")
    return CheckResult("import " + module_name, True, str(version))


def check_cuda_scatter() -> CheckResult:
    """Does a torch_scatter operation run on the GPU?"""
    import torch
    from torch_scatter import scatter

    values = torch.ones(4, device="cuda")
    index = torch.tensor([0, 0, 1, 1], device="cuda")
    total = scatter(values, index, dim=0, reduce="sum")
    return CheckResult("scatter on GPU", bool(total.sum() == 4), str(total.tolist()))


def check_neighbour_sampling() -> CheckResult:
    """Does PyG's LinkNeighborLoader sample a batch from a tiny graph?"""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    data = Data(x=torch.ones(3, 1), edge_index=edge_index)
    loader = LinkNeighborLoader(
        data, num_neighbors=[2], batch_size=2, edge_label_index=edge_index
    )
    batch = next(iter(loader))
    return CheckResult("neighbour sampling", True, "sampled %d nodes" % batch.num_nodes)


def run_safely(name: str, check: Callable[[], CheckResult]) -> CheckResult:
    """Run a check, turning any crash into a failed result."""
    try:
        return check()
    except Exception as error:  # noqa: BLE001 - a crash is a finding, not a bug here
        return CheckResult(name, False, "%s: %s" % (type(error).__name__, error))


def run_checks() -> List[CheckResult]:
    """All checks, in order. GPU checks are skipped if CUDA is already broken."""
    results = [run_safely("torch CUDA", check_torch_cuda)]
    results += [check_import(name) for name in REQUIRED_MODULES]
    if all(result.passed for result in results):
        results.append(run_safely("scatter on GPU", check_cuda_scatter))
        results.append(run_safely("neighbour sampling", check_neighbour_sampling))
    return results


def failed(results: List[CheckResult]) -> List[CheckResult]:
    """Only the checks that did not pass."""
    return [result for result in results if not result.passed]


def main() -> None:
    """Print the report; exit 1 if anything failed."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    results = run_checks()
    for result in results:
        status = "ok" if result.passed else "FAIL"
        print("%-4s %-22s %s" % (status, result.name, result.detail))
    if failed(results):
        print("\nEnvironment is NOT ready: %d problem(s)." % len(failed(results)))
        sys.exit(1)
    print("\nEnvironment is ready.")


if __name__ == "__main__":
    main()
