"""Install PyG's compiled add-ons (pyg-lib, torch-scatter, torch-sparse) to match torch.

These packages are built for one exact torch and CUDA pair. A mismatch gives
errors like "libc10_cuda.so: cannot open shared object file". PyG publishes
matching wheels at data.pyg.org, so we read the torch and CUDA versions of
the Python that is running this and ask pip for the wheels built for them.

Run with the Python of the environment you want to fix:
    python -m src.pyg_wheels              # base libraries and all three add-ons
    python -m src.pyg_wheels --only pyg_lib
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional

WHEEL_HOST = "https://data.pyg.org/whl"
ADDON_PACKAGES = ["pyg_lib", "torch_scatter", "torch_sparse"]
BASE_PACKAGES = ["torch_geometric", "wandb", "munch", "datatable"]


def cuda_tag(cuda_version: Optional[str]) -> str:
    """PyG's name for a CUDA version: "11.8" becomes "cu118"."""
    if cuda_version is None:
        raise RuntimeError("torch has no CUDA build here; is the GPU accelerator on?")
    return "cu" + cuda_version.replace(".", "")


def wheel_index_urls(torch_version: str, cuda_version: Optional[str]) -> List[str]:
    """Wheel pages for this torch: its exact version, then the x.y.0 release.

    PyG often publishes one page for the whole x.y series (for example torch
    2.5.0 for 2.5.1), and its wheels work for the patch releases.
    """
    tag = cuda_tag(cuda_version)
    exact = torch_version.split("+")[0]
    series = ".".join(exact.split(".")[:2]) + ".0"
    versions = [exact] if exact == series else [exact, series]
    return ["%s/torch-%s+%s.html" % (WHEEL_HOST, version, tag) for version in versions]


def install_command(packages: List[str], urls: List[str]) -> List[str]:
    """The pip command. Binary wheels only: a source build would take far too long."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:"]
    command += packages
    for url in urls:
        command += ["-f", url]
    return command


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ADDON_PACKAGES,
        default=None,
        help="Install just these add-ons (the base libraries are skipped)",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Install what is missing, matched to this Python's torch."""
    import torch

    arguments = parse_arguments()
    urls = wheel_index_urls(torch.__version__, torch.version.cuda)
    print("Looking for PyG wheels at:", *urls, sep="\n  ")
    packages = arguments.only or BASE_PACKAGES + ADDON_PACKAGES
    subprocess.run(install_command(packages, urls), check=True)


if __name__ == "__main__":
    main()
