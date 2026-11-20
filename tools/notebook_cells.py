"""Notebook cells shared by the Kaggle and Colab runners.

The cells are plain Python (no `!` or `%` lines) so every step goes through the
`run()` helper, which stops the notebook when a command fails.
"""

from __future__ import annotations

from typing import Any, Dict


def markdown(text: str) -> Dict[str, Any]:
    """A notebook text cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(True),
    }


def code(text: str) -> Dict[str, Any]:
    """A notebook code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


# Must stay first in the first cell, before REPO_URL: tests run just this part.
RUN_HELPER = r'''
import os, shlex, subprocess, sys, time


def run(command):
    """Run a shell command, show its output as it arrives, and stop the notebook if it fails."""
    process = subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", command],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed (exit code {code}): {command}")


REPO_URL = "https://github.com/coachuttam/aml-gnn-transfer.git"
'''

GPU_GATE = r"""
# Stop here if the GPU accelerator is off; remember which GPU we got.
run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
GPU_NAME = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).splitlines()[0].strip()
print("GPU:", GPU_NAME)
"""

CLONE_REPO = r"""
# Always run the latest pushed code: update an existing checkout instead of keeping a stale one.
if os.path.isdir(REPO):
    run(f"git -C {REPO} fetch origin && git -C {REPO} reset --hard origin/main")
else:
    run(f"git clone {REPO_URL} {REPO}")
run(f"git -C {REPO} submodule update --init")
run(f"git -C {REPO} log -1 --oneline")
run(f"git -C {REPO} submodule status")
os.chdir(REPO)
sys.path.insert(0, REPO)
"""

ENVIRONMENT_NOTE = """
Environment. First choice is Multi-GNN's own `env.yml` (micromamba, forced to a CUDA build of torch),
plus pyg-lib (the fast neighbour sampler, which `env.yml` lacks) as a wheel matched to that torch.
Whatever we get must pass `src.check_environment`: CUDA works, the PyG extensions import and run,
pyg-lib is in use. If it does not, we fall back to the notebook's own CUDA torch with all PyG add-ons
installed as matching wheels (`src.pyg_wheels`). If that also fails the check, the notebook stops.
"""

ENVIRONMENT = r"""
MICROMAMBA = f"{WORK}/bin/micromamba"
ENV_PYTHON = f"{MICROMAMBA} run -r {WORK}/mamba -n multignn python"

try:
    run(f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C {WORK} bin/micromamba")
    run(f"{MICROMAMBA} create -y -r {WORK}/mamba -n multignn -f external/Multi-GNN/env.yml 'pytorch=*=*cuda11.8*'")
    run(f"{ENV_PYTHON} -m src.pyg_wheels --only pyg_lib")
    run(f"{ENV_PYTHON} -m src.check_environment")
    PYTHON, ENV_LABEL = ENV_PYTHON, "env.yml"
except RuntimeError as problem:
    print("\nThe env.yml environment is not usable:", problem)
    print("Falling back to this notebook's torch with matching PyG wheels.\n")
    run(f"{sys.executable} -m src.pyg_wheels")
    run(f"{sys.executable} -m src.check_environment")  # raises if still wrong
    PYTHON, ENV_LABEL = sys.executable, "pip-fallback"
ENVIRONMENT_LABEL = f"{ENV_LABEL} / {GPU_NAME}"
print("Using:", ENVIRONMENT_LABEL)
"""
