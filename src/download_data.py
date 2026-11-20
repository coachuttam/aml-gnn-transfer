"""Download the two IBM datasets from Kaggle, skipping any file already on disk.

Needs a Kaggle API token (kaggle.json). On Colab the token is kept on Drive and
copied to ~/.kaggle by the notebook; this module never prints or stores it.

Example:
    python -m src.download_data --data-root /content/drive/MyDrive/aml-gnn-transfer/data
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

from src.config import DATA_ROOT, DATASETS, KAGGLE_DATASET, DatasetSpec
from src.prepare import formatted_csv_path, raw_csv_path


def needs_download(spec: DatasetSpec, data_root: Path) -> bool:
    """False if the raw or formatted file is already there (formatting needs one)."""
    return not (
        raw_csv_path(spec, data_root).exists()
        or formatted_csv_path(spec, data_root).exists()
    )


def download_command(spec: DatasetSpec, data_root: Path) -> List[str]:
    """The Kaggle CLI call that fetches one CSV into its dataset folder, unzipped."""
    folder = raw_csv_path(spec, data_root).parent
    return [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-f", spec.raw_file_name,
        "-p", str(folder),
        "--unzip",
    ]  # fmt: skip


def download_missing(data_root: Path) -> None:
    """Fetch each dataset that is not cached yet. Stops on the first failure."""
    for spec in DATASETS.values():
        if not needs_download(spec, data_root):
            print("%s already on disk, skipping download" % spec.name)
            continue
        raw_csv_path(spec, data_root).parent.mkdir(parents=True, exist_ok=True)
        print("Downloading %s from Kaggle..." % spec.name)
        subprocess.run(download_command(spec, data_root), check=True)
        if not raw_csv_path(spec, data_root).exists():
            raise FileNotFoundError(
                "Kaggle gave no %s: check the token and that the dataset terms "
                "were accepted on kaggle.com" % spec.raw_file_name
            )


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    return parser.parse_args(argv)


def main() -> None:
    """Download whatever is missing."""
    download_missing(parse_arguments().data_root)


if __name__ == "__main__":
    main()
