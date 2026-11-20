"""Prepare the IBM datasets for Multi-GNN and write the config it needs.

Steps, in order:
  1. Link the Kaggle CSV into <data root>/<dataset folder>/ (optional).
  2. Run IBM's own `format_kaggle_files.py` on it (slow, so skipped if done).
  3. Print a short summary of what the formatter produced.
  4. Write `data_config.json` into a run folder, beside a copy of
     `model_settings.json`, so Multi-GNN can be started from there without
     touching a single file inside the submodule.

Example (inside Multi-GNN's environment, because the formatter needs datatable):
    python -m src.prepare --raw-dir /kaggle/input/<the IBM AML dataset>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import (
    DATA_ROOT,
    DATASETS,
    MULTI_GNN_DIR,
    RUN_FOLDERS_NAME,
    DatasetSpec,
)
from src.records import model_folder_for_run

FORMATTED_FILE_NAME = "formatted_transactions.csv"
LAUNDERING_COLUMN = "Is Laundering"


@dataclass(frozen=True)
class FormattedSummary:
    """Size of a formatted file, for checking it against the paper."""

    transactions: int
    laundering_transactions: int


def raw_csv_path(spec: DatasetSpec, data_root: Path) -> Path:
    """Where the Kaggle CSV must sit so IBM's formatter writes its output beside it."""
    return data_root / spec.folder_name / spec.raw_file_name


def formatted_csv_path(spec: DatasetSpec, data_root: Path) -> Path:
    """Where IBM's formatter writes (and Multi-GNN reads) the formatted file."""
    return data_root / spec.folder_name / FORMATTED_FILE_NAME


def run_folder_for(run_id: str, data_root: Path) -> Path:
    """The folder Multi-GNN is started from for this run (its own logs live here)."""
    return data_root / RUN_FOLDERS_NAME / run_id


def link_raw_file(spec: DatasetSpec, raw_dir: Path, data_root: Path) -> Path:
    """Symlink the Kaggle CSV into the data folder (avoids copying ~0.5 GB).

    The formatter writes next to the path it is given and does not resolve
    symlinks, so a read-only Kaggle input folder still works.
    """
    source = raw_dir / spec.raw_file_name
    if not source.exists():
        raise FileNotFoundError("Kaggle file not found: %s" % source)
    target = raw_csv_path(spec, data_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)
    return target


def format_dataset(spec: DatasetSpec, data_root: Path) -> Path:
    """Run IBM's formatter unless its output already exists. Returns the output path."""
    output = formatted_csv_path(spec, data_root)
    if output.exists():
        print("%s already formatted, skipping: %s" % (spec.name, output))
        return output
    raw = raw_csv_path(spec, data_root)
    if not raw.exists():
        raise FileNotFoundError(
            "Put %s there first (or pass --raw-dir): %s" % (spec.raw_file_name, raw)
        )
    formatter = MULTI_GNN_DIR / "format_kaggle_files.py"
    print("Formatting %s (several minutes)..." % spec.name)
    subprocess.run([sys.executable, str(formatter), str(raw)], check=True)
    return output


def summarize_formatted_file(path: Path) -> FormattedSummary:
    """Count transactions and laundering transactions in a formatted file."""
    labels = pd.read_csv(path, usecols=[LAUNDERING_COLUMN])[LAUNDERING_COLUMN]
    return FormattedSummary(
        transactions=len(labels), laundering_transactions=int(labels.sum())
    )


def build_data_config(
    data_root: Path, model_save_dir: Path, model_load_dir: Path
) -> Dict[str, Dict[str, str]]:
    """The dictionary Multi-GNN's main.py expects inside data_config.json."""
    return {
        "paths": {
            "aml_data": str(data_root),
            "model_to_load": str(model_load_dir),
            "model_to_save": str(model_save_dir),
        }
    }


def make_run_folder(run_folder: Path, data_config: Dict[str, Dict[str, str]]) -> Path:
    """Create the folder Multi-GNN is started from, with both files it reads there.

    model_settings.json is copied unchanged from the submodule. The model
    folders are created now because torch.save does not create them.
    """
    run_folder.mkdir(parents=True, exist_ok=True)
    for key in ("model_to_load", "model_to_save"):
        Path(data_config["paths"][key]).mkdir(parents=True, exist_ok=True)
    (run_folder / "data_config.json").write_text(
        json.dumps(data_config, indent=2) + "\n"
    )
    shutil.copy(
        MULTI_GNN_DIR / "model_settings.json", run_folder / "model_settings.json"
    )
    return run_folder


def prepare_datasets(
    dataset_names: List[str], raw_dir: Optional[Path], data_root: Path
) -> None:
    """Link (if asked), format and summarize each requested dataset."""
    for name in dataset_names:
        spec = DATASETS[name]
        if raw_dir is not None:
            link_raw_file(spec, raw_dir, data_root)
        output = format_dataset(spec, data_root)
        summary = summarize_formatted_file(output)
        print(
            "%s: %d transactions, %d laundering"
            % (name, summary.transactions, summary.laundering_transactions)
        )


def parse_arguments() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir", type=Path, default=None, help="Folder holding the Kaggle CSVs"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    return parser.parse_args()


def main() -> None:
    """Format the datasets, then write the run folder."""
    arguments = parse_arguments()
    prepare_datasets(arguments.datasets, arguments.raw_dir, DATA_ROOT)
    save_dir = model_folder_for_run(arguments.run_id)
    load_dir = model_folder_for_run(arguments.load_from_run_id or arguments.run_id)
    config = build_data_config(DATA_ROOT, save_dir, load_dir)
    run_folder = make_run_folder(run_folder_for(arguments.run_id, DATA_ROOT), config)
    print("Start Multi-GNN from: %s" % run_folder)


if __name__ == "__main__":
    main()
