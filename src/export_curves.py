"""Write validation curves for records that were made before curves were recorded.

Records are never changed (AGENTS.md rule 5). Instead this reads each run's own
Multi-GNN log and writes `<out>/<run_id>.json` holding [[epoch, F1], ...]. The
logs live on Drive, so run it on Colab or copy the logs down first:

    python -m src.export_curves --logs-root <data root>/multignn_runs \
        --records-dir results/pilot/records --out results/pilot/curves

On Colab (no GPU needed) the notebook's RUN_SET = "export_curves" does this for
whole run sets, writing `<home>/<set>/curves/` next to `<home>/<set>/records/`:

    python -m src.export_curves --home <Drive project folder> --sets pilot phaseA phaseC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from src.config import RUN_FOLDERS_NAME
from src.phases import phase_folders
from src.read_log import read_validation_curve
from src.records import read_all_records

EXPORTED_SETS = [
    "pilot",
    "phaseA",
    "phaseC",
]  # runs whose records lack a trustworthy curve


def export_curves(logs_root: Path, records_dir: Path, output_dir: Path) -> List[Path]:
    """One curve file per record whose log exists. Returns the files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for record in read_all_records(records_dir):
        curve = read_validation_curve(
            logs_root / record.run_id,
            record.config.get("epochs_requested"),
            record.config.get("validations"),
        )
        if not curve:
            print("no log for %s, skipped" % record.run_id)
            continue
        path = output_dir / (record.run_id + ".json")
        path.write_text(json.dumps(curve) + "\n")
        written.append(path)
    return written


def export_sets(home: Path, set_names: List[str]) -> List[Path]:
    """Curves for each run set, from the logs in `<home>/data/multignn_runs`."""
    logs_root = home / "data" / RUN_FOLDERS_NAME
    written: List[Path] = []
    for name in set_names:
        records = phase_folders(name, home).records
        print("%s: records in %s" % (name, records))
        written += export_curves(logs_root, records, records.parent / "curves")
    return written


def main(argv: Optional[List[str]] = None) -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs-root", type=Path)
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--home", type=Path, help="Project folder (Drive on Colab)")
    parser.add_argument("--sets", nargs="+", default=EXPORTED_SETS)
    arguments = parser.parse_args(argv)
    if arguments.home:
        files = export_sets(arguments.home, arguments.sets)
    elif arguments.logs_root and arguments.records_dir and arguments.out:
        files = export_curves(arguments.logs_root, arguments.records_dir, arguments.out)
    else:
        parser.error("give --home, or all of --logs-root, --records-dir and --out")
    print("Wrote %d curve files" % len(files))


if __name__ == "__main__":
    main()
