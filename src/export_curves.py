"""Write validation curves for records that were made before curves were recorded.

Records are never changed (AGENTS.md rule 5). Instead this reads each run's own
Multi-GNN log and writes `<out>/<run_id>.json` holding [[epoch, F1], ...]. The
logs live on Drive, so run it on Colab or copy the logs down first:

    python -m src.export_curves --logs-root <data root>/multignn_runs \
        --records-dir results/pilot/records --out results/pilot/curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from src.read_log import read_validation_curve
from src.records import read_all_records


def export_curves(logs_root: Path, records_dir: Path, output_dir: Path) -> List[Path]:
    """One curve file per record whose log exists. Returns the files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for record in read_all_records(records_dir):
        curve = read_validation_curve(logs_root / record.run_id)
        if not curve:
            print("no log for %s, skipped" % record.run_id)
            continue
        path = output_dir / (record.run_id + ".json")
        path.write_text(json.dumps(curve) + "\n")
        written.append(path)
    return written


def main(argv: Optional[List[str]] = None) -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    files = export_curves(arguments.logs_root, arguments.records_dir, arguments.out)
    print("Wrote %d curve files to %s" % (len(files), arguments.out))


if __name__ == "__main__":
    main()
