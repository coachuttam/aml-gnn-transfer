"""After the smoke runs: did scratch and finetune get the same labels and budget?

For one fraction and seed the two setups must train on the identical kept
labels and the same number of gradient steps (AGENTS.md rule 4, and the
equal-step rule). This reads the two smoke records and says whether they match.

Run from the repo root:
    python -m src.check_smoke --records-dir <folder with the smoke records>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from src.config import RUN_RECORDS_DIR, SEEDS, SMOKE_LABEL_FRACTION
from src.records import RunRecord, make_run_id, read_record, record_path, smoke_run_id

MUST_MATCH = ("kept_labels_fingerprint", "gradient_steps", "labels_kept")
MUST_BE_PRESENT = MUST_MATCH + ("epochs_requested",)


def mismatches(scratch: RunRecord, finetune: RunRecord) -> List[str]:
    """Messages for every field that is missing or differs between the two records."""
    problems = []
    for name in MUST_BE_PRESENT:
        values = (scratch.config.get(name), finetune.config.get(name))
        if None in values:
            problems.append("%s is missing from a record: %s" % (name, values))
        elif name in MUST_MATCH and values[0] != values[1]:
            problems.append("%s differs: scratch %s, finetune %s" % ((name,) + values))
    if scratch.fraction != finetune.fraction or scratch.seed != finetune.seed:
        problems.append("the two runs are not the same fraction and seed")
    return problems


def load_smoke_pair(records_dir: Path, fraction: float, seed: int):
    """The scratch and finetune smoke records for one fraction and seed."""
    paths = [
        record_path(smoke_run_id(make_run_id("gin", mode, fraction, seed)), records_dir)
        for mode in ("scratch", "finetune")
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError("Smoke record missing: %s" % path)
    return read_record(paths[0]), read_record(paths[1])


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records-dir", type=Path, default=RUN_RECORDS_DIR)
    parser.add_argument("--fraction", type=float, default=SMOKE_LABEL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    return parser.parse_args(argv)


def main() -> None:
    """Print the comparison; exit 1 if the two smoke runs do not match."""
    arguments = parse_arguments()
    scratch, finetune = load_smoke_pair(
        arguments.records_dir, arguments.fraction, arguments.seed
    )
    for name in MUST_BE_PRESENT:
        row = (name, scratch.config.get(name), finetune.config.get(name))
        print("%-26s scratch %-18s finetune %s" % row)
    problems = mismatches(scratch, finetune)
    for problem in problems:
        print("PROBLEM:", problem)
    if problems:
        sys.exit(1)
    print("Scratch and finetune used the same labels and the same number of steps.")


if __name__ == "__main__":
    main()
