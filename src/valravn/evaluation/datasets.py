"""Golden dataset management for Valravn evaluation."""

from __future__ import annotations

import json
from pathlib import Path

GOLDEN_DATASET = (
    Path(__file__).parent.parent.parent.parent
    / "tests"
    / "evaluation"
    / "datasets"
    / "golden.jsonl"
)


def add_to_golden(report_path: Path) -> None:
    """Append a verified FindingsReport JSON to the golden dataset.

    Parameters
    ----------
    report_path:
        Path to a ``FindingsReport`` JSON file to add to the golden dataset.

    Side effects
    ------------
    Creates the dataset directory (and any parents) if they do not exist,
    then appends the report as a single JSON line.
    """
    report_json = json.loads(report_path.read_text())
    GOLDEN_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_DATASET, "a") as f:
        f.write(json.dumps(report_json) + "\n")
    print(f"Added {report_path.name} to golden dataset ({GOLDEN_DATASET})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Append a verified FindingsReport JSON to the golden dataset."
    )
    parser.add_argument(
        "--add",
        required=True,
        type=Path,
        metavar="REPORT_JSON",
        help="Path to the FindingsReport JSON file to add",
    )
    args = parser.parse_args()
    add_to_golden(args.add)
