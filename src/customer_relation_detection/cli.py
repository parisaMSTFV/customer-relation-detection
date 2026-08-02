"""Command-line interface for shared-address relation detection."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from customer_relation_detection.config import PROJECT_ROOT
from customer_relation_detection.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reproduce = subparsers.add_parser("reproduce", help="regenerate all artifacts")
    reproduce.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    subparsers.add_parser("smoke", help="run the pipeline in a temporary directory")
    args = parser.parse_args()
    if args.command == "smoke":
        with tempfile.TemporaryDirectory(prefix="relation-detection-") as directory:
            metrics = run_pipeline(Path(directory))
    else:
        metrics = run_pipeline(args.output_root)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
