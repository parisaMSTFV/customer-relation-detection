"""Command-line interface for shared-address relation detection."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from customer_relation_detection.config import load_config
from customer_relation_detection.pipeline import run_analysis, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce = subparsers.add_parser("reproduce", help="regenerate synthetic benchmark artifacts")
    reproduce.add_argument("--output-root", type=Path, default=Path.cwd())
    reproduce.add_argument("--config", type=Path)

    subparsers.add_parser("smoke", help="run the packaged benchmark in a temporary directory")

    analyze = subparsers.add_parser("analyze", help="analyze an unlabeled observation CSV")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--snapshot-date")
    analyze.add_argument(
        "--identifier-salt-env",
        default="RELATION_ID_SALT",
        help="environment variable containing the HMAC salt (default: RELATION_ID_SALT)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "smoke":
        with tempfile.TemporaryDirectory(prefix="relation-detection-") as directory:
            metrics = run_pipeline(Path(directory))
    elif args.command == "reproduce":
        metrics = run_pipeline(args.output_root, config=load_config(args.config))
    else:
        identifier_salt = os.environ.get(args.identifier_salt_env)
        if not identifier_salt:
            raise SystemExit(
                f"Set {args.identifier_salt_env} to a secret value of at least 16 characters"
            )
        orders = pd.read_csv(args.input)
        metrics = run_analysis(
            orders,
            args.output_root,
            identifier_salt=identifier_salt,
            config=load_config(args.config),
            snapshot_date=args.snapshot_date,
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
