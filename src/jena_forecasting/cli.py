from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .data import audit_frame, clean_and_resample_hourly, load_hourly_csv, load_jena_csv
from .pipeline import run_experiments

DEFAULT_CONFIG = Path("config/experiment.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jena-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Inspect the raw Jena CSV")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    preprocess = subparsers.add_parser(
        "preprocess", help="Create leakage-safe hourly means"
    )
    preprocess.add_argument("--input", type=Path, required=True)
    preprocess.add_argument(
        "--output", type=Path, default=Path("data/processed/jena_hourly.csv")
    )
    run = subparsers.add_parser("run", help="Run the frozen test comparisons")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        frame = load_jena_csv(args.input)
        payload = (
            json.dumps(audit_frame(frame, str(args.input)).to_dict(), indent=2) + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    if args.command == "preprocess":
        frame = load_jena_csv(args.input)
        hourly = clean_and_resample_hourly(frame)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(args.output)
        print(f"Wrote {len(hourly):,} completed hourly intervals to {args.output}")
        return 0
    if args.command == "run":
        config = ExperimentConfig.from_path(args.config)
        hourly = load_hourly_csv(args.input)
        results = run_experiments(hourly, config, args.output_dir)
        print(results.to_string(index=False))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
