#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_local_cli.pipeline import ResumeOCRPipeline
from ocr_local_cli.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resume OCR pipeline on local files.")
    parser.add_argument("inputs", nargs="+", help="Resume files or directories (PDF/PNG/JPG).")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config.yaml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.verbose:
        configure_logging(level=10)

    config = None
    if args.config:
        from ocr_local_cli.config import load_config

        config = load_config(args.config)

    pipeline = ResumeOCRPipeline(config=config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        path = Path(input_path)
        result = pipeline.run(path)
        output_path = args.output_dir / f\"{path.stem}.json\"
        output_path.write_text(json.dumps(result.document, indent=2), encoding=\"utf-8\")
        print(f\"Processed {path} -> {output_path}\")


if __name__ == \"__main__\":
    main()

