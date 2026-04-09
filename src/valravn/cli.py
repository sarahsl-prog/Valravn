from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from valravn.config import OutputConfig, load_config
from valravn.models.task import InvestigationTask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="valravn",
        description="Autonomous DFIR investigation agent",
    )
    sub = p.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("investigate", help="Run an investigation")
    inv.add_argument("--prompt", required=True, help="Natural-language investigation prompt")
    inv.add_argument(
        "--evidence",
        required=True,
        action="append",
        dest="evidence_refs",
        metavar="PATH",
        help="Evidence path (repeat for multiple)",
    )
    inv.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    inv.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Working directory for analysis/, reports/",
    )
    inv.add_argument(
        "--skip-tool-check",
        action="store_true",
        help="Skip forensic tool availability check (not recommended)",
    )

    # Tool check command
    check = sub.add_parser("check-tools", help="Verify forensic tools are available")
    check.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed tool paths",
    )

    return p


def _configure_logging() -> None:
    """Configure loguru from LOG_LEVEL env var (set in .env or shell)."""
    level = os.getenv("LOG_LEVEL", "WARNING").upper()
    logger.remove()  # remove default stderr sink
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - <level>{message}</level>",
    )


def main() -> None:
    _configure_logging()
    args = build_parser().parse_args()

    if args.command == "investigate":
        app_cfg = load_config(args.config)
        out_cfg = OutputConfig(output_dir=args.output_dir.resolve())
        out_cfg.ensure_dirs()

        try:
            task = InvestigationTask(
                prompt=args.prompt,
                evidence_refs=[str(Path(r).resolve()) for r in args.evidence_refs],
            )
        except ValidationError as e:
            msg = e.errors()[0]["msg"]
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(2)

        # Import graph here so CLI is importable without compiling graph
        from valravn.graph import run as graph_run

        exit_code = graph_run(task=task, app_cfg=app_cfg, out_cfg=out_cfg)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
