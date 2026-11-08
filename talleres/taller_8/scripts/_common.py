"""Shared argument parsing for the run_* scripts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the scripts without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medec.config import load_config, set_seed  # noqa: E402


def parse_args(description: str, splits: bool = False, extra=None) -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    if splits:
        parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Splits to process")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N texts (quick tests)")
    if extra:
        extra(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args, cfg
