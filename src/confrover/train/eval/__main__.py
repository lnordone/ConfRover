# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CLI: ``python -m confrover.train.eval`` (also ``confrover eval``).

Compares a ConfRover generation output directory against ATLAS reference data
and writes ``metrics.csv`` / ``metrics.json``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from confrover.train.eval.metrics import evaluate_dir
from confrover.utils import get_pylogger

log = get_pylogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        metavar="<path>",
        help="ConfRover generation output dir (one subdir per case_id).",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=True,
        metavar="<path>",
        help="ATLAS reference data dir (one subdir per case_id, "
        "each with <case_id>.pdb + <case_id>_prod_R*_fit.xtc).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        metavar="<path>",
        help="Where to write metrics.csv/json (defaults to --gen_dir).",
    )
    parser.add_argument(
        "--contact_cutoff",
        type=float,
        default=8.0,
        metavar="<float>",
        help="CA-CA contact cutoff in angstrom.",
    )
    parser.add_argument(
        "--max_pairwise",
        type=int,
        default=200,
        metavar="<int>",
        help="Cap on frames used for O(Fg*Fr) coverage metrics.",
    )
    return parser


def cli(args: argparse.Namespace) -> None:
    rows = evaluate_dir(
        gen_dir=Path(args.gen_dir),
        ref_dir=Path(args.ref_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        contact_cutoff=args.contact_cutoff,
        max_pairwise=args.max_pairwise,
    )
    log.info(f"Evaluated {len(rows)} cases.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="confrover-eval",
        description="Quantitative ATLAS metrics for generated ConfRover ensembles.",
    )
    parser = add_args(parser)
    args = parser.parse_args()
    cli(args)


if __name__ == "__main__":
    main()
