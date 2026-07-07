#!/usr/bin/env python
# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build ConfRover train + eval manifests from an ATLAS-style data directory.

Given a CSV of ``chain_name,seqres`` (the shape of
``tests/test_data/atlas_test_small.csv``) and an ATLAS root laid out as::

    <atlas_root>/<case_id>/<case_id>.pdb
    <atlas_root>/<case_id>/<case_id>_prod_R{1,2,3}_fit.xtc

this emits:

  * a **training** manifest (``confrover.train.dataset.TrajDataset`` format,
    multi-replicate ``xtc_fpaths``), and
  * an **eval** manifest (``confrover generate`` forward-simulation format,
    conditioning on frame 0 of replicate R1).

Paths are stored relative to ``--atlas_root`` so both the trainer
(``--relpath_to``) and the generator (``data.gen_dataset.relpath_to``) can
resolve them. Stdlib only -- no heavy deps.

Example
-------
    python scripts/build_manifests.py \\
        --csv my_atlas_subset.csv \\
        --atlas_root ~/scratch/atlas \\
        --out_dir manifests/ \\
        --n_eval 15 --n_frames 8 --strides 60 120 256 512
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


def _find_replicates(case_dir: Path, case_id: str) -> List[str]:
    reps = sorted(case_dir.glob(f"{case_id}_prod_R*_fit.xtc"))
    if not reps:
        reps = sorted(p for p in case_dir.glob("*.xtc"))
    return [p.name for p in reps]


def _read_cases(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            case_id = r.get("chain_name") or r.get("case_id")
            seqres = r["seqres"].strip()
            if case_id and seqres:
                rows.append({"case_id": case_id.strip(), "seqres": seqres})
        return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--atlas_root", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--name", default="atlas_subset")
    ap.add_argument("--n_eval", type=int, default=15, help="# cases held out for eval")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--stride_in_10ps", type=int, default=120)
    ap.add_argument(
        "--strides", type=int, nargs="*", default=[60, 120, 256, 512],
        help="Stride list for random-stride training (empty = fixed stride).",
    )
    ap.add_argument("--samples_per_epoch", type=int, default=None)
    ap.add_argument("--eval_n_replicates", type=int, default=5)
    ap.add_argument("--eval_n_frames", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = _read_cases(args.csv)

    resolved = []
    for c in cases:
        case_id = c["case_id"]
        case_dir = args.atlas_root / case_id
        pdb = case_dir / f"{case_id}.pdb"
        reps = _find_replicates(case_dir, case_id)
        if not pdb.exists() or not reps:
            print(f"[skip] {case_id}: missing pdb or xtc under {case_dir}")
            continue
        resolved.append(
            {
                "case_id": case_id,
                "seqres": c["seqres"],
                "pdb_rel": f"{case_id}/{pdb.name}",
                "xtc_rel": [f"{case_id}/{n}" for n in reps],
            }
        )

    if not resolved:
        raise SystemExit("No usable cases found -- check --csv and --atlas_root.")

    random.Random(args.seed).shuffle(resolved)
    n_eval = min(args.n_eval, max(0, len(resolved) - 1))
    eval_cases = resolved[:n_eval]
    train_cases = resolved[n_eval:]
    print(f"{len(train_cases)} train / {len(eval_cases)} eval cases")

    # ---- Training manifest (TrajDataset format) ----
    train_manifest = {
        "name": f"{args.name}_train",
        "n_frames": args.n_frames,
        "stride_in_10ps": args.stride_in_10ps,
        "strides_in_10ps": args.strides or None,
        "samples_per_epoch": args.samples_per_epoch,
        "cases": [
            {
                "case_id": c["case_id"],
                "seqres": c["seqres"],
                "pdb_fpath": c["pdb_rel"],
                "xtc_fpaths": c["xtc_rel"],
            }
            for c in train_cases
        ],
    }
    train_path = args.out_dir / f"{args.name}_train.json"
    train_path.write_text(json.dumps(train_manifest, indent=2))

    # ---- Eval manifest (confrover generate, forward simulation) ----
    eval_manifest = {
        "name": f"{args.name}_eval",
        "task_mode": "forward",
        "n_replicates": args.eval_n_replicates,
        "n_frames": args.eval_n_frames,
        "stride_in_10ps": args.stride_in_10ps,
        "cases": [
            {
                "case_id": c["case_id"],
                "seqres": c["seqres"],
                "conditions": {
                    "xtc_fpath": c["xtc_rel"][0],
                    "pdb_fpath": c["pdb_rel"],
                    "frame_idxs": 0,
                },
            }
            for c in eval_cases
        ],
    }
    eval_path = args.out_dir / f"{args.name}_eval.json"
    eval_path.write_text(json.dumps(eval_manifest, indent=2))

    print(f"Wrote:\n  {train_path}\n  {eval_path}")
    print(
        "Train with: python -m confrover.train.cli "
        f"--train_manifest {train_path} --output_dir <run> "
        f"'data.train_dataset.repr_loader.repr_root=<folding_repr>' "
        f"'data.train_dataset.relpath_to={args.atlas_root}'"
    )


if __name__ == "__main__":
    main()
