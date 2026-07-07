# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Quantitative ATLAS metrics: generated ensemble vs. reference MD trajectory.

The upstream ConfRover release is inference-only and ships no benchmark; the
ATLAS metrics in the paper ("Coming soon" in the README) are not public. This
module implements a practical subset used to judge whether a trained model has
learned protein flexibility and conformational spread:

* **Per-residue RMSF** flexibility profile -- Pearson correlation + MAE vs. the
  reference (does the model put flexibility in the right places?).
* **Radius of gyration** distribution -- mean + 1-D Wasserstein distance
  (does the ensemble breathe like the MD ensemble?).
* **CA contact map** agreement -- mean absolute error of contact frequency
  (is local/tertiary structure preserved?).
* **Ensemble coverage** -- mean over generated conformers of the minimum CA-RMSD
  (and best TM-score) to any reference frame (are generated states realistic?).

Design
------
The metric maths is pure NumPy (``ca_coords`` arrays of shape ``(n_frames, L, 3)``
in angstrom) so it is unit-testable without mdtraj. Trajectory IO (mdtraj) and
directory orchestration sit on top.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from confrover.utils import get_pylogger

logger = get_pylogger(__name__)


# =============================================================================
# Pure-numpy geometry / metric primitives
# =============================================================================


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimal rotation ``R`` (3x3) minimising ``||P @ R.T - Q||`` for centered P, Q."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def superpose(frames: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rigid-align each frame onto ``ref``.

    Parameters
    ----------
    frames : (F, L, 3)
    ref : (L, 3)

    Returns
    -------
    (F, L, 3) frames translated+rotated into ``ref``'s frame.
    """
    ref_c = ref - ref.mean(axis=0)
    out = np.empty_like(frames)
    for i, x in enumerate(frames):
        xc = x - x.mean(axis=0)
        R = _kabsch_rotation(xc, ref_c)
        out[i] = xc @ R.T + ref.mean(axis=0)
    return out


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimum CA-RMSD between two conformers ``(L, 3)`` after optimal alignment."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    R = _kabsch_rotation(Pc, Qc)
    P_rot = Pc @ R.T
    return float(np.sqrt(((P_rot - Qc) ** 2).sum(axis=1).mean()))


def rmsf_profile(ca_coords: np.ndarray) -> np.ndarray:
    """Per-residue RMSF ``(L,)`` after aligning all frames to the first frame."""
    aligned = superpose(ca_coords, ca_coords[0])
    mean = aligned.mean(axis=0)
    return np.sqrt(((aligned - mean) ** 2).sum(axis=-1).mean(axis=0))


def radius_of_gyration(ca_coords: np.ndarray) -> np.ndarray:
    """Per-frame radius of gyration ``(F,)`` (unmassed, CA only)."""
    centroid = ca_coords.mean(axis=1, keepdims=True)  # (F, 1, 3)
    return np.sqrt(((ca_coords - centroid) ** 2).sum(axis=-1).mean(axis=-1))


def contact_map(ca_coords: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    """Mean CA-CA contact frequency map ``(L, L)`` at ``cutoff`` angstrom.

    Accumulates per frame to avoid materialising an ``(F, L, L, 3)`` tensor
    (which OOMs for long reference trajectories).
    """
    F, L, _ = ca_coords.shape
    acc = np.zeros((L, L), dtype=np.float64)
    for x in ca_coords:  # x: (L, 3)
        d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))
        acc += d < cutoff
    return acc / F


def tm_score(P: np.ndarray, Q: np.ndarray) -> float:
    """TM-score between two conformers ``(L, 3)`` under fixed residue correspondence.

    Both share the same sequence, so residues correspond by index (no alignment
    search). Returns a value in (0, 1]; 1.0 is identical.
    """
    L = P.shape[0]
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    R = _kabsch_rotation(Pc, Qc)
    di = np.sqrt(((Pc @ R.T - Qc) ** 2).sum(axis=1))  # (L,)
    d0 = 1.24 * (max(L - 15, 1)) ** (1.0 / 3.0) - 1.8 if L > 15 else 0.5
    d0 = max(d0, 0.5)
    return float((1.0 / (1.0 + (di / d0) ** 2)).mean())


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein (earth-mover) distance; uses scipy if present, else a
    quantile approximation so the module has no hard scipy dependency."""
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except Exception:
        qs = np.linspace(0.0, 1.0, 101)
        return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# =============================================================================
# Ensemble comparison (numpy in, dict out)
# =============================================================================


def compare_ensembles(
    gen_ca: np.ndarray,
    ref_ca: np.ndarray,
    *,
    contact_cutoff: float = 8.0,
    max_pairwise: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Compute all metrics comparing a generated vs. reference CA ensemble.

    Parameters
    ----------
    gen_ca, ref_ca : (F, L, 3) arrays in angstrom (same L).
    max_pairwise : cap on frames used for the O(Fg*Fr) coverage metrics.
    """
    assert gen_ca.shape[1] == ref_ca.shape[1], (
        f"residue-count mismatch: gen L={gen_ca.shape[1]} vs ref L={ref_ca.shape[1]}"
    )
    rng = np.random.default_rng(seed)

    # --- RMSF flexibility profile ---
    gen_rmsf = rmsf_profile(gen_ca)
    ref_rmsf = rmsf_profile(ref_ca)

    # --- Radius of gyration ---
    gen_rg = radius_of_gyration(gen_ca)
    ref_rg = radius_of_gyration(ref_ca)

    # --- Contact map ---
    gen_cmap = contact_map(gen_ca, cutoff=contact_cutoff)
    ref_cmap = contact_map(ref_ca, cutoff=contact_cutoff)

    # --- Coverage: min RMSD / best TM-score of each gen conformer to ref set ---
    def _subsample(x: np.ndarray) -> np.ndarray:
        if len(x) <= max_pairwise:
            return x
        idx = rng.choice(len(x), size=max_pairwise, replace=False)
        return x[idx]

    gen_s = _subsample(gen_ca)
    ref_s = _subsample(ref_ca)
    min_rmsds = []
    best_tms = []
    for g in gen_s:
        rmsds = np.array([kabsch_rmsd(g, r) for r in ref_s])
        tms = np.array([tm_score(g, r) for r in ref_s])
        min_rmsds.append(float(rmsds.min()))
        best_tms.append(float(tms.max()))

    return {
        "seqlen": int(gen_ca.shape[1]),
        "n_gen_frames": int(gen_ca.shape[0]),
        "n_ref_frames": int(ref_ca.shape[0]),
        "rmsf_pearson": _pearson(gen_rmsf, ref_rmsf),
        "rmsf_mae": float(np.abs(gen_rmsf - ref_rmsf).mean()),
        "rg_gen_mean": float(gen_rg.mean()),
        "rg_ref_mean": float(ref_rg.mean()),
        "rg_wasserstein": _wasserstein_1d(gen_rg, ref_rg),
        "contact_map_mae": float(np.abs(gen_cmap - ref_cmap).mean()),
        "coverage_mean_min_rmsd": float(np.mean(min_rmsds)),
        "coverage_mean_best_tmscore": float(np.mean(best_tms)),
    }


# =============================================================================
# Trajectory IO (mdtraj)
# =============================================================================


def _ca_from_traj(traj) -> np.ndarray:
    """Extract CA coordinates ``(n_frames, L, 3)`` in angstrom from an mdtraj traj."""
    ca_idx = traj.topology.select("name CA")
    return traj.xyz[:, ca_idx, :] * 10.0  # nm -> angstrom


def load_generated_ensemble(case_dir: Path) -> Optional[np.ndarray]:
    """Load all generated conformers for one case into a CA ensemble.

    Handles both trajectory output (``*_sample*.xtc`` with matching ``.pdb``
    topology) and iid output (``*_sample*.pdb``). Returns ``None`` if nothing
    is found.
    """
    import mdtraj as md

    case_dir = Path(case_dir)
    frames: List[np.ndarray] = []

    xtcs = sorted(case_dir.glob("*_sample*.xtc"))
    for xtc in xtcs:
        top = xtc.with_suffix(".pdb")
        if not top.exists():
            logger.warning(f"no topology for {xtc.name}; skipping")
            continue
        frames.append(_ca_from_traj(md.load(str(xtc), top=str(top))))

    if not xtcs:
        # iid: each sample is a single-frame (or multi-frame) pdb
        pdbs = sorted(
            p for p in case_dir.glob("*_sample*.pdb") if "_preview" not in p.name
        )
        for pdb in pdbs:
            frames.append(_ca_from_traj(md.load(str(pdb))))

    if not frames:
        return None
    return np.concatenate(frames, axis=0)


def load_reference_ensemble(
    ref_case_dir: Path, case_id: str, max_frames: Optional[int] = 2000
) -> Optional[np.ndarray]:
    """Load the reference ATLAS trajectory (all replicates) as a CA ensemble.

    Expects ``<ref_case_dir>/<case_id>_prod_R*_fit.xtc`` and a topology
    ``<ref_case_dir>/<case_id>.pdb`` (the standard ATLAS layout, matching
    ``tests/test_data/atlas/``).
    """
    import mdtraj as md

    ref_case_dir = Path(ref_case_dir)
    top = ref_case_dir / f"{case_id}.pdb"
    if not top.exists():
        logger.warning(f"no reference topology {top}; skipping {case_id}")
        return None

    xtcs = sorted(ref_case_dir.glob(f"{case_id}_prod_R*_fit.xtc"))
    if not xtcs:
        xtcs = sorted(ref_case_dir.glob("*.xtc"))
    if not xtcs:
        logger.warning(f"no reference xtc in {ref_case_dir}; skipping {case_id}")
        return None

    frames = [_ca_from_traj(md.load(str(x), top=str(top))) for x in xtcs]
    ca = np.concatenate(frames, axis=0)
    if max_frames is not None and len(ca) > max_frames:
        idx = np.linspace(0, len(ca) - 1, max_frames).astype(int)
        ca = ca[idx]
    return ca


# =============================================================================
# Orchestration
# =============================================================================


def evaluate_case(
    gen_case_dir: Path,
    ref_case_dir: Path,
    case_id: str,
    **compare_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Evaluate one case; returns a metrics dict or ``None`` if data is missing."""
    gen_ca = load_generated_ensemble(gen_case_dir)
    if gen_ca is None:
        logger.warning(f"{case_id}: no generated conformers found; skipping")
        return None
    ref_ca = load_reference_ensemble(ref_case_dir, case_id)
    if ref_ca is None:
        return None
    if gen_ca.shape[1] != ref_ca.shape[1]:
        logger.warning(
            f"{case_id}: L mismatch gen={gen_ca.shape[1]} ref={ref_ca.shape[1]}; skip"
        )
        return None
    metrics = compare_ensembles(gen_ca, ref_ca, **compare_kwargs)
    metrics["case_id"] = case_id
    return metrics


def evaluate_dir(
    gen_dir: Path,
    ref_dir: Path,
    output_dir: Optional[Path] = None,
    **compare_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Evaluate every case present in ``gen_dir`` against ``ref_dir``.

    ``gen_dir`` is the ConfRover generation output (one subdir per case_id);
    ``ref_dir`` holds ATLAS reference data (one subdir per case_id). Writes
    ``metrics.csv`` + ``metrics.json`` and an aggregate row to ``output_dir``
    (defaults to ``gen_dir``).
    """
    gen_dir = Path(gen_dir)
    ref_dir = Path(ref_dir)
    output_dir = Path(output_dir) if output_dir is not None else gen_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for case_subdir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        case_id = case_subdir.name
        row = evaluate_case(
            case_subdir, ref_dir / case_id, case_id, **compare_kwargs
        )
        if row is not None:
            rows.append(row)
            logger.info(
                f"{case_id}: RMSF r={row['rmsf_pearson']:.3f} "
                f"min-RMSD={row['coverage_mean_min_rmsd']:.2f}A "
                f"best-TM={row['coverage_mean_best_tmscore']:.3f}"
            )

    if rows:
        _write_outputs(rows, output_dir)
    else:
        logger.warning("no cases evaluated -- check gen/ref directory layout.")
    return rows


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = [
        k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "seqlen"
    ]
    agg: Dict[str, Any] = {"case_id": "MEAN", "n_cases": len(rows)}
    for k in numeric_keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[k] = float(np.nanmean(vals))
    return agg


def _write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    agg = _aggregate(rows)
    fieldnames = list(rows[0].keys())

    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"per_case": rows, "aggregate": agg}, f, indent=2)

    logger.info(f"Wrote metrics for {len(rows)} cases to {output_dir}/metrics.csv")
    logger.info(
        "AGGREGATE  "
        + "  ".join(
            f"{k}={agg[k]:.3f}"
            for k in (
                "rmsf_pearson",
                "rmsf_mae",
                "rg_wasserstein",
                "contact_map_mae",
                "coverage_mean_min_rmsd",
                "coverage_mean_best_tmscore",
            )
            if k in agg
        )
    )
