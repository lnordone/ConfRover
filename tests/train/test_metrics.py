# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure-numpy tests for the ATLAS evaluation metrics (no mdtraj needed)."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from confrover.train.eval.metrics import (  # noqa: E402
    compare_ensembles,
    kabsch_rmsd,
    radius_of_gyration,
    rmsf_profile,
    tm_score,
)


def _random_conformer(L: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(L, 3)) * 5.0


def test_kabsch_rmsd_invariant_to_rotation_and_translation():
    P = _random_conformer(20, 0)
    # Random rotation
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    P_moved = P @ Q.T + np.array([10.0, -3.0, 7.0])
    assert kabsch_rmsd(P, P_moved) == pytest.approx(0.0, abs=1e-6)


def test_tm_score_identity_is_one():
    P = _random_conformer(50, 2)
    assert tm_score(P, P) == pytest.approx(1.0, abs=1e-6)
    # A rotated/translated copy is still identical structurally.
    P2 = P + 5.0
    assert tm_score(P, P2) == pytest.approx(1.0, abs=1e-6)


def test_rmsf_zero_for_static_ensemble():
    P = _random_conformer(15, 3)
    frames = np.stack([P + np.array([i, 0.0, 0.0]) for i in range(10)])  # pure shift
    prof = rmsf_profile(frames)
    assert prof.shape == (15,)
    # Rigid translation is removed by superposition -> ~zero fluctuation.
    assert np.allclose(prof, 0.0, atol=1e-6)


def test_radius_of_gyration_scales_with_size():
    small = _random_conformer(30, 4)[None] * 0.5
    big = small * 2.0
    assert radius_of_gyration(big)[0] == pytest.approx(
        2.0 * radius_of_gyration(small)[0], rel=1e-6
    )


def test_compare_ensembles_matches_itself():
    rng = np.random.default_rng(5)
    ens = rng.normal(size=(12, 25, 3)) * 4.0
    m = compare_ensembles(ens, ens, max_pairwise=12)
    assert m["seqlen"] == 25
    assert m["rmsf_pearson"] == pytest.approx(1.0, abs=1e-6)
    assert m["rmsf_mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["contact_map_mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["rg_wasserstein"] == pytest.approx(0.0, abs=1e-6)
    # Every generated frame has an identical match in the reference set.
    assert m["coverage_mean_min_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert m["coverage_mean_best_tmscore"] == pytest.approx(1.0, abs=1e-6)


def test_compare_ensembles_rejects_length_mismatch():
    a = np.zeros((4, 10, 3))
    b = np.zeros((4, 11, 3))
    with pytest.raises(AssertionError):
        compare_ensembles(a, b)
