# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for :mod:`confrover.train.dataset`.

The sampling-logic tests are pure (no trajectory IO); the end-to-end window
test uses the bundled ``7jfl_C`` ATLAS data + OpenFold repr. Everything is
guarded by ``importorskip`` because the module imports OpenFold/mdtraj.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("openfold")
pytest.importorskip("mdtraj")

from confrover.train.dataset import (  # noqa: E402
    TrajCaseConfig,
    TrajDataset,
    TrajDatasetConfig,
)


def _dummy_dataset(deterministic: bool = True, **cfg_kwargs) -> TrajDataset:
    case = TrajCaseConfig(
        case_id="a",
        seqres="AAAA",
        pdb_fpath="x.pdb",
        xtc_fpaths=["r1.xtc", "r2.xtc", "r3.xtc"],
    )
    cfg = TrajDatasetConfig(
        name="t", n_frames=4, stride_in_10ps=120, cases=[case], **cfg_kwargs
    )
    return TrajDataset(config=cfg, deterministic=deterministic)


# ---- Config normalisation ---------------------------------------------------


def test_case_config_single_path_normalises_to_list():
    c = TrajCaseConfig(case_id="a", seqres="AA", pdb_fpath="p.pdb", xtc_fpath="r1.xtc")
    assert c.xtc_fpaths == ["r1.xtc"]
    assert c.xtc_fpath == "r1.xtc"


def test_case_config_requires_some_xtc():
    with pytest.raises(ValueError):
        TrajCaseConfig(case_id="a", seqres="AA", pdb_fpath="p.pdb")


# ---- Sampling logic ---------------------------------------------------------


def test_sample_window_indices_deterministic():
    ds = _dummy_dataset(deterministic=True)
    idxs = ds._sample_window_indices(n_total_frames=10000, stride=120)
    assert list(idxs) == [0, 120, 240, 360]


def test_sample_window_indices_random_starts_in_range():
    ds = _dummy_dataset(deterministic=False)
    for _ in range(50):
        idxs = ds._sample_window_indices(n_total_frames=10000, stride=120)
        assert len(idxs) == 4
        assert idxs[-1] <= 9999
        assert idxs[1] - idxs[0] == 120


def test_choose_stride_filters_infeasible():
    ds = _dummy_dataset(deterministic=True, strides_in_10ps=[60, 120, 256, 512])
    # With only 100 frames and F=4, (F-1)*s <= 99 requires s <= 33 -> none fit,
    # so the smallest candidate is returned.
    assert ds._choose_stride(n_total_frames=100) == 60
    # With a long trajectory, deterministic mode returns the first feasible.
    assert ds._choose_stride(n_total_frames=10000) == 60


def test_pick_xtc_path_deterministic_is_first_replicate():
    ds = _dummy_dataset(deterministic=True)
    assert ds._pick_xtc_path(ds.cfg.cases[0]) == "r1.xtc"


def test_pick_xtc_path_random_stays_in_set():
    ds = _dummy_dataset(deterministic=False)
    picks = {ds._pick_xtc_path(ds.cfg.cases[0]) for _ in range(50)}
    assert picks <= {"r1.xtc", "r2.xtc", "r3.xtc"}


def test_n_total_frames_respects_explicit_override():
    ds = _dummy_dataset(deterministic=True)
    ds.cfg.cases[0].n_total_frames = 777
    # override wins without touching the filesystem
    assert ds._n_total_frames(ds.cfg.cases[0], "does-not-exist.xtc") == 777


# ---- End-to-end window (bundled data) --------------------------------------


def test_getitem_shapes_on_bundled_protein(repo_root):
    from confrover.data.pretrain_repr.openfold.loader import OpenFoldReprLoader

    repr_loader = OpenFoldReprLoader(
        repr_root=str(repo_root / "tests" / "test_data" / "openfold_repr"),
        num_recycles=3,
        load_single=True,
        load_pair=True,
        v1=False,
    )
    ds = TrajDataset(
        config=str(repo_root / "examples" / "train_manifest_smoke.json"),
        repr_loader=repr_loader,
        relpath_to=str(repo_root),
        deterministic=True,
    )
    item = ds[0]
    F = ds.cfg.n_frames
    L = ds.cfg.cases[0].seqlen
    assert item["aatype"].shape == (F, L)
    assert item["rigids_0"].shape == (F, L, 7)
    assert item["atom14_gt_positions"].shape == (F, L, 14, 3)
    assert item["torsion_angles_sin_cos"].shape == (F, L, 7, 2)
    assert item["pos_id"].shape == (F,)
    assert "pretrained_single" in item and "pretrained_pair" in item
