# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Trajectory dataset for ConfRover training.

Loads ATLAS-style ``.xtc`` trajectories and yields windows of ``n_frames``
consecutive frames at a given stride, each accompanied by:

- per-frame ``aatype``, ``rigids_0``, ``atom14_gt_positions``, ``pseudo_beta``,
  ``pseudo_beta_mask``, ``torsion_angles_sin_cos``, ``all_atom_mask``
- pretrained OpenFold features (``pretrained_single``, ``pretrained_pair``)
  loaded once per protein via :class:`OpenFoldReprLoader`
- ``pos_id`` (the absolute frame indices in the source trajectory)

The output dict aligns with the keys ``ConfRoverTrainable.training_step``
consumes; see ``confrover.train.module``.

WORKING NOW:
    Loading XTC frames, OpenFold preprocessing (re-using
    ``GenDataset.process_coords``), CA centering, padded collate.

TODO (left as exercises):
    * Multi-replicate sampling (ATLAS proteins typically have 3 replicates,
      each a separate ``.xtc``). The skeleton handles a single replicate per
      case for simplicity.
    * Random-stride sampling: paper trains with multiple strides; the skeleton
      uses a fixed stride. Adding random stride is a one-line change in
      ``__getitem__`` (sample stride from a list).
    * Length bucketing for efficient batching across heterogeneous proteins.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from openfold.np import residue_constants as rc
from openfold.utils import rigid_utils as ru
from torch.nn.utils.rnn import pad_sequence

from confrover.data.infer import GenDataset, LoaderConfig, all_angles_mask_with_x
from confrover.data.io.xtc import xtc_to_atom37
from confrover.utils import get_pylogger
from confrover.utils.torch.tensor import rearrange

logger = get_pylogger(__name__)


# =============================================================================
# Manifest dataclasses
# =============================================================================


@dataclass
class TrajCaseConfig:
    """One protein's trajectory configuration."""

    case_id: str
    seqres: str
    xtc_fpath: str
    pdb_fpath: str
    n_total_frames: Optional[int] = None  # populated lazily on first load

    @property
    def seqlen(self) -> int:
        return len(self.seqres)


@dataclass
class TrajDatasetConfig:
    """Top-level training-manifest config."""

    name: str
    n_frames: int  # window size (# frames per training example)
    stride_in_10ps: int  # spacing between consecutive frames in a window
    cases: List[TrajCaseConfig] = field(default_factory=list)
    samples_per_epoch: Optional[int] = None  # if None, one window per case per epoch
    relpath_to: Optional[str] = None  # base path for resolving xtc/pdb paths

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        case_subset: Optional[List[str]] = None,
        relpath_to: Optional[str] = None,
    ) -> "TrajDatasetConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        cases_raw = raw.pop("cases")
        if case_subset is not None:
            cases_raw = [c for c in cases_raw if c["case_id"] in case_subset]
        cases = [TrajCaseConfig(**c) for c in cases_raw]
        cfg = cls(cases=cases, relpath_to=relpath_to, **raw)
        if relpath_to is not None:
            base = Path(relpath_to)
            for c in cfg.cases:
                c.xtc_fpath = str(base / c.xtc_fpath)
                c.pdb_fpath = str(base / c.pdb_fpath)
        return cfg


# =============================================================================
# Dataset
# =============================================================================


class TrajDataset(torch.utils.data.Dataset):
    """ATLAS-style trajectory dataset for ConfRover training.

    Each ``__getitem__`` returns one *window* of ``n_frames`` frames sampled
    from one protein's trajectory. By default a window is ``n_frames`` frames
    spaced ``stride_in_10ps`` 10-ps steps apart, starting at a random offset.

    For overfit-smoke-test usage, set ``deterministic=True`` to fix the
    starting offset to 0 so every call returns the same window.
    """

    def __init__(
        self,
        config: str | TrajDatasetConfig,
        repr_loader=None,
        case_subset: Optional[List[str]] = None,
        relpath_to: Optional[str] = None,
        deterministic: bool = False,
        **loader_kwargs,
    ) -> None:
        if isinstance(config, TrajDatasetConfig):
            self.cfg = config
        else:
            self.cfg = TrajDatasetConfig.from_json(
                config, case_subset=case_subset, relpath_to=relpath_to
            )
        self.dataset_name = self.cfg.name
        self.repr_loader = repr_loader
        self.deterministic = deterministic
        self.loader_cfg = LoaderConfig(**loader_kwargs)

        # We re-use GenDataset.process_coords for OpenFold feature extraction;
        # rather than subclass GenDataset (which assumes inference-style
        # configs), bind it here as a small helper.
        self._process_coords = GenDataset.process_coords.__get__(self, GenDataset)

    def __len__(self) -> int:
        if self.cfg.samples_per_epoch is not None:
            return self.cfg.samples_per_epoch
        return len(self.cfg.cases)

    # ---- Frame sampling -----------------------------------------------------

    def _sample_window_indices(self, case: TrajCaseConfig) -> np.ndarray:
        """Pick which frame indices in the source XTC to use for this window.

        Frames are spaced ``cfg.stride_in_10ps`` 10-ps steps apart. The starting
        offset is random unless ``self.deterministic`` is True.
        """
        if case.n_total_frames is None:
            # Cheap one-time read of the trajectory length via mdtraj is fine
            # because XTC is indexed; but to keep this template dependency-light
            # we just trust the user-supplied value or fall back to a generous
            # upper bound. Production code should populate n_total_frames
            # eagerly (e.g. in TrajDatasetConfig.__post_init__) by calling
            # mdtraj.iterload(...) and counting frames.
            case.n_total_frames = 1000  # ATLAS trajectories are 10000 frames

        F = self.cfg.n_frames
        stride = self.cfg.stride_in_10ps
        max_start = case.n_total_frames - (F - 1) * stride - 1
        max_start = max(max_start, 0)
        if self.deterministic or max_start == 0:
            start = 0
        else:
            start = int(np.random.randint(0, max_start + 1))
        return np.arange(start, start + F * stride, stride)

    # ---- Main accessor ------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # If samples_per_epoch is set, idx is virtual; pick a case at random.
        if self.cfg.samples_per_epoch is not None:
            case = self.cfg.cases[
                int(np.random.randint(0, len(self.cfg.cases)))
                if not self.deterministic
                else (idx % len(self.cfg.cases))
            ]
        else:
            case = self.cfg.cases[idx]

        seqres = case.seqres
        seqlen = case.seqlen
        F = self.cfg.n_frames
        frame_idxs = self._sample_window_indices(case)

        # ---- Load frames as atom37 ----
        per_frame_atom37 = []
        for fi in frame_idxs:
            atom37 = xtc_to_atom37(
                xtc_path=case.xtc_fpath,
                pdb_path=case.pdb_fpath,
                seqlen=seqlen,
                frame_idx=int(fi),
                unit="A",
            )  # (L, 37, 3) numpy
            per_frame_atom37.append(atom37)
        atom_coords = np.stack(per_frame_atom37, axis=0)  # (F, L, 37, 3)

        # ---- Center each frame by its CA mean ----
        ca_mean = np.nanmean(
            atom_coords[..., rc.atom_order["CA"] : rc.atom_order["CA"] + 1, :],
            axis=1,
            keepdims=True,
        )  # (F, 1, 1, 3)
        atom_coords = atom_coords - ca_mean

        # ---- OpenFold feature extraction ----
        # process_coords expects a flat batch (F * L); replicate the
        # GenDataset's call pattern here.
        flat_atom_coords = atom_coords.reshape(F * seqlen, 37, 3)
        aatype_flat = torch.LongTensor(
            [rc.restype_order_with_x[res] for res in seqres] * F
        )
        of_feat = self._process_coords(flat_atom_coords, aatype_flat)

        # Reshape per-frame features to (F, L, ...).
        rigids_0 = ru.Rigid.from_tensor_4x4(of_feat["rigidgroups_gt_frames"])[
            :, 0
        ].to_tensor_7()  # (F*L, 7)
        rigids_0 = rearrange(rigids_0, "(F L) C -> F L C", F=F, L=seqlen)
        atom14_gt = rearrange(
            of_feat["atom14_gt_positions"], "(F L) ... -> F L ...", F=F, L=seqlen
        )
        pseudo_beta = rearrange(
            of_feat["pseudo_beta"].float(), "(F L) C -> F L C", F=F, L=seqlen
        )
        pseudo_beta_mask = rearrange(
            of_feat["pseudo_beta_mask"].float(), "(F L) -> F L", F=F, L=seqlen
        )
        torsion_sin_cos = rearrange(
            of_feat["torsion_angles_sin_cos"], "(F L) ... -> F L ...", F=F, L=seqlen
        )

        # ---- aatype (per-frame) ----
        aatype = torch.LongTensor(
            [rc.restype_order_with_x[res] for res in seqres] * F
        )  # (F * L)
        aatype = rearrange(aatype, "(F L) -> F L", F=F, L=seqlen)

        # ---- Static torsion-angles mask (depends only on residue identities) ----
        torsion_angles_mask = all_angles_mask_with_x[aatype]  # (F, L, 7)

        data: Dict[str, Any] = {
            "case_id": case.case_id,
            "seqres": seqres,
            "seqlen": seqlen,
            "n_frames": F,
            "aatype": aatype,  # (F, L)
            "rigids_0": rigids_0,  # (F, L, 7)
            "atom14_gt_positions": atom14_gt,  # (F, L, 14, 3)
            "pseudo_beta": pseudo_beta,  # (F, L, 3)
            "pseudo_beta_mask": pseudo_beta_mask,  # (F, L)
            "torsion_angles_sin_cos": torsion_sin_cos,  # (F, L, 7, 2)
            "torsion_angles_mask": torsion_angles_mask,  # (F, L, 7)
            "pos_id": torch.from_numpy(frame_idxs).long(),  # (F,)
        }

        # ---- Pretrained OpenFold representations (single + pair) ----
        if self.repr_loader is not None:
            pretrained = self.repr_loader.load(seqres=seqres)
            # repr_loader returns single (L, S) and pair (L, L, P) -- we tile to
            # (F, L, ...) and (F, L, L, ...) so the encoder treats every frame
            # the same way at the temporal-context level.
            data["pretrained_single"] = pretrained["pretrained_single"]  # (L, S)
            data["pretrained_pair"] = pretrained["pretrained_pair"]  # (L, L, P)

        return data

    # ---- Collate ------------------------------------------------------------

    @staticmethod
    def collate(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of trajectory windows into a flat training batch.

        Output shape conventions match what the encoder/decoder consume in the
        existing inference path: most tensors are ``(B*F, L, ...)``.
        """
        B = len(batch_list)
        F = batch_list[0]["n_frames"]
        seqlens = torch.tensor([d["seqlen"] for d in batch_list])
        max_L = int(seqlens.max().item())

        # Per-residue padding mask (B, max_L)
        padding_mask = torch.arange(max_L).expand(B, max_L) < seqlens.unsqueeze(1)

        def _pad_per_frame(key: str) -> torch.Tensor:
            # Each item is shape (F, L, ...). Pad along L to max_L, stack to (B, F, max_L, ...)
            out = []
            for d in batch_list:
                t = d[key]
                pad_amt = max_L - t.shape[1]
                if pad_amt > 0:
                    pad_shape = list(t.shape)
                    pad_shape[1] = pad_amt
                    t = torch.cat([t, t.new_zeros(pad_shape)], dim=1)
                out.append(t)
            return torch.stack(out, dim=0)

        def _pad_pair(key: str) -> torch.Tensor:
            # pretrained_pair is (L, L, P); pad L on both axes, expand to (B, F, max_L, max_L, P)
            out = []
            for d in batch_list:
                t = d[key]  # (L, L, P)
                L, _, P = t.shape
                if L < max_L:
                    pad = t.new_zeros(max_L, max_L, P)
                    pad[:L, :L, :] = t
                    t = pad
                out.append(t)
            return torch.stack(out, dim=0).unsqueeze(1).expand(-1, F, -1, -1, -1)

        def _pad_single(key: str) -> torch.Tensor:
            # pretrained_single is (L, S); pad to (max_L, S), tile to (B, F, max_L, S)
            out = []
            for d in batch_list:
                t = d[key]  # (L, S)
                L, S = t.shape
                if L < max_L:
                    pad = t.new_zeros(max_L, S)
                    pad[:L, :] = t
                    t = pad
                out.append(t)
            return torch.stack(out, dim=0).unsqueeze(1).expand(-1, F, -1, -1)

        per_frame_keys = [
            "aatype",
            "rigids_0",
            "atom14_gt_positions",
            "pseudo_beta",
            "pseudo_beta_mask",
            "torsion_angles_sin_cos",
            "torsion_angles_mask",
        ]
        batch: Dict[str, Any] = {}
        for key in per_frame_keys:
            stacked = _pad_per_frame(key)  # (B, F, max_L, ...)
            # Flatten (B, F) -> B*F so it matches downstream B*F shapes
            batch[key] = rearrange(stacked, "B F ... -> (B F) ...")

        if "pretrained_single" in batch_list[0]:
            ps = _pad_single("pretrained_single")  # (B, F, max_L, S)
            batch["pretrained_single"] = rearrange(ps, "B F ... -> (B F) ...")
            pp = _pad_pair("pretrained_pair")  # (B, F, max_L, max_L, P)
            batch["pretrained_pair"] = rearrange(pp, "B F ... -> (B F) ...")

        # padding_mask is (B, max_L) -> (B*F, max_L)
        batch["padding_mask"] = padding_mask.repeat_interleave(F, dim=0).float()

        # pos_id is (B, F) -> (B*F,)
        pos_id = torch.stack([d["pos_id"] for d in batch_list], dim=0)  # (B, F)
        batch["pos_id"] = rearrange(pos_id, "B F -> (B F)")

        # Static metadata
        batch["batch_size"] = B
        batch["num_frames"] = F
        batch["case_id"] = [d["case_id"] for d in batch_list]
        return batch
