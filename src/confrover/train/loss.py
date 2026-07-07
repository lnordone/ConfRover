# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SE(3) score-matching loss for ConfRover training.

This module provides :class:`SE3DiffusionLoss`, which implements the loss
function the upstream :class:`confrover.model.decoder.confdiff.ConfDiffDecoder`
expects (its :meth:`forward` already calls ``self.loss(...)`` with a fixed
keyword signature, but no concrete ``loss`` is shipped).

Status
------
This file is **a working minimum**, not a full reproduction of the paper's loss.

WORKING NOW:
    * ``loss_rot``: rotation score-matching MSE on so(3), weighted by the
      diffuser's per-time score scaling.
    * ``loss_trans``: translation score-matching MSE on R^3, weighted by the
      diffuser's per-time score scaling.
    * ``loss_bb_atom``: backbone-atom (N, CA, C, O) MSE on ``pred_atom14``,
      active only when ``t < t_bb_threshold`` (default 0.25). Enable by setting
      ``decoder.loss.weights.bb_atom > 0`` (on by default in
      ``confrover_train.yaml``).

TODO (port from https://github.com/bytedance/ConfDiff -- look for ``loss.py``
or ``losses.py`` in their model/decoder directory):
    * ``loss_dist_mat``: pairwise CA-CA distance MSE for local-geometry
      regularisation (helps small models converge).
    * ``loss_torsion``: chi-angle torsion loss (atan2-style, avoids 2pi
      discontinuity), masked by ``torsion_angles_mask``.
    * ``loss_aux_atom14``: full-atom14 MSE during late training only.

The loss components above are documented in the ConfDiff paper (Wang et al.,
2024) Section 3 and visible in their training config as ``loss_weights``.

Notes on the contract with the decoder
---------------------------------------
``ConfDiffDecoder.forward`` does **not** pass ``rigids_t`` to ``self.loss``.
We rely on the training loop (``ConfRoverTrainable.training_step``) to inject
``rigids_t`` and the per-component ground-truth scores into ``gt_feat`` before
the decoder forward runs. See the docstring on
:meth:`SE3DiffusionLoss.forward` for the exact keys we consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from confrover.utils import get_pylogger

logger = get_pylogger(__name__)


@dataclass
class LossWeights:
    """Per-component loss weights.

    Defaults are deliberately low-noise: only rot+trans contribute. Override
    when you port the auxiliary terms from ConfDiff.
    """

    rot: float = 1.0
    trans: float = 1.0
    bb_atom: float = 0.0
    dist_mat: float = 0.0
    torsion: float = 0.0
    aux_atom14: float = 0.0


class SE3DiffusionLoss(nn.Module):
    """Minimal SE(3) score-matching loss for ConfRover.

    Parameters
    ----------
    weights:
        :class:`LossWeights` controlling the contribution of each component.
        Defaults to rot+trans only; auxiliaries are TODO.
    t_bb_threshold:
        Time threshold below which backbone-atom auxiliary loss should activate
        (used by the TODO ``loss_bb_atom`` below). Mirrors ConfDiff convention.
    eps:
        Numerical floor to avoid divide-by-zero when masks have zero sum.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        t_bb_threshold: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.t_bb_threshold = t_bb_threshold
        self.eps = eps

    def forward(
        self,
        *,
        t: torch.Tensor,
        rigids_mask: torch.Tensor,
        torsion_angles_mask: torch.Tensor,
        pred_rigids_0,
        pred_torsion_sin_cos: torch.Tensor,
        pred_atom14: torch.Tensor,
        pred_rot_score: torch.Tensor,
        pred_trans_score: torch.Tensor,
        pred_sidechain_frame: torch.Tensor,
        gt_feat: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the diffusion training loss.

        Parameters
        ----------
        t:
            Diffusion times for each sample, shape ``(B*F,)``. Range
            ``[t_min, 1.0]``.
        rigids_mask:
            Per-residue mask (``padding_mask * structure_mask``), shape
            ``(B*F, L)``.
        pred_rot_score:
            Predicted rotation score (so(3) tangent vector), shape
            ``(B*F, L, 3)``. Already produced by the decoder using
            ``calc_rot_score(rigids_t.rots, pred_rigids_0.rots, t)``.
        pred_trans_score:
            Predicted translation score (R^3), shape ``(B*F, L, 3)``.
        gt_feat:
            Dict that *must* contain (injected by ``training_step``):

            * ``"gt_rot_score"``: ground-truth rotation score from
              ``calc_rot_score(rigids_t.rots, gt_rigids_0.rots, t)``,
              shape ``(B*F, L, 3)``.
            * ``"gt_trans_score"``: ground-truth translation score from
              ``calc_trans_score(rigids_t.trans, gt_rigids_0.trans, t)``,
              shape ``(B*F, L, 3)``.
            * ``"rot_score_scaling"``: per-sample scalar from
              ``so3_diffuser.score_scaling(t)``, shape ``(B*F,)``.
            * ``"trans_score_scaling"``: per-sample scalar from
              ``r3_diffuser.score_scaling(t)``, shape ``(B*F,)``.

            And *may* contain (used by the TODO auxiliaries):

            * ``"atom14_gt_positions"``: ``(B*F, L, 14, 3)``
            * ``"torsion_angles_sin_cos"``: ``(B*F, L, 7, 2)``

        Returns
        -------
        loss : torch.Tensor (scalar)
        aux  : dict of named scalar terms for logging
        """
        aux: Dict[str, torch.Tensor] = {}

        # ---- Rotation score-matching loss (so(3)) ----
        # Expected behaviour: pred_rot_score and gt_rot_score live in the same
        # tangent space at rigids_t. ConfDiff scales the per-sample residual by
        # 1 / score_scaling to give every t-slice equal weight.
        gt_rot_score = gt_feat["gt_rot_score"]
        rot_score_scaling = gt_feat["rot_score_scaling"]  # (B*F,)
        rot_resid = (pred_rot_score - gt_rot_score) * rigids_mask[..., None]
        rot_resid = rot_resid / (rot_score_scaling[:, None, None] + self.eps)
        loss_rot = (rot_resid.pow(2).sum(dim=(-1, -2))) / (
            rigids_mask.sum(dim=-1) + self.eps
        )
        loss_rot = loss_rot.mean()
        aux["loss_rot"] = loss_rot.detach()

        # ---- Translation score-matching loss (R^3) ----
        gt_trans_score = gt_feat["gt_trans_score"]
        trans_score_scaling = gt_feat["trans_score_scaling"]  # (B*F,)
        trans_resid = (pred_trans_score - gt_trans_score) * rigids_mask[..., None]
        trans_resid = trans_resid / (trans_score_scaling[:, None, None] + self.eps)
        loss_trans = (trans_resid.pow(2).sum(dim=(-1, -2))) / (
            rigids_mask.sum(dim=-1) + self.eps
        )
        loss_trans = loss_trans.mean()
        aux["loss_trans"] = loss_trans.detach()

        loss = self.weights.rot * loss_rot + self.weights.trans * loss_trans

        # ---- Backbone-atom MSE (active only when t < t_bb_threshold) ----
        # Ported per ConfDiff's ``loss_bb_atom``. Supervises the predicted
        # backbone atom positions directly (in addition to the score-matching
        # terms) but only on near-clean frames, where ``pred_atom14`` is a
        # meaningful reconstruction rather than pure noise.
        #
        # Atom indexing: in the OpenFold **atom14** layout the first four slots
        # are always N, CA, C, O for every residue type (GLY has no CB but its
        # backbone still occupies [0, 1, 2, 3]), so these indices are safe
        # across residues. NOTE: the ``[0, 1, 2, 4]`` seen in older sketches is
        # **atom37** ordering (O at 4) -- do not use it on atom14 tensors.
        if self.weights.bb_atom > 0:
            bb_idx = [0, 1, 2, 3]  # N, CA, C, O in atom14
            gt_atom14 = gt_feat["atom14_gt_positions"]
            gt_bb = gt_atom14[..., bb_idx, :]  # (B*F, L, 4, 3)
            pred_bb = pred_atom14[..., bb_idx, :]  # (B*F, L, 4, 3)
            m = rigids_mask[..., None, None]  # (B*F, L, 1, 1)
            # Per-example squared error over (residues, 4 backbone atoms, xyz),
            # normalised by the number of supervised atoms in that example.
            sq = ((pred_bb - gt_bb) * m).pow(2).sum(dim=(-1, -2, -3))  # (B*F,)
            denom = len(bb_idx) * rigids_mask.sum(dim=-1) + self.eps  # (B*F,)
            loss_bb_per = sq / denom  # (B*F,)
            # Gate: only supervise low-noise (near-clean) frames.
            active = (t < self.t_bb_threshold).to(loss_bb_per.dtype)  # (B*F,)
            loss_bb = (loss_bb_per * active).sum() / (active.sum() + self.eps)
            aux["loss_bb_atom"] = loss_bb.detach()
            loss = loss + self.weights.bb_atom * loss_bb

        # ---- TODO: pairwise CA-CA distance loss ----
        # Reference: ConfDiff loss.py, "dist_mat" or "dgram".
        if self.weights.dist_mat > 0:
            raise NotImplementedError("loss_dist_mat not yet ported.")

        # ---- TODO: torsion-angle loss ----
        # Use atan2-style loss on (sin, cos) pairs to avoid 2pi wrap.
        # Reference: OpenFold supervised_chi_loss as a starting point, plus
        # ConfDiff's adaptation.
        if self.weights.torsion > 0:
            raise NotImplementedError("loss_torsion not yet ported.")

        # ---- TODO: full atom14 auxiliary loss (late training only) ----
        if self.weights.aux_atom14 > 0:
            raise NotImplementedError("loss_aux_atom14 not yet ported.")

        aux["loss_total"] = loss.detach()
        return loss, aux
