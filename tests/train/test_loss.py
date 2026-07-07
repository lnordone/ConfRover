# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for :class:`confrover.train.loss.SE3DiffusionLoss`.

These are pure-torch (no OpenFold / GPU required) and focus on the
backbone-atom term added on top of the rot+trans score-matching baseline.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from confrover.train.loss import LossWeights, SE3DiffusionLoss


def _make_inputs(BF: int = 2, L: int = 4, *, t_value: float, seed: int = 0):
    """Build a tiny synthetic (pred, gt_feat) batch for the loss.

    Returns a kwargs dict ready to splat into ``SE3DiffusionLoss.forward``.
    All pred/gt score tensors are leaf tensors with ``requires_grad`` so the
    tests can check gradient flow.
    """
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    rigids_mask = torch.ones(BF, L)
    t = torch.full((BF,), float(t_value))

    pred_rot_score = rand(BF, L, 3).requires_grad_(True)
    pred_trans_score = rand(BF, L, 3).requires_grad_(True)
    pred_atom14 = rand(BF, L, 14, 3).requires_grad_(True)

    gt_feat = {
        "gt_rot_score": rand(BF, L, 3),
        "gt_trans_score": rand(BF, L, 3),
        "rot_score_scaling": torch.ones(BF),
        "trans_score_scaling": torch.ones(BF),
        "atom14_gt_positions": rand(BF, L, 14, 3),
    }

    return dict(
        t=t,
        rigids_mask=rigids_mask,
        torsion_angles_mask=torch.ones(BF, L, 7),
        pred_rigids_0=None,  # unused by implemented terms
        pred_torsion_sin_cos=rand(BF, L, 7, 2),
        pred_atom14=pred_atom14,
        pred_rot_score=pred_rot_score,
        pred_trans_score=pred_trans_score,
        pred_sidechain_frame=rand(BF, L, 8, 4, 4),
        gt_feat=gt_feat,
    )


def test_rot_trans_baseline_unchanged_when_bb_gated_off():
    """With all t above threshold, enabling bb_atom must not change the loss.

    The gated backbone term contributes exactly 0 when no example is
    near-clean, so rot+trans-only and rot+trans+bb losses must match.
    """
    inputs = _make_inputs(t_value=0.9)  # 0.9 > default t_bb_threshold (0.25)

    loss_baseline = SE3DiffusionLoss(LossWeights(rot=1.0, trans=1.0, bb_atom=0.0))
    loss_with_bb = SE3DiffusionLoss(LossWeights(rot=1.0, trans=1.0, bb_atom=0.25))

    l0, aux0 = loss_baseline(**inputs)
    l1, aux1 = loss_with_bb(**inputs)

    assert torch.allclose(l0, l1), "bb_atom leaked into the loss while fully gated"
    # bb term should be reported as ~0 (or absent) when gated off
    assert float(aux1.get("loss_bb_atom", torch.tensor(0.0))) == pytest.approx(
        0.0, abs=1e-6
    )
    assert torch.isfinite(l1)


def test_bb_atom_active_and_differentiable_when_near_clean():
    """With t below threshold, bb_atom is active, positive, and differentiable."""
    inputs = _make_inputs(t_value=0.05)  # 0.05 < 0.25

    loss_mod = SE3DiffusionLoss(LossWeights(rot=1.0, trans=1.0, bb_atom=0.25))
    loss, aux = loss_mod(**inputs)

    assert "loss_bb_atom" in aux
    assert float(aux["loss_bb_atom"]) > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    # Gradient must reach the predicted backbone atoms.
    grad = inputs["pred_atom14"].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0.0


def test_padding_mask_excludes_padded_residues():
    """Residues masked out by rigids_mask must not affect the backbone loss."""
    loss_mod = SE3DiffusionLoss(LossWeights(rot=0.0, trans=0.0, bb_atom=1.0))

    # Corrupt a padded residue's predictions; with it masked, loss is unchanged.
    inputs_masked = _make_inputs(t_value=0.05, L=4)
    inputs_masked["rigids_mask"][:, -1] = 0.0
    _, aux_a = loss_mod(**inputs_masked)
    with torch.no_grad():
        inputs_masked["pred_atom14"][:, -1] += 1000.0  # wild corruption on padded res
    _, aux_b = loss_mod(**inputs_masked)

    assert float(aux_a["loss_bb_atom"]) == pytest.approx(
        float(aux_b["loss_bb_atom"]), abs=1e-4
    )
