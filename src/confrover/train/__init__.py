# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ConfRover training pipeline (community reproduction).

The upstream ByteDance release is inference-only. This subpackage adds the
training plumbing that the paper describes but does not ship:

- :mod:`confrover.train.loss`: an SE(3) score-matching loss module that
  matches the signature ``ConfDiffDecoder`` already calls in its ``forward``.
- :mod:`confrover.train.dataset`: a trajectory dataset that loads ATLAS-style
  ``.xtc`` files and produces per-frame ground-truth features for diffusion
  training.
- :mod:`confrover.train.module`: ``ConfRoverTrainable``, a subclass of
  :class:`confrover.model.ConfRover` that adds ``training_step`` and
  ``configure_optimizers``.
- :mod:`confrover.train.cli`: a Lightning ``Trainer`` entrypoint.

See ``src/confrover/train/README.md`` for the porting roadmap and a list of
TODOs you should expect to fill in by reading the ConfDiff repo
(https://github.com/bytedance/ConfDiff).
"""
from __future__ import annotations

from confrover.train.dataset import TrajDataset, TrajDatasetConfig
from confrover.train.loss import SE3DiffusionLoss
from confrover.train.module import ConfRoverTrainable

__all__ = [
    "ConfRoverTrainable",
    "SE3DiffusionLoss",
    "TrajDataset",
    "TrajDatasetConfig",
]
