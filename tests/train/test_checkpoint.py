# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for checkpoint compatibility with the upstream inference loader."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("openfold")

from confrover.train.module import _to_inference_model_cfg  # noqa: E402


def test_to_inference_model_cfg_rewrites_target_and_strips_training_keys():
    trainable = {
        "_target_": "confrover.train.module.ConfRoverTrainable",
        "seed": 42,
        "t_min": 0.01,
        "t_max": 1.0,
        "optimizer_cfg": {"lr": 1e-4},
        "scheduler_cfg": {"warmup_steps": 1000},
        "encoder": {"_target_": "enc"},
        "decoder": {
            "_target_": "dec",
            "freeze_model_nn": False,
            "loss": {"_target_": "confrover.train.loss.SE3DiffusionLoss"},
        },
    }
    inf = _to_inference_model_cfg(trainable)

    assert inf["_target_"] == "confrover.model.confrover.ConfRover"
    for k in ("optimizer_cfg", "scheduler_cfg", "t_min", "t_max"):
        assert k not in inf
    assert inf["decoder"]["loss"] is None
    # Original is untouched (deepcopy).
    assert trainable["decoder"]["loss"] is not None
    assert trainable["_target_"].endswith("ConfRoverTrainable")


@pytest.mark.slow
def test_checkpoint_roundtrip_loads_into_inference_model():
    """Full round-trip: train module -> embed model_cfg -> load into ConfRover.

    Heavy (instantiates the whole model + SE3 diffuser caches), so only runs on
    a real environment. Verifies the state_dict loads strictly into a plain
    ConfRover built from the embedded config.
    """
    import hydra
    from omegaconf import OmegaConf

    from confrover import PACKAGE_ROOT
    from confrover.model.confrover import ConfRover

    model_cfg = OmegaConf.load(
        PACKAGE_ROOT / "configs" / "model" / "confrover_train.yaml"
    )
    model = hydra.utils.instantiate(model_cfg)
    model.set_model_cfg(OmegaConf.to_container(model_cfg, resolve=True))

    # Emulate what Lightning's ModelCheckpoint + our hook produce.
    ckpt = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(ckpt)
    assert "model_cfg" in ckpt

    inf_model = ConfRover.from_config(ckpt["model_cfg"])
    missing, unexpected = inf_model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
