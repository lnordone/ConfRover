# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ConfRover training CLI.

Usage::

    python -m confrover.train.cli \\
        --train_manifest examples/train_manifest_smoke.json \\
        --output_dir ./runs/smoke \\
        --max_steps 2000

Or via Hydra::

    python -m confrover.train.cli +experiment=overfit_smoke

Mirrors the structure of ``confrover.inference.cli``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import hydra
import torch
from lightning import LightningDataModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from confrover import PACKAGE_ROOT
from confrover.utils import get_pylogger, hydra_utils, log_header

log = get_pylogger(__name__)
torch.set_float32_matmul_precision("high")

TRAIN_CONFIG = PACKAGE_ROOT / "configs" / "train.yaml"


def compose_hydra_config(
    train_config,
    model_config,
    cli_overrides=None,
) -> DictConfig:
    """Merge the training config with the model config and CLI overrides."""
    train_cfg = hydra_utils.to_cfg(train_config)
    model_cfg = hydra_utils.wrap_under("model", hydra_utils.to_cfg(model_config))
    cfg = OmegaConf.merge(model_cfg, train_cfg)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    return cfg


def train(cfg: DictConfig) -> Trainer:
    seed_everything(cfg.seed, workers=True)

    log.info(log_header(log, "Instantiate datamodule"))
    log.info(f"   _target_ = {cfg.data._target_}")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(log_header(log, "Instantiate model"))
    log.info(f"   _target_ = {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)

    if cfg.get("from_pretrained_ckpt"):
        log.info(f"Loading initial weights from {cfg.from_pretrained_ckpt}")
        ckpt = torch.load(cfg.from_pretrained_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        log.info(f"   missing={len(missing)} unexpected={len(unexpected)}")

    log.info(log_header(log, "Instantiate Trainer"))
    callbacks = [
        hydra.utils.instantiate(c) for c in (cfg.get("callbacks") or {}).values()
    ]
    logger_obj = (
        hydra.utils.instantiate(cfg.logger) if cfg.get("logger") else None
    )
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger_obj
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), output_dir / "train.yaml")

    log.info(log_header(log, "Start training"))
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("resume_ckpt"))

    log.info(log_header(log, "Done"))
    return trainer


def cli(args: argparse.Namespace) -> None:
    train_cfg: DictConfig = OmegaConf.load(TRAIN_CONFIG)

    # Resolve which model config to use. Default training config inlines a
    # full model block, but we let users point at a saved pretrained ckpt
    # whose .pt embeds the model_cfg used at training time.
    if args.from_pretrained_ckpt and Path(args.from_pretrained_ckpt).exists():
        ckpt = torch.load(args.from_pretrained_ckpt, map_location="cpu")
        model_cfg = ckpt["model_cfg"]
    else:
        model_cfg = OmegaConf.load(
            PACKAGE_ROOT / "configs" / "model" / "confrover_train.yaml"
        )

    cli_overrides = []
    if args.train_manifest:
        cli_overrides.append(
            f"data.train_dataset.config={Path(args.train_manifest).resolve()}"
        )
    if args.val_manifest:
        cli_overrides.append(
            f"data.val_dataset.config={Path(args.val_manifest).resolve()}"
        )
    if args.output_dir:
        cli_overrides.append(f"output_dir={Path(args.output_dir).resolve()}")
    if args.max_steps is not None:
        cli_overrides.append(f"trainer.max_steps={args.max_steps}")
    if args.from_pretrained_ckpt:
        cli_overrides.append(f"from_pretrained_ckpt={args.from_pretrained_ckpt}")
    if args.resume_ckpt:
        cli_overrides.append(f"resume_ckpt={args.resume_ckpt}")
    cli_overrides.extend(args.extra or [])

    cfg = compose_hydra_config(train_cfg, model_cfg, cli_overrides=cli_overrides)
    train(cfg)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--from_pretrained_ckpt", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument(
        "extra",
        nargs="*",
        help="Additional Hydra-style overrides, e.g. trainer.devices=2",
    )
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="confrover-train",
        description="Train ConfRover (community reproduction).",
    )
    parser = add_args(parser)
    args = parser.parse_args()
    cli(args)


if __name__ == "__main__":
    main()
