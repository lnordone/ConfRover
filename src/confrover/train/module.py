# Copyright 2026 Lucas Nordone, Georgia Tech GCML Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ConfRoverTrainable: LightningModule subclass that adds the training loop.

The upstream :class:`confrover.model.ConfRover` only implements
``predict_step``. This subclass adds:

- :meth:`training_step` that performs teacher-forced encoding of all F frames,
  one causal LLaMA pass, per-frame diffusion noising via the
  :class:`SE3Diffuser`, and a single batched call to
  ``decoder.forward(...)`` for the loss.
- :meth:`validation_step` (basic; logs val loss).
- :meth:`configure_optimizers`: AdamW + linear warmup + cosine decay.

The training arrangement
------------------------
Given F clean ground-truth frames (``rigids_0``, ``atom14_gt_positions``, ...):

1. Encode every frame **using its clean structure** (teacher forcing).
2. Build the LLaMA input sequence by prepending a learned mask token and
   dropping the last frame. So LLaMA sees ``[mask, frame_0, ..., frame_{F-2}]``
   and outputs F hidden states; output position ``i`` is the temporal context
   for predicting frame ``i``.
3. Sample diffusion times ``t`` and apply ``diffuser.forward_marginal`` to each
   ground-truth frame to get noisy ``rigids_t``.
4. Run ``decoder.forward(...)`` once on the flat ``B*F`` batch -- it expects
   per-frame ``s``, ``z``, ``t``, ``rigids_t``, and ``gt_feat`` and returns the
   loss directly.

This mirrors the inference path's encode -> temporal -> decoder structure,
just trained in parallel across frames instead of sampled autoregressively.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from openfold.utils import rigid_utils as ru

from confrover.model.confrover import ConfRover
from confrover.utils import get_pylogger
from confrover.utils.torch.tensor import rearrange

logger = get_pylogger(__name__)


class ConfRoverTrainable(ConfRover):
    """ConfRover with training_step + configure_optimizers added.

    Parameters
    ----------
    encoder, temporal, decoder, writer, seed:
        Same as :class:`confrover.model.ConfRover`.
    optimizer_cfg:
        Dict with keys ``lr``, ``weight_decay``, ``betas``. Used by
        :meth:`configure_optimizers`.
    scheduler_cfg:
        Dict with keys ``warmup_steps``, ``total_steps``, ``min_lr_ratio``.
        Set ``warmup_steps=0`` and ``total_steps=0`` to disable scheduling.
    t_min, t_max:
        Diffusion-time sampling range. Defaults match the inference sampler.
    grad_clip_val:
        Stored for reference; actually applied by Lightning's Trainer
        (``gradient_clip_val`` arg).
    """

    def __init__(
        self,
        encoder: nn.Module,
        temporal,
        decoder,
        writer=None,
        seed: int = 42,
        optimizer_cfg: Dict[str, Any] | None = None,
        scheduler_cfg: Dict[str, Any] | None = None,
        t_min: float = 0.01,
        t_max: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            encoder=encoder,
            temporal=temporal,
            decoder=decoder,
            writer=writer,
            seed=seed,
            **kwargs,
        )
        self.optimizer_cfg = optimizer_cfg or {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "betas": (0.9, 0.999),
        }
        self.scheduler_cfg = scheduler_cfg or {
            "warmup_steps": 1_000,
            "total_steps": 100_000,
            "min_lr_ratio": 0.1,
        }
        self.t_min = t_min
        self.t_max = t_max

    # =========================================================================
    # Training step
    # =========================================================================

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, aux = self._shared_step(batch, stage="train")
        self.log("train/loss", loss, prog_bar=True, batch_size=batch["batch_size"])
        for k, v in aux.items():
            self.log(f"train/{k}", v, batch_size=batch["batch_size"])
        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss, aux = self._shared_step(batch, stage="val")
        self.log("val/loss", loss, prog_bar=True, batch_size=batch["batch_size"])
        for k, v in aux.items():
            self.log(f"val/{k}", v, batch_size=batch["batch_size"])
        return loss

    def _shared_step(
        self, batch: Dict[str, Any], stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert self.decoder.loss is not None, (
            "decoder.loss is None -- training requires a loss module. "
            "Set it via the model config (see configs/model/confrover_train.yaml)."
        )

        B = batch["batch_size"]
        F = batch["num_frames"]
        BF, L = batch["aatype"].shape  # (B*F, L)
        assert BF == B * F

        aatype = batch["aatype"]  # (B*F, L)
        padding_mask = batch["padding_mask"]  # (B*F, L)
        pseudo_beta = batch["pseudo_beta"]  # (B*F, L, 3)
        pseudo_beta_mask = batch["pseudo_beta_mask"]  # (B*F, L)
        rigids_0 = batch["rigids_0"]  # (B*F, L, 7)
        atom14_gt = batch["atom14_gt_positions"]  # (B*F, L, 14, 3)
        torsion_sin_cos = batch["torsion_angles_sin_cos"]  # (B*F, L, 7, 2)
        torsion_angles_mask = batch["torsion_angles_mask"]  # (B*F, L, 7)
        pretrained_single = batch.get("pretrained_single")  # (B*F, L, S) | None
        pretrained_pair = batch.get("pretrained_pair")  # (B*F, L, L, P) | None
        pos_id = batch.get("pos_id")  # (B*F,) | None

        # ---- Step 1: encode all F frames using clean ground-truth structure ----
        # The encoder uses pseudo_beta + pseudo_beta_mask as the structure
        # signal (see confrover/model/encoder/pseudo_beta_pair.py); rigids_0
        # is unused at this stage in the released encoder.
        no_struct_mask = torch.ones(BF, dtype=aatype.dtype, device=aatype.device)
        single_feat, pair_feat = self.encoder(
            aatype=aatype,
            padding_mask=padding_mask,
            rigids_0=rigids_0,
            batch_size=B,
            struct_mask=no_struct_mask,
            pseudo_beta=pseudo_beta,
            pseudo_beta_mask=pseudo_beta_mask,
            pretrained_single=pretrained_single,
            pretrained_pair=pretrained_pair,
        )  # single: (B*F, L, C); pair: (B*F, L, L, C)

        # ---- Step 2: encode the BOS mask token and prepend it along F ----
        # Mask token has identity rotation, zero translation, zero pseudo-beta.
        mask_pseudo_beta = self.mask_token_pseudo_beta.expand(B, L, -1)
        mask_pseudo_beta_mask = self.mask_token_pseudo_beta_mask.expand(B, L)
        mask_aatype = aatype[::F]  # one frame per batch element
        mask_padding_mask = padding_mask[::F]
        mask_pretrained_single = (
            pretrained_single[::F] if pretrained_single is not None else None
        )
        mask_pretrained_pair = (
            pretrained_pair[::F] if pretrained_pair is not None else None
        )
        mask_single, mask_pair = self.encoder(
            aatype=mask_aatype,
            padding_mask=mask_padding_mask,
            rigids_0=self.mask_token_rigids.expand(B, L, -1),
            batch_size=B,
            struct_mask=torch.zeros(B, dtype=aatype.dtype, device=aatype.device),
            pseudo_beta=mask_pseudo_beta,
            pseudo_beta_mask=mask_pseudo_beta_mask,
            pretrained_single=mask_pretrained_single,
            pretrained_pair=mask_pretrained_pair,
        )  # mask_single: (B, L, C); mask_pair: (B, L, L, C)

        # ---- Step 3: fuse single+pair into per-frame token sequences ----
        # M = L + L*L "tokens" per frame, identical to the inference path.
        fused_clean = self._fuse_single_pair((single_feat, pair_feat))  # (B*F, M, C)
        fused_mask = self._fuse_single_pair((mask_single, mask_pair))  # (B, M, C)

        # ---- Step 4: build LLaMA input [mask, frame_0, ..., frame_{F-2}] ----
        fused_clean_BFMC = rearrange(fused_clean, "(B F) M C -> B F M C", B=B)
        # drop last frame, prepend mask -> (B, F, M, C)
        llama_input_BFMC = torch.cat(
            [fused_mask.unsqueeze(1), fused_clean_BFMC[:, :-1]], dim=1
        )
        llama_input = rearrange(
            llama_input_BFMC, "B F M C -> (B M) F C"
        )  # (B*M, F, C)

        # Position ids: tile per-frame pos_id across all M tokens of that frame.
        if pos_id is None:
            pos_id_BF = (
                torch.arange(F, device=aatype.device).unsqueeze(0).expand(B, -1)
            )
        else:
            pos_id_BF = rearrange(pos_id, "(B F) -> B F", B=B)
        M = llama_input.shape[0] // B
        pos_id_for_llama = pos_id_BF.unsqueeze(1).expand(-1, M, -1)  # (B, M, F)
        pos_id_for_llama = rearrange(pos_id_for_llama, "B M F -> (B M) F")

        # ---- Step 5: causal LLaMA pass (no KV cache during training) ----
        # rigids_mask passed to the temporal block is used by the embedded
        # pairformer to mask single/pair attention. Use the (B*F, L) mask.
        temporal_out = self.temporal(
            inputs_embeds=llama_input,
            rigids_mask=padding_mask,  # (B*F, L) -- pairformer masks
            batch_size=B,
            position_ids=pos_id_for_llama,
            use_cache=False,
            return_dict=True,
        )
        # last_hidden_state: (B*M, F, C)
        hidden = temporal_out.last_hidden_state
        # Reshape back to per-frame single/pair features.
        hidden_BFMC = rearrange(hidden, "(B M) F C -> (B F) M C", B=B)
        s_out, z_out = self._split_single_pair(hidden_BFMC, seqlen=L)
        # s_out: (B*F, L, C); z_out: (B*F, L, L, C)

        # ---- Step 6: sample diffusion times and noise rigids_0 -> rigids_t ----
        # NOTE: diffuser.forward_marginal in this repo expects a scalar t. To
        # batch per-example t we'd loop or vectorize; the smoke-test version
        # samples a single t for the whole batch (acceptable -- gradient
        # diversity comes from many steps, not many t's per step).
        # TODO: port per-example t sampling from ConfDiff for production runs.
        t_scalar = float(
            torch.empty(1).uniform_(self.t_min, self.t_max).item()
        )
        t_vec = torch.full(
            (BF,), t_scalar, dtype=s_out.dtype, device=s_out.device
        )
        rigids_t, gt_rot_score, gt_trans_score, rot_scaling, trans_scaling = (
            self._diffuse_per_batch(rigids_0, t_scalar)
        )
        # All returned shapes: rigids_t (B*F, L, 7), gt_rot/trans_score (B*F, L, 3),
        # *_scaling (B*F,) (constant per-step at smoke-test fidelity).

        # ---- Step 7: assemble gt_feat for the loss ----
        gt_feat: Dict[str, Any] = {
            "rigids_0": rigids_0,
            "atom14_gt_positions": atom14_gt,
            "pseudo_beta": pseudo_beta,
            "pseudo_beta_mask": pseudo_beta_mask,
            "torsion_angles_sin_cos": torsion_sin_cos,
            "gt_rot_score": gt_rot_score,
            "gt_trans_score": gt_trans_score,
            "rot_score_scaling": rot_scaling,
            "trans_score_scaling": trans_scaling,
        }

        # ---- Step 8: decoder forward (computes per-frame loss in parallel) ----
        loss, aux_info, _output = self.decoder(
            aatype=aatype,
            s=s_out,
            z=z_out,
            t=t_vec,
            rigids_t=rigids_t,
            rigids_mask=padding_mask,  # decoder multiplies by padding mask internally
            padding_mask=padding_mask,
            gt_feat=gt_feat,
            torsion_angles_mask=torsion_angles_mask,
            pretrained_single=pretrained_single,
            pretrained_pair=pretrained_pair,
        )
        return loss, aux_info

    # =========================================================================
    # Diffusion helper
    # =========================================================================

    def _diffuse_per_batch(
        self, rigids_0_tensor: torch.Tensor, t_scalar: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply ``diffuser.forward_marginal`` and compute ground-truth scores.

        Parameters
        ----------
        rigids_0_tensor:
            ``(B*F, L, 7)`` ground-truth rigids.
        t_scalar:
            Single diffusion time used for the whole batch.

        Returns
        -------
        rigids_t : (B*F, L, 7)
        gt_rot_score : (B*F, L, 3)
        gt_trans_score : (B*F, L, 3)
        rot_score_scaling : (B*F,)
        trans_score_scaling : (B*F,)
        """
        diffuser = self.decoder.diffuser
        device = rigids_0_tensor.device
        BF, L, _ = rigids_0_tensor.shape

        rigids_0 = ru.Rigid.from_tensor_7(rigids_0_tensor)

        # diffuser.forward_marginal expects num_frames so it can broadcast
        # noise correctly. With BF flattened and num_frames=1 we noise each
        # rigid independently (which is what we want for training).
        marg = diffuser.forward_marginal(
            rigids_0=rigids_0,
            t=t_scalar,
            num_frames=1,
            as_tensor_7=True,
        )
        rigids_t = marg["rigids_t"]
        if not isinstance(rigids_t, torch.Tensor):
            rigids_t = torch.as_tensor(rigids_t)
        rigids_t = rigids_t.to(device=device, dtype=rigids_0_tensor.dtype)

        # Ground-truth scores: same calculation the decoder does with predicted
        # rigids_0, but using the GT rigids_0.
        rigids_t_obj = ru.Rigid.from_tensor_7(rigids_t)
        t_tensor = torch.full(
            (BF,), t_scalar, dtype=rigids_0_tensor.dtype, device=device
        )
        gt_rot_score = diffuser.calc_rot_score(
            rigids_t_obj.get_rots(),
            rigids_0.get_rots(),
            t_tensor,
            use_cached_score=False,
        )
        gt_trans_score = diffuser.calc_trans_score(
            rigids_t_obj.get_trans(),
            rigids_0.get_trans(),
            t_tensor[:, None, None],
            use_torch=True,
        )

        # Score scalings (used by the loss to equalize gradient magnitude
        # across t). The diffuser returns numpy scalars; broadcast to (B*F,).
        rot_scaling = float(diffuser._so3_diffuser.score_scaling(t_scalar))
        trans_scaling = float(diffuser._r3_diffuser.score_scaling(t_scalar))
        rot_scaling_t = torch.full(
            (BF,), rot_scaling, dtype=rigids_0_tensor.dtype, device=device
        )
        trans_scaling_t = torch.full(
            (BF,), trans_scaling, dtype=rigids_0_tensor.dtype, device=device
        )

        return rigids_t, gt_rot_score, gt_trans_score, rot_scaling_t, trans_scaling_t

    # =========================================================================
    # Optimizer / scheduler
    # =========================================================================

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.optimizer_cfg["lr"],
            weight_decay=self.optimizer_cfg.get("weight_decay", 0.0),
            betas=tuple(self.optimizer_cfg.get("betas", (0.9, 0.999))),
        )
        warmup = int(self.scheduler_cfg.get("warmup_steps", 0))
        total = int(self.scheduler_cfg.get("total_steps", 0))
        if warmup == 0 and total == 0:
            return opt

        min_ratio = float(self.scheduler_cfg.get("min_lr_ratio", 0.1))
        import math

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cos

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
