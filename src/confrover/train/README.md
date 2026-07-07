# `confrover.train` — community training pipeline

The upstream ByteDance ConfRover release is **inference-only**. This subpackage adds the training plumbing the paper describes but does not ship.

This started as a scaffolded template and is being built out into a runnable small-scale training pipeline. The wiring is correct end-to-end; the loss now implements rot + trans score-matching **plus** the backbone-atom auxiliary. The remaining auxiliary loss components are left as TODOs to port from the predecessor [ConfDiff](https://github.com/bytedance/ConfDiff) repo.

## What's in here

| File | Status | Purpose |
| --- | --- | --- |
| `loss.py` | **Working** | `SE3DiffusionLoss` — rot+trans score-matching MSE **and** `loss_bb_atom` (backbone N,CA,C,O MSE, gated on `t < t_bb_threshold`). Remaining aux terms (`dist_mat`, `torsion`, `aux_atom14`) are TODO — port from ConfDiff `loss.py`. |
| `dataset.py` | **Working** | `TrajDataset` — loads ATLAS XTC trajectories, samples F-frame windows (real frame count via mdtraj, multi-replicate + random-stride sampling), runs OpenFold preprocessing on every frame, batches with padding. |
| `module.py` | **Working** | `ConfRoverTrainable` — adds `training_step`, `validation_step`, `configure_optimizers`, and `on_save_checkpoint` (embeds `model_cfg` for `from_pretrained`). Teacher-forced encoding + causal LLaMA pass + **per-example** diffusion noising + a single batched decoder forward. |
| `cli.py` | **Working** | `python -m confrover.train.cli` entrypoint. Composes Hydra configs, instantiates Trainer, calls `.fit`. Pass `--val_manifest` to enable a validation split. |
| `eval/metrics.py` | **Working** | Quantitative ATLAS metrics (RMSF, CA-RMSD/TM-score, Rg, contact maps) comparing generated vs. reference trajectories. Run via `python -m confrover.train.eval`. |

Configs:
- `src/confrover/configs/train.yaml` — top-level Hydra training config.
- `src/confrover/configs/model/confrover_train.yaml` — model config that instantiates `ConfRoverTrainable` and includes a `decoder.loss` block.

Examples / scripts:
- `examples/train_manifest_smoke.json` — single-protein single-window manifest using the bundled `7jfl_C` test data.
- `examples/overfit_smoke.ipynb` — the actual overfit-one-batch smoke test (run this first).
- `examples/pace_01_component_checks.ipynb` — validate the new training code (loss, dataset, checkpoint, metrics) on bundled data, offline, on a PACE GPU node.
- `examples/pace_02_train_and_eval.ipynb` — mini end-to-end run (train → generate → eval) on the 4 bundled proteins, offline.
- `scripts/phoenix_interactive.sh` — `salloc` helper for a Phoenix interactive GPU session.
- `scripts/phoenix_train.sbatch` — SLURM batch template for a longer training run.
- `scripts/build_manifests.py` — build train + eval manifests from an ATLAS dir + a `chain_name,seqres` CSV.

**Full step-by-step for a small-scale run (data prep → train → generate → eval): see [`RUNBOOK.md`](RUNBOOK.md).**

## How to use it

### 1. Smoke test on Phoenix (do this first)

The easiest path is **Open OnDemand** at <https://ondemand-phoenix.pace.gatech.edu/> — Interactive Apps → Jupyter Notebook, request 1 GPU + 8 cores + 32 GB RAM + 2 hours walltime, QOS=`embers`. The smoke test's actual compute is < 5 minutes; the rest of the time covers env setup and downloads.

If you prefer a terminal:

```bash
ssh <gtid>@login-phoenix-slurm.pace.gatech.edu    # GT VPN required
SLURM_ACCOUNT=gts-<yourPI> bash scripts/phoenix_interactive.sh
# once on compute node:
module load anaconda3
source activate ~/scratch/envs/confrover
cd ~/scratch/ConfRover
jupyter lab --no-browser --port=8889 --ip=0.0.0.0    # then SSH-tunnel from your laptop
```

Open `examples/overfit_smoke.ipynb` and run cells top-to-bottom. Loss should drop monotonically by ≥5× over ~500 steps. If it doesn't, **stop and debug** — see "Common failure modes" below — before moving to longer runs.

### 2. Tiny-scale training run (Phase 6 of the roadmap)

Build a manifest with ~10 ATLAS proteins (mimic `examples/train_manifest_smoke.json`'s shape but with multiple cases, each pointing at one of your downloaded ATLAS XTC files), pre-compute their OpenFold features once (run `confrover.data.pretrain_repr.openfold.loader.OpenFoldReprLoader.generate_repr` for each sequence, like the inference path does), then:

```bash
sbatch scripts/phoenix_train.sbatch
# edit TRAIN_MANIFEST and OUTPUT_DIR in the script (or set them via env)
```

### 3. Full-scale training run

Same flow, but with the full ATLAS train split and probably multi-GPU DDP. Edit `scripts/phoenix_train.sbatch` to use `--gres=gpu:H100:4 --ntasks-per-node=4` and set `trainer.devices=4 trainer.strategy=ddp` in the Hydra overrides. Plan for days of GPU time.

## Porting roadmap (read this before changing anything)

Already done in this fork (were TODOs in the original scaffold):

- **Backbone-atom loss (`loss_bb_atom`)** — implemented in `loss.py`: MSE on the (N, CA, C, O) atoms of `pred_atom14` (atom14 indices `[0,1,2,3]`), gated by `t < t_bb_threshold`. Enabled by default via `decoder.loss.weights.bb_atom` in `confrover_train.yaml`.
- **Per-example `t` sampling** — `ConfRoverTrainable._diffuse_per_example` samples one `t` per frame and noises each independently.
- **Random-stride + multi-replicate sampling** — set `strides_in_10ps` on the dataset config and `xtc_fpaths` on a case; `TrajDataset` draws a feasible stride and a replicate per window. Real trajectory length comes from `_count_xtc_frames`.
- **Checkpoint format compatibility** — `ConfRoverTrainable.on_save_checkpoint` embeds an inference-shaped `model_cfg`, so checkpoints load via `ConfRover.from_pretrained` / the inference CLI.

Still to port from the [ConfDiff](https://github.com/bytedance/ConfDiff) repo (same authors, same per-frame denoiser) if you want paper-faithful curves:

1. **Pairwise CA-CA distance loss (`loss_dist_mat`)**. Helps small models learn local geometry.
2. **Torsion loss (`loss_torsion`)**. Use atan2-style loss on (sin, cos) pairs to handle 2π wrap. OpenFold's `supervised_chi_loss` is a good reference.
3. **Full atom14 auxiliary (`loss_aux_atom14`)**. Late-training-only, fine-tunes side chains.
4. **EMA of model weights**. Standard for diffusion training. Add via `lightning.pytorch.callbacks` or hand-roll inside `ConfRoverTrainable`.

## Common failure modes (and what to do)

- **`AssertionError: decoder.loss is None`** — your model config didn't include the `decoder.loss` block. Use `confrover_train.yaml`, not `confrover.yaml`.
- **`forward_marginal` returns NumPy / device mismatch** — the SE3 diffuser mixes torch and numpy internally. The fix in `_diffuse_per_example` casts back; if you see it on a different code path, do the same `torch.as_tensor(..., device=...).to(dtype)` dance.
- **Loss is exactly constant across steps** — gradients aren't reaching the decoder. Most common cause: `freeze_model_nn=true` (default in `confrover.yaml`). The training config sets it to `false`.
- **Loss → NaN at step 1** — almost always a missing/zero score-scaling. Check that `gt_feat["rot_score_scaling"]` and `gt_feat["trans_score_scaling"]` are positive scalars before the loss runs.
- **Memory blow-up on a single 50-residue protein** — the encoder produces a `(B*F, L, L, C)` pair tensor; with `F=4 L=50 C=128` that's only 1.3 MB, but with `L=200` it's 80 MB *per training example* and the LLaMA pass inflates it further. Drop `n_frames` first, then `L`.
- **`KeyError: 'gt_rot_score'`** — the loss expected `gt_feat` keys that `_shared_step` populates, but you bypassed `_shared_step`. If you're calling `decoder.forward(...)` directly, you must inject those keys yourself (or refactor the loss to accept them as top-level kwargs).

## What's intentionally NOT in this template

- **Validation-set generation metrics** (TM-score, RMSD vs. ground truth). These belong in `confrover.train.eval` once you decide on a metric set; the upstream `tests/infer/test_infer.py` is a starting reference.
- **Multi-GPU / DDP wiring**. Lightning handles it via `trainer.devices` and `trainer.strategy`; the configs default to single-GPU. Multi-node Phoenix runs need additional `srun` / `torchrun` boilerplate in the sbatch script.
- **Hyperparameter sweeps**. Use Hydra multirun (`-m`) or W&B sweeps; not pre-wired.
