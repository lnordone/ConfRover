# `confrover.train` — community training pipeline

The upstream ByteDance ConfRover release is **inference-only**. This subpackage adds the training plumbing the paper describes but does not ship.

This is a **scaffolded template**. The wiring is correct end-to-end; the loss function intentionally implements only the rotation + translation score-matching terms so that you have a runnable smoke-test starting point. Auxiliary loss components and other paper-faithful details are left as TODOs that you will fill in by reading the predecessor [ConfDiff](https://github.com/bytedance/ConfDiff) repo.

## What's in here

| File | Status | Purpose |
| --- | --- | --- |
| `loss.py` | **Working minimum** | `SE3DiffusionLoss` — implements rot+trans score-matching MSE. Auxiliary terms (`bb_atom`, `dist_mat`, `torsion`, `aux_atom14`) are TODO — port from ConfDiff `loss.py`. |
| `dataset.py` | **Working** | `TrajDataset` — loads ATLAS XTC trajectories, samples F-frame windows, runs OpenFold preprocessing on every frame, batches with padding. |
| `module.py` | **Working** | `ConfRoverTrainable` — adds `training_step`, `validation_step`, `configure_optimizers` to the upstream LightningModule. Implements teacher-forced encoding + causal LLaMA pass + per-frame diffusion noising + a single batched decoder forward. |
| `cli.py` | **Working** | `python -m confrover.train.cli` entrypoint. Composes Hydra configs, instantiates Trainer, calls `.fit`. |

Configs:
- `src/confrover/configs/train.yaml` — top-level Hydra training config.
- `src/confrover/configs/model/confrover_train.yaml` — model config that instantiates `ConfRoverTrainable` and includes a `decoder.loss` block.

Examples / scripts:
- `examples/train_manifest_smoke.json` — single-protein single-window manifest using the bundled `7jfl_C` test data.
- `examples/overfit_smoke.ipynb` — the actual overfit-one-batch smoke test (run this first).
- `scripts/phoenix_interactive.sh` — `salloc` helper for a Phoenix interactive GPU session.
- `scripts/phoenix_train.sbatch` — SLURM batch template for a longer training run.

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

The smoke-test loss is intentionally minimal (rot + trans score-matching only). To get loss curves that look like the paper's, port these components from the [ConfDiff](https://github.com/bytedance/ConfDiff) repo (same authors, same per-frame denoiser):

1. **Backbone-atom loss (`loss_bb_atom`)**. Find ConfDiff's loss file (typically `model/decoder/.../loss.py`); the function name will contain `bb_atom` or `atom4`. It's an MSE on the (N, CA, C, O) atoms of `pred_atom14`, gated by `t < t_bb_threshold`. Set `decoder.loss.weights.bb_atom > 0` in the model config; the placeholder in `loss.py` raises `NotImplementedError` until you implement the body.
2. **Pairwise CA-CA distance loss (`loss_dist_mat`)**. Helps small models learn local geometry. Same place in ConfDiff.
3. **Torsion loss (`loss_torsion`)**. Use atan2-style loss on (sin, cos) pairs to handle 2π wrap. OpenFold's `supervised_chi_loss` is a good reference.
4. **Full atom14 auxiliary (`loss_aux_atom14`)**. Late-training-only, fine-tunes side chains.

Other things you should expect to revisit:

- **Per-example `t` sampling**. The smoke-test version samples one scalar `t` per training step (acceptable for overfitting, but reduces gradient diversity for full training). Modify `ConfRoverTrainable._diffuse_per_batch` to sample `t` per example and call `forward_marginal` per example (or vectorize). ConfDiff's training loop is the reference.
- **Trajectory-window sampling strategy**. `TrajDataset._sample_window_indices` uses a fixed stride. Paper trains with multiple strides — extend to sample stride from a list (e.g. `[60, 120, 256, 512]` 10-ps).
- **Multi-replicate handling**. Each ATLAS protein has 3 replicates (`*_prod_R{1,2,3}_fit.xtc`). The current dataset only uses one replicate per case; extend `TrajCaseConfig` to carry a list of XTC paths and pick one at random per `__getitem__`.
- **EMA of model weights**. Standard for diffusion training. Add via `lightning.pytorch.callbacks` or hand-roll inside `ConfRoverTrainable`.
- **Checkpoint format compatibility**. After training, save checkpoints as `{"model_cfg": ..., "state_dict": ...}` so they're loadable via the upstream `ConfRover.from_pretrained`. The Lightning `ModelCheckpoint` callback alone doesn't do this — write a small `on_save_checkpoint` hook on `ConfRoverTrainable` to inject `model_cfg`.

## Common failure modes (and what to do)

- **`AssertionError: decoder.loss is None`** — your model config didn't include the `decoder.loss` block. Use `confrover_train.yaml`, not `confrover.yaml`.
- **`forward_marginal` returns NumPy / device mismatch** — the SE3 diffuser mixes torch and numpy internally. The fix in `_diffuse_per_batch` casts back; if you see it on a different code path, do the same `torch.as_tensor(..., device=...).to(dtype)` dance.
- **Loss is exactly constant across steps** — gradients aren't reaching the decoder. Most common cause: `freeze_model_nn=true` (default in `confrover.yaml`). The training config sets it to `false`.
- **Loss → NaN at step 1** — almost always a missing/zero score-scaling. Check that `gt_feat["rot_score_scaling"]` and `gt_feat["trans_score_scaling"]` are positive scalars before the loss runs.
- **Memory blow-up on a single 50-residue protein** — the encoder produces a `(B*F, L, L, C)` pair tensor; with `F=4 L=50 C=128` that's only 1.3 MB, but with `L=200` it's 80 MB *per training example* and the LLaMA pass inflates it further. Drop `n_frames` first, then `L`.
- **`KeyError: 'gt_rot_score'`** — the loss expected `gt_feat` keys that `_shared_step` populates, but you bypassed `_shared_step`. If you're calling `decoder.forward(...)` directly, you must inject those keys yourself (or refactor the loss to accept them as top-level kwargs).

## What's intentionally NOT in this template

- **Validation-set generation metrics** (TM-score, RMSD vs. ground truth). These belong in `confrover.train.eval` once you decide on a metric set; the upstream `tests/infer/test_infer.py` is a starting reference.
- **Multi-GPU / DDP wiring**. Lightning handles it via `trainer.devices` and `trainer.strategy`; the configs default to single-GPU. Multi-node Phoenix runs need additional `srun` / `torchrun` boilerplate in the sbatch script.
- **Hyperparameter sweeps**. Use Hydra multirun (`-m`) or W&B sweeps; not pre-wired.
