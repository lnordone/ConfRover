# ConfRover small-scale training runbook

End-to-end steps to train a *decent* model on a ~50–100 protein ATLAS subset and
evaluate it through the public inference path. The **code** for every step lives
in this repo; this runbook is the operational sequence you run on a GPU host
(e.g. PACE Phoenix). Steps that need a GPU or the ATLAS download cannot run in a
CPU-only CI container.

Prereqs: the conda env from the top-level README (`pip install .` +
`pip install --no-build-isolation .[openfold]`), and a scratch cache dir
(`export CONFROVER_CACHE_DIR=~/scratch/confrover_cache`).

---

## Phase 0 — Smoke test (overfit one batch)

Run `examples/overfit_smoke.ipynb` top-to-bottom on 1 GPU (see
`scripts/phoenix_interactive.sh`). Pass criterion: `train/loss` drops ≥5–10×
monotonically, no NaNs. With the backbone-atom loss now on by default, you
should also see a non-zero `loss_bb_atom` once `t < 0.25` frames appear. If it
fails, use the "Common failure modes" table in `README.md`.

Unit tests (CPU, fast) before/after code changes:

```bash
pytest tests/train -m "not slow"          # loss / dataset-logic / metrics
pytest tests/infer/test_infer.py          # inference regression (needs repr data)
pytest tests/train/test_checkpoint.py -m slow   # full ckpt round-trip (heavy)
```

## Phase 4 — Build the subset + precompute features

1. Assemble a CSV of `chain_name,seqres` for ~65–115 ATLAS chains (train + eval),
   same columns as `tests/test_data/atlas_test_small.csv`.
2. Download each chain's `<case_id>.pdb` + `<case_id>_prod_R{1,2,3}_fit.xtc`
   into `<atlas_root>/<case_id>/`.
3. Generate train + eval manifests (holds out `--n_eval` chains for eval):

   ```bash
   python scripts/build_manifests.py \
       --csv my_atlas_subset.csv \
       --atlas_root $HOME/scratch/atlas \
       --out_dir manifests/ \
       --n_eval 15 --n_frames 8 --strides 60 120 256 512
   ```

4. Precompute OpenFold features once per sequence (cached, reused at train +
   inference). The simplest path reuses the inference machinery:

   ```bash
   confrover openfold_repr --help    # or OpenFoldReprLoader.generate_repr(...)
   ```

   Point `--folding_repr` / `repr_root` at `$CONFROVER_CACHE_DIR/folding_repr`.

## Phase 5 — Train

```bash
python -m confrover.train.cli \
    --train_manifest manifests/atlas_subset_train.json \
    --val_manifest   manifests/atlas_subset_eval.json \
    --output_dir     $HOME/scratch/confrover_runs/subset01 \
    --max_steps      100000 \
    "data.train_dataset.relpath_to=$HOME/scratch/atlas" \
    "data.train_dataset.repr_loader.repr_root=$CONFROVER_CACHE_DIR/folding_repr"
```

(`scripts/phoenix_train.sbatch` wraps this for SLURM.) `--val_manifest` turns on
the validation split automatically (mirrored from `train_dataset`). Watch
`train/loss`, `train/loss_rot`, `train/loss_trans`, `train/loss_bb_atom`,
`val/loss` in TensorBoard (`--output_dir/confrover_train`). Healthy =
smooth decrease with `val/loss` tracking `train/loss`.

Checkpoints land in `<output_dir>/checkpoints/` and already embed a
`model_cfg` (via `on_save_checkpoint`), so they load through the inference path.
Start single-GPU; add `trainer.devices=4 trainer.strategy=ddp` only if you need
throughput.

## Phase 6a — Generate with the public inference CLI

Convert a Lightning `.ckpt` to the `.pt` the CLI expects (dict with `model_cfg`
+ `state_dict`) if needed — the embedded `model_cfg` makes this a straight
re-save — then:

```bash
confrover generate \
    --job_config manifests/atlas_subset_eval.json \
    --model      <path/to/checkpoint.pt> \
    --output     $HOME/scratch/confrover_gen \
    --diffusion_steps 200
```

This exercises the exact upstream inference code (`inference.py`, sampler,
writer). Output: `<output>/atlas_subset_eval/<case_id>/<case_id>_sample*.xtc`.

## Phase 6b — Quantitative ATLAS metrics

```bash
confrover eval \
    --gen_dir $HOME/scratch/confrover_gen/atlas_subset_eval \
    --ref_dir $HOME/scratch/atlas \
    --output_dir $HOME/scratch/confrover_gen/atlas_subset_eval
# writes metrics.csv + metrics.json (per-case + aggregate)
```

(Equivalently `python -m confrover.train.eval ...`.) Reports per-residue RMSF
correlation/MAE, radius-of-gyration Wasserstein distance, CA contact-map MAE,
and ensemble coverage (mean min CA-RMSD / best TM-score to the reference).

**Success bar** (a "decent", not SOTA, model): `rmsf_pearson` clearly positive
(≳0.5), generated Rg distribution overlapping the reference, and coverage
`best_tmscore` well above what a random/static structure scores.
