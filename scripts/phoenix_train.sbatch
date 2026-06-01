#!/bin/bash
# ============================================================================
# PACE Phoenix SLURM batch script for ConfRover training.
#
# Phoenix uses --qos (inferno=paid, embers=backfill) instead of --partition.
# See https://docs.pace.gatech.edu/phoenix_cluster/slurm_guide_phnx/.
#
# Submit:    sbatch scripts/phoenix_train.sbatch
# Inspect:   squeue --me ; tail -f logs/<jobname>-<jobid>.out
# Cancel:    scancel <jobid>
#
# Edit the #SBATCH lines marked with CHANGE before submitting.
# ============================================================================

#SBATCH --job-name=confrover-train
#SBATCH --account=gts-yourPI               # CHANGE: run `pace-quota` to find yours
#SBATCH --qos=embers                       # embers (free/backfill) or inferno (paid)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1                  # H100 / H200 / L40S / A100 / RTX_6000
#SBATCH --mem=64G
#SBATCH --time=12:00:00                    # max 8 days on inferno; embers may be preempted
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=NONE                   # set to BEGIN,END,FAIL if you want emails

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load anaconda3                      # PACE-managed Anaconda
# CUDA module: name varies; on RHEL9 nodes try cuda/12.4 or cuda/12.1.
# Run `module avail cuda` on a compute node to see what's installed.
module load cuda/12.1 || module load cuda/12.4 || true

# CHANGE: path to your conda env. Phoenix scratch is /storage/scratch1/<X>/<gtid>
# (typically symlinked as $HOME/scratch).
CONDA_ENV="${CONDA_ENV:-$HOME/scratch/envs/confrover}"
source activate "$CONDA_ENV"

# CHANGE: path to your repo (also typically on scratch)
REPO_DIR="${REPO_DIR:-$HOME/scratch/ConfRover}"
cd "$REPO_DIR"

# Cache: keep weights, MSAs, OpenFold reps on scratch (NOT in $HOME, which is
# only 10 GB on Phoenix).
export CONFROVER_CACHE_DIR="${CONFROVER_CACHE_DIR:-$HOME/scratch/confrover_cache}"
mkdir -p "$CONFROVER_CACHE_DIR"

echo "Host: $(hostname)"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-(none)}"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
TRAIN_MANIFEST="${TRAIN_MANIFEST:-examples/train_manifest_smoke.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/scratch/confrover_runs/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}}"
MAX_STEPS="${MAX_STEPS:-100000}"

mkdir -p "$OUTPUT_DIR"
echo "Output: $OUTPUT_DIR"

python -m confrover.train.cli \
    --train_manifest "$TRAIN_MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    "data.train_dataset.repr_loader.repr_root=${CONFROVER_CACHE_DIR}/folding_repr"
