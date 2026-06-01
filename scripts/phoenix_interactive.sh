#!/usr/bin/env bash
# Helper to grab an interactive GPU session on PACE Phoenix for ConfRover work.
#
# Phoenix uses --qos (inferno=paid, embers=backfill) instead of --partition.
# See https://docs.pace.gatech.edu/phoenix_cluster/slurm_guide_phnx/.
#
# Usage:
#   1. SSH into a login node:
#        ssh <gtid>@login-phoenix-slurm.pace.gatech.edu
#      (you must be on the GT VPN or eduroam).
#   2. cd into the repo, then:
#        bash scripts/phoenix_interactive.sh
#
# To override defaults inline, e.g.:
#   QOS=inferno GPU_TYPE=H100 WALLTIME=01:00:00 bash scripts/phoenix_interactive.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Edit SLURM_ACCOUNT for your lab/PI.
# Run `pace-quota` to see which accounts you have access to.
# Free-tier accounts look like   gts-<PI>
# Paid accounts look like        gts-<PI>-paid
# ---------------------------------------------------------------------------
SLURM_ACCOUNT="${SLURM_ACCOUNT:-gts-yourPI}"

# QOS: "embers" is backfill (free or near-free, can be preempted by inferno
# jobs but rarely is for short windows). "inferno" is the paid regular queue.
# Recommended for the smoke test: embers.
QOS="${QOS:-embers}"

# GPU type. Phoenix has (as of FY26): H200, H100, L40S, A100, RTX_6000.
# H200/H100/L40S require RHEL9 nodes. Smoke test fits on any of these
# (model is 20M params, seq len ~50). Default H100 because it's plentiful.
GPU_TYPE="${GPU_TYPE:-H100}"
NUM_GPUS="${NUM_GPUS:-1}"

# Walltime, cores, memory. Defaults are sized for the overfit smoke test:
# - 02:00:00 covers conda env setup and the notebook with margin.
# - 8 CPUs is plenty for one GPU (Phoenix caps depend on GPU type).
# - 32G RAM is enough for the 20M model + dataloader + Jupyter.
WALLTIME="${WALLTIME:-02:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"

echo "Requesting interactive shell:"
echo "  account=$SLURM_ACCOUNT  qos=$QOS  gres=gpu:${GPU_TYPE}:${NUM_GPUS}"
echo "  walltime=$WALLTIME  cpus-per-task=$CPUS_PER_TASK  mem=$MEM"
echo ""
echo "Once the prompt drops you onto a compute node, you'll see something like"
echo "  [<gtid>@atl1-1-02-009-5-1 ~]\$"
echo "Then run \`module load anaconda3\` and start your work."

salloc \
    -A "$SLURM_ACCOUNT" \
    -q "$QOS" \
    --gres="gpu:${GPU_TYPE}:${NUM_GPUS}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    -t "$WALLTIME"
