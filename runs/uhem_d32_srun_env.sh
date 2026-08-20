#!/bin/bash

# Translate Slurm's static task coordinates into torch.distributed env:// names.

set -euo pipefail

: "${SLURM_PROCID:?}"
: "${SLURM_LOCALID:?}"
: "${SLURM_NTASKS:?}"
: "${MASTER_ADDR:?}"
: "${MASTER_PORT:?}"

export RANK="$SLURM_PROCID"
# --gpus-per-task=1 remaps the assigned GPU to local CUDA device zero.
export LOCAL_RANK=0
export NODE_LOCAL_RANK="$SLURM_LOCALID"
export WORLD_SIZE="$SLURM_NTASKS"
export LOCAL_WORLD_SIZE="${NPROC_PER_NODE:-4}"

exec "$@"
