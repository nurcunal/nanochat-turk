#!/bin/bash

# Four-rank, one-node srun/NCCL clean-exit probe. No training is performed.

#SBATCH --job-name=tr-d32-static-launch
#SBATCH --account=nakane
#SBATCH --partition=a100x4q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
cd "$CODE_DIR"

if [ "${SLURM_NNODES:-0}" -ne 1 ] || [ "${SLURM_NTASKS:-0}" -ne 4 ]; then
    echo "Static launcher probe requires exactly one 4xA100 node and four tasks" >&2
    exit 2
fi
mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${SLURM_HOSTS[0]}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
export NPROC_PER_NODE=4
FAMILY_ID="${FAMILY_ID:-$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["family_id"])' "$RECIPE")}"
RUN_ID="${FAMILY_ID}_launcher_probe_ws4"
CONTROL_ROOT="$NANOCHAT_BASE_DIR/control/d32/launcher/job${SLURM_JOB_ID}"
RANK_DIR="$CONTROL_ROOT/static_nccl_probe"
RAW_RECEIPT="$CONTROL_ROOT/static_nccl_probe.json"
if [ -e "$CONTROL_ROOT" ]; then
    echo "Refusing to reuse launcher probe directory: $CONTROL_ROOT" >&2
    exit 2
fi
mkdir -p "$RANK_DIR"

srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=16 \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    bash runs/uhem_d32_srun_env.sh .venv/bin/python -m scripts.d32_static_launch_probe \
    --recipe="$RECIPE" --receipt-dir="$RANK_DIR" \
    --run-id="$RUN_ID" --phase=static_nccl_probe
.venv/bin/python scripts/d32_family_workflow.py seal-static-probe \
    --recipe="$RECIPE" --receipt-dir="$RANK_DIR" --run-id="$RUN_ID" \
    --phase=static_nccl_probe --slurm-job-id="$SLURM_JOB_ID" \
    --world-size=4 --nodes=1 --srun-exit-code=0 --output="$RAW_RECEIPT"

echo "Raw launcher probe complete; its afterok finalizer must verify Slurm COMPLETED."
