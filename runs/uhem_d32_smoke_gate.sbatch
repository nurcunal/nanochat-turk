#!/bin/bash

# Seal the 8-to-16-GPU throughput gate. No GPU is requested.

#SBATCH --job-name=tr-d32-smoke-gate
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v1.json}"
cd "$CODE_DIR"

compare_args=(
    --recipe="$RECIPE"
    --smoke-8gpu="$NANOCHAT_BASE_DIR/control/d32/smoke_ws8.json"
    --output="$NANOCHAT_BASE_DIR/control/d32/production_topology_gate.json"
)
smoke16="$NANOCHAT_BASE_DIR/control/d32/smoke_ws16.json"
if [ -f "$smoke16" ]; then
    compare_args+=(--smoke-16gpu="$smoke16")
else
    echo "No clean sealed 16-GPU smoke supplied; sealing the explicit 8-GPU fallback."
fi
.venv/bin/python scripts/d32_family_workflow.py compare-smokes "${compare_args[@]}"
