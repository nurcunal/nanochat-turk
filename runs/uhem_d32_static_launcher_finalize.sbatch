#!/bin/bash

# CPU finalizer for uhem_d32_static_launcher_probe.sbatch.

#SBATCH --job-name=tr-d32-static-final
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${PROBE_JOB_ID:?Set the completed parent launcher-probe job ID}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v4.json}"
PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32_v4/preflight.json}"
cd "$CODE_DIR"

.venv/bin/python scripts/d32_family_workflow.py finalize-static-probe \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --raw-probe-receipt="$NANOCHAT_BASE_DIR/control/d32_v4/launcher/job${PROBE_JOB_ID}/static_nccl_probe.json" \
    --slurm-job-id="$PROBE_JOB_ID" --nodes=1 \
    --output="$NANOCHAT_BASE_DIR/control/d32_v4/static_launcher_gate_ws4.json"
