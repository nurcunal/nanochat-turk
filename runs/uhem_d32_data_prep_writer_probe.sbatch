#!/bin/bash

# Bounded production-writer probe. Submit after the packed sample cluster.
# It creates the pre-safety backend projection consumed by the resource bridge.

#SBATCH --job-name=tr-d32-writer-probe
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${QUOTA_HEADROOM_BYTES:?Set current project-quota headroom in bytes}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v3.json}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v3}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
SAMPLE_RUN_DIR="${SAMPLE_RUN_DIR:-$NANOCHAT_BASE_DIR/data_backend/resource_sample_v3}"
BACKEND_RESOURCE_REPORT="${BACKEND_RESOURCE_REPORT:-$CONTROL_DIR/backend_resource_report.json}"
WRITER_PROBE="${WRITER_PROBE:-$CONTROL_DIR/post_cluster_writer_probe.json}"
PYTHON="${PYTHON:-$CODE_DIR/environments/turkish-data/.venv/bin/python}"
SCRATCH_DIR="${SCRATCH_DIR:-$NANOCHAT_BASE_DIR/scratch/data_prep_writer_probe/job${SLURM_JOB_ID:?}}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
test -f "$POLICY"
test -f "$RECIPE"
test -f "$SOURCE_PLAN"
test -f "$CALIBRATION"
test -f "$SAMPLE_RUN_DIR/cluster_receipt.json"
mkdir -p "$SCRATCH_DIR"
test -d "$SCRATCH_DIR"
test ! -L "$SCRATCH_DIR"
BASE_DEVICE="$("$PYTHON" -c 'import os,sys; print(os.stat(sys.argv[1]).st_dev)' "$NANOCHAT_BASE_DIR")"
SCRATCH_DEVICE="$("$PYTHON" -c 'import os,sys; print(os.stat(sys.argv[1]).st_dev)' "$SCRATCH_DIR")"
test "$BASE_DEVICE" = "$SCRATCH_DEVICE"
test ! -e "$BACKEND_RESOURCE_REPORT"
test ! -e "$WRITER_PROBE"

"$PYTHON" scripts/turkish_data_backend.py resource-report \
    --policy "$POLICY" --source-plan "$SOURCE_PLAN" \
    --calibration "$CALIBRATION" --sample-run-dir "$SAMPLE_RUN_DIR" \
    --quota-headroom-bytes "$QUOTA_HEADROOM_BYTES" \
    --billable-cpus-per-job 128 --safety-factor 1 \
    --output "$BACKEND_RESOURCE_REPORT"

"$PYTHON" scripts/d32_family_workflow.py seal-data-prep-writer-probe \
    --recipe "$RECIPE" --policy "$POLICY" --source-plan "$SOURCE_PLAN" \
    --calibration "$CALIBRATION" --sample-run-dir "$SAMPLE_RUN_DIR" \
    --backend-resource-report "$BACKEND_RESOURCE_REPORT" \
    --scratch-dir "$SCRATCH_DIR" --output "$WRITER_PROBE"

echo "Pre-safety backend resource report: $BACKEND_RESOURCE_REPORT"
echo "Sealed bounded post-cluster writer probe: $WRITER_PROBE"
