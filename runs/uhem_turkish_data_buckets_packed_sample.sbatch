#!/bin/bash

# Run all fourteen fixed DataTrove MinHash buckets concurrently inside one
# exclusive cpu2dq allocation.  Each Slurm task owns exactly one disjoint
# bucket rank/output.  This interface is sample-only and accepts no approval.

#SBATCH --job-name=tr-data-bucket14
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --ntasks-per-node=14
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${OBJECT_SAMPLE_LAUNCH_RECEIPT:?Set the verified packed object launch receipt}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$CODE_DIR/runs/uhem_d32_require_clean_code.sh"
d32_require_clean_committed_code "$CODE_DIR"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v4.json}"
D32_PATH_CONTRACT="${D32_PATH_CONTRACT:-v4}"
case "$D32_PATH_CONTRACT" in
    v3|v4) source "$CODE_DIR/runs/uhem_d32_${D32_PATH_CONTRACT}_paths.sh" ;;
    *) echo "D32_PATH_CONTRACT must equal v3 or v4" >&2; exit 2 ;;
esac
CONTROL_DIR="${CONTROL_DIR:-$D32_CONTROL_DIR}"
d32_bind_canonical_path CONTROL_DIR "$D32_CONTROL_DIR"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
SAMPLE_RANKS="${SAMPLE_RANKS:-$CONTROL_DIR/resource_sample_ranks.json}"
LANE_PLAN="${LANE_PLAN:-$CONTROL_DIR/resource_sample_lane_plan.json}"
d32_bind_canonical_path SOURCE_PLAN "$D32_SOURCE_PLAN"
d32_bind_canonical_path CALIBRATION "$D32_CALIBRATION"
d32_bind_canonical_path SAMPLE_RANKS "$D32_SAMPLE_RANKS"
d32_bind_canonical_path LANE_PLAN "$D32_LANE_PLAN"
DATA_RUN_DIR="${DATA_RUN_DIR:-$D32_SAMPLE_RUN_DIR}"
d32_bind_canonical_path DATA_RUN_DIR "$D32_SAMPLE_RUN_DIR"
PYTHON="$CODE_DIR/environments/turkish-data/.venv/bin/python"
LAUNCH_ROOT="$DATA_RUN_DIR/packed_bucket_launches/job${SLURM_JOB_ID:?}"
TASK_RECEIPT_DIR="$LAUNCH_ROOT/tasks"
LAUNCH_RECEIPT="$LAUNCH_ROOT/launch_receipt.json"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
test -f "$POLICY"
test -f "$SOURCE_PLAN"
test -f "$CALIBRATION"
test -f "$SAMPLE_RANKS"
test -f "$LANE_PLAN"
test -f "$OBJECT_SAMPLE_LAUNCH_RECEIPT"
if [ "${SLURM_NNODES:-0}" -ne 1 ] || \
   [ "${SLURM_NTASKS:-0}" -ne 14 ] || \
   [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
    echo "Packed buckets require exactly one node, 14 tasks, and 8 CPUs/task" >&2
    exit 2
fi
if [ -e "$LAUNCH_ROOT" ]; then
    echo "Refusing to reuse packed bucket launch evidence: $LAUNCH_ROOT" >&2
    exit 2
fi
mkdir -p "$TASK_RECEIPT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

set +e
srun --nodes=1 --ntasks=14 --ntasks-per-node=14 --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    --output="$LAUNCH_ROOT/bucket-%t.out" \
    --error="$LAUNCH_ROOT/bucket-%t.err" \
    "$PYTHON" -m scripts.turkish_packed_sample run-bucket \
    --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
    --calibration="$CALIBRATION" --sample-ranks="$SAMPLE_RANKS" \
    --lane-plan="$LANE_PLAN" \
    --object-launch-receipt="$OBJECT_SAMPLE_LAUNCH_RECEIPT" \
    --run-dir="$DATA_RUN_DIR" --receipt-dir="$TASK_RECEIPT_DIR"
srun_rc=$?
set -e
if [ "$srun_rc" -ne 0 ]; then
    echo "Packed buckets failed collectively (srun rc=$srun_rc); inspect $LAUNCH_ROOT" >&2
    exit "$srun_rc"
fi

"$PYTHON" -m scripts.turkish_packed_sample finalize-buckets \
    --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
    --calibration="$CALIBRATION" --sample-ranks="$SAMPLE_RANKS" \
    --lane-plan="$LANE_PLAN" \
    --object-launch-receipt="$OBJECT_SAMPLE_LAUNCH_RECEIPT" \
    --run-dir="$DATA_RUN_DIR" --receipt-dir="$TASK_RECEIPT_DIR" \
    --job-id="$SLURM_JOB_ID" --output="$LAUNCH_RECEIPT"
test -f "$LAUNCH_RECEIPT"
echo "Verified packed MinHash sample buckets: $LAUNCH_RECEIPT"
