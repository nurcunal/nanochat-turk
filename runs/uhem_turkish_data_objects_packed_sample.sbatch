#!/bin/bash

# One full cpu2dq node runs the deterministic v2 object resource sample as
# thirty-two disjoint lanes. Each worker is single-core in practice, while four
# CPUs/task preserve one full exclusive 128-CPU allocation. This launcher has
# no production mode: every object is processed with sample_mode=True and no
# resource approval can be supplied.

#SBATCH --job-name=tr-data-sample32
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v2}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v2.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v2}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
SAMPLE_RANKS="${SAMPLE_RANKS:-$CONTROL_DIR/resource_sample_ranks.json}"
LANE_PLAN="${LANE_PLAN:-$CONTROL_DIR/resource_sample_lane_plan.json}"
GLOTLID_MODEL="${GLOTLID_MODEL:-$NANOCHAT_BASE_DIR/source_models/glotlid-v3/model_v3.bin}"
DATA_RUN_DIR="${DATA_RUN_DIR:-$NANOCHAT_BASE_DIR/data_backend/resource_sample_v2}"
PYTHON="$CODE_DIR/environments/turkish-data/.venv/bin/python"
LAUNCH_ROOT="$DATA_RUN_DIR/packed_sample_launches/job${SLURM_JOB_ID:?}"
LANE_RECEIPT_DIR="$LAUNCH_ROOT/lanes"
LAUNCH_RECEIPT="$LAUNCH_ROOT/launch_receipt.json"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
test -f "$POLICY"
test -f "$SOURCE_PLAN"
test -f "$CALIBRATION"
test -f "$SAMPLE_RANKS"
test -f "$GLOTLID_MODEL"
if [ "${SLURM_NNODES:-0}" -ne 1 ] || \
   [ "${SLURM_NTASKS:-0}" -ne 32 ] || \
   [ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]; then
    echo "Packed sample requires exactly one node, 32 tasks, and 4 CPUs/task" >&2
    exit 2
fi
if [ -e "$LAUNCH_ROOT" ]; then
    echo "Refusing to reuse packed sample launch evidence: $LAUNCH_ROOT" >&2
    exit 2
fi
mkdir -p "$DATA_RUN_DIR" "$LANE_RECEIPT_DIR"

"$PYTHON" -m scripts.turkish_packed_sample seal-plan \
    --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
    --calibration="$CALIBRATION" --sample-ranks="$SAMPLE_RANKS" \
    --lane-plan="$LANE_PLAN"

# Each rank is a single Python worker. Numerical runtimes are forced to one
# thread so thirty-two workers cannot silently oversubscribe their cpusets.
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
srun --nodes=1 --ntasks=32 --ntasks-per-node=32 --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    --output="$LAUNCH_ROOT/lane-%t.out" \
    --error="$LAUNCH_ROOT/lane-%t.err" \
    "$PYTHON" -m scripts.turkish_packed_sample run-lane \
    --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
    --calibration="$CALIBRATION" --sample-ranks="$SAMPLE_RANKS" \
    --lane-plan="$LANE_PLAN" --model="$GLOTLID_MODEL" \
    --run-dir="$DATA_RUN_DIR" --receipt-dir="$LANE_RECEIPT_DIR"
srun_rc=$?
set -e
if [ "$srun_rc" -ne 0 ]; then
    echo "Packed sample failed collectively (srun rc=$srun_rc); inspect $LAUNCH_ROOT" >&2
    exit "$srun_rc"
fi

"$PYTHON" -m scripts.turkish_packed_sample finalize \
    --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
    --calibration="$CALIBRATION" --sample-ranks="$SAMPLE_RANKS" \
    --lane-plan="$LANE_PLAN" --run-dir="$DATA_RUN_DIR" \
    --receipt-dir="$LANE_RECEIPT_DIR" --job-id="$SLURM_JOB_ID" \
    --output="$LAUNCH_RECEIPT"
test -f "$LAUNCH_RECEIPT"
echo "Verified packed v2 object sample: $LAUNCH_RECEIPT"
