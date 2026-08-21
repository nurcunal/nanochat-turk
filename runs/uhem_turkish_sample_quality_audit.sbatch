#!/bin/bash

# Read-only QA pass over the completed bounded active-policy Turkish sample.
# Success seals evidence for manual review; it never approves the mixture or
# authorizes production materialization.

#SBATCH --job-name=tr-data-qa
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

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
SAMPLE_RUN_DIR="${SAMPLE_RUN_DIR:-$D32_SAMPLE_RUN_DIR}"
d32_bind_canonical_path SAMPLE_RUN_DIR "$D32_SAMPLE_RUN_DIR"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$CONTROL_DIR/sample_quality_audit}"
d32_bind_canonical_path SOURCE_PLAN "$D32_SOURCE_PLAN"
d32_bind_canonical_path CALIBRATION "$D32_CALIBRATION"
d32_bind_canonical_path AUDIT_OUTPUT_DIR "$D32_AUDIT_OUTPUT_DIR"
PYTHON="${PYTHON:-$CODE_DIR/environments/turkish-data/.venv/bin/python}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
test -f "$POLICY"
test -f "$SOURCE_PLAN"
test -f "$CALIBRATION"
test -f "$SAMPLE_RUN_DIR/cluster_receipt.json"

"$PYTHON" -m scripts.audit_turkish_backend_sample \
    --policy="$POLICY" \
    --source-plan="$SOURCE_PLAN" \
    --calibration="$CALIBRATION" \
    --run-dir="$SAMPLE_RUN_DIR" \
    --output-dir="$AUDIT_OUTPUT_DIR"

test -f "$AUDIT_OUTPUT_DIR/sample_quality_audit_report.json"
echo "Sealed bounded sample QA evidence: $AUDIT_OUTPUT_DIR"
echo "Manual accepted/rejected-example review remains mandatory; no mixture was approved."
