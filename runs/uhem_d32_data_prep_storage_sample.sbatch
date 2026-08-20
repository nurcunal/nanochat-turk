#!/bin/bash

# Submit with afterok dependencies on the MaCoCu preparation, bootstrap,
# packed object sample, packed bucket sample, cluster, and writer probe.
# Every job ID below must name one exact completed allocation. This job seals evidence and runs the live gate; it
# does not auto-approve the backend resource report or launch production work.

#SBATCH --job-name=tr-d32-prep-gate
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${MACOCU_JOB_ID:?Set the completed MaCoCu preparation allocation ID}"
: "${BOOTSTRAP_JOB_ID:?Set the completed bootstrap allocation ID}"
: "${SAMPLE_OBJECT_JOB_ID:?Set the completed packed object-sample allocation ID}"
: "${SAMPLE_BUCKET_JOB_ID:?Set the completed packed bucket-sample allocation ID}"
: "${SAMPLE_CLUSTER_JOB_ID:?Set the completed cluster allocation ID}"
: "${SAMPLE_QUALITY_AUDIT_JOB_ID:?Set the completed bounded quality-audit allocation ID}"
: "${WRITER_PROBE_JOB_ID:?Set the completed writer-probe allocation ID}"
: "${WRITER_PROBE:?Set the sealed post-cluster writer-probe receipt}"
: "${RESOURCE_APPROVAL:?Set the accepted backend resource approval}"
: "${MIXTURE_QUALITY_APPROVAL:?Set the accepted manual mixture-quality approval}"
: "${PRODUCTION_DATA_NODES:?Choose the explicit cpu2dq node count for packed production data work}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v2}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v2.json}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v2.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v2}"
SAMPLE_RUN_DIR="${SAMPLE_RUN_DIR:-$NANOCHAT_BASE_DIR/data_backend/resource_sample_v2}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
SAMPLE_RANKS="${SAMPLE_RANKS:-$CONTROL_DIR/resource_sample_ranks.json}"
SAMPLE_LANE_PLAN="${SAMPLE_LANE_PLAN:-$CONTROL_DIR/resource_sample_lane_plan.json}"
SAMPLE_OBJECT_LAUNCH_RECEIPT="${SAMPLE_OBJECT_LAUNCH_RECEIPT:-$SAMPLE_RUN_DIR/packed_sample_launches/job$SAMPLE_OBJECT_JOB_ID/launch_receipt.json}"
SAMPLE_BUCKET_LAUNCH_RECEIPT="${SAMPLE_BUCKET_LAUNCH_RECEIPT:-$SAMPLE_RUN_DIR/packed_bucket_launches/job$SAMPLE_BUCKET_JOB_ID/launch_receipt.json}"
MACOCU_MANIFEST="${MACOCU_MANIFEST:-$NANOCHAT_BASE_DIR/source_data/macocu_genre_tr_v1/manifest.json}"
BACKEND_RESOURCE_REPORT="${BACKEND_RESOURCE_REPORT:-$CONTROL_DIR/backend_resource_report.json}"
PRODUCTION_PACK_PLAN="${PRODUCTION_PACK_PLAN:-$CONTROL_DIR/production_source_pack_plan.json}"
STORAGE_SAMPLE="${STORAGE_SAMPLE:-$CONTROL_DIR/d32_data_prep_storage_sample.json}"
STORAGE_GATE="${STORAGE_GATE:-$NANOCHAT_BASE_DIR/control/d32/data_prep_storage_gate.json}"
WORK_DIR="${WORK_DIR:-$NANOCHAT_BASE_DIR}"
PYTHON="${PYTHON:-$CODE_DIR/environments/turkish-data/.venv/bin/python}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
for input in "$POLICY" "$RECIPE" "$SOURCE_PLAN" "$CALIBRATION" \
    "$SAMPLE_RANKS" "$SAMPLE_LANE_PLAN" "$SAMPLE_OBJECT_LAUNCH_RECEIPT" \
    "$SAMPLE_BUCKET_LAUNCH_RECEIPT" \
    "$MACOCU_MANIFEST" "$BACKEND_RESOURCE_REPORT" "$WRITER_PROBE" \
    "$RESOURCE_APPROVAL" "$MIXTURE_QUALITY_APPROVAL"; do
    test -f "$input"
done
test -d "$SAMPLE_RUN_DIR"
test -d "$WORK_DIR"
test ! -e "$PRODUCTION_PACK_PLAN"
test ! -e "$STORAGE_SAMPLE"
test ! -e "$STORAGE_GATE"
mkdir -p "$(dirname "$STORAGE_GATE")"

"$PYTHON" scripts/d32_family_workflow.py seal-data-prep-pack-plan \
    --recipe "$RECIPE" --policy "$POLICY" --source-plan "$SOURCE_PLAN" \
    --calibration "$CALIBRATION" --nodes "$PRODUCTION_DATA_NODES" \
    --output "$PRODUCTION_PACK_PLAN"

"$PYTHON" scripts/d32_family_workflow.py seal-data-prep-storage-sample \
    --recipe "$RECIPE" --policy "$POLICY" --source-plan "$SOURCE_PLAN" \
    --calibration "$CALIBRATION" --sample-ranks "$SAMPLE_RANKS" \
    --sample-lane-plan "$SAMPLE_LANE_PLAN" --sample-run-dir "$SAMPLE_RUN_DIR" \
    --sample-object-launch-receipt "$SAMPLE_OBJECT_LAUNCH_RECEIPT" \
    --sample-bucket-launch-receipt "$SAMPLE_BUCKET_LAUNCH_RECEIPT" \
    --backend-resource-report "$BACKEND_RESOURCE_REPORT" \
    --resource-approval "$RESOURCE_APPROVAL" \
    --mixture-quality-approval "$MIXTURE_QUALITY_APPROVAL" \
    --macocu-manifest "$MACOCU_MANIFEST" \
    --production-pack-plan "$PRODUCTION_PACK_PLAN" \
    --writer-probe "$WRITER_PROBE" --macocu-job-id "$MACOCU_JOB_ID" \
    --bootstrap-job-id "$BOOTSTRAP_JOB_ID" \
    --sample-object-job-id "$SAMPLE_OBJECT_JOB_ID" \
    --sample-bucket-job-id "$SAMPLE_BUCKET_JOB_ID" \
    --sample-cluster-job-id "$SAMPLE_CLUSTER_JOB_ID" \
    --sample-quality-audit-job-id "$SAMPLE_QUALITY_AUDIT_JOB_ID" \
    --writer-probe-job-id "$WRITER_PROBE_JOB_ID" --output "$STORAGE_SAMPLE"

"$PYTHON" scripts/d32_family_workflow.py data-prep-storage-gate \
    --recipe "$RECIPE" --policy "$POLICY" \
    --sample-measurement "$STORAGE_SAMPLE" \
    --work-dir "$WORK_DIR" --output "$STORAGE_GATE"

echo "Accepted backend resource report/approval verified: $BACKEND_RESOURCE_REPORT"
echo "Sealed unique-allocation storage sample: $STORAGE_SAMPLE"
echo "Passed live storage/quota gate: $STORAGE_GATE"
