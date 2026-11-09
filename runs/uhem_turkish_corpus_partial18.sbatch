#!/bin/bash

# Materialize the 18 complete candidate Parquets preserved from the cancelled
# resource sample. This is an explicitly non-production, no-quality-claim lane.

#SBATCH --job-name=tr-p18-corpus
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${CODE_REVISION:?Export the exact committed Git revision}"
test "${SLURM_JOB_ACCOUNT:-}" = nakane
test "${SLURM_JOB_PARTITION:-}" = cpu2dq
test "${SLURM_NNODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 128
test "${SLURM_MEM_PER_NODE:-}" = 196608
source "$CODE_DIR/runs/uhem_d32_require_clean_code.sh"
d32_require_clean_committed_code "$CODE_DIR"

: "${INPUT_ROOT:?Set the cancelled resource_sample_v4 run directory}"
: "${TOKENIZER_DIR:?Set the verified partial18 tokenizer package directory}"
RUN_ROOT="${RUN_ROOT:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/pretrain_data/tr_general_candidates_partial18_v1}"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v4.json}"
SOURCE_PLAN="${SOURCE_PLAN:-$RUN_ROOT/control/data_v4/source_plan.json}"
SAMPLE_RANKS="${SAMPLE_RANKS:-$RUN_ROOT/control/data_v4/resource_sample_ranks.json}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-512374}"
TOKENIZER_JOB_ID="${TOKENIZER_JOB_ID:-512413}"
EXPECTED_PACKAGE_SHA256="${EXPECTED_PACKAGE_SHA256:-909bfa20516c79b7349d3e35aacd655ef584aac431c055966366cf6e1545d871}"
EXPECTED_RECEIPT_SHA256="${EXPECTED_RECEIPT_SHA256:-5014766c50fee069fde94806a9eb82de9c42e78d8d55632c4e37f2ebed94c445}"
EXPECTED_SOURCE_PLAN_SHA256="${EXPECTED_SOURCE_PLAN_SHA256:-53ca06b8686049c4a52926bc4b1418b0380785c64f3f0ffe3925b9e2a4a09570}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
test -d "$INPUT_ROOT/objects"
test -f "$TOKENIZER_DIR/package_manifest.json"
test -f "$SOURCE_PLAN"
test -f "$SAMPLE_RANKS"
test ! -e "$OUTPUT_DIR"
OUTPUT_PARENT="$(dirname "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_PARENT"
available_kib="$(df -Pk "$OUTPUT_PARENT" | awk 'NR == 2 {print $4}')"
case "$available_kib" in ''|*[!0-9]*) exit 2 ;; esac
available_bytes="$((available_kib * 1024))"
test "$available_bytes" -ge 100000000000

source_rows="$(sacct -n -X -P -j "$SOURCE_JOB_ID" \
    -o JobIDRaw,State,Partition,AllocCPUS | sed '/^$/d')"
test "$(printf '%s\n' "$source_rows" | wc -l)" -eq 1
source_raw="$(printf '%s\n' "$source_rows" | cut -d'|' -f1)"
source_state="$(printf '%s\n' "$source_rows" | cut -d'|' -f2)"
source_partition="$(printf '%s\n' "$source_rows" | cut -d'|' -f3)"
source_cpus="$(printf '%s\n' "$source_rows" | cut -d'|' -f4)"
test "$source_raw" = "$SOURCE_JOB_ID"
case "$source_state" in CANCELLED*) ;; *) exit 2 ;; esac
test "$source_partition" = cpu2dq
test "$source_cpus" = 128

tokenizer_rows="$(sacct -n -X -P -j "$TOKENIZER_JOB_ID" \
    -o JobIDRaw,State,ExitCode,Partition,AllocCPUS | sed '/^$/d')"
test "$(printf '%s\n' "$tokenizer_rows" | wc -l)" -eq 1
tokenizer_raw="$(printf '%s\n' "$tokenizer_rows" | cut -d'|' -f1)"
tokenizer_state="$(printf '%s\n' "$tokenizer_rows" | cut -d'|' -f2)"
tokenizer_exit="$(printf '%s\n' "$tokenizer_rows" | cut -d'|' -f3)"
tokenizer_partition="$(printf '%s\n' "$tokenizer_rows" | cut -d'|' -f4)"
tokenizer_cpus="$(printf '%s\n' "$tokenizer_rows" | cut -d'|' -f5)"
test "$tokenizer_raw" = "$TOKENIZER_JOB_ID"
test "$tokenizer_state" = COMPLETED
test "$tokenizer_exit" = 0:0
test "$tokenizer_partition" = cpu2dq
test "$tokenizer_cpus" = 128

"$PYTHON" scripts/materialize_partial18_turkish_corpus.py \
    --policy "$POLICY" \
    --input-root "$INPUT_ROOT" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --source-plan "$SOURCE_PLAN" \
    --sample-ranks "$SAMPLE_RANKS" \
    --expected-source-plan-sha256 "$EXPECTED_SOURCE_PLAN_SHA256" \
    --output-dir "$OUTPUT_DIR" \
    --expected-tokenizer-package-sha256 "$EXPECTED_PACKAGE_SHA256" \
    --expected-tokenizer-receipt-sha256 "$EXPECTED_RECEIPT_SHA256" \
    --source-cancelled-job-id "$SOURCE_JOB_ID" \
    --source-job-state "$source_state" \
    --source-job-partition "$source_partition" \
    --source-job-allocated-cpus "$source_cpus" \
    --tokenizer-job-id "$TOKENIZER_JOB_ID" \
    --tokenizer-threads 32

test -f "$OUTPUT_DIR/fineweb2_manifest.json"
test -f "$OUTPUT_DIR/partial18_corpus_manifest.json"
test -f "$OUTPUT_DIR/partial18_capacity_receipt.json"
test -f "$OUTPUT_DIR/partial18_materialization_receipt.json"
test -f "$OUTPUT_DIR/validation.parquet"
test -f "$OUTPUT_DIR/test/test.parquet"
