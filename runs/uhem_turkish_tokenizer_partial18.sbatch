#!/bin/bash

# Explicitly non-production raw BPE training from the complete candidate
# Parquets preserved after cancellation of the 18-object resource sample.

#SBATCH --job-name=tr-tok-p18
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
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
source "$CODE_DIR/runs/uhem_d32_require_clean_code.sh"
d32_require_clean_committed_code "$CODE_DIR"

: "${INPUT_ROOT:?Set the cancelled sample run directory}"
: "${OUTPUT_DIR:?Set a new, provenance-labelled tokenizer output directory}"
CANCELLED_JOB_ID="${CANCELLED_JOB_ID:-512374}"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v4.json}"
TOKENIZER_NAME="${TOKENIZER_NAME:-tr_general_raw_bpe_32k_partial18_v1}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
test -d "$INPUT_ROOT/objects"
test ! -e "$OUTPUT_DIR"

cancelled_rows="$(sacct -n -X -P -j "$CANCELLED_JOB_ID" \
    -o JobIDRaw,State,Partition,AllocCPUS | sed '/^$/d')"
test "$(printf '%s\n' "$cancelled_rows" | wc -l)" -eq 1
job_raw="$(printf '%s\n' "$cancelled_rows" | cut -d'|' -f1)"
job_state="$(printf '%s\n' "$cancelled_rows" | cut -d'|' -f2)"
job_partition="$(printf '%s\n' "$cancelled_rows" | cut -d'|' -f3)"
job_cpus="$(printf '%s\n' "$cancelled_rows" | cut -d'|' -f4)"
test "$job_raw" = "$CANCELLED_JOB_ID"
case "$job_state" in CANCELLED*) ;; *) exit 2 ;; esac
test "$job_partition" = cpu2dq
test "$job_cpus" = 128

"$PYTHON" scripts/train_partial_turkish_bpe.py \
    --policy "$POLICY" \
    --input-root "$INPUT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --name "$TOKENIZER_NAME" \
    --cancelled-job-id "$CANCELLED_JOB_ID"

test -f "$OUTPUT_DIR/package_manifest.json"
test -f "$OUTPUT_DIR/tokenizer.pkl"
test -f "$OUTPUT_DIR/tokenizer.tiktoken"
test -f "$OUTPUT_DIR/token_bytes.pt"
test -f "$OUTPUT_DIR/tokenizer_config.json"
test -f "$OUTPUT_DIR/partial_training_receipt.json"
