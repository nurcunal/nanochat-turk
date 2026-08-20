#!/bin/bash

# Bounded, tokenizer-aware crop-retention calibration. This never materializes
# the full training corpus and never auto-approves its recommendation.

#SBATCH --job-name=tr-pack-plan
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
: "${POOL_DIR:?Set POOL_DIR to the manually QA-approved production pool}"
: "${TOKENIZER_DIR:?Set TOKENIZER_DIR to the verified raw-BPE package}"
: "${PACKING_CONTROL_DIR:?Set PACKING_CONTROL_DIR to a new control directory}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v1.json}"
PYTHON_BIN="${PYTHON_BIN:-$CODE_DIR/.venv/bin/python}"
mkdir -p "$PACKING_CONTROL_DIR"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" scripts/build_turkish_pretrain_corpus.py \
    --policy "$POLICY" packing-preflight \
    --pool-dir "$POOL_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --max-documents-per-rank-and-mixture "${PACKING_SAMPLE_DOCUMENTS:-8192}" \
    --projection-safety-factor "${PACKING_SAFETY_FACTOR:-1.08}" \
    --output "$PACKING_CONTROL_DIR/packing_preflight_report.json"

echo "Review the sealed report, then create packing_preflight_approval.json with the approve-packing-preflight command."
