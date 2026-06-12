#!/bin/bash -l

# Measure the Zemberek MorphBPE tokenizer on the same 50k TRmorph-reference
# sample used by the checked-in tokenizer comparison table.

#SBATCH --job-name=nanochat-zemb50k-metrics
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

REPO_DIR="${REPO_DIR:-$HOME/nanochat-turk}"
ZEMBEREK_BASE_DIR="${ZEMBEREK_BASE_DIR:-$HOME/nanochat-turk-morphbpe-zemberek-32768}"
TRMORPH_BASE_DIR="${TRMORPH_BASE_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$ZEMBEREK_BASE_DIR/tokenizers/morphbpe_zemberek_32768}"
SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$TRMORPH_BASE_DIR/datasets/trmorph-segmented}"
OUTDIR="${OUTDIR:-$ZEMBEREK_BASE_DIR/report/tokenizer_metrics_50k_trmorph_reference}"
MAX_DOCS="${MAX_DOCS:-50000}"
MAX_WORD_METRICS="${MAX_WORD_METRICS:-500000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOKENIZER_THREADS_PER_WORKER="${TOKENIZER_THREADS_PER_WORKER:-1}"

cd "$REPO_DIR"
source .venv/bin/activate
mkdir -p "$OUTDIR"

python -m scripts.tokenizer_metrics \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --data-dir "$SEGMENTED_DATA_DIR" \
    --text-column segmented_text \
    --input-has-morph-boundaries \
    --max-docs "$MAX_DOCS" \
    --max-word-metrics "$MAX_WORD_METRICS" \
    --workers "$NUM_WORKERS" \
    --num-threads "$TOKENIZER_THREADS_PER_WORKER" \
    --progress-every 20 \
    --no-report \
    --output "$OUTDIR/morphbpe_zemberek_32768_metrics.json"

echo "Wrote $OUTDIR/morphbpe_zemberek_32768_metrics.json"
