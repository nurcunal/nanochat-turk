#!/bin/bash -l

# Parallel TRmorph segmentation for the MorphBPE 32k tokenizer.
#
# This array job materializes one segmented parquet shard per task. Completed
# shards are committed atomically as:
#
#   $SEGMENTED_DATA_DIR/<raw-stem>.segmented.parquet
#   $SEGMENTED_DATA_DIR/<raw-stem>.segmented.parquet.done.json
#
# After the whole array succeeds, run the finalize/tokenizer job:
#
#   sbatch --dependency=afterok:<array_jobid> \
#     runs/uhem_nakane_finalize_morphbpe_trmorph_32k.sbatch

#SBATCH --job-name=nanochat-seg-trmorph32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --array=0-29%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    # Batch shells on Altay do not always initialize Environment Modules.
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$NANOCHAT_BASE_DIR/segmented/trmorph}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

export TRMORPH_SEGMENT_FST="${TRMORPH_SEGMENT_FST:-$HOME/TRmorph/segment.fst}"
export TRMORPH_FLOOKUP_CMD="${TRMORPH_FLOOKUP_CMD:-flookup}"
export TRMORPH_FLOOKUP_FLAGS="${TRMORPH_FLOOKUP_FLAGS:--x}"
export TRMORPH_PICK="${TRMORPH_PICK:-fewest_segments}"
export TRMORPH_MAX_OUTPUT_LINES_PER_WORD="${TRMORPH_MAX_OUTPUT_LINES_PER_WORD:-512}"
export TRMORPH_MAX_ANALYSES_PER_WORD="${TRMORPH_MAX_ANALYSES_PER_WORD:-64}"
export SEGMENT_BATCH_SIZE="${SEGMENT_BATCH_SIZE:-2048}"
export WORD_CACHE_SIZE="${WORD_CACHE_SIZE:-500000}"

mkdir -p "$SEGMENTED_DATA_DIR"

if [ ! -f "$RAW_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing raw FineWeb-2 Turkish manifest at $RAW_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi
if [ ! -f "$TRMORPH_SEGMENT_FST" ]; then
    echo "Missing TRmorph FST at $TRMORPH_SEGMENT_FST" >&2
    exit 2
fi
if ! command -v "$TRMORPH_FLOOKUP_CMD" >/dev/null 2>&1; then
    echo "Missing flookup command: $TRMORPH_FLOOKUP_CMD" >&2
    exit 2
fi
if [ ! -d ".venv" ]; then
    echo "Missing .venv in $PWD. Run the setup/prep job once before array segmentation." >&2
    exit 2
fi
source .venv/bin/activate

python -m scripts.morph_segment_corpus \
    --backend trmorph \
    --data-dir "$RAW_DATA_DIR" \
    --output-dir "$SEGMENTED_DATA_DIR" \
    --compact \
    --resume \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --segment-batch-size "$SEGMENT_BATCH_SIZE" \
    --word-cache-size "$WORD_CACHE_SIZE"

echo "Completed segmented shard index $SLURM_ARRAY_TASK_ID"
