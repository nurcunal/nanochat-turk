#!/bin/bash -l

# Generic CPU tokenizer finalizer for the Turkish tokenizer ablation.
#
# This trains one tokenizer and writes a GitHub-ready bundle under:
#   $REPO_DIR/artifacts/tokenizers/$NANOCHAT_TOKENIZER_NAME
#
# Examples:
#   sbatch --export=ALL,VOCAB_SIZE=65536,TOKENIZER_FAMILY=bpe \
#     runs/uhem_nakane_finalize_tokenizer_ablation.sbatch
#
#   sbatch --export=ALL,VOCAB_SIZE=131072,TOKENIZER_FAMILY=morphbpe,SEGMENTER=trmorph \
#     runs/uhem_nakane_finalize_tokenizer_ablation.sbatch

#SBATCH --job-name=nanochat-finalize-tokenizer
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --mem=240G
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

DEFAULT_REPO_DIR="$HOME/nanochat-turk"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$REPO_DIR/.venv" ] && [ -d "$DEFAULT_REPO_DIR/.venv" ]; then
    REPO_DIR="$DEFAULT_REPO_DIR"
fi
cd "$REPO_DIR"

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "REPO_DIR=$REPO_DIR"
echo "PYTHON=$(command -v python3)"
python3 --version

export PATH="$HOME/.local/bin:$PATH"
export VOCAB_SIZE="${VOCAB_SIZE:?Set VOCAB_SIZE, e.g. 65536 or 131072}"
export TOKENIZER_FAMILY="${TOKENIZER_FAMILY:-bpe}"
export SEGMENTER="${SEGMENTER:-none}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export TOKENIZER_CHARS="${TOKENIZER_CHARS:-2000000000}"
export TOKENIZER_METRIC_DOCS="${TOKENIZER_METRIC_DOCS:-10000}"
export TOKENIZER_PUBLISH_ROOT="${TOKENIZER_PUBLISH_ROOT:-$REPO_DIR/artifacts/tokenizers}"

case "$TOKENIZER_FAMILY" in
    bpe)
        export SEGMENTER="none"
        export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_${VOCAB_SIZE}}"
        export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-bpe-${VOCAB_SIZE}}"
        TRAIN_IMPLEMENTATION="bpe"
        TRAIN_DATA_DIR="$RAW_DATA_DIR"
        TRAIN_TEXT_COLUMN="text"
        ;;
    morphbpe)
        case "$SEGMENTER" in
            trmorph)
                DEFAULT_SEGMENTED_DATA_DIR="$HOME/nanochat-turk-morphbpe-trmorph-32768/segmented/trmorph"
                ;;
            zemberek)
                DEFAULT_SEGMENTED_DATA_DIR="$HOME/nanochat-turk-morphbpe-zemberek-32768/segmented/zemberek-segmented"
                ;;
            tdelight)
                DEFAULT_SEGMENTED_DATA_DIR="$HOME/nanochat-turk-morphbpe-tdelight-32768/segmented/tdelight-segmented"
                ;;
            *)
                echo "For TOKENIZER_FAMILY=morphbpe, SEGMENTER must be trmorph, zemberek, or tdelight." >&2
                exit 2
                ;;
        esac
        export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$DEFAULT_SEGMENTED_DATA_DIR}"
        export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-morphbpe_${SEGMENTER}_${VOCAB_SIZE}}"
        export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-${SEGMENTER}-${VOCAB_SIZE}}"
        TRAIN_IMPLEMENTATION="morphbpe"
        TRAIN_DATA_DIR="$SEGMENTED_DATA_DIR"
        TRAIN_TEXT_COLUMN="segmented_text"
        ;;
    *)
        echo "TOKENIZER_FAMILY must be bpe or morphbpe, got: $TOKENIZER_FAMILY" >&2
        exit 2
        ;;
esac

export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANOCHAT_BASE_DIR/uv-cache}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

echo "TOKENIZER_FAMILY=$TOKENIZER_FAMILY"
echo "SEGMENTER=$SEGMENTER"
echo "VOCAB_SIZE=$VOCAB_SIZE"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "NANOCHAT_TOKENIZER_NAME=$NANOCHAT_TOKENIZER_NAME"
echo "RAW_DATA_DIR=$RAW_DATA_DIR"
echo "TRAIN_DATA_DIR=$TRAIN_DATA_DIR"
echo "TRAIN_TEXT_COLUMN=$TRAIN_TEXT_COLUMN"
echo "TOKENIZER_PUBLISH_ROOT=$TOKENIZER_PUBLISH_ROOT"

if [ ! -d ".venv" ]; then
    echo "Missing .venv in $REPO_DIR. Run the main prep job before tokenizer finalization." >&2
    exit 2
fi
source .venv/bin/activate

if [ ! -f "$RAW_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing raw dataset manifest at $RAW_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi
if [ "$TOKENIZER_FAMILY" = "morphbpe" ]; then
    if [ ! -f "$SEGMENTED_DATA_DIR/fineweb2_manifest.json" ] && [ ! -f "$SEGMENTED_DATA_DIR/manifest.json" ]; then
        echo "Missing segmented corpus manifest under $SEGMENTED_DATA_DIR" >&2
        exit 2
    fi
fi

python -m scripts.tok_train \
    --implementation "$TRAIN_IMPLEMENTATION" \
    --data-dir "$TRAIN_DATA_DIR" \
    --text-column "$TRAIN_TEXT_COLUMN" \
    --vocab-size "$VOCAB_SIZE" \
    --max-chars "$TOKENIZER_CHARS" \
    --tokenizer-name "$NANOCHAT_TOKENIZER_NAME"

python -m scripts.tokenizer_metrics \
    --data-dir "$RAW_DATA_DIR" \
    --text-column text \
    --max-docs "$TOKENIZER_METRIC_DOCS" \
    --no-report \
    --output "$NANOCHAT_BASE_DIR/${NANOCHAT_TOKENIZER_NAME}_raw_metrics.json"

PUBLISH_ARGS=(
    --base-dir "$NANOCHAT_BASE_DIR"
    --tokenizer-name "$NANOCHAT_TOKENIZER_NAME"
    --local-output-dir "$TOKENIZER_PUBLISH_ROOT/$NANOCHAT_TOKENIZER_NAME"
    --metrics-path "$NANOCHAT_BASE_DIR/${NANOCHAT_TOKENIZER_NAME}_raw_metrics.json"
    --require-metrics
    --no-hf-upload
)
if [ "$TOKENIZER_FAMILY" = "morphbpe" ]; then
    PUBLISH_ARGS+=(
        --segmentation-manifest "$SEGMENTED_DATA_DIR/manifest.json"
        --segmented-dataset-manifest "$SEGMENTED_DATA_DIR/fineweb2_manifest.json"
    )
fi
python -m scripts.publish_tokenizer_artifact "${PUBLISH_ARGS[@]}"

echo "Prepared tokenizer under:"
echo "  $NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME"
echo "Published local bundle under:"
echo "  $TOKENIZER_PUBLISH_ROOT/$NANOCHAT_TOKENIZER_NAME"
du -sh "$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME" "$TOKENIZER_PUBLISH_ROOT/$NANOCHAT_TOKENIZER_NAME" || true
