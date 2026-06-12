#!/bin/bash

# Submit the 64k and 128k tokenizer-ablation grid:
#   - raw BPE
#   - MorphBPE + TRmorph
#   - MorphBPE + Zemberek
#   - MorphBPE + TurkishDelightNLP
#
# Each GPU training job depends on its tokenizer finalizer. If TurkishDelight
# 32k finalization is still pending, set TDELIGHT_READY_DEPENDENCY to that job
# id so the 64k/128k TurkishDelight tokenizer jobs wait for the completed
# segmented corpus.
#
# Example:
#   TDELIGHT_READY_DEPENDENCY=494159 \
#     bash runs/uhem_submit_64k_128k_tokenizer_ablation.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"

TOKENIZER_SCRIPT="${TOKENIZER_SCRIPT:-runs/uhem_nakane_finalize_tokenizer_ablation.sbatch}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-runs/uhem_nakane_a100x4_tokenizer_ablation.sbatch}"
TDELIGHT_READY_DEPENDENCY="${TDELIGHT_READY_DEPENDENCY:-}"

if [ ! -f "$TOKENIZER_SCRIPT" ]; then
    echo "Missing tokenizer script: $TOKENIZER_SCRIPT" >&2
    exit 2
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Missing train script: $TRAIN_SCRIPT" >&2
    exit 2
fi

depth_for_vocab() {
    case "$1" in
        65536) echo 16 ;;
        131072) echo 12 ;;
        *)
            echo "No predefined depth for vocab $1" >&2
            return 2
            ;;
    esac
}

label_for_vocab() {
    case "$1" in
        65536) echo "64k" ;;
        131072) echo "128k" ;;
        *) echo "$1" ;;
    esac
}

submit_tokenizer() {
    local vocab="$1"
    local family="$2"
    local segmenter="$3"
    local label
    label="$(label_for_vocab "$vocab")"

    local job_name
    local dep_args=()
    if [ "$family" = "bpe" ]; then
        job_name="tok-bpe-${label}"
    else
        job_name="tok-mbpe-${segmenter}-${label}"
        if [ "$segmenter" = "tdelight" ] && [ -n "$TDELIGHT_READY_DEPENDENCY" ]; then
            dep_args=(--dependency="afterok:$TDELIGHT_READY_DEPENDENCY")
        fi
    fi

    sbatch --parsable \
        --job-name="$job_name" \
        "${dep_args[@]}" \
        --export=ALL,VOCAB_SIZE="$vocab",TOKENIZER_FAMILY="$family",SEGMENTER="$segmenter" \
        "$TOKENIZER_SCRIPT"
}

submit_train() {
    local vocab="$1"
    local family="$2"
    local segmenter="$3"
    local tokenizer_job="$4"
    local depth label job_name wandb_run
    depth="$(depth_for_vocab "$vocab")"
    label="$(label_for_vocab "$vocab")"

    if [ "$family" = "bpe" ]; then
        job_name="tr-d${depth}-bpe-${label}"
        wandb_run="tr-d${depth}-bpe-${label}"
    else
        job_name="tr-d${depth}-mbpe-${segmenter}-${label}"
        wandb_run="tr-d${depth}-morphbpe-${segmenter}-${label}"
    fi

    sbatch --parsable \
        --job-name="$job_name" \
        --dependency="afterok:$tokenizer_job" \
        --export=ALL,VOCAB_SIZE="$vocab",TOKENIZER_FAMILY="$family",SEGMENTER="$segmenter",DEPTH="$depth",WANDB_RUN="$wandb_run" \
        "$TRAIN_SCRIPT"
}

printf "vocab\tfamily\tsegmenter\ttokenizer_job\ttrain_job\n"
for vocab in 65536 131072; do
    for spec in "bpe:none" "morphbpe:trmorph" "morphbpe:zemberek" "morphbpe:tdelight"; do
        family="${spec%%:*}"
        segmenter="${spec#*:}"
        tokenizer_job="$(submit_tokenizer "$vocab" "$family" "$segmenter")"
        train_job="$(submit_train "$vocab" "$family" "$segmenter" "$tokenizer_job")"
        printf "%s\t%s\t%s\t%s\t%s\n" "$vocab" "$family" "$segmenter" "$tokenizer_job" "$train_job"
    done
done
