#!/bin/bash

# One-time shared CPU preparation for the official MaCoCu-Genre Turkish gzip.
# The command verifies the pinned 14.4 GB object, retains it, and atomically
# emits deterministic zstd JSONL shards plus a sealed provenance manifest.

#SBATCH --job-name=tr-macocu-prep
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v4.json}"
MACOCU_DIR="${MACOCU_DIR:-$NANOCHAT_BASE_DIR/source_data/macocu_genre_tr_v1}"
TARGET_UNCOMPRESSED_BYTES="${TARGET_UNCOMPRESSED_BYTES:-536870912}"
MACOCU_UPSTREAM_FILE="${MACOCU_UPSTREAM_FILE:-}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$UV_BIN"
test -x environments/turkish-data/.venv/bin/python

MACOCU_SOURCE_ARGS=()
if [[ -n "$MACOCU_UPSTREAM_FILE" ]]; then
    test -f "$MACOCU_UPSTREAM_FILE"
    MACOCU_SOURCE_ARGS+=(--upstream-file "$MACOCU_UPSTREAM_FILE")
fi

"$UV_BIN" run --project environments/turkish-data --locked python \
    scripts/turkish_data_backend.py prepare-macocu \
    --policy "$POLICY" \
    --output-dir "$MACOCU_DIR" \
    --target-uncompressed-bytes "$TARGET_UNCOMPRESSED_BYTES" \
    "${MACOCU_SOURCE_ARGS[@]}"

test -f "$MACOCU_DIR/manifest.json"
echo "Sealed MaCoCu preparation: $MACOCU_DIR/manifest.json"
