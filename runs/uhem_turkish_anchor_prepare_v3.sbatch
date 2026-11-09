#!/bin/bash

# Prepare the pinned MOT and ParlaMint Turkish native-text anchors. Run once
# with MODE=discovery, review/seal exact count acceptances, then rerun with
# MODE=production and both acceptance paths supplied.

#SBATCH --job-name=tr-v3-anchor-prep
#SBATCH --account=nakane
#SBATCH --partition=defq
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
MODE="${MODE:-discovery}"
PYTHON="${PYTHON:-$CODE_DIR/environments/turkish-data/.venv/bin/python}"
RAW_DIR="${RAW_DIR:-$NANOCHAT_BASE_DIR/source_data/anchors/raw}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v3/anchors}"
SOURCE_DIR="${SOURCE_DIR:-$NANOCHAT_BASE_DIR/source_data}"

MOT_AIS="$RAW_DIR/tur_amerikaninsesi.tgz"
MOT_VOA="$RAW_DIR/tur_voaturkce.tgz"
PARLAMINT="$RAW_DIR/ParlaMint-TR.tgz"
MOT_ACQUISITION="$CONTROL_DIR/mot_acquisition.json"
PARLAMINT_ACQUISITION="$CONTROL_DIR/parlamint_acquisition.json"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$PYTHON"
test -f "$MOT_AIS"
test -f "$MOT_VOA"
test -f "$PARLAMINT"
mkdir -p "$CONTROL_DIR" "$SOURCE_DIR"

acquired_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ! -f "$MOT_ACQUISITION" ]; then
    "$PYTHON" scripts/prepare_turkish_anchors.py seal-mot-acquisition \
        --amerikaninsesi-tgz "$MOT_AIS" \
        --voaturkce-tgz "$MOT_VOA" \
        --output "$MOT_ACQUISITION" --reviewer codex \
        --acquired-at-utc "$acquired_at" --decision accepted \
        --notes "Pinned v1.11 assets staged on UHeM; native article JSON only."
fi
if [ ! -f "$PARLAMINT_ACQUISITION" ]; then
    "$PYTHON" scripts/prepare_turkish_anchors.py seal-parlamint-acquisition \
        --archive-tgz "$PARLAMINT" \
        --output "$PARLAMINT_ACQUISITION" --reviewer codex \
        --acquired-at-utc "$acquired_at" --decision accepted \
        --notes "Pinned v5.0 asset staged on UHeM; native structured text only."
fi

if [ "$MODE" = discovery ]; then
    MOT_OUTPUT="${MOT_OUTPUT:-$SOURCE_DIR/mot_tr_v1_11_discovery}"
    PARLAMINT_OUTPUT="${PARLAMINT_OUTPUT:-$SOURCE_DIR/parlamint_tr_v5_0_discovery}"
    if [ ! -f "$MOT_OUTPUT/manifest.json" ]; then
        "$PYTHON" scripts/prepare_turkish_anchors.py mot-v1.11 \
            --amerikaninsesi-tgz "$MOT_AIS" \
            --voaturkce-tgz "$MOT_VOA" \
            --acquisition-receipt "$MOT_ACQUISITION" \
            --output-dir "$MOT_OUTPUT" --discovery
    fi
    if [ ! -f "$PARLAMINT_OUTPUT/manifest.json" ]; then
        "$PYTHON" scripts/prepare_turkish_anchors.py parlamint-tr-v5.0 \
            --archive-tgz "$PARLAMINT" \
            --acquisition-receipt "$PARLAMINT_ACQUISITION" \
            --output-dir "$PARLAMINT_OUTPUT" --discovery
    fi
elif [ "$MODE" = production ]; then
    : "${MOT_COUNT_ACCEPTANCE:?Set MOT_COUNT_ACCEPTANCE}"
    : "${PARLAMINT_COUNT_ACCEPTANCE:?Set PARLAMINT_COUNT_ACCEPTANCE}"
    MOT_OUTPUT="${MOT_OUTPUT:-$SOURCE_DIR/mot_tr_v1_11}"
    PARLAMINT_OUTPUT="${PARLAMINT_OUTPUT:-$SOURCE_DIR/parlamint_tr_v5_0}"
    if [ ! -f "$MOT_OUTPUT/manifest.json" ]; then
        "$PYTHON" scripts/prepare_turkish_anchors.py mot-v1.11 \
            --amerikaninsesi-tgz "$MOT_AIS" \
            --voaturkce-tgz "$MOT_VOA" \
            --acquisition-receipt "$MOT_ACQUISITION" \
            --output-dir "$MOT_OUTPUT" \
            --count-acceptance "$MOT_COUNT_ACCEPTANCE"
    fi
    if [ ! -f "$PARLAMINT_OUTPUT/manifest.json" ]; then
        "$PYTHON" scripts/prepare_turkish_anchors.py parlamint-tr-v5.0 \
            --archive-tgz "$PARLAMINT" \
            --acquisition-receipt "$PARLAMINT_ACQUISITION" \
            --output-dir "$PARLAMINT_OUTPUT" \
            --count-acceptance "$PARLAMINT_COUNT_ACCEPTANCE"
    fi
else
    echo "MODE must be discovery or production" >&2
    exit 2
fi

test -f "$MOT_OUTPUT/manifest.json"
test -f "$PARLAMINT_OUTPUT/manifest.json"
"$PYTHON" scripts/prepare_turkish_anchors.py validate \
    --output-dir "$MOT_OUTPUT"
"$PYTHON" scripts/prepare_turkish_anchors.py validate \
    --output-dir "$PARLAMINT_OUTPUT"

echo "MOT preparation: $MOT_OUTPUT/manifest.json"
echo "ParlaMint preparation: $PARLAMINT_OUTPUT/manifest.json"
