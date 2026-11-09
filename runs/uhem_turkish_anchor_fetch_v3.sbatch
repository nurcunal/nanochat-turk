#!/bin/bash

# Fetch only the three pinned, native-text Turkish anchor archives. The
# downstream preparer verifies their immutable size/checksum contracts and
# emits separate acquisition receipts before reading any archive member.

#SBATCH --job-name=tr-v3-anchor-fetch
#SBATCH --account=nakane
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$CODE_DIR/runs/uhem_d32_require_clean_code.sh"
d32_require_clean_committed_code "$CODE_DIR"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RAW_DIR="${RAW_DIR:-$NANOCHAT_BASE_DIR/source_data/anchors/raw}"
mkdir -p "$RAW_DIR"

download_exact() {
    local url="$1"
    local destination="$2"
    local expected_size="$3"
    local partial="${destination}.part"
    local curl_args=(--fail --location --retry 5 --continue-at -)
    if [ -n "${CA_BUNDLE:-}" ]; then
        curl_args+=(--cacert "$CA_BUNDLE")
    fi
    if [ ! -f "$destination" ]; then
        curl "${curl_args[@]}" --output "$partial" "$url"
        mv "$partial" "$destination"
    fi
    test ! -L "$destination"
    test "$(stat -c %s "$destination")" -eq "$expected_size"
}

MOT_AIS="$RAW_DIR/tur_amerikaninsesi.tgz"
MOT_VOA="$RAW_DIR/tur_voaturkce.tgz"
PARLAMINT="$RAW_DIR/ParlaMint-TR.tgz"

download_exact \
    "https://github.com/bltlab/mot/releases/download/v1.11/tur_amerikaninsesi.tgz" \
    "$MOT_AIS" 219280046
download_exact \
    "https://github.com/bltlab/mot/releases/download/v1.11/tur_voaturkce.tgz" \
    "$MOT_VOA" 264239626
download_exact \
    "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2004/ParlaMint-TR.tgz?sequence=28&isAllowed=y" \
    "$PARLAMINT" 297184431

parlamint_md5="$(md5sum "$PARLAMINT")"
test "${parlamint_md5%% *}" = "9b0f2d5588c689e648555957f2668ff1"

stat -c '%s %n' "$MOT_AIS" "$MOT_VOA" "$PARLAMINT"
