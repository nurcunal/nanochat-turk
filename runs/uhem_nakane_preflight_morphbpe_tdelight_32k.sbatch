#!/bin/bash -l

# CPU-side preflight for the TurkishDelightNLP MorphBPE 32k tokenizer pipeline.
#
# This job validates the copied TurkishDelight source/model assets, creates a
# Linux venv for the DyNet runtime if needed, and verifies a tiny segmentation
# call before the expensive shard array is submitted.

#SBATCH --job-name=nanochat-preflight-tdlt32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=96G
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

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version
cmake --version | head -1 || true
gcc --version | head -1 || true
g++ --version | head -1 || true

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-tdelight-32768}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$NANOCHAT_BASE_DIR/segmented/tdelight-segmented}"

export TDELIGHT_REPO_DIR="${TDELIGHT_REPO_DIR:-$PWD/dev-ignore/vendor/turkish-delight-nlp-api}"
export TDELIGHT_VENV="${TDELIGHT_VENV:-$PWD/dev-ignore/venvs/tdelight-uhem-py312}"
export TDELIGHT_DYNET_SRC="${TDELIGHT_DYNET_SRC:-$PWD/dev-ignore/vendor/dynet-sdist/dyNET-2.1.2}"
export TDELIGHT_EIGEN_DIR="${TDELIGHT_EIGEN_DIR:-$PWD/dev-ignore/vendor/dynet-eigen/eigen-b2e267dc99d4}"
export TDELIGHT_DYNET_LIB_DIR="${TDELIGHT_DYNET_LIB_DIR:-$TDELIGHT_DYNET_SRC/build/py3.12-64bit/dynet}"
export TDELIGHT_MODEL_NAME="${TDELIGHT_MODEL_NAME:-Turkish-jointAll-MTAG_COMP=w_sum-MORPH_COMP=w_sum-POS_COMP=w_sum-COMP_ALPHA=0.1-trialmodel}"
export TDELIGHT_SEGMENT_CMD="${TDELIGHT_SEGMENT_CMD:-$TDELIGHT_VENV/bin/python scripts/tdelight_segment_cmd.py --repo-dir $TDELIGHT_REPO_DIR --model-name $TDELIGHT_MODEL_NAME}"
export LD_LIBRARY_PATH="$TDELIGHT_DYNET_LIB_DIR:${LD_LIBRARY_PATH:-}"

mkdir -p "$NANOCHAT_BASE_DIR" "$SEGMENTED_DATA_DIR" "$NANOCHAT_BASE_DIR/datasets"

if [ ! -f "$RAW_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing raw FineWeb-2 Turkish manifest at $RAW_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi
if [ ! -d "$TDELIGHT_REPO_DIR" ]; then
    echo "Missing TurkishDelight source at $TDELIGHT_REPO_DIR" >&2
    echo "Copy dev-ignore/vendor/turkish-delight-nlp-api to UHeM before this preflight." >&2
    exit 2
fi
if [ ! -f "$TDELIGHT_REPO_DIR/data/$TDELIGHT_MODEL_NAME" ] || [ ! -f "$TDELIGHT_REPO_DIR/data/$TDELIGHT_MODEL_NAME.params" ]; then
    echo "Missing TurkishDelight model files under $TDELIGHT_REPO_DIR/data" >&2
    exit 2
fi
if [ ! -d "$TDELIGHT_DYNET_SRC" ] || [ ! -d "$TDELIGHT_EIGEN_DIR" ]; then
    echo "Missing DyNet source or Eigen vendor directory." >&2
    echo "Expected $TDELIGHT_DYNET_SRC and $TDELIGHT_EIGEN_DIR" >&2
    exit 2
fi

# The copied local DyNet tree may contain a macOS build cache. If no Linux
# shared library exists yet, force a clean Linux build on UHeM.
if [ ! -f "$TDELIGHT_DYNET_LIB_DIR/libdynet.so" ]; then
    rm -rf "$TDELIGHT_DYNET_SRC/build"
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -x "$TDELIGHT_VENV/bin/python" ]; then
    uv venv --python "$(command -v python3)" "$TDELIGHT_VENV"
fi

uv pip install --python "$TDELIGHT_VENV/bin/python" \
    --upgrade pip setuptools wheel Cython packaging numpy scipy gensim==4.4.0 smart_open

if ! "$TDELIGHT_VENV/bin/python" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("dynet") else 1)
PY
then
    export EIGEN3_INCLUDE_DIR="$TDELIGHT_EIGEN_DIR"
    export MAKE_FLAGS="-j ${SLURM_CPUS_PER_TASK:-16}"
    uv pip install --python "$TDELIGHT_VENV/bin/python" "$TDELIGHT_DYNET_SRC"
    export LD_LIBRARY_PATH="$TDELIGHT_DYNET_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

"$TDELIGHT_VENV/bin/python" - <<'PY'
import dynet  # noqa: F401
import gensim  # noqa: F401
print("TurkishDelight runtime imports OK")
PY

if [ ! -d ".venv" ]; then
    echo "Missing repo .venv in $PWD. Run the main setup/prep job first." >&2
    exit 2
fi

.venv/bin/python - <<'PY'
from nanochat.morphology import create_segmenter

segmenter = create_segmenter("tdelight", strict=False, timeout=300)
words = ["evlerden", "okula", "geldik", "çalışıyorum"]
out = segmenter.segment_words(words)
for seg in out:
    print(seg.word, "=>", " + ".join(seg.pieces), "fallback=", seg.fallback)
if not any(len(seg.pieces) > 1 for seg in out):
    raise SystemExit("TurkishDelight preflight did not split any smoke-test word")
PY

ln -sfn "$RAW_DATA_DIR" "$NANOCHAT_BASE_DIR/datasets/raw-unsegmented-fineweb2-tur-latn"
ln -sfn ../segmented/tdelight-segmented "$NANOCHAT_BASE_DIR/datasets/tdelight-segmented"

echo "TurkishDelight preflight OK"
echo "Segmented corpus path:"
echo "  $SEGMENTED_DATA_DIR"
echo "Dataset aliases:"
ls -lah "$NANOCHAT_BASE_DIR/datasets"
