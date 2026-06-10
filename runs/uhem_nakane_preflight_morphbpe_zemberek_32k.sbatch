#!/bin/bash -l

# CPU-side preflight for the Zemberek MorphBPE 32k tokenizer pipeline.
#
# This job prepares the directory layout, installs the Zemberek Python binding
# into the existing repo .venv if needed, and verifies a tiny segmentation call.

#SBATCH --job-name=nanochat-preflight-zemb32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
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

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-zemberek-32768}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$NANOCHAT_BASE_DIR/segmented/zemberek-segmented}"
export ZEMBEREK_PIP_PACKAGE="${ZEMBEREK_PIP_PACKAGE:-zemberek-python}"

mkdir -p "$NANOCHAT_BASE_DIR" "$SEGMENTED_DATA_DIR" "$NANOCHAT_BASE_DIR/datasets"

if [ ! -f "$RAW_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing raw FineWeb-2 Turkish manifest at $RAW_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi
if [ ! -d ".venv" ]; then
    echo "Missing .venv in $PWD. Run the main setup/prep job once before Zemberek preflight." >&2
    exit 2
fi
source .venv/bin/activate

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("zemberek") else 1)
PY
then
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv pip install --python .venv/bin/python "$ZEMBEREK_PIP_PACKAGE"
fi

python - <<'PY'
from nanochat.morphology import create_segmenter

segmenter = create_segmenter("zemberek", strict=False)
words = ["evlerden", "okula", "geldik", "çalışıyorum"]
out = segmenter.segment_words(words)
for seg in out:
    print(seg.word, "=>", " + ".join(seg.pieces), "fallback=", seg.fallback)
if not any(len(seg.pieces) > 1 for seg in out):
    raise SystemExit("Zemberek preflight did not split any smoke-test word")
PY

ln -sfn "$RAW_DATA_DIR" "$NANOCHAT_BASE_DIR/datasets/raw-unsegmented-fineweb2-tur-latn"
ln -sfn ../segmented/zemberek-segmented "$NANOCHAT_BASE_DIR/datasets/zemberek-segmented"

echo "Zemberek preflight OK"
echo "Segmented corpus path:"
echo "  $SEGMENTED_DATA_DIR"
echo "Dataset aliases:"
ls -lah "$NANOCHAT_BASE_DIR/datasets"
