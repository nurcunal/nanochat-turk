#!/bin/bash

# Recreate the isolated, locked CPU environment used for corpus preparation.

#SBATCH --job-name=tr-data-env
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PROJECT_DIR="$CODE_DIR/environments/turkish-data"
cd "$CODE_DIR"

if [ ! -x "$UV_BIN" ]; then
    echo "uv is missing or not executable: $UV_BIN" >&2
    exit 2
fi
if [ "$("$UV_BIN" --version | awk '{print $2}')" != 0.11.29 ]; then
    echo "The Turkish-data environment requires uv 0.11.29" >&2
    exit 2
fi
if [ "$(sha256sum "$PROJECT_DIR/pyproject.toml" | awk '{print $1}')" != e93ae5c2474870bc94e6bda12496bbbac2ee25c8189aef0bbcc69273fb41a25f ]; then
    echo "Turkish-data pyproject.toml hash mismatch" >&2
    exit 2
fi
if [ "$(sha256sum "$PROJECT_DIR/uv.lock" | awk '{print $1}')" != a5fa81e681f32696d1d659499da9f490bc28326c44c24c9d223e04f5b2c6bfb4 ]; then
    echo "Turkish-data uv.lock hash mismatch" >&2
    exit 2
fi

module_python="$(command -v python3)"
"$UV_BIN" sync --project "$PROJECT_DIR" --locked --python "$module_python" \
    --reinstall-package fasttext-numpy2-wheel
"$UV_BIN" run --project "$PROJECT_DIR" --locked python -c '
import importlib.metadata as metadata
import inspect
import platform
import fasttext

assert platform.python_version() == "3.12.4"
assert metadata.version("datatrove") == "0.10.0"
assert metadata.version("fasttext-numpy2-wheel") == "0.9.2"
assert metadata.version("pyarrow") == "23.0.1"
predict_source = inspect.getsource(fasttext.FastText._FastText.predict)
assert "np.asarray(probs)" in predict_source
assert "copy=False" not in predict_source
print({
    "python": platform.python_version(),
    "datatrove": metadata.version("datatrove"),
    "fasttext-numpy2-wheel": metadata.version("fasttext-numpy2-wheel"),
    "fasttext_numpy2_predict": True,
    "pyarrow": metadata.version("pyarrow"),
})'

echo "Pinned Turkish-data environment is ready: $PROJECT_DIR/.venv"
