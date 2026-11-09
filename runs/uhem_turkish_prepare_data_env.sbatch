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
if [ "$(sha256sum "$PROJECT_DIR/pyproject.toml" | awk '{print $1}')" != 7ba5aecacc2720a71307c3e033baf53feeb382cf9eabb483146ca504a2ecce63 ]; then
    echo "Turkish-data pyproject.toml hash mismatch" >&2
    exit 2
fi
if [ "$(sha256sum "$PROJECT_DIR/uv.lock" | awk '{print $1}')" != 7c53680fb9f85fbb117f29e611bb2a08294f3b539426b72df3295bc7959010a7 ]; then
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
assert metadata.version("spacy") == "3.8.15"
try:
    metadata.version("fasttext-wheel")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("legacy fasttext-wheel must not be installed")
predict_source = inspect.getsource(fasttext.FastText._FastText.predict)
assert "np.asarray(probs)" in predict_source
assert "copy=False" not in predict_source
print({
    "python": platform.python_version(),
    "datatrove": metadata.version("datatrove"),
    "fasttext-numpy2-wheel": metadata.version("fasttext-numpy2-wheel"),
    "fasttext_numpy2_predict": True,
    "pyarrow": metadata.version("pyarrow"),
    "spacy": metadata.version("spacy"),
})'

echo "Pinned Turkish-data environment is ready: $PROJECT_DIR/.venv"
