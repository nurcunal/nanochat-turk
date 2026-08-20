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

set -euo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

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
if [ "$(sha256sum "$PROJECT_DIR/pyproject.toml" | awk '{print $1}')" != 78c149b8bfa7773b0d180296e8bf3b82fb9cab130da5f0dec1ca6d4c667a3dfa ]; then
    echo "Turkish-data pyproject.toml hash mismatch" >&2
    exit 2
fi
if [ "$(sha256sum "$PROJECT_DIR/uv.lock" | awk '{print $1}')" != a5ecf95e9d4aa45bf2273dcbb063f173b3a0f30208df7214c307c59d2bcb0a54 ]; then
    echo "Turkish-data uv.lock hash mismatch" >&2
    exit 2
fi

module_python="$(command -v python3)"
"$UV_BIN" sync --project "$PROJECT_DIR" --locked --python "$module_python"
"$UV_BIN" run --project "$PROJECT_DIR" --locked python -c '
import importlib.metadata as metadata
import platform

assert platform.python_version() == "3.12.4"
assert metadata.version("datatrove") == "0.10.0"
assert metadata.version("fasttext-wheel") == "0.9.2"
assert metadata.version("pyarrow") == "23.0.1"
print({
    "python": platform.python_version(),
    "datatrove": metadata.version("datatrove"),
    "fasttext-wheel": metadata.version("fasttext-wheel"),
    "pyarrow": metadata.version("pyarrow"),
})'

echo "Pinned Turkish-data environment is ready: $PROJECT_DIR/.venv"
