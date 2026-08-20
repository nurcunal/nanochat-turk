#!/bin/bash

# Recreate the exact pinned-upstream GPU environment before any A100 job.

set -euo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$CODE_DIR"

expected_uv_version=0.11.29
actual_uv_version="$(uv --version | awk '{print $2}')"
if [ "$actual_uv_version" != "$expected_uv_version" ]; then
    echo "uv must be exactly $expected_uv_version, found $actual_uv_version" >&2
    exit 2
fi

expected_pyproject_sha=c91fdd03ae9705565572eee31924d4c0bca24bf5431a8eabff4c061882f94929
expected_lock_sha=de7891b832854162111208644ddb72685069ce8128e2bed9dbb7993aa6af5861
actual_pyproject_sha="$(sha256sum pyproject.toml | awk '{print $1}')"
actual_lock_sha="$(sha256sum uv.lock | awk '{print $1}')"
if [ "$actual_pyproject_sha" != "$expected_pyproject_sha" ]; then
    echo "pyproject.toml differs from pinned Nanochat upstream" >&2
    exit 2
fi
if [ "$actual_lock_sha" != "$expected_lock_sha" ]; then
    echo "uv.lock differs from pinned Nanochat upstream" >&2
    exit 2
fi

# The upstream lock uses a relative seven-day exclude-newer policy. `--locked`
# ages out and attempts a re-resolution; `--frozen` consumes the reviewed lock
# byte-for-byte. The exact pyproject and lock hashes above make this fail closed.
module_python="$(command -v python3)"
if [ "$("$module_python" -c 'import platform; print(platform.python_version())')" != 3.12.4 ]; then
    echo "The loaded UHeM Python module is not Python 3.12.4" >&2
    exit 2
fi
uv sync --frozen --extra gpu --python "$module_python"

.venv/bin/python -c 'import platform, torch, kernels, pyarrow, rustbpe
assert platform.python_version() == "3.12.4"
assert torch.__version__.split("+")[0] == "2.9.1"
assert torch.version.cuda == "12.8"
print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})'

echo "Pinned d32 training environment is ready: $CODE_DIR/.venv"
