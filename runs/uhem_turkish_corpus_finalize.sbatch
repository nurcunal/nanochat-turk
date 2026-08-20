#!/bin/bash

# Full promotion plus exact ws8/ws16 finite capacity simulation. The source
# pool remains intact. Cleanup is a separate explicit command after this passes.

#SBATCH --job-name=tr-corpus-final
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u
: "${POOL_DIR:?}"
: "${TOKENIZER_DIR:?}"
: "${TOKENIZER_QUALITY_DIR:?}"
: "${PACKING_CONTROL_DIR:?}"
: "${FINAL_CORPUS_DIR:?}"
: "${APPROVED_SOURCE_TOKENS:?Set this to approved_source_token_target exactly}"
: "${QUOTA_HEADROOM_BYTES:?Set scheduler/project quota headroom in bytes}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v1.json}"
PYTHON_BIN="${PYTHON_BIN:-$CODE_DIR/.venv/bin/python}"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" scripts/build_turkish_pretrain_corpus.py \
    --policy "$POLICY" finalize \
    --pool-dir "$POOL_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --tokenizer-quality-dir "$TOKENIZER_QUALITY_DIR" \
    --packing-preflight-dir "$PACKING_CONTROL_DIR" \
    --output-dir "$FINAL_CORPUS_DIR" \
    --target-tokens "$APPROVED_SOURCE_TOKENS" \
    --shard-target-tokens "${SHARD_TARGET_TOKENS:-250000000}" \
    --quota-headroom-bytes "$QUOTA_HEADROOM_BYTES"

echo "Capacity passed. Pool data is still retained; run cleanup-pool explicitly only after verifying downstream preflight."
