#!/usr/bin/env bash
# Private, interactive inference only. Surviving calibration D32 SFT@32; no training.
#SBATCH --job-name=dsai-3b-instruct
#SBATCH --account=nakane
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
unset PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE
set -u

readonly CHAT_CONTROL="${DSAI_CHAT_CONTROL:?remote deployment directory is required}"
readonly CHAT_ENTRY="$CHAT_CONTROL/chat_web.py"
readonly CORE_DIR=/ari/users/nunal/nanochat-turk-d32-capability-calibration-code-956be38bd880
readonly PYTHON_BIN=/ari/users/nunal/nanochat-turk/.venv/bin/python
readonly MODEL_BASE=/ari/users/nunal/nanochat-turk-d32-fineweb2-bpe32k
readonly MODEL_TAG=tr_d32_bpe32k_capability_calibration_eff020_v1
readonly CHECKPOINT_DIR="$MODEL_BASE/chatsft_checkpoints/$MODEL_TAG"
: "${EXPECTED_LAUNCH_SHA256:?exact submitted launcher hash is required}"
[[ "$EXPECTED_LAUNCH_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$SLURM_JOB_PARTITION" == a100q && "$SLURM_NNODES" == 1 && "$SLURM_NTASKS" == 1 ]]
[[ "$SLURM_CPUS_PER_TASK" == 8 ]]
[[ -f "$CHAT_ENTRY" && ! -L "$CHAT_ENTRY" && -x "$PYTHON_BIN" ]]
[[ "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)" == "$EXPECTED_LAUNCH_SHA256" ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c A100)" == 1 ]]

echo "D32_CHAT_START job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST source=sft step=32 gpus=1 time_limit=01:00:00 bind=127.0.0.1:8000"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

# Authenticate only the files used by inference, on the allocated compute node.
sha256sum --check --strict <<CHECKSUMS
4238d6271c8723460d20abec1ae88bb95da44cdae2087e01c90bd75d7bc49bf0  $CHECKPOINT_DIR/model_000032.pt
7912e121be3e820f0bdcd2bff4855337b2229b298f7ac4af8108a3348384447e  $CHECKPOINT_DIR/meta_000032.json
b1b32ba10d02599b27cfa31b9d0097aa3142716636d9abedb27f67ae196c344a  $CHECKPOINT_DIR/complete_000032.json
61a490134a6f372bed918f4059b515084ff769533a9190017482441186885839  $CHAT_ENTRY
8950542a6da1c102f8463aebaa74e1403ccc06fb6b287903470d0b2a5869c286  $CORE_DIR/nanochat/common.py
009db344f93136c511ebc08e8143d6586d11a78e0465f5859cb7513430bd9a0c  $CORE_DIR/nanochat/checkpoint_manager.py
e37167c576db9cd8af876ca94502690e039465368060643ef79292153732e75e  $CORE_DIR/nanochat/gpt.py
ed349be4b21a1af24958a378ba8cbd90a28dfb41033acf2573b29de41989fe93  $CORE_DIR/nanochat/flash_attention.py
7108b3385585f037b00149080eff95808128f28ff8138cd2f02da89f83f84500  $CORE_DIR/nanochat/tokenizer.py
d53fc041180c1e8fe6ca2e507b11b57e6422258d70a35a994fced16d51820a20  $CORE_DIR/nanochat/engine.py
4cc4ed098960bbce579d4926b27d2d241712b060488e53c2d336d432f68f4f5b  $CORE_DIR/nanochat/optim.py
72e8ec856a91ca90851b6dedbbe5ce11fb51aecfbcb793d4147a13b34b6434cc  $CORE_DIR/nanochat/d32_chat_contract.py
c8e538df233633528779ac1ccbf7bea0862ee9f73fa97789341cf2f1c6d2d85f  $CORE_DIR/nanochat/sft_data.py
e0490070d5b649900b1ea0f6f3a8961501ed726e73a745642b3e51f7aa9ccb6c  $CORE_DIR/nanochat/ui.html
513eeb7bf11f152f4edb466d6ff40fc3f7fae7adb90ac965996d216f5e2e5e1a  $CORE_DIR/nanochat/logo.svg
3b7f00d8697a303bb0ac7379b497fec6b76a61bd5bb694825264eecd4cb62a20  $MODEL_BASE/tokenizers/bpe_32768/tokenizer.pkl
b381631bad8ca489b85dbab686d63f0691b6455300f2ae43f8af43ad3b80a916  $MODEL_BASE/tokenizers/bpe_32768/tokenizer_config.json
ba345dc7eea5f07b3200f213ccf61ea6e9990d15d81d25e2e323009f4fba502e  $MODEL_BASE/tokenizers/bpe_32768/token_bytes.pt
CHECKSUMS

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export NANOCHAT_BASE_DIR="$MODEL_BASE" NANOCHAT_TOKENIZER_NAME=bpe_32768 NANOCHAT_DTYPE=bfloat16
export PYTHONPATH="$CORE_DIR" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export D32_CHAT_ENTRY="$CHAT_ENTRY" D32_CHAT_MODEL_TAG="$MODEL_TAG"
cd "$CORE_DIR"

exec "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import json
import os
import runpy
import sys
import torch
from nanochat import flash_attention

assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0)
flash_attention.USE_FA3 = False
print('D32_CHAT_RUNTIME ' + json.dumps({
    'torch': torch.__version__,
    'fastapi': importlib.metadata.version('fastapi'),
    'pydantic': importlib.metadata.version('pydantic'),
    'uvicorn': importlib.metadata.version('uvicorn'),
    'attention': 'sdpa', 'source': 'sft', 'step': 32,
    'model_tag': os.environ['D32_CHAT_MODEL_TAG'],
    'network': 'compute-node loopback only; access through SSH tunnel',
    'training': False, 'automatic_followon': False,
}), flush=True)
sys.argv = [os.environ['D32_CHAT_ENTRY'], '--device-type', 'cuda', '--num-gpus', '1',
            '--source', 'sft', '--model-tag', os.environ['D32_CHAT_MODEL_TAG'],
            '--step', '32', '--temperature', '0.7', '--top-k', '50',
            '--max-tokens', '512', '--host', '127.0.0.1', '--port', '8000']
runpy.run_path(os.environ['D32_CHAT_ENTRY'], run_name='__main__')
PY
