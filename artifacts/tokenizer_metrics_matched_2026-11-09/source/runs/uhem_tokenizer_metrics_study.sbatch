#!/bin/bash -l
# One CPU node, bounded concurrent tokenizer processes, equal train/val samples.
#SBATCH --job-name=tok-study-matched
#SBATCH --account=tl0002
#SBATCH --partition=core40q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:30:00
#SBATCH --mem=160G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the isolated evaluation source directory}"
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
source "${METRICS_VENV:-/ari/users/nunal/nanochat-turk/.venv}/bin/activate"
export NANOCHAT_BASE_DIR="$PWD/runtime-cache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NANOCHAT_DTYPE=float32
source_data="${SEGMENTED_DATA_DIR:-/ari/users/nunal/nanochat-turk-morphbpe-trmorph-32768/datasets/trmorph-segmented}"
mkdir results
python -m scripts.prepare_tokenizer_metric_sample --source "$source_data" --output sample --max-docs 50000 > results/sample_manifest.json
python - <<'PY'
import importlib.metadata, json, os, platform, time
from pathlib import Path
import sklearn
data = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(), "host": platform.node(), "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"), "packages": {name: importlib.metadata.version(name) for name in ("torch", "tiktoken", "tokenizers", "pyarrow", "scikit-learn", "rustbpe")}}
Path("results/environment.json").write_text(json.dumps(data, indent=2) + "\n")
PY
run_one() {
    local tokenizer_name="$1"
    for split in train val; do
        mkdir -p "results/$split"
        python -m scripts.tokenizer_metrics \
            --tokenizer-dir "$PWD/artifacts/tokenizers/$tokenizer_name" \
            --data-dir "$PWD/sample" --text-column segmented_text \
            --input-has-morph-boundaries --split "$split" --max-docs 50000 \
            --max-word-metrics 500000 --max-morph-metrics 500000 \
            --morph-consistency-max-words 50000 --morph-consistency-clusters 100 \
            --morph-consistency-pairs-per-cluster 50 --morph-consistency-resamples 10 \
            --morph-consistency-seed 13 --num-threads 8 --workers 1 --no-report \
            --output "results/$split/${tokenizer_name}_metrics.json" \
            > "results/$split/${tokenizer_name}.log" 2>&1
    done
}
pids=()
parallelism="${METRIC_PARALLELISM:-4}"
wait_batch() {
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    pids=()
    if [ "$failed" != 0 ]; then echo "At least one tokenizer process failed; inspect per-tokenizer logs." >&2; return 1; fi
}
for size in 32768 65536 131072; do
    for family in bpe morphbpe_trmorph morphbpe_zemberek morphbpe_tdelight; do
        run_one "${family}_${size}" &
        pids+=("$!")
        if [ "${#pids[@]}" -ge "$parallelism" ]; then wait_batch; fi
    done
done
wait_batch
python -m scripts.finalize_tokenizer_metric_study --directory results --source-manifest source_manifest.json
