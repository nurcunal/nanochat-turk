#!/bin/bash -l
# All twelve archived tokenizer bundles, one reference and identical caps.
# Historical train-prefix reproduction is explicit; set METRICS_SPLIT=val for
# a separate held-out evaluation. Outputs never overwrite historical JSONs.
#SBATCH --job-name=tok-matched-metrics
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi
if type module >/dev/null 2>&1; then
    module purge
    module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768/datasets/trmorph-segmented}"
METRICS_DIR="${METRICS_DIR:-$REPO_DIR/dev-ignore/tokenizer_metrics_matched}"
METRICS_SPLIT="${METRICS_SPLIT:-train}"
MAX_DOCS="${MAX_DOCS:-50000}"
MAX_WORD_METRICS="${MAX_WORD_METRICS:-500000}"
NUM_THREADS="${NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$METRICS_DIR"

# Refuse partial/mixed reruns: choose a fresh output directory for each run.
if compgen -G "$METRICS_DIR/*_metrics.json" >/dev/null; then
    echo "Existing metrics in $METRICS_DIR; set METRICS_DIR to a fresh directory." >&2
    exit 2
fi
for size in 32768 65536 131072; do
    for family in bpe morphbpe_trmorph morphbpe_zemberek morphbpe_tdelight; do
        tokenizer_name="${family}_${size}"
        python -m scripts.tokenizer_metrics \
            --tokenizer-dir "artifacts/tokenizers/$tokenizer_name" \
            --data-dir "$SEGMENTED_DATA_DIR" \
            --text-column segmented_text --input-has-morph-boundaries \
            --split "$METRICS_SPLIT" --max-docs "$MAX_DOCS" \
            --max-word-metrics "$MAX_WORD_METRICS" \
            --max-morph-metrics "$MAX_WORD_METRICS" \
            --morph-consistency-max-words 50000 \
            --morph-consistency-clusters 100 \
            --morph-consistency-pairs-per-cluster 50 \
            --morph-consistency-resamples 10 --morph-consistency-seed 13 \
            --num-threads "$NUM_THREADS" --workers 1 --no-report \
            --output "$METRICS_DIR/${tokenizer_name}_metrics.json"
    done
done

python - "$METRICS_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
signatures = set()
for path in sorted(root.glob("*_metrics.json")):
    data = json.loads(path.read_text())
    preservation = data["morphology"]["preservation"]
    coverage = data["coverage"]
    signatures.add((
        data["split"], data["docs"], data["bytes"], data["chars"], data["words"],
        coverage["word_sequence_sha256"], preservation["reference_sequence_sha256"],
    ))
    rows.append({
        "source": path.name,
        "tokenizer_name": data["tokenizer_name"],
        "vocab_size": data["vocabulary"]["vocab_size"],
        "split": data["split"],
        "docs": data["docs"],
        "document_tokens_per_word": data["tokens_per_word"],
        "isolated_word_tokens_per_word": data["word_fertility_isolated"]["tokens_per_word"],
        "coverage": coverage,
        "morpheme_preservation": preservation,
        "document_morph_boundaries": data["morph_boundary"],
        "historical_paper_metrics": data["paper_metrics"],
    })
if len(rows) != 12 or len(signatures) != 1:
    raise SystemExit("Expected 12 results with identical document counts and word/reference hashes")
(root / "requested_metrics.json").write_text(json.dumps({
    "schema_version": 1,
    "comparison": "matched_reference_and_word_caps",
    "rows": rows,
}, ensure_ascii=False, indent=2) + "\n")
print(f"Verified all 12 samples; wrote {root / 'requested_metrics.json'}")
PY
