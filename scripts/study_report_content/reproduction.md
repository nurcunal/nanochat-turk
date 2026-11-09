# Reproduction and branch provenance

`main` is the development and results entry point. Consolidating branches does
not turn their distinct training runtimes into one controlled experiment. The
reported tokenizer ablation, the d32 production design, and future training
from consolidated `main` have separate provenance.

## Immutable recovery points

| Revision | Role | What it establishes |
| --- | --- | --- |
| [`955c3809c94c04731bf35a9695cc0a04c28e35e9`](https://github.com/nurcunal/nanochat-turk/tree/955c3809c94c04731bf35a9695cc0a04c28e35e9) | `main` immediately before consolidation | Historical tokenizer bundles, ablation results, legacy training and Atlas SFT tooling. This is a recoverable source snapshot, **not proof that every June run used this exact commit**. |
| [`0ea4231e26208be92bce765e4fdde94eeedb21fe`](https://github.com/nurcunal/nanochat-turk/tree/0ea4231e26208be92bce765e4fdde94eeedb21fe) | d32 branch tip incorporated into `main` | Frozen d32 configurations and additive production workflow; preserves their original code/environment pins. |
| [`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`](https://github.com/nurcunal/nanochat-turk/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) | Upstream core pinned by d32 | Exact core-file contract against which the original d32 production preflight validates. |
| [`8e7d0bbe3fcdd979b14ee148d178c1987bf63e8c`](https://github.com/nurcunal/nanochat-turk/commit/8e7d0bbe3fcdd979b14ee148d178c1987bf63e8c) | Expanded CETVEL results import | Publication of the eleven-model June 22 summary; an evidence-import revision, not a training-run revision. |

All branch history is retained. Use full commit IDs when recovering source;
branch names can move. The consolidated report records source-file SHA-256
hashes in [model_results.json](model_results.json) and
[tokenizer_results.json](tokenizer_results.json). Exact checkpoint and raw
CETVEL paths remain attached to each model row.

The November recovery also found a completed August model named
`tr_d32_bpe_32768_fineweb2_all_v1`, whose launch uses a content-addressed source
snapshot based on `955c3809c94c04731bf35a9695cc0a04c28e35e9`. Its snapshot hash
is `37503db09be2617d53d3f201d2fc776f7eb96844c7115818f02f38a3e0f83a97`; it
records `strict_mode=false` and `lr_schedule=nanochat_linear`. This completed
FineWeb2 run is distinct from the frozen Turkish-general WSD v4 design at the
d32 branch tip. Its validation overlaps training. See
[later model results](later_model_results.md) for that lineage and its
protocol-specific evaluation records.

## Rebuild the report versus rerun an experiment

Rebuild this complete page and its data exports from saved measurements:

```bash
python -m scripts.build_study_report
```

To refresh figures as well (requires matplotlib and NumPy):

```bash
MPLCONFIGDIR=/tmp/nanochat-mpl python -m scripts.build_study_report --plots
```

The optional PDF brief is regenerated with `python -m scripts.build_study_brief`
(requires reportlab 4.4.9 and matplotlib). None of these commands trains a model,
reruns CETVEL or submits UHEM jobs. Individual builders write temporary section
drafts under ignored `dev-ignore/study-sections/`; only the assembled README is
the public report. CSV/JSON exports live under `data/`, and original fresh run
evidence lives under `provenance/fresh_tokenizer_metrics/`.

The fresh-results builder rejects missing or unmatched samples and verifies
all 24 raw outputs against their manifests. The old matched and fresh reports
both described job `558060`; they are now merged into one fresh section/export,
including their distinct diagnostic fields. The original executed-source and
recovery receipts remain available as evidence, not as a second experiment.

Historical 50k/100k/1M/full-corpus measurements and SFT screens are included in
expandable sections on this page. SFT selection probes are not independent
capability evaluations and use only raw BPE 32k. Git retains the old file paths
and every previous report version; the cleanup changes presentation and paths,
not the measurements.

For a historical training replay, first recover the run's actual source
revision, complete effective configuration, dependency lock, seed, dataset
manifest/order, tokenizer bytes and all optimizer shards from its archived
metadata/logs. Use those together in an isolated checkout/environment. The
pre-consolidation `main` snapshot preserves the older implementation for
inspection, but is not a substitute for missing per-run provenance. Identical
model names and command-line hyperparameters alone do not establish a replay.
The recovered original ablation metadata contains effective model/user
configuration but lacks a complete training Git/environment fingerprint; the
raw-BPE 32k step-17,100 metadata is now missing at its original path.

Recovered August evaluator code documents two deviations in the June adapter:
generation kwargs were interpreted incorrectly, and likelihood continuation
boundaries were not validated. Original benchmark numbers remain historical
evidence; a corrected evaluation must be versioned separately. Model weights
and evaluator identity are both necessary to reproduce a score.

Historical model-weight inference and continued optimizer training are also
different operations. Loading weights successfully under a newer runtime does
not establish identical evaluation behavior or an exact training resume.
Record a new evaluation's runtime, task revisions, tokenizer and settings, and
retain the original result alongside it.

## The optimizer change is material

The pre-consolidation `main` optimizer and the d32 pinned core differ in update
equations, not just organization:

| Component | Historical `main` snapshot | d32 core adopted during consolidation |
| --- | --- | --- |
| Muon update | Polar Express orthogonalization plus NorMuon variance reduction | Adds MuonEq row equilibration before orthogonalization and Muon+ Frobenius-norm renormalization afterward |
| Optimizer classes | Separate `MuonAdamW` and `DistMuonAdamW`; model chooses by distributed mode | One `MuonAdamW` handles single-rank and distributed cases |
| AdamW arithmetic | Older parameter/state update implementation | Explicit float32 arithmetic and casts back to stored dtypes |

The additional Muon transforms execute in the adopted implementation. No
result in the historical ablation should be attributed retroactively to this
newer optimizer. An old optimizer shard that happens to load is not evidence
of identical updates. Future training with consolidated `main` is a new
runtime condition unless an explicitly reviewed compatibility implementation
and parity checks establish otherwise.

The change can be inspected without modifying a checkout:

```bash
git diff 955c3809c94c04731bf35a9695cc0a04c28e35e9 \
  0ea4231e26208be92bce765e4fdde94eeedb21fe -- nanochat/optim.py nanochat/gpt.py
```

## Frozen d32 recipes remain strict

The v1–v4 family JSON files are historical, sealed contracts. Consolidation
does **not** refresh their file hashes, canonical hashes, environment pins,
family IDs or approvals. The v4 recipe pins eleven core files, `pyproject.toml`
and `uv.lock`; those thirteen hashes match the original d32 tip. Its
environment additionally requires Python 3.12.4 and frozen dependency sync.

Consolidated `main` restores Turkish/tokenizer compatibility and other
historical features in shared files. Consequently it differs from the frozen
d32 contract, including the tokenizer, dataset/dataloader, checkpoint and
base-training paths and dependency metadata. The existing d32 preflight also
compares `core_scope` against its pinned upstream revision. Its strict runtime
checks exact file/environment hashes, clean code and ancestry. **The old d32
production recipe is expected to reject consolidated `main`; this rejection
preserves provenance.** Passing unit tests does not waive it.

Inspect the pins with the report-only diagnostic:

```bash
python -m scripts.check_study_provenance
python -m scripts.check_study_provenance \
  --revision 0ea4231e26208be92bce765e4fdde94eeedb21fe
python -m scripts.check_study_provenance --json
```

This command neither alters files nor adds a launch gate. Its successful exit
means the diagnostic ran; consult `checked_all_pins_match` or the printed
`MATCH`/`DRIFT` rows for compatibility. It is not a production-readiness check.

To continue the original d32 family, use its immutable compatible source and
environment and satisfy all existing artifact, topology, storage, quota and
acceptance gates. If consolidated `main` is to become a new d32 training
runtime, define a new reviewed recipe/lineage and regenerate the required
parity, environment and production receipts. Do not repair a rejection by
silently replacing hashes in v1–v4 or carrying old acceptance receipts into a
different runtime.

## What the merge does and does not prove

The report preserves twelve tokenizer configurations and eleven completed
base-model cells. The d32 workflow is a different model scale, corpus policy,
optimizer/runtime and schedule; its presence on `main` does not add a result
to the controlled d20/d16/d12 tokenizer ablation. Likewise, new tokenizer-only
evaluation does not create a missing language-model result.

The local test suite can check interfaces, tokenizer loading, checkpoint
handling and strict-recipe refusal paths. It cannot prove that historical
training was rerun, that optimizer trajectories match, or that new GPU
production gates have passed. Use [model results](model_results.md) for
measured outcomes, [tokenizer results](tokenizer_results.md) for intrinsic
measurements, and the [d32 runbook](../tr_d32_turkish_wsd_uhem.md) for its
separate operational requirements.
