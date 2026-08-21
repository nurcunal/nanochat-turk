# Turkish d32 WSD family on UHeM

This workflow prepares one shared d32 trunk and three independently cooled,
stable base-model checkpoints: s12 at 20,132,659,200 scheduled positions, s20
at 33,554,432,000, and s40 at 67,108,864,000. The s12 and s20 cooldowns fork
immutable trunk checkpoints; after each fork, the trunk continues toward the
next boundary. They are therefore three final models without paying for three
independent pretraining runs.

No automatic end-to-end production submitter is exposed by the repository.
The operator and user must review the sealed gates and topology decision before
submitting each bounded production launcher.

## Frozen model and data contract

- d32, width 2,048, 16 query/KV heads, head dimension 128, context 2,048.
- Vocabulary 32,768 from the new Turkish-only tokenizer package.
- 2,818,575,450 total parameters; 1,677,724,672 Nanochat scaling parameters.
- Global batch is fixed at 2,097,152 scheduled positions for ws8 and ws16.
- Data is Turkish-only, code corpora are forbidden, and validation is a fixed,
  finite whole-document manifest.
- The production policy is `tr_general_clean_v4`; v1/v2/v3 remain historical and
  cannot be renamed or promoted. It uses the official Turkish MaCoCu-Genre release
  (CLARIN handle `11356/1969`). A one-time CPU job verifies its 14,448,949,031-
  byte gzip and MD5 `abe376c21256798ded30e54770666aa0`, then streams it into
  deterministic sealed zstd JSONL shards. Object-array ranks consume those
  shards; they never redownload or independently decompress the full gzip.
- MaCoCu has separate frozen lanes for `Forum` (0.8%),
  `Opinion/Argumentation` (1.7%), `Information/Explanation` (2.0%),
  `Instruction` (1.5%), and bounded `News` (4.0%). Promotion,
  prose/lyrics, legal, other, and mixed genres are excluded. HPLT is not treated
  as a conversation source because the bounded register audit did not support
  that interpretation.
- The frozen v4 planning mix is FineWeb2-HQ 15%,
  `fineweb2_strict_tr_v3` 73.65%, the MaCoCu lanes above (10% total),
  FineWiki 0.4%, native MOT article JSON 0.8%, and native ParlaMint structured
  dialogue 0.15%. The strict lane acquires all 30 Turkish FineWeb2 objects and
  admits only rows that pass every local language, quality, corruption, code,
  and dedup gate. The bounded audit
  disqualified FinePDF because manually reviewed Turkish rows contained OCR,
  reading-order, and layout corruption. It also found that the raw-FineWeb
  structural filters were too permissive for semantic junk. No production
  corpus, tokenizer, or model may descend from that v2 mixture. V4 omits HPLT,
  FinePDF and the historical direct `fineweb2_tr` identity. All 30 immutable
  Turkish FineWeb2 objects are nevertheless acquired under the distinct
  `fineweb2_strict_tr_v3` derivation; its exact 30-object/134,789,283,815-byte
  inventory, upstream commit, processing hash, and audit-policy hash are
  receipt-bound, and only rows passing every gate become candidates. There is
  no direct raw fallback. The fresh policy/source-plan lineage must pass the
  accepted-token-yield, dedup-overlap, language, quality, and genre gates from
  scratch. PDF- and OCR-derived text is out of scope for this family;
  replacing FinePDF with a different OCR corpus is not an allowed fallback.
  The production launcher additionally requires every selected source to
  declare `text_origin` as exactly `born_digital_text` or `structured_text`;
  missing, unknown, mixed, PDF-extracted, and OCR-derived origins fail closed.
  It also requires zero-tolerance document gates for Unicode replacement
  characters, high-confidence Turkish mojibake sequences, C1 controls, and
  surrogate code points. These are corruption backstops for native text, not a
  mechanism for admitting OCR.
- Do not ingest the aggregate BellaTurca collection: its AkademikDerlem subset
  is PDF/OCR-derived, while its OSCAR/mC4 subsets substantially overlap the web
  families already being audited. OzenliDerlem and explicitly nontechnical
  ForumSohbetleri subsets were separately audited as text-native candidates.
  Neither aggregate is approved: only narrow Ozenli publisher files remain
  candidates, and no ForumSohbetleri configuration is currently admitted.
  `memurlar` may be reconsidered only through a bounded conversation selector
  and a new manual audit. See `docs/tr_d32_native_text_candidate_audit.md`.
- The project-local audit rejects substantial mixed-script text only when it
  exceeds both 32 letters and 2% of alphabetic text. Contextual APK/download,
  commerce, cookie-interface, legal-policy, taxonomy/search, and strong SEO
  template gates use narrowly calibrated phrases and URL context; generic words
  such as “uygulama”, “ürün”, “fiyat”, “KVKK”, or “gizlilik” do not reject text.
- The exact upstream best-fit repeat simulation runs through s12, s20, s40,
  and s40 plus 2% for ws8 and ws16, including buffer prefetch across epoch
  boundaries. At least 34,225,520,640 first-epoch packed positions is the
  preferred at-most-two-epoch tier. Between 17,112,760,320 and that value is a
  manual-risk at-most-four-epoch tier; below the hard floor is a no-go. Only the
  complete composite pool may repeat—small sources never cycle independently.
  The manual-risk tier remains defined so capacity receipts can report and
  audit it, but it cannot authorize this production run: every production gate
  and launcher requires the preferred at-most-two-epoch tier.
- The upstream training core at commit
  `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` is immutable and every relied-on
  upstream file is exact-hash pinned. The production lane is additive only.

## Required gate order

1. Measure and seal data-preparation CPU/storage extrapolation.
2. Build corpus and tokenizer; seal dataset, tokenizer, capacity, validation,
   and all ws-specific exposure plans.
3. Run the family preflight from a clean descendant of the pinned upstream
   commit. It rechecks live BeeGFS space and UHeM CPU-saat quota.
4. Run the one-node, four-rank static `srun`/NCCL clean-exit probe.
5. Run the one-A100 attention probe. Prefer upstream-auto-selected FA3+SSSL
   only after the actual d32 BF16 forward/backward succeeds; otherwise require
   the actual d32 SDPA+L fallback to succeed.
6. Run the packed d12 screening and d20 confirmation weight-decay proxy.
7. Run the bounded one-node SIGUSR1 → transactional checkpoint → collective
   exit 75 → Slurm requeue → exact resume gate.
8. Run the production-identical ws8 smoke, including a forced checkpoint and
   resume. It must pass loader, storage, NCCL, protocol, and checkpoint checks.
9. Run the ws16 smoke when four nodes are available. Select ws16 only if its
   clean measured speedup over ws8 is at least 1.7; otherwise seal ws8 as the
   fallback. One selected world size is locked for the entire family lineage.
10. Review every sealed receipt before constructing any production stage
    submission. The repository deliberately provides no umbrella
    production-submit mode.

All distributed work uses direct Slurm `srun` tasks, one task and one visible
GPU per rank. It does not use `torchrun`, elastic rendezvous, or `mpirun`.
Slurm 20.11 time-limit signalling uses `#SBATCH --signal=USR1@900` without
`B:`; the bounded test targets the exact srun step with
`scancel --signal=USR1 JOBID.0`.

## Four-node decision and cost

The four-node/ws16 plan is reasonable only after the measured gate. A 1.7x
speedup halves the requested wall-clock resources but increases billed GPU
hours by about 17.6% versus ws8; waiting for four nodes is worthwhile when
shorter elapsed time is more valuable than that extra cost. If four nodes are
scarce or speedup is below 1.7x, ws8 is the sealed fallback and remains valid.

The shared three-final training lineage is estimated at 1,224–1,948 aggregate
A100-hours. With the 15% reserve this is 22,522–35,844 CPU-saat for training;
adding the 4,000 CPU-saat proxy/kernel/smoke reserve gives a complete training
package of **26,522–39,844 CPU-saat**. UHeM bills this GPU partition at 16
CPU-saat per fully utilized A100-hour (64 CPU-saat per 4xA100 node-hour).
Data preparation is deliberately excluded: its sampled/extrapolated sealed
CPU-saat value is added before preflight because inventing a fixed estimate
before the real source mix exists would be unsafe.

The v4 storage envelope keeps all three stable forks and all three cooled
finals as full resumable checkpoint transactions on UHeM. The static training
preflight therefore requires 601,295,421,440 bytes (560 GiB) free, and the
corpus/tokenizer-inclusive project envelope is 850,403,524,608 bytes (792 GiB).
Choosing a model-only Hugging Face export never deletes optimizer, loader, or
RNG state from the local finals.

The data `resource-report` command requires `--billable-cpus-per-job 128`.
Its sealed `cpu2dq` contract bills projected stage wall time at 128 CPUs for
each one-node job; process CPU time is retained only as an efficiency diagnostic.

The ws8 calibration corresponds to roughly 153.0–206.9 training wall-hours.
At the minimum accepted 1.7x ws8-to-ws16 speedup, ws16 corresponds to roughly
90.0–121.7 wall-hours (or 76.5–103.4 hours at ideal 2x scaling). The
1,224–1,948 A100-hour envelope above is deliberately cross-topology and must
not be divided by one fixed world size. Stage boundaries, validation,
checkpoint writes, queueing, and the prerequisite gates add elapsed time.

## Safe operator entry point

Start from the clean repository root and route every Slurm stream outside the
checkout before the first submission. The strict preflight and trainers treat
any untracked checkout file as provenance drift, including an otherwise inert
Slurm log:

```bash
export CODE_DIR="$(pwd -P)"
export NANOCHAT_BASE_DIR=/ari/users/nunal/nanochat-turk-d32-general-v3
export D32_PATH_CONTRACT=v4
SBATCH_LOG_DIR="$NANOCHAT_BASE_DIR/logs/d32_v4"
mkdir -p "$SBATCH_LOG_DIR"
SBATCH_LOG_ARGS=(--output="$SBATCH_LOG_DIR/%x-%j.out" --error="$SBATCH_LOG_DIR/%x-%j.err")
SBATCH_ARRAY_LOG_ARGS=(--output="$SBATCH_LOG_DIR/%x-%A_%a.out" --error="$SBATCH_LOG_DIR/%x-%A_%a.err")
```

Prepare the isolated CPU corpus environment with
`sbatch "${SBATCH_LOG_ARGS[@]}" runs/uhem_turkish_prepare_data_env.sbatch`; it accepts only uv 0.11.29
and the reviewed Turkish-data project and lock hashes. Every producer below
also refuses a dirty checkout or a `CODE_REVISION` different from committed
`HEAD`; commit the reviewed v4 code before submitting any data job.

The project base deliberately remains the existing shared v3 BeeGFS root. Reuse
the already verified MaCoCu, MOT, ParlaMint, and 1.69-GB GlotLID artifacts in
place—do not download or copy them into a second project tree. These are the
exact overrides; the v4 bootstrap and packed workers verify their receipts,
size, and hashes before use:

```bash
SHARED_SOURCE_BASE=/ari/users/nunal/nanochat-turk-d32-general-v3
export MACOCU_MANIFEST="$SHARED_SOURCE_BASE/source_data/macocu_genre_tr_v1/manifest.json"
export MOT_MANIFEST="$SHARED_SOURCE_BASE/source_data/mot_tr_v1_11/manifest.json"
export PARLAMINT_MANIFEST="$SHARED_SOURCE_BASE/source_data/parlamint_tr_v5_0/manifest.json"
export MODEL_DIR="$SHARED_SOURCE_BASE/source_models/glotlid-v3"
export GLOTLID_MODEL="$MODEL_DIR/model_v3.bin"
test -f "$MACOCU_MANIFEST" && test -f "$MOT_MANIFEST" && \
  test -f "$PARLAMINT_MANIFEST" && test -f "$GLOTLID_MODEL"
```

Only if one of those verified artifacts is genuinely absent, run
`sbatch "${SBATCH_LOG_ARGS[@]}" runs/uhem_turkish_data_prepare_macocu.sbatch`
once on the shared
filesystem. It atomically writes the retained verified gzip, reasonably sized
prepared shards, per-shard SHA-256/row/genre accounting, and `manifest.json`.
If the official gzip is already present, set `MACOCU_UPSTREAM_FILE` to that
regular, non-symlink file before submission. The preparer verifies its exact
size and official MD5, seals its SHA-256, and verifies the staged copy against
both checksums before parsing; with the variable unset, the pinned HTTPS path
is unchanged.
If the shared MOT or ParlaMint production manifest is absent, fetch and prepare
both native-text anchors before bootstrap, in this exact order:

1. Submit
   `sbatch "${SBATCH_LOG_ARGS[@]}" runs/uhem_turkish_anchor_fetch_v3.sbatch`
   to stage the pinned two
   MOT v1.11 archives and ParlaMint-TR v5.0 archive.
2. Submit
   `sbatch "${SBATCH_LOG_ARGS[@]}" --export="ALL,MODE=discovery,CODE_DIR=$CODE_DIR,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR" runs/uhem_turkish_anchor_prepare_v3.sbatch`.
   Inspect each discovery `manifest.json`, its native-text
   evidence, counts, exclusions, Unicode audit, and archive-member provenance
   under `source_data/mot_tr_v1_11_discovery` and
   `source_data/parlamint_tr_v5_0_discovery`.
3. Only after that manual inspection, seal two independent count acceptances:

   ```bash
   environments/turkish-data/.venv/bin/python scripts/prepare_turkish_anchors.py accept-counts \
     --discovery-output-dir "$NANOCHAT_BASE_DIR/source_data/mot_tr_v1_11_discovery" \
     --reviewer "$REVIEWER" --reviewed-at-utc "$REVIEWED_AT_UTC" \
     --decision accepted \
     --output "$NANOCHAT_BASE_DIR/control/data_v3/anchors/mot_count_acceptance.json"

   environments/turkish-data/.venv/bin/python scripts/prepare_turkish_anchors.py accept-counts \
     --discovery-output-dir "$NANOCHAT_BASE_DIR/source_data/parlamint_tr_v5_0_discovery" \
     --reviewer "$REVIEWER" --reviewed-at-utc "$REVIEWED_AT_UTC" \
     --decision accepted \
     --output "$NANOCHAT_BASE_DIR/control/data_v3/anchors/parlamint_count_acceptance.json"
   ```

4. Submit the same preparation launcher with the external log arguments,
   `MODE=production`, `MOT_COUNT_ACCEPTANCE` set to the MOT receipt above, and
   `PARLAMINT_COUNT_ACCEPTANCE` set to the ParlaMint receipt above:

   ```bash
   MODE=production \
   MOT_COUNT_ACCEPTANCE="$NANOCHAT_BASE_DIR/control/data_v3/anchors/mot_count_acceptance.json" \
   PARLAMINT_COUNT_ACCEPTANCE="$NANOCHAT_BASE_DIR/control/data_v3/anchors/parlamint_count_acceptance.json" \
     sbatch "${SBATCH_LOG_ARGS[@]}" runs/uhem_turkish_anchor_prepare_v3.sbatch
   ```

   It writes
   the production manifests at `source_data/mot_tr_v1_11/manifest.json` and
   `source_data/parlamint_tr_v5_0/manifest.json`; discovery outputs are never
   admitted directly.

Once MaCoCu and both accepted-production anchor manifests exist, submit the
captured v4 bootstrap/sample chain below. The bootstrap passes both manifests
as repeatable `--prepared-source-manifest` arguments and binds them with MaCoCu
into the immutable source plan before sealing the pinned GlotLID
model/calibration and deterministic sample ranks. Historical
v1/v2/v3 plans and their existing audit artifacts remain separate; never reuse
their plan, calibration, approval, receipt, pool, tokenizer, or packing receipt
with v4. Reusing the immutable prepared-source manifests and GlotLID model bytes
is allowed; reusing a v3 source plan or downstream receipt is not.

The object resource sample runs through
`runs/uhem_turkish_data_objects_packed_sample.sbatch`: one exclusive cpu2dq
allocation contains thirty-two 4-CPU worker lanes, and every sample rank belongs to
exactly one serial lane. The post-cluster writer-probe job is the sole producer
of the backend report and deliberately seals it with safety factor 1.0. Run
`runs/uhem_turkish_sample_quality_audit.sbatch` against that same completed
sample/cluster lineage, inspect its accepted and rejected JSONL/plaintext
examples, and explicitly seal the manual mixture-quality approval. The audit
must contain an accepted row and an accepted review example for every selected
source-object rank; sparse accepted/rejected
strata remain explicit insufficiency records. A clean source cannot mask a bad
rank. The approval stores a relative evidence-bundle path and validators reopen
the actual sealed report, both JSONL files, both plaintext files, the cluster
receipt, and the packed object/bucket launch receipts. Keep that directory and
the approval together; a hand-written receipt containing plausible 64-hex
strings is intentionally rejected.

Approval validation also reopens the live sample `cluster_receipt.json` and
every listed cluster Parquet through stable file descriptors. It verifies each
bounded file hash, repeats the full-text policy audit (including examples whose
human-readable text is truncated), and reconstructs the smallest content-bound
SHA-256 example selection from the actual rows. Example records bind the exact
backend row, full text, URL, metrics, document ID, and dedup cluster ID. Small
JSON/JSONL/plaintext evidence is read once into capped immutable byte snapshots;
path replacement or in-place mutation fails closed. The resource report,
mixture-quality approval, resource approval, storage sample/gate, and production
cluster launch all carry the same `sample_cluster_receipt_sha256`; crossing two
sample lineages is invalid even when every individual receipt is self-hashed.

Source quality and language confidence are separate signals. New object
receipts explicitly attest that `quality_score` contains only source-provided
quality; GlotLID probabilities never choose a same-priority dedup winner. For
compatibility with a bounded object sample produced before that attestation was
added, the cluster stage treats its stored score as zero and rewrites the
merged score to zero. The cluster receipt lists every such neutralized rank;
the audit and production seal independently verify that list. This permits
reuse of already-computed signatures without silently trusting legacy LID-based
scores.

Each packed bucket worker receives a descriptor-backed DataTrove input folder
for its one disjoint signature bucket. Its sealed inventory duplicates the
already verified file descriptors and reads them with independent `pread`
positions; DataTrove never reopens a public signature pathname. This handoff
creates no temporary signature copies, so the resource report retains its
schema-v2 arithmetic over the five real peak components (largest raw object,
candidates, signatures, duplicate edges, and backend output) and counts the
on-disk signature corpus exactly once.

Use this one canonical v4 namespace for every command below. The project base
stays on the shared v3 BeeGFS root so the immutable source preparations and
GlotLID model are reused in place; all mutable data/family controls and derived
artifacts use v4-only paths:

```bash
export CODE_DIR="$(pwd -P)"
export NANOCHAT_BASE_DIR=/ari/users/nunal/nanochat-turk-d32-general-v3
export D32_PATH_CONTRACT=v4
DATA_PYTHON="$CODE_DIR/environments/turkish-data/.venv/bin/python"
TRAIN_PYTHON="$CODE_DIR/.venv/bin/python"
POLICY="$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v4.json"
RECIPE="$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v4.json"
DATA_CONTROL_DIR="$NANOCHAT_BASE_DIR/control/data_v4"
FAMILY_CONTROL_DIR="$NANOCHAT_BASE_DIR/control/d32_v4"
SOURCE_PLAN="$DATA_CONTROL_DIR/source_plan.json"
CALIBRATION="$DATA_CONTROL_DIR/backend_calibration.json"
SAMPLE_RANKS="$DATA_CONTROL_DIR/resource_sample_ranks.json"
SAMPLE_LANE_PLAN="$DATA_CONTROL_DIR/resource_sample_lane_plan.json"
SAMPLE_RUN_DIR="$NANOCHAT_BASE_DIR/data_backend/resource_sample_v4"
BACKEND_RESOURCE_REPORT="$DATA_CONTROL_DIR/backend_resource_report.json"
WRITER_PROBE="$DATA_CONTROL_DIR/post_cluster_writer_probe.json"
MIXTURE_QUALITY_APPROVAL="$DATA_CONTROL_DIR/mixture_quality_approval.json"
RESOURCE_APPROVAL="$DATA_CONTROL_DIR/resource_approval.json"
PRODUCTION_NODE_SELECTION="$DATA_CONTROL_DIR/production_data_node_selection.json"
PACK_PLAN="$DATA_CONTROL_DIR/production_source_pack_plan.json"
DATA_PREP_STORAGE_GATE="$FAMILY_CONTROL_DIR/data_prep_storage_gate.json"
DATA_RUN_DIR="$NANOCHAT_BASE_DIR/data_backend/production_v4"
POOL_DIR="$NANOCHAT_BASE_DIR/data_v4/filtered_pool"
TOKENIZER_SAMPLE_DIR="$NANOCHAT_BASE_DIR/control/tokenizer/tr_general_raw_bpe_32k_v4/sample"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizers/tr_general_raw_bpe_32k_v4"
TOKENIZER_QUALITY_DIR="$NANOCHAT_BASE_DIR/control/tokenizer/tr_general_raw_bpe_32k_v4/quality"
PACKING_CONTROL_DIR="$NANOCHAT_BASE_DIR/control/packing/tr_general_clean_v4"
FINAL_CORPUS_DIR="$NANOCHAT_BASE_DIR/pretrain_data/tr_general_clean_v4"
BASELINE_TOKENIZER_DIR=/ari/users/nunal/nanochat-turk-d20-bpe32k/tokenizers/bpe_32768
SBATCH_LOG_DIR="$NANOCHAT_BASE_DIR/logs/d32_v4"
mkdir -p "$SBATCH_LOG_DIR"
SBATCH_LOG_ARGS=(--output="$SBATCH_LOG_DIR/%x-%j.out" --error="$SBATCH_LOG_DIR/%x-%j.err")
SBATCH_ARRAY_LOG_ARGS=(--output="$SBATCH_LOG_DIR/%x-%A_%a.out" --error="$SBATCH_LOG_DIR/%x-%A_%a.err")
```

Historical v3 remains usable through its preserved policy, recipe, runbook
revision, and `runs/uhem_d32_v3_paths.sh`; explicitly set
`D32_PATH_CONTRACT=v3` when invoking a preserved v3 artifact launcher. Never
combine a v3 policy/control receipt with this v4 command block.

From the clean v4 checkout, execute this exact environment → bootstrap →
packed object sample → packed bucket sample → cluster → QA chain. Every fixed
path is exported explicitly, so an inherited v3 shell alias cannot redirect a
v4 producer. All scheduler logs stay outside `CODE_DIR`:

```bash
: "${MACOCU_MANIFEST:?set the verified shared MaCoCu manifest}"
: "${MOT_MANIFEST:?set the verified shared MOT manifest}"
: "${PARLAMINT_MANIFEST:?set the verified shared ParlaMint manifest}"
: "${MODEL_DIR:?set the verified shared GlotLID directory}"
: "${GLOTLID_MODEL:?set the verified shared GlotLID model}"
CODE_REVISION="$(git -C "$CODE_DIR" rev-parse HEAD)"
test "${#CODE_REVISION}" -eq 40
test -z "$(git -C "$CODE_DIR" status --porcelain --untracked-files=all)"

DATA_ENV_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --export="ALL,CODE_DIR=$CODE_DIR,CODE_REVISION=$CODE_REVISION" \
  runs/uhem_turkish_prepare_data_env.sbatch)"

BOOTSTRAP_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$DATA_ENV_JOB_ID" \
  --export="ALL,CODE_DIR=$CODE_DIR,CODE_REVISION=$CODE_REVISION,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,D32_PATH_CONTRACT=v4,POLICY=$POLICY,CONTROL_DIR=$DATA_CONTROL_DIR,SOURCE_PLAN=$SOURCE_PLAN,CALIBRATION=$CALIBRATION,SAMPLE_RANKS=$SAMPLE_RANKS,MACOCU_MANIFEST=$MACOCU_MANIFEST,MOT_MANIFEST=$MOT_MANIFEST,PARLAMINT_MANIFEST=$PARLAMINT_MANIFEST,MODEL_DIR=$MODEL_DIR" \
  runs/uhem_turkish_data_bootstrap.sbatch)"

SAMPLE_OBJECT_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$BOOTSTRAP_JOB_ID" \
  --export="ALL,CODE_DIR=$CODE_DIR,CODE_REVISION=$CODE_REVISION,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,D32_PATH_CONTRACT=v4,POLICY=$POLICY,CONTROL_DIR=$DATA_CONTROL_DIR,SOURCE_PLAN=$SOURCE_PLAN,CALIBRATION=$CALIBRATION,SAMPLE_RANKS=$SAMPLE_RANKS,LANE_PLAN=$SAMPLE_LANE_PLAN,DATA_RUN_DIR=$SAMPLE_RUN_DIR,GLOTLID_MODEL=$GLOTLID_MODEL" \
  runs/uhem_turkish_data_objects_packed_sample.sbatch)"
OBJECT_SAMPLE_LAUNCH_RECEIPT="$SAMPLE_RUN_DIR/packed_sample_launches/job$SAMPLE_OBJECT_JOB_ID/launch_receipt.json"

SAMPLE_BUCKET_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$SAMPLE_OBJECT_JOB_ID" \
  --export="ALL,CODE_DIR=$CODE_DIR,CODE_REVISION=$CODE_REVISION,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,D32_PATH_CONTRACT=v4,POLICY=$POLICY,CONTROL_DIR=$DATA_CONTROL_DIR,SOURCE_PLAN=$SOURCE_PLAN,CALIBRATION=$CALIBRATION,SAMPLE_RANKS=$SAMPLE_RANKS,LANE_PLAN=$SAMPLE_LANE_PLAN,DATA_RUN_DIR=$SAMPLE_RUN_DIR,OBJECT_SAMPLE_LAUNCH_RECEIPT=$OBJECT_SAMPLE_LAUNCH_RECEIPT" \
  runs/uhem_turkish_data_buckets_packed_sample.sbatch)"
SAMPLE_BUCKET_LAUNCH_RECEIPT="$SAMPLE_RUN_DIR/packed_bucket_launches/job$SAMPLE_BUCKET_JOB_ID/launch_receipt.json"

SAMPLE_CLUSTER_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$SAMPLE_BUCKET_JOB_ID" \
  --export="ALL,CODE_DIR=$CODE_DIR,CODE_REVISION=$CODE_REVISION,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,D32_PATH_CONTRACT=v4,POLICY=$POLICY,CONTROL_DIR=$DATA_CONTROL_DIR,SOURCE_PLAN=$SOURCE_PLAN,CALIBRATION=$CALIBRATION,DATA_RUN_DIR=$SAMPLE_RUN_DIR,SAMPLE=1" \
  runs/uhem_turkish_data_cluster.sbatch)"

SAMPLE_QUALITY_AUDIT_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$SAMPLE_CLUSTER_JOB_ID" \
  --export="ALL,CODE_DIR=$CODE_DIR,CODE_REVISION=$CODE_REVISION,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,D32_PATH_CONTRACT=v4,POLICY=$POLICY,CONTROL_DIR=$DATA_CONTROL_DIR,SOURCE_PLAN=$SOURCE_PLAN,CALIBRATION=$CALIBRATION,SAMPLE_RUN_DIR=$SAMPLE_RUN_DIR,AUDIT_OUTPUT_DIR=$DATA_CONTROL_DIR/sample_quality_audit" \
  runs/uhem_turkish_sample_quality_audit.sbatch)"

printf 'data_env=%s bootstrap=%s sample_objects=%s sample_buckets=%s sample_cluster=%s sample_qa=%s\n' \
  "$DATA_ENV_JOB_ID" "$BOOTSTRAP_JOB_ID" "$SAMPLE_OBJECT_JOB_ID" \
  "$SAMPLE_BUCKET_JOB_ID" "$SAMPLE_CLUSTER_JOB_ID" \
  "$SAMPLE_QUALITY_AUDIT_JOB_ID"
```

After the packed sample cluster completes, obtain a fresh effective BeeGFS
headroom value through the same UID/storage-pool CSV parser used by the storage
gate. This command runs the exact pinned
`beegfs-ctl --getquota --uid 4500 --storagepoolid=1 --csv` query and prints the
minimum of finite user-quota remaining and physical filesystem free bytes. Then
submit the sole writer-probe/report producer and the read-only quality audit:

```bash
: "${SAMPLE_CLUSTER_JOB_ID:?set the completed sample-cluster allocation ID}"
QUOTA_HEADROOM_BYTES="$("$DATA_PYTHON" scripts/d32_data_prep_operator.py \
  live-beegfs-headroom --recipe "$RECIPE" --work-dir "$NANOCHAT_BASE_DIR")"

WRITER_PROBE_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$SAMPLE_CLUSTER_JOB_ID" \
  --export="ALL,CODE_DIR=$CODE_DIR,CODE_REVISION=$CODE_REVISION,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,D32_PATH_CONTRACT=v4,POLICY=$POLICY,RECIPE=$RECIPE,CONTROL_DIR=$DATA_CONTROL_DIR,SOURCE_PLAN=$SOURCE_PLAN,CALIBRATION=$CALIBRATION,SAMPLE_RUN_DIR=$SAMPLE_RUN_DIR,BACKEND_RESOURCE_REPORT=$BACKEND_RESOURCE_REPORT,WRITER_PROBE=$WRITER_PROBE,QUOTA_HEADROOM_BYTES=$QUOTA_HEADROOM_BYTES" \
  runs/uhem_d32_data_prep_writer_probe.sbatch)"
```

Wait for the writer-probe and already captured QA allocations to complete.
Inspect the backend report and all four
accepted/rejected JSONL/plaintext files under
`$DATA_CONTROL_DIR/sample_quality_audit`; confirm the report's automatic gate
passed, every sampled retained source/rank is acceptable, and HPLT coverage is
explicitly empty. Only then
seal the two manual decisions with the pinned Turkish-data interpreter:

```bash
: "${REVIEWER:?set the reviewer identity}"
: "${REVIEWED_AT_UTC:?set RFC3339 UTC, for example 2026-08-21T12:34:56Z}"

"$DATA_PYTHON" scripts/d32_family_workflow.py seal-mixture-quality-approval \
  --policy "$POLICY" --source-plan "$SOURCE_PLAN" \
  --calibration "$CALIBRATION" \
  --audit-report "$DATA_CONTROL_DIR/sample_quality_audit/sample_quality_audit_report.json" \
  --reviewer "$REVIEWER" --reviewed-at-utc "$REVIEWED_AT_UTC" \
  --decision accepted --output "$MIXTURE_QUALITY_APPROVAL"

"$DATA_PYTHON" scripts/turkish_data_backend.py approve-resources \
  --policy "$POLICY" --source-plan "$SOURCE_PLAN" \
  --calibration "$CALIBRATION" --report "$BACKEND_RESOURCE_REPORT" \
  --mixture-quality-approval "$MIXTURE_QUALITY_APPROVAL" \
  --reviewer "$REVIEWER" --reviewed-at-utc "$REVIEWED_AT_UTC" \
  --decision accepted --output "$RESOURCE_APPROVAL"
```

Do not guess `PRODUCTION_DATA_NODES`. The selector builds ephemeral plans with
the existing lane balancer and evaluates every valid positive node count with
the same backend CPU, 1.35 safety, 48/24-hour, and 192-GiB arithmetic used by
the real gate. It seals the smallest passing count; the storage job recomputes
the complete evaluation and rejects a stale or hand-edited receipt:

```bash
PRODUCTION_DATA_NODES="$("$DATA_PYTHON" scripts/d32_data_prep_operator.py \
  select-production-nodes --recipe "$RECIPE" --policy "$POLICY" \
  --source-plan "$SOURCE_PLAN" --calibration "$CALIBRATION" \
  --sample-run-dir "$SAMPLE_RUN_DIR" \
  --backend-resource-report "$BACKEND_RESOURCE_REPORT" \
  --writer-probe "$WRITER_PROBE" --output "$PRODUCTION_NODE_SELECTION")"
```

Seal the pack plan, allocation ledger, storage sample, and live storage/quota
gate in one allocation. The seven IDs must be the exact completed allocations
that produced the named evidence:

```bash
: "${MACOCU_JOB_ID:?}"
: "${BOOTSTRAP_JOB_ID:?}"
: "${SAMPLE_OBJECT_JOB_ID:?}"
: "${SAMPLE_BUCKET_JOB_ID:?}"
: "${SAMPLE_CLUSTER_JOB_ID:?}"
: "${SAMPLE_QUALITY_AUDIT_JOB_ID:?}"
: "${WRITER_PROBE_JOB_ID:?}"

STORAGE_GATE_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$MACOCU_JOB_ID:$BOOTSTRAP_JOB_ID:$SAMPLE_OBJECT_JOB_ID:$SAMPLE_BUCKET_JOB_ID:$SAMPLE_CLUSTER_JOB_ID:$SAMPLE_QUALITY_AUDIT_JOB_ID:$WRITER_PROBE_JOB_ID" \
  --export="ALL,CODE_DIR=$CODE_DIR,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,MACOCU_JOB_ID=$MACOCU_JOB_ID,BOOTSTRAP_JOB_ID=$BOOTSTRAP_JOB_ID,SAMPLE_OBJECT_JOB_ID=$SAMPLE_OBJECT_JOB_ID,SAMPLE_BUCKET_JOB_ID=$SAMPLE_BUCKET_JOB_ID,SAMPLE_CLUSTER_JOB_ID=$SAMPLE_CLUSTER_JOB_ID,SAMPLE_QUALITY_AUDIT_JOB_ID=$SAMPLE_QUALITY_AUDIT_JOB_ID,WRITER_PROBE_JOB_ID=$WRITER_PROBE_JOB_ID,WRITER_PROBE=$WRITER_PROBE,RESOURCE_APPROVAL=$RESOURCE_APPROVAL,MIXTURE_QUALITY_APPROVAL=$MIXTURE_QUALITY_APPROVAL,PRODUCTION_NODE_SELECTION=$PRODUCTION_NODE_SELECTION,PRODUCTION_DATA_NODES=$PRODUCTION_DATA_NODES" \
  runs/uhem_d32_data_prep_storage_sample.sbatch)"
```

The gate reopens the same approval evidence and applies 1.35 exactly once to
the seven storage peaks and future CPU work. Historical MaCoCu/bootstrap/sample
allocations are not charged twice. Thirty-two serial lanes share each 128-CPU
object node, and billing uses the slowest lane per node.

Any mixture weight, source selector, or accepted-source policy change invalidates
the sample ranks, bounded quality audit, mixture-quality approval, resource
report/approval, pack plan, and storage gate. Re-run and re-seal that entire
lineage; never attach an old approval to a revised mixture.

### Executable post-gate data and tokenizer order

The passed `data_prep_storage_gate.json` is authorization for this exact chain,
not a reusable capacity estimate. Every production launcher re-queries BeeGFS
and UHeM quota immediately before work, verifies the sealed policy, source plan,
calibration, pack plan, resource approval, manual mixture-quality approval, and
gate, and refuses output directories outside the gated BeeGFS tree/device.

After `STORAGE_GATE_JOB_ID` completes successfully, derive the array size from
the sealed pack plan—not from the earlier shell variable—and submit the exact
object → bucket → cluster → pool dependency chain:

```bash
PRODUCTION_DATA_NODES="$("$DATA_PYTHON" -c \
  'import sys
from nanochat.experiment_manifest import load_json_strict, verify_manifest_hash
x = load_json_strict(sys.argv[1]); verify_manifest_hash(x); n = x.get("node_count")
if x.get("kind") != "d32_data_prep_production_pack_plan" or isinstance(n, bool) or not isinstance(n, int) or n <= 0:
    raise SystemExit("invalid production pack-plan identity/node count")
print(n)' \
  "$PACK_PLAN")"
OBJECT_ARRAY_MAX=$((PRODUCTION_DATA_NODES - 1))
COMMON_DATA_EXPORTS="CODE_DIR=$CODE_DIR,NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR,RESOURCE_APPROVAL=$RESOURCE_APPROVAL,MIXTURE_QUALITY_APPROVAL=$MIXTURE_QUALITY_APPROVAL,DATA_PREP_STORAGE_GATE=$DATA_PREP_STORAGE_GATE"

OBJECT_JOB_ID="$(sbatch --parsable "${SBATCH_ARRAY_LOG_ARGS[@]}" --array="0-$OBJECT_ARRAY_MAX" \
  --dependency="afterok:$STORAGE_GATE_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS" \
  runs/uhem_turkish_data_objects_packed_production.sbatch)"

OBJECT_NODE_RECEIPT_DIR="$DATA_RUN_DIR/packed_production_objects/node_receipts"
BUCKET_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" --dependency="afterok:$OBJECT_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS,OBJECT_NODE_RECEIPT_DIR=$OBJECT_NODE_RECEIPT_DIR" \
  runs/uhem_turkish_data_buckets_packed_production.sbatch)"

BUCKET_LAUNCH_RECEIPT="$DATA_RUN_DIR/packed_production_buckets/job$BUCKET_JOB_ID/launch_receipt.json"
CLUSTER_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" --dependency="afterok:$BUCKET_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS,SAMPLE=0,SOURCE_PLAN=$SOURCE_PLAN,CALIBRATION=$CALIBRATION,PACK_PLAN=$PACK_PLAN,DATA_RUN_DIR=$DATA_RUN_DIR,OBJECT_NODE_RECEIPT_DIR=$OBJECT_NODE_RECEIPT_DIR,BUCKET_LAUNCH_RECEIPT=$BUCKET_LAUNCH_RECEIPT" \
  runs/uhem_turkish_data_cluster.sbatch)"

CLUSTER_LAUNCH_RECEIPT="$DATA_RUN_DIR/packed_production_cluster/job$CLUSTER_JOB_ID/launch_receipt.json"
POOL_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" --dependency="afterok:$CLUSTER_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS,CLUSTER_LAUNCH_RECEIPT=$CLUSTER_LAUNCH_RECEIPT" \
  runs/uhem_turkish_production_pool.sbatch)"
```

When the pool job is complete, review both `$POOL_DIR/qa/qa_examples.jsonl` and
`$POOL_DIR/qa/qa_examples.txt`, then seal its approval (the output path is fixed
to `$POOL_DIR/qa/qa_approval.json`):

```bash
"$DATA_PYTHON" scripts/build_turkish_pretrain_corpus.py --policy "$POLICY" \
  approve-qa --pool-dir "$POOL_DIR" --reviewer "$REVIEWER" \
  --reviewed-at-utc "$REVIEWED_AT_UTC" --decision accepted
```

Prepare the main Nanochat environment, submit the deterministic sample, and run
the pinned baseline inventory/hash check before requesting tokenizer training.
The baseline check touches no output directory and validates the three policy-
pinned files `token_bytes.pt`, `tokenizer.pkl`, and `tokenizer_config.json`:

```bash
bash runs/uhem_d32_prepare_training_env.sh

TOKENIZER_SAMPLE_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" --dependency="afterok:$POOL_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS,CLUSTER_LAUNCH_RECEIPT=$CLUSTER_LAUNCH_RECEIPT" \
  runs/uhem_turkish_tokenizer_sample.sbatch)"

"$TRAIN_PYTHON" scripts/train_turkish_raw_bpe.py --policy "$POLICY" \
  --sample-dir "$TOKENIZER_SAMPLE_DIR" --output-dir "$TOKENIZER_DIR" \
  --quality-output-dir "$TOKENIZER_QUALITY_DIR" \
  --baseline-tokenizer-dir "$BASELINE_TOKENIZER_DIR" --baseline-preflight-only

TOKENIZER_TRAIN_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$TOKENIZER_SAMPLE_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS,CLUSTER_LAUNCH_RECEIPT=$CLUSTER_LAUNCH_RECEIPT,BASELINE_TOKENIZER_DIR=$BASELINE_TOKENIZER_DIR" \
  runs/uhem_turkish_tokenizer_train.sbatch)"
```

After training completes, inspect
`$TOKENIZER_QUALITY_DIR/quality_report.json`. Automatic quality must already
pass; only then submit the manual accepted decision and packing preflight:

```bash
TOKENIZER_QUALITY_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$TOKENIZER_TRAIN_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS,CLUSTER_LAUNCH_RECEIPT=$CLUSTER_LAUNCH_RECEIPT,TOKENIZER_REVIEWER=$REVIEWER,TOKENIZER_REVIEWED_AT_UTC=$REVIEWED_AT_UTC,TOKENIZER_DECISION=accepted" \
  runs/uhem_turkish_tokenizer_quality.sbatch)"

PACKING_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" \
  --dependency="afterok:$TOKENIZER_QUALITY_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS,CLUSTER_LAUNCH_RECEIPT=$CLUSTER_LAUNCH_RECEIPT" \
  runs/uhem_turkish_packing_preflight.sbatch)"
```

Review the packing report and explicitly seal its recommendation. Then extract
the exact approved target from the verified approval and launch the finalizer.
The allocation itself recomputes live BeeGFS headroom immediately before
materialization; no pre-submit `QUOTA_HEADROOM_BYTES` is accepted:

```bash
"$TRAIN_PYTHON" scripts/build_turkish_pretrain_corpus.py --policy "$POLICY" \
  approve-packing-preflight \
  --report "$PACKING_CONTROL_DIR/packing_preflight_report.json" \
  --output "$PACKING_CONTROL_DIR/packing_preflight_approval.json" \
  --reviewer "$REVIEWER" --reviewed-at-utc "$REVIEWED_AT_UTC" \
  --decision accepted

APPROVED_SOURCE_TOKENS="$("$TRAIN_PYTHON" -c \
  'import sys
from nanochat.experiment_manifest import load_json_strict, verify_manifest_hash
x = load_json_strict(sys.argv[1]); verify_manifest_hash(x); n = x.get("approved_source_token_target")
if x.get("kind") != "turkish_packing_preflight_approval" or x.get("decision") != "accepted" or isinstance(n, bool) or not isinstance(n, int) or n <= 0:
    raise SystemExit("invalid accepted packing approval/source-token target")
print(n)' \
  "$PACKING_CONTROL_DIR/packing_preflight_approval.json")"
FINAL_CORPUS_JOB_ID="$(sbatch --parsable "${SBATCH_LOG_ARGS[@]}" --dependency="afterok:$PACKING_JOB_ID" \
  --export="ALL,$COMMON_DATA_EXPORTS,CLUSTER_LAUNCH_RECEIPT=$CLUSTER_LAUNCH_RECEIPT,APPROVED_SOURCE_TOKENS=$APPROVED_SOURCE_TOKENS" \
  runs/uhem_turkish_corpus_finalize.sbatch)"
```

The launchers themselves enforce the canonical pool, tokenizer sample/package/
quality, packing-control, and final-corpus paths above. The final corpus must
match the exact pool QA, tokenizer quality, packing approval, and production
chain before family preflight can pass. Preserve `tokenizer.pkl`, canonical
`tokenizer.tiktoken`, `token_bytes.pt`, configuration, receipts, report, and
approval as one upload inventory.

Never continue from a partially written output directory. Leave it quarantined
for audit, choose a new output/launch directory, and keep all prior artifacts;
these scripts intentionally refuse overwrite and never auto-delete data. The
family `preflight` command now requires `--cluster-launch-receipt`, and the HF
export requires `CLUSTER_LAUNCH_RECEIPT`, so a merely well-formed 64-hex hash
cannot stand in for the completed production allocation.

Before creating the family preflight, run
`bash runs/uhem_d32_prepare_training_env.sh`. It requires uv 0.11.29 and
consumes the byte-exact upstream `pyproject.toml` and `uv.lock` with
`uv sync --frozen --extra gpu`. The frozen mode is intentional: upstream's
relative seven-day `exclude-newer` lock metadata eventually makes `--locked`
request a re-resolution, while the recipe and preflight independently pin both
input-file hashes.

`runs/uhem_submit_d32_family.sh --plan` performs no submission and reports the
gate order and currently present artifacts. It exposes only bounded static,
proxy, signal/resume, and smoke submissions. It cannot submit production.

The Hugging Face family export defaults to a private, dry-run verification via
`runs/uhem_d32_family_upload.sbatch`. Stable fork transactions always include
optimizer/loader/RNG state, and all three local cooled-final transactions remain
fully resumable. Before export, inspect the terminal fixed-validation BPB and
complete validation evidence for s12, s20, and s40, then seal one explicit
manual decision:

```bash
.venv/bin/python scripts/d32_family_workflow.py seal-final-quality-approval \
  --recipe "$RECIPE" --preflight-receipt "$FAMILY_CONTROL_DIR/preflight.json" \
  --gate "$FAMILY_CONTROL_DIR/production_topology_gate.json" \
  --base-dir "$NANOCHAT_BASE_DIR" --lineage-dir "$FAMILY_CONTROL_DIR/lineage" \
  --reviewer "$REVIEWER" --reviewed-at-utc "$REVIEWED_AT_UTC" \
  --decision accepted --notes "$NOTES" \
  --output "$FAMILY_CONTROL_DIR/final_quality_approval.json"
```

No automatic BPB cutoff is invented: a rejected decision is retained for
audit, while publication fails closed unless the exact checkpoint, lineage,
curve-log, and full-validation evidence has an accepted approval. The export
also includes and hash-binds the MaCoCu, MOT, and ParlaMint preparation
manifests plus the source plan, backend calibration, backend resource report,
mixture/resource approvals, production pack plan, writer probe, storage
sample/gate, and the fixed ws8/ws16 smoke receipts used by the topology gate.
The MaCoCu export check rehashes every prepared shard and the retained upstream
gzip; a valid manifest by itself is insufficient.

The bounded sample-audit report, copied cluster/object/bucket launch receipts,
and accepted/rejected JSONL and plaintext review examples are exported beneath
the mixture approval's original relative evidence root. This makes its
relative links usable after download. The live sample Parquets and the full
training corpus are intentionally not copied into the model repository; their
immutable hashes and source URIs remain in the archived receipts. Set
`BACKEND_RESOURCE_REPORT`, `DATA_PREP_STORAGE_SAMPLE`, and `WRITER_PROBE` only
when their defaults under `control/data_v4` do not apply.

The `reviewer` fields are trusted-operator self-attestations, not
cryptographically authenticated identities. Canonical hashes bind each
decision to exact evidence and reject later drift or stale substitution, but
they cannot prevent a person or process with write access to the control tree
from creating a new self-consistent approval under another reviewer name.
Restrict write access, preserve scheduler/audit logs, and perform publication
from the reviewed operator account. A detached signature would require a
separate, explicitly designed key-verification policy; this workflow does not
claim to provide one.

Final optimizer inclusion or omission must be an explicit
`FINAL_OPTIMIZER_POLICY` choice, and remote mutation additionally requires
`DRY_RUN=0`.
