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
- The v2 policy replaces Cosmos with the official Turkish MaCoCu-Genre release
  (CLARIN handle `11356/1969`). A one-time CPU job verifies its 14,448,949,031-
  byte gzip and MD5 `abe376c21256798ded30e54770666aa0`, then streams it into
  deterministic sealed zstd JSONL shards. Object-array ranks consume those
  shards; they never redownload or independently decompress the full gzip.
- MaCoCu conversation selects only `Forum` and `Opinion/Argumentation`; general
  selects only `Information/Explanation`, `Instruction`, and `News`. Promotion,
  prose/lyrics, legal, other, and mixed genres are excluded. HPLT is not treated
  as a conversation source because the bounded register audit did not support
  that interpretation.
- The v2 sampling mix is a candidate, not a final production approval: HPLT
  general 35%, FineWeb2-HQ 30%, raw FineWeb2 12%, MaCoCu conversation/general
  6%/4%, FinePDF 8%, and FineWiki 5%. Freeze or revise it only after the bounded
  accepted-token-yield, dedup-overlap, language, quality, and genre audit.
- The project-local v2 audit rejects substantial mixed-script text only when it
  exceeds both 32 letters and 2% of alphabetic text. Contextual APK/download,
  commerce, cookie-interface, legal-policy, taxonomy/search, and strong SEO
  template gates use narrowly calibrated phrases and URL context; generic words
  such as “uygulama”, “ürün”, “fiyat”, “KVKK”, or “gizlilik” do not reject text.
- The finite best-fit simulation must prove no epoch wrap through 40x plus a
  2% margin for both ws8 and ws16. It also checks the retained post-cropping
  mixture, rather than treating raw source-token totals as usable capacity.
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

Prepare the isolated CPU corpus environment with
`sbatch runs/uhem_turkish_prepare_data_env.sbatch`; it accepts only uv 0.11.29
and the reviewed Turkish-data project and lock hashes.
Then run `sbatch runs/uhem_turkish_data_prepare_macocu.sbatch` once on the shared
filesystem. It atomically writes the retained verified gzip, reasonably sized
prepared shards, per-shard SHA-256/row/genre accounting, and `manifest.json`.
Only after that succeeds, run `sbatch runs/uhem_turkish_data_bootstrap.sbatch`
to bind the preparation manifest into the immutable v2 source plan and seal the
pinned GlotLID model/calibration and deterministic sample ranks. The v1 policy
and its existing audit artifacts remain separate; never reuse a v1 plan,
calibration, approval, receipt, pool, tokenizer, or packing receipt with v2.

The object resource sample runs through
`runs/uhem_turkish_data_objects_packed_sample.sbatch`: one exclusive cpu2dq
allocation contains thirty-two 4-CPU worker lanes, and every sample rank belongs to
exactly one serial lane. The post-cluster writer-probe job is the sole producer
of the backend report and deliberately seals it with safety factor 1.0. Run
`runs/uhem_turkish_sample_quality_audit.sbatch` against that same completed
sample/cluster lineage, inspect its accepted and rejected JSONL/plaintext
examples, and explicitly seal the manual mixture-quality approval. Seal the
resource approval only after that approval exists, because the resource
decision binds it. Then submit `runs/uhem_d32_data_prep_storage_sample.sbatch`
with an `afterok` dependency and the exact MaCoCu, bootstrap, object-sample,
bucket-sample, cluster, quality-audit, and writer-probe allocation IDs. It also
requires `RESOURCE_APPROVAL`, `MIXTURE_QUALITY_APPROVAL`, and an explicit
`PRODUCTION_DATA_NODES`; it consumes the already sealed backend report and
does not create or auto-approve either decision.
The family gate then
applies the recipe's 1.35 factor exactly once to seven explicit projected
storage peaks and to future production CPU work. MaCoCu preparation, bootstrap,
sample, and writer-probe allocations remain sealed historical evidence and are
not charged against the future-work projection a second time. Production
source-object work uses an explicit node plan: thirty-two serial lanes share each
128-CPU node, and CPU-saat is projected from the slowest lane on each node,
not from thirty-two independent node charges.

Any mixture weight, source selector, or accepted-source policy change invalidates
the sample ranks, bounded quality audit, mixture-quality approval, resource
report/approval, pack plan, and storage gate. Re-run and re-seal that entire
lineage; never attach an old approval to a revised mixture.

### Executable post-gate data order

The passed `data_prep_storage_gate.json` is authorization for this exact chain,
not a reusable capacity estimate. Every production launcher re-queries BeeGFS
and UHeM quota immediately before work, verifies the sealed policy, source plan,
calibration, pack plan, resource approval, manual mixture-quality approval, and
gate, and refuses output directories outside the gated BeeGFS tree/device.

1. Submit `runs/uhem_turkish_data_objects_packed_production.sbatch` as the exact
   zero-based array declared by `production_source_pack_plan.json`. Each array
   allocation is one 128-CPU node with 32 tasks × 4 CPUs and a 48-hour maximum.
   Set `RESOURCE_APPROVAL`, `MIXTURE_QUALITY_APPROVAL`,
   `DATA_PREP_STORAGE_GATE`, and the standard `SOURCE_PLAN`, `CALIBRATION`,
   `PACK_PLAN`, and `DATA_RUN_DIR` overrides only when needed.
2. After every object-node receipt exists, run
   `runs/uhem_turkish_data_buckets_packed_production.sbatch`. It is one node,
   14 tasks × 8 CPUs, with a 24-hour maximum. Set
   `OBJECT_NODE_RECEIPT_DIR` to the completed packed-object receipt directory.
3. Run `runs/uhem_turkish_data_cluster.sbatch` with `SAMPLE=0`, the exact object
   receipt directory, and `BUCKET_LAUNCH_RECEIPT`. Its single 192-GiB node has
   a 48-hour maximum. Submission is forbidden unless sampled/projected wall
   time and measured/projected peak RSS, after the one 1.35 safety factor, fit
   those limits. Preserve its `CLUSTER_LAUNCH_RECEIPT`; all later artifacts and
   the family preflight bind its canonical hash.
4. Run `runs/uhem_turkish_production_pool.sbatch` with that cluster-launch
   receipt. It seals source/backend receipts and materializes the production
   pool on one node with a 48-hour maximum. Review `qa/qa_examples.jsonl` and
   `qa/qa_examples.txt`, then create the explicit accepted pool QA approval.
5. Run, in order, `runs/uhem_turkish_tokenizer_sample.sbatch` (12 hours),
   `runs/uhem_turkish_tokenizer_train.sbatch` (24 hours), and—only after human
   review of `quality_report.json`—`runs/uhem_turkish_tokenizer_quality.sbatch`
   (12-hour ceiling). Each command requires `POOL_DIR` and the same exact
   `CLUSTER_LAUNCH_RECEIPT`; the sample, package, training receipt, quality
   report, and approval carry the parent-pool, QA, policy, and production chain.
   Before tokenizer training or quality review, prepare the main Nanochat
   `.venv` with `bash runs/uhem_d32_prepare_training_env.sh`; the separate
   Turkish-data environment does not contain the tokenizer runtime. Set
   `BASELINE_TOKENIZER_DIR` defaults to the pinned prior tokenizer at
   `/ari/users/nunal/nanochat-turk-d20-bpe32k/tokenizers/bpe_32768`; its exact
   three-file inventory is verified before any output directory is created.
   The new `tr_general_raw_bpe_32k_v2` tokenizer is trained from the
   deterministic full-pool row-group traversal, while its
   fixed val-only holdout contains at least 50,000 documents or 128 MiB and
   targets at least 32 documents in every available mixture/source/register
   stratum (recording explicit source insufficiency). Automatic quality
   must pass before a human can accept it. Preserve `tokenizer.pkl`, the
   canonical `tokenizer.tiktoken`, `token_bytes.pt`, configuration, receipts,
   report, and approval as one upload inventory.
6. Run `runs/uhem_turkish_packing_preflight.sbatch` (12 hours), manually review
   and seal `packing_preflight_approval.json`, then run
   `runs/uhem_turkish_corpus_finalize.sbatch` (48 hours). Set
   `APPROVED_SOURCE_TOKENS` to the exact approved source-token target in the
   packing approval and obtain a fresh scheduler/project quota reading for
   `QUOTA_HEADROOM_BYTES` immediately before submitting the finalizer. The final corpus must
   match the exact pool, QA approval, tokenizer, quality approval, packing
   approval, and production chain before family preflight can pass.

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
optimizer/loader/RNG state. Final optimizer inclusion or omission must be an
explicit `FINAL_OPTIMIZER_POLICY` choice, and remote mutation additionally
requires `DRY_RUN=0`.
