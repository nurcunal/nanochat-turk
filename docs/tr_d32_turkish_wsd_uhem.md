# Turkish d32 WSD family on UHeM

This workflow prepares one shared d32 trunk and three independently cooled,
stable base-model checkpoints: s12 at 20,132,659,200 scheduled positions, s20
at 33,554,432,000, and s40 at 67,108,864,000. The s12 and s20 cooldowns fork
immutable trunk checkpoints; after each fork, the trunk continues toward the
next boundary. They are therefore three final models without paying for three
independent pretraining runs.

No production submission is exposed by the repository. The operator and user
must review the sealed gates and topology decision first.

## Frozen model and data contract

- d32, width 2,048, 16 query/KV heads, head dimension 128, context 2,048.
- Vocabulary 32,768 from the new Turkish-only tokenizer package.
- 2,818,575,450 total parameters; 1,677,724,672 Nanochat scaling parameters.
- Global batch is fixed at 2,097,152 scheduled positions for ws8 and ws16.
- Data is Turkish-only, code corpora are forbidden, and validation is a fixed,
  finite whole-document manifest.
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
    submission. The repository deliberately provides no production-submit
    mode.

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
Then run `sbatch runs/uhem_turkish_data_bootstrap.sbatch` to seal the immutable
source plan, pinned GlotLID model/calibration, and deterministic sample ranks.

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
