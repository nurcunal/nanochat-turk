# Study summary

This study compares **12 tokenizers**: raw BPE and MorphBPE with TRmorph,
Zemberek and TurkishDelightNLP boundaries at **32,768 / 65,536 / 131,072**
vocabulary sizes. The original model matrix has **11 completed models**;
32k TurkishDelight remains tokenizer-only. Work continues on **main**, and all
historical branch commits and measured evidence are retained.

- **Fresh intrinsic assessment:** all requested metrics are measured on matched
  50,000-document training and validation samples, with 500,000 word occurrences
  per tokenizer/split. The validation shard belongs to the original tokenizer
  split; later all-shard d32 training also used this text.
- **Morphology:** TRmorph leads validation morpheme preservation and boundary F1
  at all three vocabulary sizes. At 32k, preservation is **44.30% versus 31.57%**
  for raw BPE, and boundary F1 is **38.38% versus 24.96%**. The reference is
  automatic TRmorph, not independent human-gold annotation. The older pairwise
  consistency metric has a different winner at 128k on training/full-corpus data:
  TurkishDelight scores **0.3971**, versus TRmorph's **0.3786**.
- **Compression:** raw BPE uses the fewest document tokens per word within every
  vocabulary tier. Isolated-word fertility is a different objective: Zemberek
  is slightly best at 32k validation; TRmorph is best at 64k/128k.
- **Pretraining:** raw BPE has the lowest archived validation BPB in every tier
  (**0.6232 / 0.6409 / 0.6749**). Architecture is held fixed within each tier;
  across tiers depth changes d20/d16/d12. Equal token budgets do not establish
  equal raw-text exposure or identical evaluation spans.
- **Extrinsic results:** the original CETVEL rankings are **provisional** because
  the evaluator had generation and likelihood-boundary bugs. Correcting the
  evaluator changes the same d20 checkpoint's XQuAD F1 from **3.0985 to 11.3369**.
  A complete corrected eleven-model rerun remains outstanding.
- **Later experiments:** the d20 continuation and the **dsai-3b-base** d32 model
  change training budget or architecture. d32's **0.519040 BPB overlaps training
  text** and cannot establish held-out superiority. Its **37.1709/100**
  table-compatible CETVEL receipt has its own scoring protocol. SFT screens use
  only raw BPE 32k and selection probes, so they do not add a multi-tokenizer SFT
  comparison.

The sections below preserve both current and historical measurements, including
full-corpus, 100k/1M, external-tokenizer, analyzer and SFT evidence. There is one
fresh tokenizer report/export for job **558060**; its formerly separate matched
report supplied additional diagnostic columns that are now included here.
