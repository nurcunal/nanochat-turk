# d32 native-text candidate audit

This note records bounded source-selection evidence for the fresh, PDF/OCR-free
Turkish d32 production lineage. It is not a production approval. All candidates
must still pass immutable acquisition, independent Turkish/no-code filtering,
global cross-source deduplication, manual mixture review, and capacity gates.

## OzenliDerlem exploratory audit

- Dataset: `turkish-nlp-suite/OzenliDerlem`
- Pinned Hugging Face commit: `539fc60c4565fc556fbb5b1dab6cb872e7fb5db6`
- Declared origin: handpicked native web pages; no PDF/OCR source is selected
  by this audit.
- Declared scale: 1,391,239 documents, 4.6 GB of text, and 557
  million words across eleven topical configurations.
- Local sample: complete contents of fourteen pinned JSONL files spanning
  travel, culture, fairy tales, film, lifestyle, books, viral media, and
  popular science; 3,722 documents total. Files were deliberately selected
  across small and mid-sized publishers, so this is a source-diversity probe,
  not an estimate with a statistical confidence interval for the full corpus.
- Manual sample: the two documents with the smallest SHA-256 key per sampled
  file, where the key binds URL and full text; 28 documents total.

The current structural audit accepted 3,709/3,722 documents. The new strict
encoding backstop rejected three additional lifestyle documents containing six
Unicode replacement characters, leaving 3,706/3,722 accepted (99.57%). This
high automatic pass rate is not evidence of high semantic quality. Manual
inspection still found, among other issues:

- crypto and generic entertainment advertisements under film-review sources;
- medical/cosmetic marketing claims in lifestyle material;
- a Kurdish product description that passed the lightweight Turkish heuristic;
- awkward, apparently spun or translated viral-media prose;
- product-catalog and promotional copy mixed with book commentary; and
- templated/repetitive modern fairy-tale prose of uncertain authorship.

Decision: **do not ingest the aggregate dataset or any complete configuration**.
The following are bounded candidates only:

- selected first-person travel publishers, for colloquial narrative;
- selected culture/literature publishers, for broad general knowledge;
- selected film-review publishers after path-level advertisement exclusion; and
- selected popular-science publishers after publisher-level factual review.

`TeknoYazilar` is excluded by the no-coding/no-technical-domain requirement.
`ViralMedya`, `SusluTrendler`, `Serzenisler`, `MasalMasal`, and the product-heavy
parts of `YazarinKaleminden` are excluded unless later evidence overturns the
observed noise, marketing, privacy, or provenance risks. `Havadis` is not an
automatic addition: it must demonstrate useful nonduplicate coverage beyond
the already selected news/web sources and remain capped so news does not crowd
out everyday Turkish.

Even the full declared 557-million-word corpus is only a small supplement to a
67.1B-token run. Any admitted slice is therefore a diversity anchor, not the
scale backbone. Its value must be judged by manual quality and unique retained
tokens after global deduplication, not raw row count.

## Production invariants derived from the audit

- Every selected source must declare `text_origin` as exactly
  `born_digital_text` or `structured_text`.
- PDF-extracted, OCR-derived, mixed, missing, and unknown origins fail closed.
- Production documents permit zero Unicode replacement characters, known
  Turkish mojibake sequences, C1 controls, or surrogate code points after the
  normalization pipeline.
- Passing automatic language/structure filters never substitutes for bounded
  manual semantic review at source-file granularity.
- Candidate aggregations are decomposed into explicit publisher/source files;
  provenance, rights, quality, and removal decisions remain source-specific.

## ForumSohbetleri exploratory audit

- Dataset: `turkish-nlp-suite/ForumSohbetleri`
- Pinned Hugging Face commit: `eba610eec21598d1c32c066938ebb5770ffc8a87`
- Declared origin: native Turkish forum threads stored as URL plus an ordered
  list of posts; no PDF/OCR extraction is involved.
- Local sample: three fixed 2 MB byte ranges from one pinned shard in each of
  `turkiyeforum`, `forumum`, `kadinlarklubu`, and `memurlar`; 24 MB total.
  Complete JSONL records wholly contained in each range yielded 7,988 threads
  and 87,989 posts. Byte-range sampling is reproducible and covers the start,
  middle, and end of each selected shard, but is not a thread-uniform estimate
  with a statistical confidence interval.

The strict structural/encoding audit accepted 7,707/7,988 joined threads
(96.48%). Manual review again showed that this is not a semantic-quality score:

- `turkiyeforum` contains copied articles/poetry, persistent user signatures,
  markup, short chat fragments, and 36 encoding-corrupt threads in this probe;
- `forumum` is dominated in sampled ranges by single-post questions, copied
  news/product/lifestyle pages, and systematic loss of Turkish diacritics;
- `kadinlarklubu` contains authentic informal dialogue, but also extreme typo
  density, signatures/usernames, sensitive medical and family disclosures, and
  long threads that are difficult to de-identify reliably; and
- `memurlar` contains useful multi-party everyday discussion, but is narrowly
  concentrated around public employment/appointments and includes insults,
  copied material, scores/rankings, and other privacy/noise risks.

Decision: **no ForumSohbetleri configuration is currently approved**. Hardware,
webmaster, and security-oriented configurations are excluded outright by the
no-code/no-specialized-technical-data requirement. `turkiyeforum`, `forumum`,
and `kadinlarklubu` are excluded on the observed quality, encoding, or privacy
evidence. `memurlar` remains only a bounded research candidate: it may receive
at most a very small share if a deterministic conversation selector (multiple
substantive posts, bounded post dominance/length, Turkish per-post checks,
PII/no-code gates, and copied-article rejection) passes a new independent
manual audit. Until then, the already audited MaCoCu `Forum` and
`Opinion/Argumentation` genres are the preferred conversational source.

## Web-scale source conclusions

### FineWeb2-HQ Turkish

- Pinned Hugging Face commit: `c0c06e94` (full commit recorded in the source
  manifest before any production fetch).
- Upstream construction: the highest-scoring ten percent selected by the
  FineWeb2-HQ MLP/XLM-R quality classifier. The published Turkish release is
  8,578,808 documents / 100 GB.
- Local probe: Parquet ranks 31, 57, and 83 contained 261,931 candidate
  documents and 750,329,443 candidate characters in total.
- Automatic bounded sample: 182/192 documents passed the current structural,
  language, and no-code gates.
- Manual sample: roughly five of 24 documents appeared translated or
  machine-translated and roughly seven of 24 showed scraper/SEO overlap. No
  code-heavy material was observed, but very little was genuinely
  conversational.
- Observed footer score ranges were compressed near the selection floor
  (approximately 0.9006 to 1.00001), so a second arbitrary score threshold is
  not treated as a quality proof.

Decision: **retain FineWeb2-HQ only as a scale candidate**, not as the gold or
conversational tier. Admission and weight require a deterministic score/host
stratified audit plus measured unique retained tokens after global
deduplication. The upstream paper evaluates its classifier in several
languages but not Turkish and explicitly cautions that cross-language transfer
can vary; Turkish quality therefore remains an empirical local question.

### HPLT Turkish

The audited Web Document Segmentation bins are not monotonically clean:

- WDS 8 retained 57/64 sampled documents and was the cleanest inspected bin,
  although dominated by news/lifestyle prose rather than dialogue;
- WDS 9 retained 58/64 automatically, but manual inspection found
  technical/code material, product SEO, advertorials, and cookie text; and
- WDS 10 retained only 48/64 and showed boilerplate, SEO, document-hosting,
  promotional, and adult-page contamination.

Decision: exclude WDS 10. Keep HPLT as a bounded supporting candidate capped
at 20--25% of the eventual mix, with an initial target of 15--20% from WDS 8
and 5--7% from WDS 9. Exact weights remain contingent on retained unique-token
yield and global deduplication; nominally higher WDS rank does not override the
Turkish audit.

### MaCoCu Turkish

The bounded audit found more natural Turkish and no sampled code or obvious
machine translation, but the source is still news-heavy: approximately 59%
`News`, 13.5% `Information/Explanation`, 12% `Opinion/Argumentation`, 10%
`Instruction`, and 5.7% `Forum` in the observed sample. The current automatic
gate retained 149/192 documents.

Decision: MaCoCu is the preferred web-scale everyday-Turkish candidate.
`Forum` and `Opinion/Argumentation` form its conversational lane;
`Information/Explanation`, `Instruction`, and a capped `News` slice provide
general knowledge. Promotion, prose, legal, other, and mixed genres remain
excluded. Genre labels are selectors, not substitutes for document-level
quality and deduplication gates.
