# Historical normalizing-tokenizer diagnostics

These three original JSON files were previously published in Git commit `2ff53058540b5d2d00a9e832e7f02a19582faed4` and removed when the main comparison narrowed to lossless raw-text BPE references (`2e6a714a18e0ed9b53f4c82c6d09696da2122307`). They are recovered byte-for-byte for completeness; the manifest records immutable GitHub source URLs and SHA-256 hashes.

BERTurk, TURNA and VBART were measured on 50,000 training documents and 500,000 isolated word occurrences. Their observed document roundtrip failure rates are 99.292%, 95.214% and 95.214%. Normalization/decoding behavior and unrecorded UNK incidence make token counts and one-token-word counts non-equivalent to lossless BPE compression and coverage. Their recorded boundary-crossing values are preserved for historical inspection, not used as a comparable morphology ranking. No trained-model results or later MorphBPE edit/consistency metrics are present in these files.

See the separate historical section in [archived tokenizer results](../../docs/study_results/tokenizer_results.md).
