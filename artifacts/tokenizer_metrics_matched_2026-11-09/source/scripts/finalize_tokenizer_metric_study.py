"""Verify matched samples and package fresh tokenizer measurements with hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path


def finalize(directory: Path, source_manifest: Path) -> dict:
    manifest = {
        "schema_version": 1,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": json.loads(source_manifest.read_text()),
        "environment": json.loads((directory / "environment.json").read_text()),
        "sample": json.loads((directory / "sample_manifest.json").read_text()),
        "reference": "automatic TRmorph segmentation; validation is a reserved source shard, not independent human morphology annotation",
        "word_cap_per_tokenizer_per_split": 500000,
        "morphology_cap_per_tokenizer_per_split": 500000,
        "splits": {},
    }
    for split in ("train", "val"):
        rows, signatures = [], set()
        for path in sorted((directory / split).glob("*_metrics.json")):
            raw = path.read_bytes()
            data = json.loads(raw)
            if data["morphology"]["consistency"]["clustering"] != "sklearn_minibatch_kmeans":
                raise ValueError(f"{path}: expected sklearn clustering, refusing fallback comparison")
            preservation = data["morphology"]["preservation"]
            coverage = data["coverage"]
            signatures.add((data["split"], data["docs"], data["bytes"], data["chars"], data["words"], coverage["word_sequence_sha256"], preservation["reference_sequence_sha256"]))
            rows.append({
                "source": path.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
                "tokenizer_name": data["tokenizer_name"], "vocab_size": data["vocabulary"]["vocab_size"],
                "split": data["split"], "docs": data["docs"],
                "document_tokens_per_word": data["tokens_per_word"],
                "isolated_word_tokens_per_word": data["word_fertility_isolated"]["tokens_per_word"],
                "coverage": coverage, "morpheme_preservation": preservation,
                "document_morph_boundaries": data["morph_boundary"], "paper_metrics": data["paper_metrics"],
            })
        if len(rows) != 12 or len(signatures) != 1:
            raise ValueError(f"{split}: expected 12 matched measurements, got {len(rows)} rows and {len(signatures)} sample signatures")
        signature = next(iter(signatures))
        source_sample = manifest["sample"]["splits"][split]
        if signature[:5] != (split, source_sample["documents"], source_sample["raw_bytes"], source_sample["raw_chars"], source_sample["regex_words"]):
            raise ValueError(f"{split}: metric corpus totals differ from materialized source sample")
        payload = {"schema_version": 1, "comparison": "matched_reference_and_word_caps", "rows": rows}
        output = directory / split / "requested_metrics.json"
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"measurements": len(rows), "word_sequence_sha256": signature[5], "reference_sequence_sha256": signature[6], "requested_metrics_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "files": {row["source"]: row["source_sha256"] for row in rows}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.directory, args.source_manifest), indent=2, ensure_ascii=False))
