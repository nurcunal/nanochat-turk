"""
Measure tokenizer compression, fertility, and encode throughput on the dataset.

This is intentionally tokenizer-implementation agnostic: future MorphBPE or
SentencePiece tokenizers only need to expose the same nanochat tokenizer methods.
"""

import argparse
import json
import os
import re
import time

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer, get_tokenizer_dir, get_tokenizer_name


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def load_tokenizer(tokenizer_dir):
    if tokenizer_dir:
        return RustBPETokenizer.from_directory(tokenizer_dir)
    return get_tokenizer()


def main():
    parser = argparse.ArgumentParser(description="Tokenizer ablation metrics")
    parser.add_argument("--tokenizer-dir", type=str, default="", help="Tokenizer directory. Default = active nanochat tokenizer.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir or get_tokenizer_dir()
    tokenizer = load_tokenizer(args.tokenizer_dir)

    docs = []
    for batch in parquets_iter_batched(split=args.split):
        for doc in batch:
            docs.append(doc)
            if len(docs) >= args.max_docs:
                break
        if len(docs) >= args.max_docs:
            break

    if not docs:
        raise RuntimeError("No documents found for tokenizer metrics")

    t0 = time.time()
    encoded = tokenizer.encode(docs, num_threads=args.num_threads)
    elapsed = time.time() - t0

    total_tokens = sum(len(ids) for ids in encoded)
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_chars = sum(len(doc) for doc in docs)
    total_words = sum(len(WORD_RE.findall(doc)) for doc in docs)

    metrics = {
        "tokenizer_name": get_tokenizer_name(),
        "tokenizer_dir": tokenizer_dir,
        "split": args.split,
        "docs": len(docs),
        "bytes": total_bytes,
        "chars": total_chars,
        "words": total_words,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "chars_per_token": total_chars / total_tokens,
        "tokens_per_word": total_tokens / max(total_words, 1),
        "encode_seconds": elapsed,
        "encode_docs_per_sec": len(docs) / elapsed,
        "encode_tokens_per_sec": total_tokens / elapsed,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    from nanochat.report import get_report
    get_report().log(section="Tokenizer metrics", data=[metrics])


if __name__ == "__main__":
    main()
