"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import json
import time
import argparse
import torch
import pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import DATA_DIR, list_parquet_files
from nanochat.morphology import (
    MORPHEME_BOUNDARY,
    MorphBPEIteratorStats,
    display_boundary,
    iter_morphbpe_training_stream,
    strip_morpheme_boundaries,
)
from nanochat.tokenizer import get_tokenizer_dir, get_tokenizer_name

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
parser.add_argument('--tokenizer-name', type=str, default=None, help='Optional tokenizer experiment name. Writes to $NANOCHAT_BASE_DIR/tokenizers/<name>.')
parser.add_argument('--implementation', type=str, default='bpe', choices=['bpe', 'morphbpe', 'preseg_bpe'], help='Tokenizer implementation.')
parser.add_argument('--data-dir', type=str, default='', help='Optional parquet directory. Default = nanochat dataset dir.')
parser.add_argument('--text-column', type=str, default=os.environ.get("NANOCHAT_TEXT_COLUMN", "text"), help='Parquet text column to train on.')
parser.add_argument('--morph-boundary', type=str, default=MORPHEME_BOUNDARY, help='Internal morpheme boundary marker for --implementation morphbpe/preseg_bpe.')
parser.add_argument('--allow-missing-boundary', action='store_true', help='Allow morphology-aware training docs without the boundary marker.')
args = parser.parse_args()
if args.tokenizer_name:
    os.environ["NANOCHAT_TOKENIZER_NAME"] = args.tokenizer_name
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
print(f"tokenizer_name: {get_tokenizer_name()}")
print(f"implementation: {args.implementation}")
resolved_data_dir = args.data_dir or DATA_DIR
print(f"data_dir: {resolved_data_dir}")
print(f"text_column: {args.text_column}")
if args.implementation in ("morphbpe", "preseg_bpe"):
    print(f"morph_boundary: {display_boundary(args.morph_boundary)}")

# -----------------------------------------------------------------------------
# Text iterator

def parquet_text_batches(split="train"):
    parquet_paths = list_parquet_files(args.data_dir or None)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run `python -m nanochat.dataset -n 8`?"
    parquet_paths = parquet_paths[:-1] if split == "train" and len(parquet_paths) > 1 else parquet_paths
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        if args.text_column not in pf.schema_arrow.names:
            raise KeyError(f"Column {args.text_column!r} not found in {filepath}; columns: {pf.schema_arrow.names}")
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=[args.text_column])
            yield rg.column(args.text_column).to_pylist()


iterator_stats = {
    "docs": 0,
    "chars": 0,
    "docs_with_boundary": 0,
}
morphbpe_stats = MorphBPEIteratorStats()


def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquet_text_batches(split="train"):
        for doc in batch:
            doc_text = "" if doc is None else str(doc)
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            iterator_stats["docs"] += 1
            iterator_stats["chars"] += len(doc_text)
            if args.implementation in ("morphbpe", "preseg_bpe") and args.morph_boundary in doc_text:
                iterator_stats["docs_with_boundary"] += 1
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
decode_strip = args.morph_boundary if args.implementation == "preseg_bpe" else ""
tokenizer_training_iter = text_iter
if args.implementation == "morphbpe":
    tokenizer_training_iter = iter_morphbpe_training_stream(
        text_iter,
        boundary=args.morph_boundary,
        stats=morphbpe_stats,
    )
tokenizer = RustBPETokenizer.train_from_iterator(
    tokenizer_training_iter,
    args.vocab_size,
    decode_strip=decode_strip,
)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

if args.implementation in ("morphbpe", "preseg_bpe") and not args.allow_missing_boundary:
    assert iterator_stats["docs_with_boundary"] > 0, (
        f"{args.implementation} training did not see any morpheme boundary markers. "
        "Check --data-dir, --text-column, and --morph-boundary, or pass "
        "--allow-missing-boundary for a deliberate control run."
    )
if args.implementation == "morphbpe":
    assert morphbpe_stats.training_chunks > 0, "MorphBPE did not produce any training chunks."

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = get_tokenizer_dir()
tokenizer.save(tokenizer_dir)
tokenizer_config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
with open(tokenizer_config_path, "w", encoding="utf-8") as f:
    json.dump({
        "name": get_tokenizer_name(),
        "implementation": args.implementation,
        "vocab_size": args.vocab_size,
        "max_chars": args.max_chars,
        "doc_cap": args.doc_cap,
        "data_dir": resolved_data_dir,
        "text_column": args.text_column,
        "morph_boundary": args.morph_boundary if args.implementation in ("morphbpe", "preseg_bpe") else "",
        "morph_boundary_codepoints": display_boundary(args.morph_boundary) if args.implementation in ("morphbpe", "preseg_bpe") else "",
        "requires_runtime_segmentation": args.implementation == "preseg_bpe",
        "training_uses_morph_boundaries": args.implementation in ("morphbpe", "preseg_bpe"),
        "training_boundary_semantics": (
            "merge_constraint_only"
            if args.implementation == "morphbpe"
            else "visible_presegmented_control"
            if args.implementation == "preseg_bpe"
            else ""
        ),
        "decode_strip": decode_strip,
        "iterator_stats": iterator_stats,
        "morphbpe_iterator_stats": vars(morphbpe_stats) if args.implementation == "morphbpe" else {},
    }, f, indent=2)
print(f"Saved tokenizer config to {tokenizer_config_path}")

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
test_text += "\nTürkçe: İstanbul'da çalışıyorum; ğ, ü, ş, ı, ö, ç karakterleri doğru çözülmeli."
if args.implementation == "preseg_bpe":
    test_text += f"\nMorphBPE: ev{args.morph_boundary}ler{args.morph_boundary}den çalış{args.morph_boundary}ıyor{args.morph_boundary}um."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
expected_decoded = (
    strip_morpheme_boundaries(test_text, args.morph_boundary)
    if args.implementation == "preseg_bpe"
    else test_text
)
assert decoded == expected_decoded

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
