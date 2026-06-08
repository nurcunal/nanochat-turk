"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import json
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool
from urllib.parse import quote

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The Turkish fork trains on FineWeb-2 Turkish, preserving the parquet order from
# the Hugging Face tree for reproducibility.
DATASET_REPO = os.environ.get("NANOCHAT_DATASET_REPO", "HuggingFaceFW/fineweb-2")
DATASET_REVISION = os.environ.get("NANOCHAT_DATASET_REVISION", "main")
DATASET_CONFIG = os.environ.get("NANOCHAT_FINEWEB2_CONFIG", "tur_Latn")
DATASET_PATH = os.environ.get("NANOCHAT_DATASET_PATH", f"data/{DATASET_CONFIG}/train")
TEXT_COLUMN = os.environ.get("NANOCHAT_TEXT_COLUMN", "text")
base_dir = get_base_dir()
DATA_DIR = os.environ.get("NANOCHAT_DATA_DIR", os.path.join(base_dir, "base_data_fineweb2_tur_latn"))
MANIFEST_FILE = "fineweb2_manifest.json"

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def _manifest_path(data_dir):
    return os.path.join(data_dir, MANIFEST_FILE)

def _load_manifest(data_dir):
    path = _manifest_path(data_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_manifest(data_dir, filenames, revision=DATASET_REVISION):
    os.makedirs(data_dir, exist_ok=True)
    manifest = {
        "repo": DATASET_REPO,
        "revision": revision,
        "path": DATASET_PATH,
        "text_column": TEXT_COLUMN,
        "filenames": filenames,
    }
    with open(_manifest_path(data_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def _headers():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}

def _fetch_remote_filenames(revision=DATASET_REVISION):
    """Return FineWeb-2 parquet filenames in stable tree order."""
    api_url = f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/{revision}/{DATASET_PATH}"
    filenames = []
    next_url = api_url
    params = {"recursive": "false", "expand": "false"}
    while next_url:
        response = requests.get(next_url, params=params, headers=_headers(), timeout=60)
        response.raise_for_status()
        entries = response.json()
        for entry in entries:
            path = entry.get("path", "")
            if path.endswith(".parquet"):
                filenames.append(os.path.basename(path))
        next_url = response.links.get("next", {}).get("url")
        params = None
    ordered_filenames = []
    seen = set()
    for filename in filenames:
        if filename not in seen:
            ordered_filenames.append(filename)
            seen.add(filename)
    if not ordered_filenames:
        raise RuntimeError(f"No parquet files found in hf://datasets/{DATASET_REPO}/{DATASET_PATH}@{revision}")
    return ordered_filenames

def _list_parquet_files_in_dir(data_dir):
    manifest = _load_manifest(data_dir)
    if manifest is not None:
        return [
            os.path.join(data_dir, name)
            for name in manifest.get("filenames", [])
            if os.path.exists(os.path.join(data_dir, name))
        ]

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    ])
    return [os.path.join(data_dir, f) for f in parquet_files]

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  TURKISH DATASET NOT FOUND")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  This branch expects FineWeb-2 Turkish parquet files.")
            print("  Download a reproducible prefix plus a held-out final shard with:")
            print()
            print("    python -m nanochat.dataset -n 8       # tokenizer/debug prefix")
            print("    python -m nanochat.dataset -n -1      # all Turkish shards")
            print()
            print("=" * 80)
            print()
        return []
    return _list_parquet_files_in_dir(data_dir)

def parquets_iter_batched(split, start=0, step=1, data_dir=None, text_column=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - data_dir/text_column can override the default dataset for tokenizer
      experiments, e.g. MorphBPE segmented shards with a segmented_text column.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    text_column = TEXT_COLUMN if text_column is None else text_column
    parquet_paths = list_parquet_files(data_dir)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run `python -m nanochat.dataset -n 8`?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        if text_column not in pf.schema_arrow.names:
            raise KeyError(f"Column {text_column!r} not found in {filepath}; columns: {pf.schema_arrow.names}")
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx, columns=[text_column])
            texts = rg.column(text_column).to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def _resolve_url(filename, revision=DATASET_REVISION):
    path = quote(f"{DATASET_PATH}/{filename}", safe="/")
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{revision}/{path}"

def download_single_file(item):
    """ Downloads a single parquet file, with some backoff. """
    if isinstance(item, tuple):
        filename, revision = item
    else:
        filename, revision = item, DATASET_REVISION

    # Construct the local filepath for this file and skip if it already exists
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = _resolve_url(filename, revision=revision)
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60, headers=_headers())
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-2 Turkish pretraining shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download. -1 = all shards.")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument("--revision", type=str, default=DATASET_REVISION, help="Hugging Face revision/commit (default: main)")
    parser.add_argument("--list-local", action="store_true", help="List local parquet files and exit")
    parser.add_argument("--list-remote", action="store_true", help="List remote parquet files and exit")
    args = parser.parse_args()

    if args.list_local:
        for path in list_parquet_files():
            print(path)
        raise SystemExit(0)

    remote_filenames = _fetch_remote_filenames(revision=args.revision)
    if args.list_remote:
        for name in remote_filenames:
            print(name)
        raise SystemExit(0)

    # Prepare the output directory only when we are going to download/write.
    os.makedirs(DATA_DIR, exist_ok=True)

    # The user specifies the number of train shards to download via -n.
    # In addition, the final shard in the remote order is always downloaded and
    # used as validation, matching nanochat's "last shard is val" convention.
    train_pool = remote_filenames[:-1]
    val_file = remote_filenames[-1]
    if args.num_files == -1:
        train_filenames = train_pool
    else:
        train_filenames = train_pool[:min(args.num_files, len(train_pool))]
    filenames_to_download = train_filenames + [val_file]
    _write_manifest(DATA_DIR, filenames_to_download, revision=args.revision)

    # Download the shards
    print(f"Dataset: hf://datasets/{DATASET_REPO}/{DATASET_PATH}@{args.revision}")
    print(f"Downloading {len(filenames_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        download_items = [(name, args.revision) for name in filenames_to_download]
        results = pool.map(download_single_file, download_items)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(filenames_to_download)} shards to {DATA_DIR}")
