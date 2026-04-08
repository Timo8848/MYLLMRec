#!/usr/bin/env python3
"""
Extract item text embeddings via the SiliconFlow API (OpenAI-compatible).

For the Steam-700-Dense dataset (2119 items), pulls bge-m3 embeddings (1024-dim)
for each item's text and saves them as a `text_feat.npy` drop-in replacement.

Cache: the final .npy is the cache. If `--out` already exists, the script
loads it and exits without making any API calls.

Output target (default): a new sibling dataset directory under
`LLMRec/data/steam_new_warm_start_demo_bgem3/` containing the new
`text_feat.npy` plus symlinks back to all other unchanged artifacts
(train/val/test json, train_mat, image_feat.npy, user/item attribute
pickles, etc.). This lets you train with `--dataset steam_new_warm_start_demo_bgem3`
to compare against the existing all-MiniLM-L6-v2 baseline.

Usage:
    python NewData/extract_api_embeddings.py
    python NewData/extract_api_embeddings.py --batch-size 32 --sleep 0.3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np


# ---- Defaults ----------------------------------------------------------
DEFAULT_BENCH_DIR = Path("/Volumes/FirstDrive/project/NewData/processed/benchmarks_demo/warm_start_demo")
DEFAULT_SOURCE_DATA_DIR = Path("/Volumes/FirstDrive/project/LLMRec/data/steam_new_warm_start_demo")
DEFAULT_OUT_DATA_DIR = Path("/Volumes/FirstDrive/project/LLMRec/data/steam_new_warm_start_demo_e5")

API_BASE_URL = "https://api.together.xyz/v1"
API_KEY = "tgp_v1_-qdXu_ZOxpc4N72DHo0BL2eTJxhWybn_-QXrtjM2NsY"
MODEL = "intfloat/multilingual-e5-large-instruct"
EMBED_DIM = 1024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR,
                   help="Directory containing items.csv + item_id_map.csv")
    p.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA_DIR,
                   help="Existing LLMRec dataset dir to inherit non-text artifacts from")
    p.add_argument("--out-data-dir", type=Path, default=DEFAULT_OUT_DATA_DIR,
                   help="New dataset dir to write text_feat.npy + links into")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Items per API request (SiliconFlow allows up to 64 per call for bge-m3)")
    p.add_argument("--sleep", type=float, default=0.2,
                   help="Seconds to sleep between batches to avoid rate-limiting")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if cache exists")
    return p.parse_args()


def load_item_id_map(path: Path) -> dict[int, str]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["item_idx"])] = row["app_id"]
    return out


def load_items(items_csv: Path) -> dict[str, dict]:
    out = {}
    with items_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["app_id"]] = row
    return out


def build_item_text(row: dict) -> str:
    """Same format as prepare_newdata.py so we keep an apples-to-apples comparison."""
    parts = []
    title = (row.get("title") or row.get("app_name") or "").strip()
    genres = (row.get("genres") or "").strip()
    tags = (row.get("tags") or "").strip()
    if title:
        parts.append(f"title: {title}")
    if genres:
        parts.append(f"genres: {genres}")
    if tags:
        parts.append(f"tags: {tags}")
    return "\n".join(parts) if parts else "unknown game"


def collect_item_texts(bench_dir: Path) -> tuple[list[str], int]:
    idx_to_appid = load_item_id_map(bench_dir / "item_id_map.csv")
    item_meta = load_items(bench_dir / "items.csv")
    n_items = len(idx_to_appid)
    texts = []
    for idx in range(n_items):
        appid = idx_to_appid.get(idx, "")
        meta = item_meta.get(appid, {})
        texts.append(build_item_text(meta))
    return texts, n_items


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (mat / norms).astype(np.float32)


def call_api_batch(client, texts: list[str], max_retries: int) -> np.ndarray:
    delay = 1.0
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=MODEL, input=texts)
            vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            if vecs.shape != (len(texts), EMBED_DIM):
                raise RuntimeError(
                    f"Unexpected embedding shape {vecs.shape}, expected ({len(texts)}, {EMBED_DIM})")
            return vecs
        except Exception as e:  # pragma: no cover - network-dependent
            last_err = e
            print(f"  retry {attempt+1}/{max_retries} after error: {e}", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_err}")


def extract_embeddings(texts: list[str], batch_size: int, sleep_s: float, max_retries: int) -> np.ndarray:
    try:
        from openai import OpenAI
    except ImportError as e:
        sys.exit(f"openai package missing: {e}\nInstall with: pip install openai")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    n = len(texts)
    out = np.zeros((n, EMBED_DIM), dtype=np.float32)
    print(f"Calling {API_BASE_URL} ({MODEL}) for {n} items in batches of {batch_size}...")
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = texts[start:end]
        vecs = call_api_batch(client, chunk, max_retries)
        out[start:end] = vecs
        elapsed = time.time() - t0
        print(f"  [{end:>5}/{n}] +{end-start} items  ({elapsed:.1f}s elapsed)", flush=True)
        if end < n:
            time.sleep(sleep_s)
    return out


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def materialize_dataset_dir(source_dir: Path, out_dir: Path, text_feat: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "text_feat.npy", text_feat)
    print(f"Saved text_feat.npy: {text_feat.shape} -> {out_dir/'text_feat.npy'}")

    inherit = [
        "train.json", "val.json", "test.json",
        "image_feat.npy",
        "train_mat",
        "augmented_atttribute_embedding_dict",
        "augmented_user_init_embedding",
        "augmented_user_init_embedding_pooled",
        "augmented_user_init_embedding_history_summary",
        "augmented_user_init_embedding_structured_profile",
        "augmented_sample_dict",
        "candidate_indices",
    ]
    missing = []
    for name in inherit:
        src = source_dir / name
        if not src.exists():
            missing.append(name)
            continue
        link_or_copy(src.resolve(), out_dir / name)
    if missing:
        print(f"WARNING: source artifacts missing (will need regeneration): {missing}")
    else:
        print(f"Linked {len(inherit)} unchanged artifacts from {source_dir}")


def main():
    args = parse_args()
    cache_path = args.out_data_dir / "text_feat.npy"

    if cache_path.exists() and not args.force:
        print(f"Cache hit: {cache_path}")
        cached = np.load(cache_path)
        print(f"  shape: {cached.shape}, dtype: {cached.dtype}")
        # Still ensure inherited links are in place in case the user nuked them
        materialize_dataset_dir(args.source_data_dir, args.out_data_dir, cached)
        return

    print(f"Loading item texts from {args.bench_dir} ...")
    texts, n_items = collect_item_texts(args.bench_dir)
    print(f"  {n_items} items")
    print(f"  example: {texts[0][:120]!r}")

    embs = extract_embeddings(texts, args.batch_size, args.sleep, args.max_retries)
    embs = normalize_rows(embs)  # cosine-friendly, matches prepare_newdata.py default
    materialize_dataset_dir(args.source_data_dir, args.out_data_dir, embs)
    print("Done.")


if __name__ == "__main__":
    main()
