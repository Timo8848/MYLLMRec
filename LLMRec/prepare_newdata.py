#!/usr/bin/env python3
"""
Generate LLMRec training artifacts from NewData benchmark directories.

For each benchmark (warm_start, cold_start, long_tail), reads the existing
train.json / val.json / test.json and items.csv, then produces:
  - text_feat.npy          (item text features via sentence encoder)
  - image_feat.npy         (placeholder zeros)
  - train_mat              (scipy sparse matrix, pickle)
  - augmented_user_init_embedding_pooled        (pickle)
  - augmented_user_init_embedding               (alias of pooled)
  - augmented_user_init_embedding_history_summary
  - augmented_user_init_embedding_structured_profile
  - augmented_atttribute_embedding_dict         (pickle)
  - augmented_sample_dict                       (pickle)
  - candidate_indices                           (pickle)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "about", "after", "against", "all", "also", "and", "are", "around",
    "because", "been", "being", "between", "but", "can", "each", "features",
    "from", "game", "games", "has", "have", "into", "its", "just", "more",
    "most", "new", "not", "now", "off", "one", "online", "out", "over",
    "play", "player", "players", "playing", "same", "set", "steam", "team",
    "that", "the", "their", "them", "then", "there", "these", "this",
    "through", "time", "variety", "when", "which", "while", "with", "world",
    "your",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks-dir", default="../NewData/processed/benchmarks")
    p.add_argument("--output-base", default="./data")
    p.add_argument("--benchmarks", nargs="+", default=["warm_start", "cold_start", "long_tail"])
    p.add_argument("--image-dim", type=int, default=256)
    p.add_argument("--text-encoder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--text-encoder-device", default="auto")
    p.add_argument("--text-encoder-batch-size", type=int, default=64)
    p.add_argument("--text-max-length", type=int, default=256)
    p.add_argument("--candidate-k", type=int, default=10)
    p.add_argument("--profile-history-max-items", type=int, default=20)
    return p.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def extract_top_keywords(texts: list[str], limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for tok in tokenize(text):
            if len(tok) >= 4 and tok not in STOPWORDS:
                counts[tok] += 1
    return [t for t, _ in counts.most_common(limit)]


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (mat / norms).astype(np.float32)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def encode_texts(texts, model_name, device, batch_size, max_length):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    if max_length > 0:
        model.max_seq_length = max_length
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)


def load_items(items_csv: Path) -> dict[int, dict]:
    """Load item metadata keyed by item_idx from item_id_map + items.csv."""
    items = {}
    with items_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items[row["app_id"]] = row
    return items


def load_item_id_map(path: Path) -> dict[int, str]:
    """Returns {item_idx: app_id}."""
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["item_idx"])] = row["app_id"]
    return mapping


def build_item_text(row: dict) -> str:
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
    return "\n".join(parts)


def build_history_summary(user_idx, train_items, idx_to_appid, item_meta, max_items):
    items = train_items[:max_items]
    if not items:
        return "User positive history summary: no positive training interactions available."
    lines = [f"User positive history summary ({len(items)} training interactions):"]
    genre_counts = Counter()
    descs = []
    for i, iidx in enumerate(items, 1):
        appid = idx_to_appid.get(iidx, "")
        meta = item_meta.get(appid, {})
        title = (meta.get("title") or meta.get("app_name") or "").strip()
        genres = (meta.get("genres") or "").strip()
        if genres:
            genre_counts.update(g.strip() for g in genres.split("|") if g.strip())
        desc_parts = []
        if genres:
            desc_parts.append(f"genres: {genres}")
        lines.append(f"{i}. {title}" + (f" | {' | '.join(desc_parts)}" if desc_parts else ""))
        descs.append(title + " " + genres)
    if genre_counts:
        top_genres = [g for g, _ in genre_counts.most_common(4)]
        lines.append(f"Recurring genres: {', '.join(top_genres)}.")
    kw = extract_top_keywords(descs, 6)
    if kw:
        lines.append(f"Recurring themes: {', '.join(kw)}.")
    return "\n".join(lines)


def build_structured_profile(user_idx, train_items, idx_to_appid, item_meta, max_items):
    items = train_items[:max_items]
    if not items:
        return "user_profile {\ninteraction_count: 0\npreference_summary: No positive training history available.\n}"
    genre_counts = Counter()
    titles = []
    descs = []
    for iidx in items:
        appid = idx_to_appid.get(iidx, "")
        meta = item_meta.get(appid, {})
        title = (meta.get("title") or meta.get("app_name") or "").strip()
        genres = (meta.get("genres") or "").strip()
        if genres:
            genre_counts.update(g.strip() for g in genres.split("|") if g.strip())
        if title:
            titles.append(title)
        descs.append(title + " " + genres)
    fav_genres = [g for g, _ in genre_counts.most_common(4)]
    themes = extract_top_keywords(descs, 6)
    lines = ["user_profile {", f"interaction_count: {len(items)}"]
    if fav_genres:
        lines.append(f"favorite_genres: {', '.join(fav_genres)}")
    if titles[:5]:
        lines.append(f"representative_games: {'; '.join(titles[:5])}")
    if themes:
        lines.append(f"gameplay_themes: {', '.join(themes)}")
    stmts = []
    if fav_genres:
        stmts.append(f"leans toward {', '.join(fav_genres[:3])} games")
    if themes:
        stmts.append(f"responds to themes like {', '.join(themes[:4])}")
    lines.append("preference_summary: This user " + "; ".join(stmts) + "." if stmts else "preference_summary: Weak signal.")
    lines.append("}")
    return "\n".join(lines)


def process_benchmark(bench_dir: Path, output_dir: Path, args):
    print(f"\n{'='*60}")
    print(f"Processing: {bench_dir.name}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train = json.loads((bench_dir / "train.json").read_text())
    val = json.loads((bench_dir / "val.json").read_text())
    test = json.loads((bench_dir / "test.json").read_text())
    idx_to_appid = load_item_id_map(bench_dir / "item_id_map.csv")
    item_meta = load_items(bench_dir / "items.csv")

    n_items = len(idx_to_appid)
    n_users = max(int(k) for k in train.keys()) + 1

    print(f"  n_users={n_users}, n_items={n_items}")

    # --- 1. text_feat.npy ---
    print("  Building text features...")
    item_texts = []
    for idx in range(n_items):
        appid = idx_to_appid.get(idx, "")
        meta = item_meta.get(appid, {})
        item_texts.append(build_item_text(meta))

    device = resolve_device(args.text_encoder_device)
    text_feats = encode_texts(item_texts, args.text_encoder_model, device,
                              args.text_encoder_batch_size, args.text_max_length)
    np.save(output_dir / "text_feat.npy", text_feats)
    print(f"  text_feat.npy: {text_feats.shape}")

    # --- 2. image_feat.npy (placeholder) ---
    image_feats = np.zeros((n_items, args.image_dim), dtype=np.float32)
    np.save(output_dir / "image_feat.npy", image_feats)
    print(f"  image_feat.npy: {image_feats.shape}")

    # --- 3. train_mat ---
    print("  Building train_mat...")
    rows, cols = [], []
    for uid_str, items in train.items():
        uid = int(uid_str)
        for iid in items:
            rows.append(uid)
            cols.append(iid)
    vals = np.ones(len(rows), dtype=np.float32)
    train_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    with open(output_dir / "train_mat", "wb") as f:
        pickle.dump(train_mat, f)
    print(f"  train_mat: {train_mat.shape}, nnz={train_mat.nnz}")

    # --- 4. Item attribute embeddings ---
    print("  Building item attribute embeddings...")
    title_texts = []
    genre_texts = []
    desc_texts = []
    for idx in range(n_items):
        appid = idx_to_appid.get(idx, "")
        meta = item_meta.get(appid, {})
        title = (meta.get("title") or meta.get("app_name") or "").strip()
        genres = (meta.get("genres") or "").strip()
        tags = (meta.get("tags") or "").strip()
        title_texts.append(f"title: {title}" if title else "")
        genre_texts.append(f"genres: {genres}" if genres else "")
        desc_texts.append(f"tags: {tags}" if tags else "")

    title_embs = encode_texts(title_texts, args.text_encoder_model, device,
                              args.text_encoder_batch_size, args.text_max_length)
    genre_embs = encode_texts(genre_texts, args.text_encoder_model, device,
                              args.text_encoder_batch_size, args.text_max_length)
    desc_embs = encode_texts(desc_texts, args.text_encoder_model, device,
                             args.text_encoder_batch_size, args.text_max_length)

    item_attr_dict = {
        "title": [row for row in title_embs],
        "genre": [row for row in genre_embs],
        "description": [row for row in desc_embs],
    }
    with open(output_dir / "augmented_atttribute_embedding_dict", "wb") as f:
        pickle.dump(item_attr_dict, f)
    print(f"  item attribute embedding dim: {title_embs.shape[1]}")

    # --- 5. User profile embeddings ---
    print("  Building user profile embeddings...")
    pooled_item_embs = normalize_rows((title_embs + genre_embs + desc_embs) / 3.0)

    pooled_user_embs = []
    history_summary_texts = []
    structured_profile_texts = []

    for uid in range(n_users):
        titems = train.get(str(uid), [])
        if titems:
            vec = pooled_item_embs[titems].mean(axis=0)
        else:
            vec = np.zeros(pooled_item_embs.shape[1], dtype=np.float32)
        pooled_user_embs.append(normalize_vector(vec))

        history_summary_texts.append(
            build_history_summary(uid, titems, idx_to_appid, item_meta, args.profile_history_max_items))
        structured_profile_texts.append(
            build_structured_profile(uid, titems, idx_to_appid, item_meta, args.profile_history_max_items))

    with open(output_dir / "augmented_user_init_embedding", "wb") as f:
        pickle.dump(pooled_user_embs, f)
    with open(output_dir / "augmented_user_init_embedding_pooled", "wb") as f:
        pickle.dump(pooled_user_embs, f)
    print(f"  pooled user embeddings: {len(pooled_user_embs)} x {len(pooled_user_embs[0])}")

    print("  Encoding history summary profiles...")
    hs_embs = encode_texts(history_summary_texts, args.text_encoder_model, device,
                           args.text_encoder_batch_size, args.text_max_length)
    hs_embs = normalize_rows(hs_embs)
    with open(output_dir / "augmented_user_init_embedding_history_summary", "wb") as f:
        pickle.dump(hs_embs, f)

    print("  Encoding structured profiles...")
    sp_embs = encode_texts(structured_profile_texts, args.text_encoder_model, device,
                           args.text_encoder_batch_size, args.text_max_length)
    sp_embs = normalize_rows(sp_embs)
    with open(output_dir / "augmented_user_init_embedding_structured_profile", "wb") as f:
        pickle.dump(sp_embs, f)
    print(f"  user profile embeddings: history_summary={hs_embs.shape}, structured={sp_embs.shape}")

    # --- 6. Candidate indices ---
    print("  Building candidate indices...")
    popularity = Counter()
    for items in train.values():
        for iid in items:
            popularity[iid] += 1
    popular_items = [iid for iid, _ in popularity.most_common()]
    k = min(args.candidate_k, n_items - 1)
    candidates = np.zeros((n_users, k), dtype=np.int64)
    for uid in range(n_users):
        seen = set(train.get(str(uid), []))
        ranked = [iid for iid in popular_items if iid not in seen]
        if len(ranked) < k:
            ranked.extend(iid for iid in range(n_items) if iid not in seen and iid not in ranked)
        candidates[uid] = np.array(ranked[:k], dtype=np.int64)
    with open(output_dir / "candidate_indices", "wb") as f:
        pickle.dump(candidates, f)

    # --- 7. Augmented sample dict ---
    print("  Building augmented sample dict...")
    aug_samples = {}
    for uid in range(n_users):
        uid_s = str(uid)
        pos_source = test.get(uid_s) or val.get(uid_s) or train.get(uid_s, [])
        if not pos_source:
            continue
        seen = set(train.get(uid_s, [])) | set(val.get(uid_s, [])) | set(test.get(uid_s, []))
        neg = next((iid for iid in candidates[uid] if iid not in seen), None)
        if neg is None:
            neg = next((iid for iid in range(n_items) if iid not in seen), 0)
        aug_samples[uid] = {0: int(pos_source[0]), 1: int(neg)}
    with open(output_dir / "augmented_sample_dict", "wb") as f:
        pickle.dump(aug_samples, f)

    # --- 8. Copy JSON splits ---
    import shutil
    for fname in ["train.json", "val.json", "test.json"]:
        src = bench_dir / fname
        dst = output_dir / fname
        if src != dst:
            shutil.copy2(src, dst)

    print(f"  Done: {output_dir}")


def main():
    args = parse_args()
    benchmarks_dir = Path(args.benchmarks_dir).expanduser().resolve()
    output_base = Path(args.output_base).expanduser().resolve()

    for bench in args.benchmarks:
        bench_dir = benchmarks_dir / bench
        if not bench_dir.exists():
            print(f"Skipping {bench}: {bench_dir} not found")
            continue
        # Output to e.g. ./data/steam_new_warm_start/
        output_dir = output_base / f"steam_new_{bench}"
        process_benchmark(bench_dir, output_dir, args)

    print("\nAll done!")


if __name__ == "__main__":
    main()
