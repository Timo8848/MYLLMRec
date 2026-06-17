#!/usr/bin/env python3
"""
Pre-compute Top-20 recommendations for all 700 demo users and write
recommendations.json consumed by hf_demo/app.py (Gradio).

Usage (from LLMRec/):
    python export_recommendations.py
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
REPO = Path("/Volumes/FirstDrive/project/LLMRec")
DATA_DIR = REPO / "data" / "steam_new_warm_start_demo"
BENCH_DIR = Path("/Volumes/FirstDrive/project/NewData/processed/benchmarks_demo/warm_start_demo")
EMB_DIR = REPO / "train_output" / "demo_export"
OUT_PATH = REPO / "hf_demo" / "recommendations.json"

TOP_K = 20
HISTORY_SHOW = 8  # max history items shown per user

# ── Load embeddings ─────────────────────────────────────────────────────────
user_emb = np.load(EMB_DIR / "user_embedding_best.npy")   # (700, 32)
item_emb = np.load(EMB_DIR / "item_embedding_best.npy")   # (2119, 32)
n_users, n_items = user_emb.shape[0], item_emb.shape[0]

# ── Load interaction data ───────────────────────────────────────────────────
with open(DATA_DIR / "train.json") as f:
    train_data = json.load(f)   # {"0": [item_idx, ...], ...}
with open(DATA_DIR / "val.json") as f:
    val_data = json.load(f)
with open(DATA_DIR / "test.json") as f:
    test_data = json.load(f)    # {"0": [item_idx], ...}  (one item per user)

train_sets = {int(u): set(map(int, items)) for u, items in train_data.items()}
val_sets   = {int(u): set(map(int, items)) for u, items in val_data.items()}
test_sets  = {int(u): set(map(int, items)) for u, items in test_data.items()}

# ── Item metadata ───────────────────────────────────────────────────────────
item_id_map = pd.read_csv(BENCH_DIR / "item_id_map.csv")   # item_idx, app_id
items_meta  = pd.read_csv(BENCH_DIR / "items.csv")         # app_id, title, genres, tags, ...

# Build item_idx → metadata dict
idx2appid = dict(zip(item_id_map["item_idx"], item_id_map["app_id"].astype(str)))
def _str(val) -> str:
    return str(val) if pd.notna(val) and str(val) != "nan" else ""

# Games present in interaction data but missing from metadata fetch
STEAM_FALLBACK = {
    "72850":  {"title": "The Elder Scrolls V: Skyrim", "genres": "Action | RPG",      "tags": ["RPG", "Open World", "Fantasy", "Singleplayer", "Adventure"]},
    "43110":  {"title": "Metro 2033",                  "genres": "Action",             "tags": ["Horror", "Shooter", "Post-apocalyptic", "Singleplayer", "FPS"]},
    "221080": {"title": "cs_tools",                    "genres": "Action",             "tags": []},
    "107600": {"title": "Terraria (Beta)",             "genres": "Indie | Adventure",  "tags": ["Sandbox", "Survival", "Indie"]},
}

appid2meta = {}
for _, row in items_meta.iterrows():
    aid = str(row["app_id"])
    title = _str(row.get("title", "")) or _str(row.get("app_name", ""))
    if not title and aid in STEAM_FALLBACK:
        fb = STEAM_FALLBACK[aid]
        appid2meta[aid] = {**fb, "app_id": aid}
        continue
    appid2meta[aid] = {
        "title":  title or STEAM_FALLBACK.get(aid, {}).get("title", f"Game {aid}"),
        "genres": _str(row.get("genres", "")),
        "tags":   [t for t in _str(row.get("tags", "")).split(" | ") if t][:5],
        "app_id": aid,
    }

def item_info(item_idx: int) -> dict:
    app_id = idx2appid.get(item_idx, "")
    base = appid2meta.get(app_id, {"title": f"Item {item_idx}", "genres": "", "tags": [], "app_id": app_id})
    return dict(base)  # copy so mutation in rec loop doesn't bleed into history

# ── Pop@20 baseline ─────────────────────────────────────────────────────────
# Count training interactions per item
item_counts = np.zeros(n_items, dtype=np.int64)
for items in train_sets.values():
    for i in items:
        if i < n_items:
            item_counts[i] += 1

pop_top20_idx = np.argsort(item_counts)[::-1][:TOP_K].tolist()
pop_top20 = [{"rank": r + 1, **item_info(idx)} for r, idx in enumerate(pop_top20_idx)]

# ── Score all users at once ─────────────────────────────────────────────────
# scores: (700, 2119)
scores = user_emb @ item_emb.T

# ── Build per-user records ──────────────────────────────────────────────────
users_out = {}
pop_recall_hits = 0
llm_recall_hits = 0

for uid in range(n_users):
    train_items  = train_sets.get(uid, set())
    val_items    = val_sets.get(uid, set())
    test_items   = test_sets.get(uid, set())
    exclude      = train_items | val_items

    # LLMRec top-K (exclude train + val)
    raw_scores = scores[uid].copy()
    raw_scores[list(exclude)] = -1e9
    top_idx = np.argsort(raw_scores)[::-1][:TOP_K].tolist()

    recs = []
    for rank, idx in enumerate(top_idx):
        info = item_info(idx)
        info["rank"] = rank + 1
        info["score"] = float(raw_scores[idx])
        info["is_test_hit"] = idx in test_items
        recs.append(info)

    # History: sort by item popularity so recognisable games appear first
    history_idx = sorted(train_items, key=lambda i: -item_counts[i] if i < n_items else 0)
    history = [item_info(i) for i in history_idx[:HISTORY_SHOW]]

    # Test item
    test_item_idx = list(test_items)[0] if test_items else None
    test_item = item_info(test_item_idx) if test_item_idx is not None else None

    # Hit@20 metrics
    llm_hit = any(r["is_test_hit"] for r in recs)
    llm_recall_hits += int(llm_hit)

    pop_recs_for_user = [idx for idx in pop_top20_idx if idx not in exclude][:TOP_K]
    pop_hit = test_item_idx in set(pop_recs_for_user) if test_item_idx is not None else False
    pop_recall_hits += int(pop_hit)

    users_out[str(uid)] = {
        "history":       history,
        "history_total": len(train_items),
        "recommendations": recs,
        "test_item":     test_item,
        "llm_hit_at_20": llm_hit,
        "pop_hit_at_20": pop_hit,
    }

stats = {
    "n_users":          n_users,
    "n_items":          n_items,
    "llmrec_recall_20": round(llm_recall_hits / n_users, 5),
    "pop_recall_20":    round(pop_recall_hits / n_users, 5),
    "top_k":            TOP_K,
}
print(f"LLMRec Recall@20: {stats['llmrec_recall_20']}  Pop@20: {stats['pop_recall_20']}")

# ── Write output ─────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
payload = {"stats": stats, "pop_top20": pop_top20, "users": users_out}
with open(OUT_PATH, "w") as f:
    json.dump(payload, f, separators=(",", ":"))

size_kb = OUT_PATH.stat().st_size // 1024
print(f"Written: {OUT_PATH}  ({size_kb} KB, {n_users} users)")
