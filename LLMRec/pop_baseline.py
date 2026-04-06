#!/usr/bin/env python3
"""
Pop@20 baseline: recommend the 20 most popular items from the training set
to every test user and compute Recall@20.
"""

import json
from collections import Counter
from pathlib import Path


def pop_at_k(data_dir: str, k: int = 20):
    data_dir = Path(data_dir)
    train = json.loads((data_dir / "train.json").read_text())
    test = json.loads((data_dir / "test.json").read_text())
    val = json.loads((data_dir / "val.json").read_text())

    # Count item popularity in training set
    item_counts: Counter = Counter()
    for items in train.values():
        for iid in items:
            item_counts[iid] += 1

    top_k_items = set(iid for iid, _ in item_counts.most_common(k))
    print(f"Top-{k} most popular items (by train interactions):")
    for rank, (iid, cnt) in enumerate(item_counts.most_common(k), 1):
        print(f"  {rank:3d}. item {iid:5d}  ({cnt} interactions)")

    # Compute Recall@K for test set
    hits = 0
    total_users = 0
    for uid_str, test_items in test.items():
        if not test_items:
            continue
        total_users += 1
        # A hit if any test item is in the top-k
        for iid in test_items:
            if iid in top_k_items:
                hits += 1

    recall = hits / total_users if total_users > 0 else 0.0
    print(f"\nPop@{k} Baseline:")
    print(f"  Test users: {total_users}")
    print(f"  Hits: {hits}")
    print(f"  Recall@{k}: {recall:.5f}")

    # Also compute for val set
    val_hits = 0
    val_total = 0
    for uid_str, val_items in val.items():
        if not val_items:
            continue
        val_total += 1
        for iid in val_items:
            if iid in top_k_items:
                val_hits += 1

    val_recall = val_hits / val_total if val_total > 0 else 0.0
    print(f"\n  Val users: {val_total}")
    print(f"  Val Hits: {val_hits}")
    print(f"  Val Recall@{k}: {val_recall:.5f}")

    return recall


if __name__ == "__main__":
    data_dir = "/Volumes/FirstDrive/project/LLMRec/data/steam_new_warm_start_demo"
    pop_at_k(data_dir, k=20)
