#!/usr/bin/env python3
"""
Popularity baseline for the Steam-700-Dense dataset.

Recommends the top-K most popular items (by training interactions) to every
test user. Computes Recall@K and NDCG@K under the Leave-One-Out evaluation
protocol (one ground-truth item per user).

For LOO with one relevant item:
    NDCG@K = 1 / log2(rank + 1)   if the item is in the top-K (rank 1-indexed)
           = 0                    otherwise
(IDCG = 1 because the ideal ranking puts the single relevant item at rank 1.)
"""

import json
import math
from collections import Counter
from pathlib import Path


def pop_baseline(data_dir: str, k: int = 20):
    data_dir = Path(data_dir)
    train = json.loads((data_dir / "train.json").read_text())
    val = json.loads((data_dir / "val.json").read_text())
    test = json.loads((data_dir / "test.json").read_text())

    # Rank items by training popularity (descending)
    item_counts: Counter = Counter()
    for items in train.values():
        for iid in items:
            item_counts[iid] += 1

    # Ordered top-k recommendation list (same for every user)
    ranked_items = [iid for iid, _ in item_counts.most_common()]
    top_k = ranked_items[:k]
    item_to_rank = {iid: rank + 1 for rank, iid in enumerate(top_k)}  # 1-indexed

    def evaluate(split, split_name):
        n_users = 0
        recall_sum = 0.0
        ndcg_sum = 0.0
        for uid, gt_items in split.items():
            if not gt_items:
                continue
            n_users += 1
            # LOO: one ground-truth item per user
            target = gt_items[0]
            # Exclude items already seen in train (Pop baseline isn't user-personalized
            # but for consistency with model evaluation we filter seen items here too).
            # In practice the popularity ranking is the same for everyone, so seen
            # items just shift recommendations down — we still take the top-k of
            # whatever's left. To match the model's evaluation protocol exactly:
            seen = set(train.get(uid, []))
            personal_topk = [iid for iid in ranked_items if iid not in seen][:k]
            personal_rank = {iid: rank + 1 for rank, iid in enumerate(personal_topk)}

            if target in personal_rank:
                rank = personal_rank[target]
                recall_sum += 1.0
                ndcg_sum += 1.0 / math.log2(rank + 1)

        recall = recall_sum / n_users if n_users else 0.0
        ndcg = ndcg_sum / n_users if n_users else 0.0
        print(f"  {split_name}: users={n_users}, Recall@{k}={recall:.5f}, NDCG@{k}={ndcg:.5f}")
        return recall, ndcg

    print(f"Pop@{k} Baseline on {data_dir.name}")
    print(f"Top-{k} most popular items in train (no per-user filtering):")
    for rank, (iid, cnt) in enumerate(item_counts.most_common(k), 1):
        print(f"  {rank:3d}. item {iid:5d}  ({cnt} train interactions)")
    print()
    val_r, val_n = evaluate(val, "Val ")
    test_r, test_n = evaluate(test, "Test")
    return {"recall": test_r, "ndcg": test_n, "val_recall": val_r, "val_ndcg": val_n}


if __name__ == "__main__":
    data_dir = "/Volumes/FirstDrive/project/LLMRec/data/steam_new_warm_start_demo"
    pop_baseline(data_dir, k=20)
