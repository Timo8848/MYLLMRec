#!/usr/bin/env python3
"""
Dense subgraph sampling (k-core filtering) for NewData benchmarks.

Produces a small, interaction-dense demo dataset suitable for fast
training iteration. The algorithm:

1. Load train/val/test splits
2. Merge all interactions to build the full user-item bipartite graph
3. Iterative k-core pruning: remove users with < k_user interactions
   and items with < k_item interactions until stable
4. From the dense core, sample target_users users (preferring diverse
   interaction counts)
5. Keep only items those sampled users interact with (post-filter items
   with < min_item_degree interactions in the sample)
6. Re-index users and items to consecutive IDs
7. Write subsampled benchmark files + copy item metadata

Usage:
    python subsample_dense.py --benchmark warm_start --target-users 700
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks-dir", default="processed/benchmarks")
    p.add_argument("--output-dir", default="processed/benchmarks_demo")
    p.add_argument("--benchmark", default="warm_start",
                   help="Which benchmark to subsample")
    p.add_argument("--target-users", type=int, default=700)
    p.add_argument("--k-user", type=int, default=10,
                   help="Min interactions per user in k-core")
    p.add_argument("--k-item", type=int, default=10,
                   help="Min interactions per item in k-core")
    p.add_argument("--min-item-degree", type=int, default=3,
                   help="Min item degree after user sampling")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def k_core_filter(user_items: dict[int, set[int]], k_user: int, k_item: int):
    """Iteratively prune users and items below degree thresholds."""
    iteration = 0
    while True:
        iteration += 1
        # Count item degrees
        item_degrees: Counter[int] = Counter()
        for items in user_items.values():
            for item in items:
                item_degrees[item] += 1

        # Remove low-degree items from all users
        valid_items = {i for i, d in item_degrees.items() if d >= k_item}
        pruned_users = {}
        for uid, items in user_items.items():
            filtered = items & valid_items
            if len(filtered) >= k_user:
                pruned_users[uid] = filtered

        n_users_before = len(user_items)
        n_items_before = len(item_degrees)
        n_users_after = len(pruned_users)
        n_items_after = len(valid_items & {i for items in pruned_users.values() for i in items})

        print(f"  k-core iteration {iteration}: "
              f"users {n_users_before}->{n_users_after}, "
              f"items {n_items_before}->{n_items_after}")

        if pruned_users == user_items:
            break
        user_items = pruned_users

        if not user_items:
            print("  WARNING: k-core collapsed to empty! Loosen thresholds.")
            break

    return user_items


def main():
    args = parse_args()
    random.seed(args.seed)

    bench_dir = Path(args.benchmarks_dir) / args.benchmark
    out_dir = Path(args.output_dir) / f"{args.benchmark}_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {bench_dir}...")
    train = json.loads((bench_dir / "train.json").read_text())
    val = json.loads((bench_dir / "val.json").read_text())
    test = json.loads((bench_dir / "test.json").read_text())

    # Convert to int keys
    train = {int(k): v for k, v in train.items()}
    val = {int(k): v for k, v in val.items()}
    test = {int(k): v for k, v in test.items()}

    # Merge all interactions for k-core
    all_user_items: dict[int, set[int]] = {}
    for uid, items in train.items():
        all_user_items.setdefault(uid, set()).update(items)
    for uid, items in val.items():
        all_user_items.setdefault(uid, set()).update(items)
    for uid, items in test.items():
        all_user_items.setdefault(uid, set()).update(items)

    print(f"Original: {len(all_user_items)} users, "
          f"{len({i for items in all_user_items.values() for i in items})} items, "
          f"{sum(len(v) for v in all_user_items.values())} interactions")

    # Step 1: k-core filtering
    print(f"\nk-core filtering (k_user={args.k_user}, k_item={args.k_item})...")
    dense_core = k_core_filter(all_user_items, args.k_user, args.k_item)

    if not dense_core:
        print("ERROR: dense core is empty")
        return

    core_users = sorted(dense_core.keys())
    print(f"\nDense core: {len(core_users)} users")

    # Step 2: Sample target_users from dense core
    if len(core_users) <= args.target_users:
        sampled_users = set(core_users)
        print(f"Core smaller than target, keeping all {len(sampled_users)} users")
    else:
        sampled_users = set(random.sample(core_users, args.target_users))
        print(f"Sampled {len(sampled_users)} users from dense core")

    # Step 3: Collect items from sampled users, filter low-degree
    item_counts: Counter[int] = Counter()
    for uid in sampled_users:
        for item in dense_core[uid]:
            item_counts[item] += 1

    valid_items = {i for i, c in item_counts.items() if c >= args.min_item_degree}
    print(f"Items after min_degree={args.min_item_degree} filter: {len(valid_items)}")

    # Step 4: Build re-index mappings
    old_to_new_user = {}
    for new_idx, old_uid in enumerate(sorted(sampled_users)):
        old_to_new_user[old_uid] = new_idx

    old_to_new_item = {}
    for new_idx, old_iid in enumerate(sorted(valid_items)):
        old_to_new_item[old_iid] = new_idx

    # Step 5: Re-map train/val/test
    def remap_split(split_data):
        remapped = {}
        for uid, items in split_data.items():
            if uid not in old_to_new_user:
                continue
            new_items = [old_to_new_item[i] for i in items if i in old_to_new_item]
            if new_items:
                remapped[str(old_to_new_user[uid])] = new_items
        return remapped

    new_train = remap_split(train)
    new_val = remap_split(val)
    new_test = remap_split(test)

    # Stats
    n_users = len(old_to_new_user)
    n_items = len(old_to_new_item)
    n_train = sum(len(v) for v in new_train.values())
    n_test = sum(len(v) for v in new_test.values())
    n_val = sum(len(v) for v in new_val.values())
    density = (n_train + n_test + n_val) / (n_users * n_items) if n_users * n_items > 0 else 0

    print(f"\n{'='*50}")
    print(f"Demo dataset summary:")
    print(f"  Users:        {n_users}")
    print(f"  Items:        {n_items}")
    print(f"  Train:        {n_train} interactions ({n_train/max(len(new_train),1):.1f} per user)")
    print(f"  Val:          {n_val} interactions")
    print(f"  Test:         {n_test} interactions")
    print(f"  Density:      {density:.4f} ({density*100:.2f}%)")
    print(f"  Train users:  {len(new_train)}")
    print(f"  Test users:   {len(new_test)}")
    print(f"{'='*50}")

    # Step 6: Write outputs
    (out_dir / "train.json").write_text(json.dumps(new_train, indent=2))
    (out_dir / "val.json").write_text(json.dumps(new_val, indent=2))
    (out_dir / "test.json").write_text(json.dumps(new_test, indent=2))

    # Re-map item_id_map.csv
    old_item_id_map = {}
    with (bench_dir / "item_id_map.csv").open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_item_id_map[int(row["item_idx"])] = row["app_id"]

    with (out_dir / "item_id_map.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_idx", "app_id"])
        writer.writeheader()
        for old_iid in sorted(valid_items):
            writer.writerow({
                "item_idx": old_to_new_item[old_iid],
                "app_id": old_item_id_map.get(old_iid, ""),
            })

    # Re-map user_id_map.csv
    old_user_id_map = {}
    if (bench_dir / "user_id_map.csv").exists():
        with (bench_dir / "user_id_map.csv").open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                old_user_id_map[int(row["user_idx"])] = row["user_id"]

    with (out_dir / "user_id_map.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_idx", "user_id"])
        writer.writeheader()
        for old_uid in sorted(sampled_users):
            writer.writerow({
                "user_idx": old_to_new_user[old_uid],
                "user_id": old_user_id_map.get(old_uid, str(old_uid)),
            })

    # Copy items.csv as-is (item metadata lookup is by app_id)
    if (bench_dir / "items.csv").exists():
        shutil.copy2(bench_dir / "items.csv", out_dir / "items.csv")

    # Write summary
    summary = {
        "source_benchmark": args.benchmark,
        "k_user": args.k_user,
        "k_item": args.k_item,
        "min_item_degree": args.min_item_degree,
        "target_users": args.target_users,
        "seed": args.seed,
        "n_users": n_users,
        "n_items": n_items,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "density": density,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOutput written to: {out_dir}")


if __name__ == "__main__":
    main()
