#!/usr/bin/env python3
"""
Re-split train/test into train/val/test using Leave-One-Out with validation.

For each user:
- If user has test item(s) AND >= 2 train items:
    val = [last train item], train = train[:-1], test unchanged
- If user has no test item AND >= 3 train items:
    test = [last train item], val = [second-to-last train item], train = train[:-2]
- Otherwise: keep as-is (user may only appear in train)

This modifies the benchmark_demo directory in-place, then the LLMRec data directory.
"""

import json
import sys
from pathlib import Path


def resplit(bench_dir: Path):
    train = json.loads((bench_dir / "train.json").read_text())
    test = json.loads((bench_dir / "test.json").read_text())
    val_old = json.loads((bench_dir / "val.json").read_text()) if (bench_dir / "val.json").exists() else {}

    new_train = {}
    new_val = {}
    new_test = {}

    all_uids = set(train.keys()) | set(test.keys())

    stats = {"val_from_train": 0, "val_and_test_from_train": 0, "unchanged": 0, "total": 0}

    for uid in sorted(all_uids, key=lambda x: int(x)):
        t_items = train.get(uid, [])
        te_items = test.get(uid, [])
        stats["total"] += 1

        if te_items and len(t_items) >= 2:
            # Has test, take last train item as val
            new_train[uid] = t_items[:-1]
            new_val[uid] = [t_items[-1]]
            new_test[uid] = te_items
            stats["val_from_train"] += 1
        elif not te_items and len(t_items) >= 3:
            # No test, create both val and test from train
            new_train[uid] = t_items[:-2]
            new_val[uid] = [t_items[-2]]
            new_test[uid] = [t_items[-1]]
            stats["val_and_test_from_train"] += 1
        else:
            # Not enough items, keep as-is
            new_train[uid] = t_items
            if te_items:
                new_test[uid] = te_items
            stats["unchanged"] += 1

    print(f"Re-split stats for {bench_dir.name}:")
    print(f"  Total users: {stats['total']}")
    print(f"  Val from last train item (had test): {stats['val_from_train']}")
    print(f"  Val+Test from train (no test): {stats['val_and_test_from_train']}")
    print(f"  Unchanged (too few items): {stats['unchanged']}")
    print(f"  New train users: {sum(1 for v in new_train.values() if v)}")
    print(f"  New val users: {sum(1 for v in new_val.values() if v)}")
    print(f"  New test users: {sum(1 for v in new_test.values() if v)}")
    print(f"  Train interactions: {sum(len(v) for v in new_train.values())}")
    print(f"  Val interactions: {sum(len(v) for v in new_val.values())}")
    print(f"  Test interactions: {sum(len(v) for v in new_test.values())}")

    (bench_dir / "train.json").write_text(json.dumps(new_train, indent=2))
    (bench_dir / "val.json").write_text(json.dumps(new_val, indent=2))
    (bench_dir / "test.json").write_text(json.dumps(new_test, indent=2))
    print(f"  Written to {bench_dir}")


if __name__ == "__main__":
    # Re-split benchmark demo
    demo_bench = Path("/Volumes/FirstDrive/project/NewData/processed/benchmarks_demo/warm_start_demo")
    resplit(demo_bench)

    # Also re-split the LLMRec data directory (which has copies)
    llmrec_data = Path("/Volumes/FirstDrive/project/LLMRec/data/steam_new_warm_start_demo")
    resplit(llmrec_data)

    print("\nDone! Now re-run prepare_newdata.py to regenerate train_mat and other artifacts.")
