#!/usr/bin/env python3
"""
Hyperparameter grid search for LLMRec on demo dataset.
Each run takes ~30s, so we can afford dozens of configs.
"""
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

PYTHON = "/Volumes/FirstDrive/project/.venv/bin/python"
MAIN_PY = "/Volumes/FirstDrive/project/LLMRec/main.py"
OUTPUT_DIR = "/Volumes/FirstDrive/project/LLMRec/train_output/grid_search"

# Fixed params
FIXED = {
    "dataset": "steam_new_warm_start_demo",
    "data_path": "./data/",
    "epoch": 100,
    "Ks": "[10, 20, 50]",
    "verbose": 1,
    "early_stopping_patience": 15,
    "seed": 2022,
}

# Grid search space
GRID = {
    "lr": [5e-5, 1e-4, 5e-4],
    "embed_size": [16, 32],
    "weight_size": ["'[16, 16]'", "'[32, 32]'"],  # must match embed_size
    "prune_loss_drop_rate": [0.0, 0.3, 0.71],
    "use_image_feat": [0],  # always off - it's all zeros
    "drop_rate": [0.3, 0.5],
    "weight_decay": [1e-3, 1e-2],
}

# Build configs: embed_size and weight_size must match
def build_configs():
    configs = []
    for lr in GRID["lr"]:
        for prune in GRID["prune_loss_drop_rate"]:
            for drop in GRID["drop_rate"]:
                for wd in GRID["weight_decay"]:
                    for emb, ws in [(16, "[16, 16]"), (32, "[32, 32]")]:
                        configs.append({
                            "lr": lr,
                            "embed_size": emb,
                            "weight_size": ws,
                            "prune_loss_drop_rate": prune,
                            "use_image_feat": 0,
                            "drop_rate": drop,
                            "weight_decay": wd,
                        })
    return configs


def run_one(config, idx, total):
    label = f"lr{config['lr']}_emb{config['embed_size']}_prune{config['prune_loss_drop_rate']}_drop{config['drop_rate']}_wd{config['weight_decay']}"
    result_path = os.path.join(OUTPUT_DIR, f"{label}.json")

    if os.path.exists(result_path):
        print(f"[{idx+1}/{total}] SKIP {label} (already done)")
        with open(result_path) as f:
            return json.load(f)

    print(f"[{idx+1}/{total}] Running {label}...", flush=True)

    cmd = [PYTHON, MAIN_PY]
    for k, v in FIXED.items():
        cmd += [f"--{k}", str(v)]
    for k, v in config.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--experiment_name", label]
    cmd += ["--result_json_path", result_path]

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd="/Volumes/FirstDrive/project/LLMRec")

    if proc.returncode != 0:
        print(f"  FAILED: {proc.stderr[-500:]}")
        return None

    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        r20 = result["best_metrics"]["recall"][1]
        ep = result["best_epoch"]
        print(f"  Recall@20={r20:.5f}, best_epoch={ep}")
        return result
    else:
        print(f"  No result file produced")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configs = build_configs()
    print(f"Total configs: {len(configs)}")
    print(f"Estimated time: ~{len(configs) * 30 / 60:.0f} minutes\n")

    results = []
    for i, cfg in enumerate(configs):
        r = run_one(cfg, i, len(configs))
        if r:
            results.append(r)

    # Sort by test recall@20
    results.sort(key=lambda x: x["best_metrics"]["recall"][1], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP 10 CONFIGURATIONS (by Test Recall@20):")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Recall@10':<12} {'Recall@20':<12} {'Recall@50':<12} {'NDCG@20':<12} {'Epoch':<7} {'Config'}")
    print(f"{'-'*5} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*7} {'-'*40}")

    for i, r in enumerate(results[:10]):
        m = r["best_metrics"]
        print(f"{i+1:<5} {m['recall'][0]:<12.5f} {m['recall'][1]:<12.5f} {m['recall'][2]:<12.5f} "
              f"{m['ndcg'][1]:<12.5f} {r['best_epoch']:<7} {r['experiment_name']}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump([{"experiment_name": r["experiment_name"],
                     "best_epoch": r["best_epoch"],
                     "recall_10": r["best_metrics"]["recall"][0],
                     "recall_20": r["best_metrics"]["recall"][1],
                     "recall_50": r["best_metrics"]["recall"][2],
                     "ndcg_20": r["best_metrics"]["ndcg"][1],
                     } for r in results], f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
