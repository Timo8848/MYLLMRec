#!/usr/bin/env python3
"""Mini grid search for the e5 ablation — same search space shape as the MiniLM grid."""
import itertools
import json
import os
import subprocess
from pathlib import Path

PYTHON = "/Volumes/FirstDrive/project/.venv/bin/python"
MAIN_PY = "/Volumes/FirstDrive/project/LLMRec/main.py"
OUTPUT_DIR = Path("/Volumes/FirstDrive/project/LLMRec/train_output/e5_grid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED = {
    "dataset": "steam_new_warm_start_demo_e5",
    "data_path": "./data/",
    "epoch": 100,
    "Ks": "[10, 20, 50]",
    "verbose": 1,
    "early_stopping_patience": 15,
    "seed": 2022,
    "use_image_feat": 0,
    "embed_size": 32,
    "weight_size": "[32, 32]",
    "drop_rate": 0.5,
    "prune_loss_drop_rate": 0.71,
}

# Smaller grid centered on lower lr (e5 is unstable at 5e-4)
GRID = list(itertools.product(
    [1e-4, 5e-5, 2e-5],  # lr
    [1e-3, 1e-2],        # weight_decay
))

results = []
for i, (lr, wd) in enumerate(GRID):
    label = f"lr{lr}_wd{wd}"
    out_path = OUTPUT_DIR / f"{label}.json"
    if out_path.exists():
        print(f"[{i+1}/{len(GRID)}] SKIP {label}")
        results.append(json.loads(out_path.read_text()))
        continue

    cfg = dict(FIXED)
    cfg["lr"] = lr
    cfg["weight_decay"] = wd
    cfg["experiment_name"] = label
    cfg["result_json_path"] = str(out_path)

    cmd = [PYTHON, MAIN_PY]
    for k, v in cfg.items():
        cmd += [f"--{k}", str(v)]

    print(f"[{i+1}/{len(GRID)}] {label}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd="/Volumes/FirstDrive/project/LLMRec")
    if proc.returncode != 0:
        print(f"  FAILED: {proc.stderr[-400:]}")
        continue
    if out_path.exists():
        r = json.loads(out_path.read_text())
        m = r["best_metrics"]
        print(f"  R@20={m['recall'][1]:.5f} N@20={m['ndcg'][1]:.5f} epoch={r['best_epoch']}")
        results.append(r)

results.sort(key=lambda x: x["best_metrics"]["recall"][1], reverse=True)
print("\n" + "=" * 70)
print(f"{'Label':<25}{'R@20':<12}{'N@20':<12}{'Epoch':<8}")
for r in results:
    m = r["best_metrics"]
    print(f"{r['experiment_name']:<25}{m['recall'][1]:<12.5f}{m['ndcg'][1]:<12.5f}{r['best_epoch']:<8}")

(OUTPUT_DIR / "_summary.json").write_text(json.dumps(results, indent=2))
