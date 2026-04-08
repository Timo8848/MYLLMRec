#!/usr/bin/env python3
"""
t-SNE visualization of LLMRec's learned item embeddings on Steam-700-Dense.

Pipeline:
  1. If the item-embedding .npy doesn't exist yet, retrain the best MiniLM
     config (Recall@20 = 0.13143) with `--save_item_embedding_path` to dump
     the propagated item embeddings at the best-val epoch.
  2. Load items.csv and assign each item its dominant genre. The top-N most
     common genres get their own color; the rest are bucketed into "Other".
  3. t-SNE to 2D (perplexity=30, random_state=42).
  4. Seaborn scatter plot, clean academic style. Save as 300-dpi PNG + PDF.

Run:
    python LLMRec/visualize_embeddings.py
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ---- Paths --------------------------------------------------------------
REPO = Path("/Volumes/FirstDrive/project/LLMRec")
DATA_DIR = REPO / "data" / "steam_new_warm_start_demo"
BENCH_DIR = Path("/Volumes/FirstDrive/project/NewData/processed/benchmarks_demo/warm_start_demo")
OUT_DIR = REPO / "train_output" / "viz"
EMB_PATH = OUT_DIR / "item_embedding_best.npy"
PNG_PATH = OUT_DIR / "item_embeddings_tsne.png"
PDF_PATH = OUT_DIR / "item_embeddings_tsne.pdf"

PYTHON = "/Volumes/FirstDrive/project/.venv/bin/python"
MAIN_PY = REPO / "main.py"

# Best MiniLM config (see EXPERIMENTS.md)
BEST_CONFIG = {
    "dataset": "steam_new_warm_start_demo",
    "data_path": "./data/",
    "epoch": 100,
    "batch_size": 1024,
    "lr": 5e-4,
    "embed_size": 32,
    "weight_size": "[32, 32]",
    "prune_loss_drop_rate": 0.71,
    "drop_rate": 0.5,
    "weight_decay": 1e-3,
    "use_image_feat": 0,
    "Ks": "[10, 20, 50]",
    "early_stopping_patience": 15,
    "seed": 2022,
    "experiment_name": "viz_embedding_dump",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k-genres", type=int, default=7,
                   help="Number of distinct genres to color; the rest collapse into 'Other'")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--force-retrain", action="store_true")
    return p.parse_args()


def train_and_dump_embeddings():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(MAIN_PY)]
    for k, v in BEST_CONFIG.items():
        cmd += [f"--{k}", str(v)]
    cmd += [
        "--result_json_path", str(OUT_DIR / "viz_run.json"),
        "--save_item_embedding_path", str(EMB_PATH),
    ]
    print(f"Training best MiniLM config to dump item embeddings -> {EMB_PATH}")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO))
    if proc.returncode != 0:
        sys.exit(f"Training failed with code {proc.returncode}")
    if not EMB_PATH.exists():
        sys.exit(f"Training finished but {EMB_PATH} was not written")


GENERIC_TAGS = {"Action", "Indie", "Casual", "Adventure", "Great Soundtrack",
                "Singleplayer", "Multiplayer", "2D", "3D", "Atmospheric",
                "Story Rich", "Classic", "Open World", "Free to Play"}


def load_item_labels():
    """Returns list[idx] of the most discriminative tag for each item.

    Steam's `genres` field is too coarse (~48% of items get 'Action'), so we
    use the `tags` field instead. We pick the first *non-generic* tag — i.e.
    the first tag that is not in GENERIC_TAGS. Falls back to the first tag
    if all are generic, or the dominant genre if tags are missing.
    """
    idx_to_appid: dict[int, str] = {}
    with (BENCH_DIR / "item_id_map.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx_to_appid[int(row["item_idx"])] = row["app_id"]

    meta: dict[str, dict] = {}
    with (BENCH_DIR / "items.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["app_id"]] = row

    n_items = len(idx_to_appid)
    dominant = []
    for idx in range(n_items):
        row = meta.get(idx_to_appid.get(idx, ""), {})
        tags_raw = row.get("tags", "") or ""
        tags = [t.strip() for t in tags_raw.split("|") if t.strip()]
        pick = next((t for t in tags if t not in GENERIC_TAGS), None)
        if pick is None and tags:
            pick = tags[0]
        if pick is None:
            genres_raw = row.get("genres", "") or ""
            gparts = [g.strip() for g in genres_raw.split("|") if g.strip()]
            pick = gparts[0] if gparts else "Unknown"
        dominant.append(pick)
    return dominant


def bucket_labels(dominant: list[str], top_k: int) -> tuple[list[str], list[str]]:
    counts = Counter(g for g in dominant if g != "Unknown")
    top = [g for g, _ in counts.most_common(top_k)]
    out = [g if g in top else "Other" for g in dominant]
    return out, top


def run_tsne(emb: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    print(f"t-SNE: {emb.shape} -> 2D (perplexity={perplexity})")
    # For a small n (~2k), tsne runs in a few seconds
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
    ).fit_transform(emb)
    return coords


def plot(coords: np.ndarray, labels: list[str], top_genres: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Genre": labels,
    })

    # Order legend: top genres first (by frequency), then "Other"
    order = list(top_genres)
    if "Other" in set(labels):
        order.append("Other")
    palette = sns.color_palette("tab10", n_colors=len(top_genres))
    palette_map = dict(zip(top_genres, palette))
    palette_map["Other"] = (0.75, 0.75, 0.75)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=df, x="x", y="y", hue="Genre",
        hue_order=order, palette=palette_map,
        s=28, alpha=0.82, linewidth=0, ax=ax,
    )
    ax.set_title(
        "LLMRec item embeddings — t-SNE projection\n"
        "(Steam-700-Dense, 2119 items, 32-d → 2-d, colored by dominant Steam tag)",
        fontsize=12, pad=14,
    )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        title="Dominant tag",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        markerscale=1.3,
    )
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {PDF_PATH}")


def main():
    args = parse_args()

    if args.force_retrain or not EMB_PATH.exists():
        train_and_dump_embeddings()
    else:
        print(f"Reusing cached embeddings: {EMB_PATH}")

    emb = np.load(EMB_PATH)
    print(f"Loaded embeddings: {emb.shape}")

    dominant = load_item_labels()
    assert len(dominant) == emb.shape[0], \
        f"label list ({len(dominant)}) != embedding rows ({emb.shape[0]})"
    labels, top_genres = bucket_labels(dominant, args.top_k_genres)

    counts = Counter(labels)
    print(f"Genre buckets ({args.top_k_genres} + Other):")
    for g in top_genres + ["Other"]:
        print(f"  {g:<20} {counts.get(g, 0)}")

    coords = run_tsne(emb, args.perplexity, args.random_state)
    plot(coords, labels, top_genres)


if __name__ == "__main__":
    main()
