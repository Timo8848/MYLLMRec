#!/usr/bin/env python3
"""
Pure LightGCN baseline for the Steam-700-Dense dataset.

No LLM/text features. No prune loss. No contrastive loss. No augmentation.
Just User/Item ID embeddings -> LightGCN propagation -> BPR loss + L2 reg.

Usage (single run):
    python run_lightgcn_baseline.py --lr 1e-3 --weight_decay 1e-4 --n_layers 3

Usage (grid search):
    python run_lightgcn_baseline.py --grid
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# Set sys.argv before importing batch_test (which reads args at import time).
DATASET = "steam_new_warm_start_demo"
DATA_PATH = "./data/"
KS = "[10, 20, 50]"
BATCH_SIZE = 1024

_orig_argv = sys.argv
sys.argv = [
    "run_lightgcn_baseline.py",
    "--dataset", DATASET,
    "--data_path", DATA_PATH,
    "--Ks", KS,
    "--batch_size", str(BATCH_SIZE),
]
from utility.batch_test import test_torch, data_generator  # noqa: E402
sys.argv = _orig_argv


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_norm_adj(data):
    """Symmetric normalized adjacency: D^-1/2 (A) D^-1/2, where A is the bipartite UI graph."""
    n_users = data.n_users
    n_items = data.n_items
    R = data.R.tocsr().astype(np.float32)

    n = n_users + n_items
    adj = sp.lil_matrix((n, n), dtype=np.float32)
    adj[:n_users, n_users:] = R
    adj[n_users:, :n_users] = R.T
    adj = adj.tocsr()

    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = (D_inv_sqrt @ adj @ D_inv_sqrt).tocoo()

    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_size, n_layers, norm_adj):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, embed_size)
        self.item_emb = nn.Embedding(n_items, embed_size)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.register_buffer("norm_adj", norm_adj.to_dense() if False else None)
        self.norm_adj_sparse = norm_adj  # set on the right device externally

    def propagate(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj_sparse, all_emb)
            embs.append(all_emb)
        out = torch.stack(embs, dim=0).mean(dim=0)
        u_out, i_out = torch.split(out, [self.n_users, self.n_items], dim=0)
        return u_out, i_out

    def bpr_loss(self, users, pos_items, neg_items, weight_decay):
        u_out, i_out = self.propagate()
        u_e = u_out[users]
        pos_e = i_out[pos_items]
        neg_e = i_out[neg_items]

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 reg on the *initial* embeddings of the sampled batch (standard LightGCN)
        u0 = self.user_emb(users)
        p0 = self.item_emb(pos_items)
        n0 = self.item_emb(neg_items)
        reg = (u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()) / (2 * users.shape[0])
        loss = bpr + weight_decay * reg
        return loss, bpr.item(), reg.item()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(lr, weight_decay, n_layers, embed_size=32, epochs=200, patience=15,
              seed=2022, label=""):
    set_seed(seed)
    device = get_device()
    print(f"\n=== LightGCN run: lr={lr}, wd={weight_decay}, n_layers={n_layers}, "
          f"emb={embed_size}, device={device} ===", flush=True)

    data = data_generator
    norm_adj = build_norm_adj(data).to(device)

    model = LightGCN(data.n_users, data.n_items, embed_size, n_layers, norm_adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_batches = data.n_train // BATCH_SIZE + 1

    best_val_recall20 = -1.0
    best_test = None
    best_epoch = -1
    bad_epochs = 0

    val_users = list(data.val_set.keys())
    test_users = list(data.test_set.keys())

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        for _ in range(n_batches):
            users, pos_items, neg_items = data.sample()
            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)
            loss, _, _ = model.bpr_loss(users_t, pos_t, neg_t, weight_decay)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        train_time = time.time() - t0

        # Eval on val
        model.eval()
        with torch.no_grad():
            u_out, i_out = model.propagate()
        val_res = test_torch(u_out, i_out, val_users, is_val=True)
        val_r20 = val_res["recall"][1]
        val_n20 = val_res["ndcg"][1]

        msg = (f"Epoch {epoch:3d} | loss={loss_sum/n_batches:.4f} | "
               f"val Recall@20={val_r20:.5f} NDCG@20={val_n20:.5f} | {train_time:.1f}s")

        if val_r20 > best_val_recall20:
            best_val_recall20 = val_r20
            best_epoch = epoch
            with torch.no_grad():
                test_res = test_torch(u_out, i_out, test_users, is_val=False)
            best_test = {
                "recall": test_res["recall"].tolist(),
                "ndcg": test_res["ndcg"].tolist(),
                "precision": test_res["precision"].tolist(),
                "hit_ratio": test_res["hit_ratio"].tolist(),
            }
            msg += (f"  *test Recall@20={test_res['recall'][1]:.5f} "
                    f"NDCG@20={test_res['ndcg'][1]:.5f}")
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(msg, flush=True)

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    print(f"BEST {label}: epoch={best_epoch}, val_recall@20={best_val_recall20:.5f}, "
          f"test={best_test}", flush=True)
    return {
        "label": label,
        "lr": lr,
        "weight_decay": weight_decay,
        "n_layers": n_layers,
        "embed_size": embed_size,
        "best_epoch": best_epoch,
        "best_val_recall_20": best_val_recall20,
        "best_test": best_test,
    }


def grid_search():
    out_dir = Path("./train_output/lightgcn_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = []
    for lr in [1e-3, 5e-4]:
        for wd in [1e-4, 1e-3]:
            for n_layers in [2, 3]:
                grid.append((lr, wd, n_layers))

    print(f"LightGCN grid search: {len(grid)} configs")
    results = []
    for i, (lr, wd, n_layers) in enumerate(grid):
        label = f"lr{lr}_wd{wd}_L{n_layers}"
        out_path = out_dir / f"{label}.json"
        if out_path.exists():
            print(f"[{i+1}/{len(grid)}] SKIP {label}")
            results.append(json.loads(out_path.read_text()))
            continue
        print(f"\n[{i+1}/{len(grid)}] {label}")
        r = train_one(lr=lr, weight_decay=wd, n_layers=n_layers, label=label)
        out_path.write_text(json.dumps(r, indent=2))
        results.append(r)

    results.sort(key=lambda x: x["best_test"]["recall"][1] if x["best_test"] else -1,
                 reverse=True)
    print("\n" + "=" * 80)
    print("LightGCN Grid Search Results (sorted by test Recall@20):")
    print("=" * 80)
    print(f"{'Rank':<5}{'Label':<30}{'Recall@20':<12}{'NDCG@20':<12}{'Epoch':<8}")
    for rank, r in enumerate(results, 1):
        m = r["best_test"]
        if m is None:
            continue
        print(f"{rank:<5}{r['label']:<30}{m['recall'][1]:<12.5f}{m['ndcg'][1]:<12.5f}"
              f"{r['best_epoch']:<8}")

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved to {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    if args.grid:
        grid_search()
    else:
        train_one(args.lr, args.weight_decay, args.n_layers,
                  embed_size=args.embed_size, epochs=args.epochs, patience=args.patience,
                  label="single")
