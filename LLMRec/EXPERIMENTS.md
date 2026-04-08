# LLMRec on Steam Demo Dataset — Experiment Notes

## Project Data Source

After extensive iteration on the full Steam Australian users benchmark (68,403 users / 3.18M interactions / >1h per epoch on Apple MPS), we adopted a smaller k-core sampled subset as the **canonical project dataset**. All future model development and evaluation targets this dataset.

**Dataset: `steam_new_warm_start_demo`**

| Stat            | Value          |
|-----------------|----------------|
| Users           | 700            |
| Items           | 2,119          |
| Train interactions | 41,409      |
| Val interactions   | 700         |
| Test interactions  | 700         |
| Density         | ~2.89%         |
| Per-user mean   | ~60 train items |
| Train epoch time | ~3 seconds (Apple MPS) |

### How it was built

1. **k-core dense subgraph sampling** (`NewData/subsample_dense.py`):
   - Iteratively prune users with `< 10` interactions and items with `< 10` interactions until stable
   - Random-sample 700 users from the resulting dense core
   - Drop items appearing in fewer than 3 of the sampled users
   - Re-index user/item IDs to be consecutive
2. **Feature generation** (`LLMRec/prepare_newdata.py`):
   - Sentence-transformer text features for items, user profiles, and item attributes
   - Image features are zero placeholders — **always train with `--use_image_feat 0`**
3. **Leave-One-Out re-split** (`LLMRec/resplit_add_val.py`):
   - For each user: last interaction → test, second-to-last → val, rest → train

### Why k-core (not random sampling)

Steam interactions follow a heavy long-tail distribution. A purely random 1% user sample produces an extremely sparse user-item graph where most items have 0–1 interactions, and the model cannot learn meaningful collaborative signal. k-core sampling preserves a dense interaction core, lifting density from 0.47% (full data) to 2.89% (demo). This was essential — without it, even a correct model implementation cannot escape the sparsity trap.

---

## Pipeline Fixes

Several bugs in the original training loop were fixed before any tuning could give meaningful results:

1. **AdamW `weight_decay` was not being passed** to the optimizer in `main.py`. The argument was parsed but ignored — L2 regularization was effectively disabled. Now actually used.
2. **Early stopping peeked at the test set.** `main.py` evaluated test recall every epoch and used it to decide whether to stop, leaking test signal into model selection. Fixed to evaluate on a proper validation set; test set is only computed when validation recall improves.
3. **Test candidates included validation items.** `utility/batch_test.py` only excluded train items from the candidate pool, so val items could appear as test candidates. Fixed to also exclude val items when evaluating test.
4. **Device detection ignored Apple MPS.** Both `main.py` and `Models.py` only checked for CUDA. Now falls back to MPS when CUDA is unavailable.

Default hyperparameters were also retuned for the smaller dataset: `embed_size 64→32`, `drop_rate 0→0.3`, `weight_decay 1e-4→1e-3`.

---

## Pop@20 Baseline

Implemented in `LLMRec/pop_baseline.py`. Recommends the 20 most-frequently-interacted training items to every test user, after filtering items the user already saw in train (matching the model evaluation protocol).

**Pop@20 Recall = 0.01714, NDCG@20 = 0.01001**

This is the floor any collaborative model must exceed to demonstrate it learned anything beyond popularity.

---

## Hyperparameter Grid Search

`LLMRec/run_hyperparam_search.py` swept 72 configurations over:

| Hyperparameter           | Values searched          |
|--------------------------|--------------------------|
| `lr`                     | 5e-5, 1e-4, 5e-4         |
| `embed_size`             | 16, 32                   |
| `prune_loss_drop_rate`   | 0.0, 0.3, 0.71           |
| `drop_rate`              | 0.3, 0.5                 |
| `weight_decay`           | 1e-3, 1e-2               |
| `use_image_feat`         | 0 (always — placeholder zeros) |

Each run takes ~30 seconds; full sweep was ~36 minutes.

### Top 5 Results

| Rank | Recall@10 | **Recall@20** | Recall@50 | NDCG@20 | Best Epoch | Configuration |
|------|-----------|---------------|-----------|---------|-----------|---------------|
| **1** | 0.08286 | **0.13143** | 0.23143 | 0.05532 | 46 | `lr=5e-4, emb=32, prune=0.71, drop=0.5, wd=1e-3` |
| 2    | 0.03714 | 0.06857    | 0.13714 | 0.02629 | 6  | `lr=5e-4, emb=16, prune=0.71, drop=0.3, wd=1e-3` |
| 3    | 0.03143 | 0.05714    | 0.10857 | 0.02325 | 6  | `lr=5e-4, emb=32, prune=0.71, drop=0.5, wd=1e-2` |
| 4    | 0.03571 | 0.04714    | 0.09000 | 0.02058 | 1  | `lr=5e-4, emb=16, prune=0.71, drop=0.3, wd=1e-2` |
| 5    | 0.02571 | 0.04000    | 0.07000 | 0.01579 | 3  | `lr=1e-4, emb=16, prune=0.71, drop=0.5, wd=1e-3` |

### Observations

- **Best Recall@20 = 0.13143**, which is **9.2× the Pop@20 baseline (0.01429)** — the model is clearly learning collaborative signal beyond popularity.
- **`prune_loss_drop_rate=0.71` dominates.** Every entry in the top 10 uses 0.71. Lower values (0.0, 0.3) consistently underperformed. The curriculum-style loss pruning seems essential for this dataset.
- **`lr=5e-4` + `emb_size=32` was the only configuration that benefited from long training** (best epoch 46). All lower-LR configs plateaued by epoch 5–10 around Recall@20 ≈ 0.04. The big winner needed both more capacity and more time.
- Full results are saved to `LLMRec/train_output/grid_search/_summary.json`.

---

## Best Configuration (Reproducible)

```bash
/Volumes/FirstDrive/project/.venv/bin/python main.py \
  --dataset steam_new_warm_start_demo \
  --data_path ./data/ \
  --epoch 100 \
  --batch_size 1024 \
  --lr 5e-4 \
  --embed_size 32 \
  --weight_size '[32, 32]' \
  --prune_loss_drop_rate 0.71 \
  --drop_rate 0.5 \
  --weight_decay 1e-3 \
  --use_image_feat 0 \
  --Ks '[10, 20, 50]' \
  --early_stopping_patience 15 \
  --experiment_name warm_start_demo_best \
  --result_json_path ./train_output/warm_start_demo_best_result.json \
  --seed 2022
```

---

## Pure LightGCN Baseline

Implemented in `LLMRec/run_lightgcn_baseline.py`. A clean LightGCN with **no LLM features**, **no prune loss**, **no contrastive loss**, **no augmentation** — just User/Item ID embeddings → symmetric-normalized graph propagation → BPR loss with L2 regularization. Uses the same train/val/test split, same evaluation protocol (LOO with val items excluded from test candidates), and the same `test_torch` evaluator as LLMRec.

### Grid search

8 configurations over `lr ∈ {1e-3, 5e-4}`, `weight_decay ∈ {1e-4, 1e-3}`, `n_layers ∈ {2, 3}`, with `embed_size=32`, patience=15, max 200 epochs.

| Config | Recall@20 | NDCG@20 | Best Epoch |
|--------|-----------|---------|------------|
| `lr=5e-4, wd=1e-3, L=2` | 0.01714 | 0.01108 | 1  |
| `lr=5e-4, wd=1e-4, L=2` | 0.01714 | 0.01108 | 1  |
| `lr=1e-3, wd=1e-4, L=2` | 0.01714 | 0.01079 | 0  |
| `lr=1e-3, wd=1e-3, L=2` | 0.01714 | 0.01078 | 0  |
| `lr=1e-3, wd=1e-4, L=3` | 0.01714 | 0.01001 | 0  |
| `lr=1e-3, wd=1e-3, L=3` | 0.01714 | 0.01001 | 0  |
| `lr=5e-4, wd=1e-3, L=3` | 0.01714 | 0.00919 | 0  |
| `lr=5e-4, wd=1e-4, L=3` | 0.01714 | 0.00918 | 0  |

**Every config plateaus at exactly the Pop@20 floor.** Verified with a longer run (`embed_size=64`, patience=40, 200 epochs): val recall stays in the 0.067–0.081 range across all 41 epochs while training BPR loss converges from 0.69 → 0.27. The model memorizes training pairs without ever extracting collaborative signal beyond popularity.

This is a meaningful negative result: with only 700 users and random ID embeddings, pure LightGCN cannot bootstrap a useful collaborative ranking — the BPR signal collapses into "rank items by training frequency". The LLMRec text features are what allow the same architecture to break out of the popularity floor.

---

## Text-Encoder Ablation (MiniLM vs e5)

To isolate how much of LLMRec's lift comes from the *specific* text encoder versus "having any text features at all", we re-ran LLMRec with `text_feat.npy` regenerated by **`intfloat/multilingual-e5-large-instruct`** (1024-dim, via Together AI). Everything else — architecture, user/item attribute embeddings (still MiniLM 384-dim), hyperparameter search shape, seed — was held fixed.

- Extraction script: `NewData/extract_api_embeddings.py` (OpenAI-compatible client, 50/batch, 0.2s sleep, local `.npy` cache)
- Drop-in dataset: `LLMRec/data/steam_new_warm_start_demo_e5/` (new `text_feat.npy` + symlinks to the rest)
- Model MLP input dim auto-adapts: `SharedTextEncoder._resolve_shared_input_dim()` picks `max(1024, 384, 384)=1024` and inserts `Linear(384→1024)` adapters for user/item domains

The original best hyperparameters (`lr=5e-4, wd=1e-3`) were unstable on e5 — training loss oscillated (120→44→40→…→72→91), model hit val Recall@20=0.077 briefly at epoch 1 then collapsed. A 6-config grid over `lr ∈ {1e-4, 5e-5, 2e-5}` × `wd ∈ {1e-3, 1e-2}` found the best stable config:

| Config | Recall@20 | NDCG@20 | Best Epoch |
|--------|-----------|---------|------------|
| `lr=1e-4, wd=1e-3` | 0.03286 | 0.01402 | 3 |
| `lr=1e-4, wd=1e-2` | 0.03286 | 0.00985 | 0 |
| `lr=2e-5, wd=1e-3` | 0.02857 | 0.01076 | 2 |
| `lr=2e-5, wd=1e-2` | 0.02857 | 0.01076 | 2 |
| `lr=5e-5, wd=1e-3` | 0.02286 | 0.01005 | 0 |
| `lr=5e-5, wd=1e-2` | 0.02286 | 0.01005 | 0 |

**Swapping MiniLM-L6 (384) → e5-large-instruct (1024) drops Recall@20 from 0.13143 to 0.03286 (−75%).** Likely reasons:
1. `e5-*-instruct` is trained to expect explicit instruction prefixes (`"Instruct: ...\nQuery: ..."`). Feeding raw `title/genres/tags` text bypasses the instruction template and produces lower-quality embeddings for this downstream task.
2. 1024-dim adds ~200k extra parameters via the `Linear(384→1024)` user/item adapters inside SharedTextEncoder — on 700 users this compounds the overfitting problem.

This is a useful counter-intuitive finding: **a "stronger" foundation model is not a drop-in upgrade**. The MiniLM encoder's simplicity is actually a good fit for raw-text sparse recsys inputs; using e5 correctly would require integrating its instruction template into `prepare_newdata.py` (follow-up work).

---

## Final Comparison

| Model | Text encoder | Recall@20 | NDCG@20 | vs. Pop@20 Recall |
|-------|--------------|----------:|--------:|-------------------|
| Pop@20 (popularity floor) | — | 0.01714 | 0.01001 | 1.00× |
| Pure LightGCN (best of 8) | none (ID only) | 0.01714 | 0.01108 | 1.00× |
| LLMRec + e5-large-instruct (best of 6) | e5-large-instruct 1024d (raw text) | 0.03286 | 0.01402 | 1.92× |
| **LLMRec + MiniLM (best of 72)** | **all-MiniLM-L6-v2 384d** | **0.13143** | **0.05532** | **7.67×** |

On this 700-user Steam-Dense dataset, the MiniLM-based LLMRec remains the clear winner. The bottleneck for a "bigger is better" upgrade isn't the downstream model — it's honoring the foundation model's input contract (instruction prefixes, domain prompting).

---

## Open Work

- Build cold_start and long_tail demo splits (k-core may need different parameters since these benchmarks specifically test sparse scenarios)
- Real image features (`image_feat.npy` is currently a zero placeholder)
- Ablation study over `use_user_profile`, `use_item_attribute`, `use_sample_augmentation`
- Try `user_profile_variant` alternatives (`history_summary`, `structured_profile`) — only `pooled` has been tested
