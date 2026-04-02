# 项目交接总结

## 1. 项目目标

这个项目的核心目标，是把开源 `LLMRec` 框架迁移到 Steam 场景，并围绕 Steam 数据做两件事：

1. 先用一条小规模 MVP 链路，快速验证 “文本特征 / item attribute / user profile” 这些增强信号在 Steam 上是否有效。
2. 再把更完整的 Steam 原始数据整理成可复用的 benchmark，支持后续在 `warm_start / cold_start / long_tail` 三种设置下继续训练和评测。

当前仓库里，以上两条线都已经打通了，但成熟度不同：

- 小规模 MVP 链路已经完成数据准备、模型改造和消融实验。
- 大规模 benchmark 链路已经完成数据清洗、benchmark 切分和 LLMRec 输入包打包，但还没有看到对应的大盘训练结果沉淀在仓库里。


## 2. 当前仓库里最重要的文件和目录

### 根目录数据处理脚本

- `eda_review_quality.py`
  对评论文本做质量扫描，识别空文本、超短文本、重复文本、符号垃圾文本等。
- `fetch_steam_games_metadata.py`
  根据评论里的 `app_id` 拉 Steam metadata，生成 `games.csv`，并使用 `steam_appdetails_cache.json` 做缓存。
- `merge_reviews_with_games.py`
  把清洗后的评论数据和游戏 metadata 合并成 `merged_data.csv`。
- `organize_newdata.py`
  把 `NewData/` 下的原始 Steam dump 规范化成多张 CSV 表。
- `build_steam_benchmarks.py`
  基于规范化后的 Steam 表构建 `warm_start / cold_start / long_tail` 三套 benchmark。

### LLMRec 相关

- `LLMRec/main.py`
  训练入口，已经改造成支持 Steam 数据、feature toggle、profile variant 选择和结构化结果输出。
- `LLMRec/Models.py`
  模型主体，新增了 shared text encoder / residual adapter / feature enable 开关。
- `LLMRec/utility/parser.py`
  训练参数入口，新增了和 Steam 适配有关的一批参数。
- `LLMRec/prepare_steam_mvp.py`
  把 `merged_data.csv` 打成 `LLMRec/data/steam` 可直接训练的数据包。
- `LLMRec/prepare_steam_benchmark_packages.py`
  把 benchmark 切分结果继续打成 `LLMRec/data/steam_warm_start`、`steam_cold_start`、`steam_long_tail`。
- `LLMRec/run_ablation_matrix.py`
  跑 4 个核心配置 x 多 seed 的消融矩阵。
- `LLMRec/run_user_profile_ablation.py`
  跑 user profile 来源/表示方式对比。

### 关键产物目录

- `eda_output/`
  评论质量分析报告。
- `NewData/processed/`
  原始 Steam dump 规范化后的表。
- `NewData/processed/benchmarks/`
  benchmark 切分结果。
- `LLMRec/data/steam/`
  小规模 MVP 的 LLMRec 数据包。
- `LLMRec/data/steam_warm_start/`
- `LLMRec/data/steam_cold_start/`
- `LLMRec/data/steam_long_tail/`
  三套 benchmark 的 LLMRec 数据包。
- `LLMRec/ablation_results/`
  小规模 MVP 实验的结构化结果。
- `LLMRec/logs/`
  训练日志。


## 3. 这段时间实际做了什么

### 3.1 第一阶段：先做小规模 MVP 验证链路

这条链路的目标不是做最终 benchmark，而是快速判断 LLMRec 在 Steam 评论数据上是不是有信号。

#### 3.1.1 对评论数据做质量扫描

输入文件是根目录的 `output.csv`，字段很简单：

- `id`
- `app_id`
- `content`
- `author_id`
- `is_positive`

`eda_review_quality.py` 的作用是先判断评论文本是否值得拿来做后续建模，输出包括：

- `eda_output/review_quality_summary.json`
- `eda_output/review_quality_report.md`
- `eda_output/suspected_useless_reviews.csv`
- `eda_output/top_duplicate_reviews.csv`

根据现有报告，得到的关键信息是：

- 总评论数：`201,151`
- 疑似无效评论：`67,172`，占 `33.39%`
- 重复文本：`37,693`
- distinct app 数：`50`
- 文本长度中位数：`41` 字符，`8` 个词

主要问题类型：

- `short_text`: `81,804`
- `very_short_text`: `59,164`
- `duplicate_content`: `43,117`
- 还有大量 symbol-only、ascii art、repeated punctuation

结论很明确：原始评论噪声很重，所以先清洗再做建模是必要的。

脚本设计上会把清洗后的结果写到 `data.csv`，但当前仓库根目录里 **没有保留这个文件**。后面如果要复现这条链路，需要重新跑一遍清洗，或者去找原始生成时的 `data.csv`。

#### 3.1.2 拉取游戏 metadata 并做合并

在评论清洗之后，做了两步：

1. `fetch_steam_games_metadata.py`
   从 Steam appdetails API 拉 `name / short_description / genres`
2. `merge_reviews_with_games.py`
   把评论和 metadata 按 `app_id` 合并

生成的产物：

- `games.csv`
- `steam_appdetails_cache.json`
- `merged_data.csv`

`games.csv` 当前只有 `50` 行游戏 metadata，和上面的 50 个 app 对应，说明这条 MVP 链路本身就是一个小规模数据集。

`merged_data.csv` 的字段是：

- `id`
- `app_id`
- `name`
- `short_description`
- `genres`
- `content`
- `author_id`
- `is_positive`

#### 3.1.3 把 MVP 数据打成 LLMRec 可训练格式

之后用 `LLMRec/prepare_steam_mvp.py` 把 `merged_data.csv` 变成 `LLMRec/data/steam/`。

这一步做了几件事：

- 只保留正向交互，构建 user-item interaction
- 做 iterative k-core，过滤极稀疏 user/item
- 构建 train / val / test
- 为 item 构建 `text_feat.npy`
- 为 item 构建 `item_attribute` embedding
- 为 user 构建三种 profile 表示
- 生成 `candidate_indices`
- 生成 `train_mat`、`train.json`、`val.json`、`test.json`

当前 `LLMRec/data/steam/summary.json` 反映出的结果：

- k-core 前正向交互：`64,889`
- k-core 后正向交互：`6,978`
- 用户数：`1,785`
- item 数：`33`
- train：`3,408`
- val：`1,785`
- test：`1,785`

这说明小样本链路虽然能跑通，但数据规模被压得很小，最后只剩 `33` 个 item。它适合验证方法方向，不适合直接当最终业务结论。

#### 3.1.4 在 LLMRec 上做了 Steam 适配

为了让原始 LLMRec 能吃这个 Steam 数据，我们对训练部分做了几类改造：

1. 在 `LLMRec/utility/parser.py` 里补了 feature 开关
   - `use_image_feat`
   - `use_text_feat`
   - `use_item_attribute`
   - `use_user_profile`
   - `use_sample_augmentation`
   - `user_profile_variant`
   - `user_profile_path`
   - `experiment_name`
   - `result_json_path`

2. 在 `LLMRec/Models.py` 里补了更灵活的文本投影方式
   - `SharedTextEncoder`
   - `ResidualAdapter`
   - `text_encoder_mode=separate/shared/shared_adapter`

3. 在 `LLMRec/main.py` 里补了运行时逻辑
   - 可以按开关关闭某类 feature
   - 可以从不同的 user profile embedding 文件加载
   - 训练结束能把最优结果写成 JSON
   - log 文件名里会带上实验名和 seed

4. 在数据准备脚本里补了三种 user profile 表示
   - `pooled`
   - `history_summary`
   - `structured_profile`

#### 3.1.5 跑了核心消融实验

`2026-03-27` 跑了 `LLMRec/run_ablation_matrix.py`，结果在：

- `LLMRec/ablation_results/core_matrix_20260327/summary.json`
- `LLMRec/ablation_results/core_matrix_20260327/summary.md`

实验配置是 4 组 x 3 个 seed（`2022 / 2023 / 2024`）：

- `text_only`
- `text_plus_item_attribute`
- `text_plus_user_profile`
- `text_plus_item_attribute_plus_user_profile`

按 `Recall@20 / NDCG@20` 汇总的结果：

| 实验 | Recall@20 mean±std | NDCG@20 mean±std |
| --- | --- | --- |
| text_only | 0.902148 ± 0.020417 | 0.445089 ± 0.003394 |
| text_plus_item_attribute | 0.902708 ± 0.019676 | 0.445779 ± 0.003524 |
| text_plus_user_profile | 0.953128 ± 0.014881 | 0.448483 ± 0.011512 |
| text_plus_item_attribute_plus_user_profile | 0.947339 ± 0.013194 | 0.445609 ± 0.010802 |

这里的结论很重要：

- `user_profile` 是最明显有效的增强信号。
- `item_attribute` 单独加进去，提升很小，接近于 marginal gain。
- `item_attribute + user_profile` 一起上，并没有继续提升，反而略低于 `text_plus_user_profile`。

从日志上看，`text_plus_item_attribute_plus_user_profile` 很多 run 在第 `1~5` 个 epoch 就达到最好结果，后面迅速 early stop，说明在这个小数据集上存在明显的过拟合或 feature 冲突风险。

#### 3.1.6 补做了 user profile 表示方式 smoke test

`2026-03-29` 补跑了 user profile variant 的 smoke：

- `pooled`
- `history_summary`
- `structured_profile`

当前保存在 `LLMRec/logs/` 里的 seed=2022 结果大致如下：

- `pooled`: Recall@20 = `0.93109`, NDCG@20 = `0.45291`
- `history_summary`: Recall@20 = `0.93221`, NDCG@20 = `0.44485`
- `structured_profile`: Recall@20 = `0.89132`, NDCG@20 = `0.39817`

注意两点：

1. 这个对比目前只是 smoke，不是完整 multi-seed 正式结论。
2. 但至少能看出 `structured_profile` 明显偏弱，优先级最低；`pooled` 和 `history_summary` 更值得继续。


### 3.2 第二阶段：做全量 Steam benchmark 数据链路

这条线的目标，是把 `NewData/` 下更完整的 Steam 原始 dump 变成可长期复用的 benchmark。

#### 3.2.1 先把原始 dump 规范化

用 `organize_newdata.py` 处理了以下原始文件：

- `NewData/steam_games.json`
- `NewData/australian_users_items.json`
- `NewData/australian_user_reviews.json`
- `NewData/steam_new.json`
- `NewData/bundle_data.json`

输出到 `NewData/processed/`：

- `item_catalog.csv`
- `user_library.csv`
- `reviews.csv`
- `bundle_items.csv`
- `dataset_summary.json`

从 `dataset_summary.json` 看，整理后的核心规模如下：

- item catalog：`32,132`
- user library user 数：`87,626`
- user library interaction 数：`5,153,209`
- 正向 playtime 交互：`3,285,246`
- review 总数：`7,852,374`
- review 覆盖 item：`16,283`
- bundle 数：`615`

其中 review 来源拆开看：

- `australian_user_reviews`: `59,305`
- `steam_new`: `7,793,069`

这是当前项目里真正的大盘数据基础。

#### 3.2.2 构建三套 benchmark

用 `build_steam_benchmarks.py` 基于正向 library playtime 和 review context，构建了：

- `warm_start`
- `cold_start`
- `long_tail`

输出目录是 `NewData/processed/benchmarks/`，总 manifest 在：

- `NewData/processed/benchmarks/benchmark_manifest.json`

当前使用的筛选阈值：

- `min_train_interactions = 3`
- `long_tail_min_item_support = 2`
- `long_tail_max_item_support = 3`
- `warm_min_item_support = 5`
- `max_cold_evals_per_user = 1`

三套 benchmark 的定义和规模如下：

#### warm_start

- 定义：seen-user / seen-item
- test 正例来自正向 library interaction，且原始 support 至少为 `5`
- train：`3,184,943` interactions
- test：`61,409` positives
- test users：`61,409`
- test items：`6,924`

#### cold_start

- 定义：seen-user / unseen-item
- test 正例来自 `australian_user_reviews` 里显式正向评论，且该 item 在 library 里没有正向 playtime support
- train：`3,246,352` interactions
- test：`1,979` positives
- test users：`1,979`
- test items：`89`

#### long_tail

- 定义：seen-user / rare-item
- test 正例来自原始 support 在 `2~3` 之间的正向 library interaction
- train：`3,245,055` interactions
- test：`1,297` positives
- test users：`1,297`
- test items：`817`

这一步除了 train/test 交互外，还把 item side context 也整理好了，包括：

- bundle 关联
- review 统计
- metadata
- user / item id map
- `train.json / val.json / test.json`

#### 3.2.3 把 benchmark 继续打成 LLMRec 输入包

之后用 `LLMRec/prepare_steam_benchmark_packages.py`，把三套 benchmark 转成：

- `LLMRec/data/steam_warm_start`
- `LLMRec/data/steam_cold_start`
- `LLMRec/data/steam_long_tail`

总体 manifest 在：

- `LLMRec/data/steam_benchmark_feature_packages.json`

当前三套包的规模如下：

| 数据集 | users | items | train | test |
| --- | --- | --- | --- | --- |
| steam_warm_start | 68,403 | 10,050 | 3,184,943 | 61,409 |
| steam_cold_start | 68,403 | 10,139 | 3,246,352 | 1,979 |
| steam_long_tail | 68,403 | 10,050 | 3,245,055 | 1,297 |

这里有几个实现细节，接手的人一定要知道：

1. 这三套 benchmark 包默认用的是 `hash` 文本特征，而不是 encoder 特征。
   原因很现实：全量数据规模太大，先用 hash 把链路打通。

2. `image_feat.npy` 目前是占位用的零向量。
   小规模 `steam` 包和 benchmark 包都是这样生成的，并没有真实图像特征。

3. `val.json` 目前是空的。
   benchmark 切分只提供了 train/test，没有单独再切 val。

4. 仍然保留了三种 user profile 变体：
   - `pooled`
   - `history_summary`
   - `structured_profile`

5. benchmark 包里也会生成：
   - `candidate_indices`
   - `augmented_atttribute_embedding_dict`
   - `augmented_user_init_embedding_*`
   - `user_profile_texts.json`


## 4. 当前项目状态判断

截至当前工作区里的产物，可以把项目状态概括成下面几条。

### 已经完成的

- 已把原始 LLMRec 改造成可以吃 Steam 数据。
- 已打通小规模 Steam MVP 链路。
- 已完成小规模 4x3 核心消融。
- 已初步测试 3 种 user profile 表示方式。
- 已打通全量 Steam 数据规范化链路。
- 已完成 warm/cold/long-tail benchmark 构建。
- 已把三套 benchmark 转成 LLMRec 可直接读取的数据包。

### 还没有完成的

- 还没有看到 `steam_warm_start / steam_cold_start / steam_long_tail` 的正式训练结果沉淀到仓库中。
- `data.csv` 这个清洗后的中间文件当前不在根目录，导致小规模链路不能无脑一步复现。
- 还没有统一的 one-click pipeline 或 Makefile。
- 也没有 git 历史可直接查；当前目录不是一个 Git 工作树。


## 5. 我对现阶段结果的判断

### 方向上成立的点

- 在小规模 Steam 数据上，`user_profile` 明显有用。
- 把用户历史转成语义表示，再喂给 LLMRec，确实比单纯 `text_only` 更强。
- 大规模 benchmark 的数据组织方式是合理的，三种评测场景也区分清楚了。

### 需要保守看待的点

- 小规模 `steam` 包最终只剩 `33` 个 item，实验更像方法 smoke，不适合拿来做最终结论。
- `item_attribute` 的收益很弱，目前不能说它对 Steam 一定有效。
- `item_attribute + user_profile` 组合在小规模数据上反而不稳定，值得在大盘上复核。
- benchmark 包当前使用 `hash` 特征，和 MVP 使用的 encoder 特征不是同一设定，结果不可直接横向类比。


## 6. 新同事接手时建议优先做的事

### 第一优先级

把 benchmark 包真正跑起来，至少补齐这三套数据集的正式训练结果：

- `steam_warm_start`
- `steam_cold_start`
- `steam_long_tail`

建议先从最稳的配置开始：

- `use_text_feat=1`
- `use_user_profile=1`
- `use_item_attribute=0`
- `use_image_feat=0`
- `use_sample_augmentation=0`
- `user_profile_variant=pooled`

### 第二优先级

把小规模链路补成可复现状态：

- 重新生成或找回 `data.csv`
- 明确 `output.csv -> data.csv -> games.csv -> merged_data.csv -> LLMRec/data/steam` 的完整命令链
- 最好写成脚本或 README

### 第三优先级

在全量 benchmark 上验证以下问题：

1. `pooled` 是否仍然是最优 user profile 变体
2. `history_summary` 在大盘上是否能超过 `pooled`
3. `item_attribute` 在大盘上到底有没有增益
4. `shared / separate / shared_adapter` 三种 text encoder mode 是否有差别


## 7. 关键复现命令备忘

### 小规模 MVP 链路

```bash
python eda_review_quality.py --input output.csv --output-dir eda_output --clean-output data.csv
python fetch_steam_games_metadata.py --input data.csv --output games.csv --cache steam_appdetails_cache.json
python merge_reviews_with_games.py --reviews data.csv --games games.csv --output merged_data.csv
cd LLMRec
python prepare_steam_mvp.py --input ../merged_data.csv --output-dir ./data/steam --text-encoder-device cpu
python run_ablation_matrix.py --dataset steam
python run_user_profile_ablation.py --dataset steam --use-item-attribute 0
```

### 全量 benchmark 链路

```bash
python organize_newdata.py --input-dir ./NewData --output-dir ./NewData/processed
python build_steam_benchmarks.py --input-dir ./NewData/processed --output-dir ./NewData/processed/benchmarks
python LLMRec/prepare_steam_benchmark_packages.py \
  --benchmark-root ./NewData/processed/benchmarks \
  --output-root ./LLMRec/data \
  --dataset-prefix steam
```

### 后续推荐先跑的训练命令

```bash
cd LLMRec
python main.py \
  --dataset steam_warm_start \
  --use_image_feat 0 \
  --use_text_feat 1 \
  --use_item_attribute 0 \
  --use_user_profile 1 \
  --use_sample_augmentation 0 \
  --user_profile_variant pooled \
  --experiment_name steam_warm_start_text_plus_user_profile
```


## 8. 最后补一句实际交接建议

如果只能先抓一件事，不要先回头抠小样本 `steam` 的指标，先去跑 `steam_warm_start / cold_start / long_tail`。原因很简单：

- 小样本链路已经足够说明方向；
- 真正还缺的是大盘结果；
- 现在仓库里最有价值、最稀缺的产物，其实是已经整理好的 benchmark 包。

