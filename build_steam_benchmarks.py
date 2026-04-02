#!/usr/bin/env python3
"""
Build warm-start, cold-start, and long-tail benchmark splits from the normalized
Steam tables under NewData/processed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build warm-start, cold-start, and long-tail Steam benchmarks."
    )
    parser.add_argument("--input-dir", default="./NewData/processed")
    parser.add_argument("--output-dir", default="./NewData/processed/benchmarks")
    parser.add_argument(
        "--min-train-interactions",
        type=int,
        default=3,
        help="Minimum positive library interactions left in train for an evaluation user.",
    )
    parser.add_argument(
        "--long-tail-quantile",
        type=float,
        default=0.20,
        help="Quantile used to derive the long-tail max support when --long-tail-max-item-support is omitted.",
    )
    parser.add_argument(
        "--long-tail-max-item-support",
        type=int,
        default=None,
        help="Explicit maximum positive-library support for long-tail items.",
    )
    parser.add_argument(
        "--long-tail-min-item-support",
        type=int,
        default=2,
        help="Minimum positive-library support for long-tail evaluation items.",
    )
    parser.add_argument(
        "--warm-min-item-support",
        type=int,
        default=None,
        help=(
            "Minimum positive-library support for warm-start evaluation items before holdout. "
            "Defaults to long_tail_max_item_support + 2 so warm and long-tail stay disjoint after holdout."
        ),
    )
    parser.add_argument(
        "--max-cold-evals-per-user",
        type=int,
        default=1,
        help="Maximum number of cold-start positives kept per user.",
    )
    return parser.parse_args()


def compact_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def sort_key(value: str) -> tuple[int, int | str]:
    text = str(value or "").strip()
    return (0, int(text)) if text.isdigit() else (1, text)


def to_int(value: str) -> int:
    try:
        return int(float((value or "").strip()))
    except (TypeError, ValueError):
        return 0


def parse_iso_date(value: str) -> str:
    text = compact_whitespace(value)
    if not text:
        return ""
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return ""


def join_unique_sorted(values: set[str]) -> str:
    cleaned = {compact_whitespace(value) for value in values if compact_whitespace(value)}
    return " | ".join(sorted(cleaned))


def quantile_support(counts: list[int], quantile: float) -> int:
    if not counts:
        return 0
    bounded = min(1.0, max(0.0, quantile))
    index = int((len(counts) - 1) * bounded)
    return counts[index]


def load_item_catalog(path: Path) -> dict[str, dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            app_id = compact_whitespace(row.get("app_id"))
            if not app_id:
                continue
            items[app_id] = {key: compact_whitespace(value) for key, value in row.items()}
    return items


def load_bundle_stats(path: Path) -> dict[str, dict[str, str | int]]:
    bundle_ids_by_item: dict[str, set[str]] = defaultdict(set)
    bundle_names_by_item: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            app_id = compact_whitespace(row.get("app_id"))
            if not app_id:
                continue
            bundle_id = compact_whitespace(row.get("bundle_id"))
            bundle_name = compact_whitespace(row.get("bundle_name"))
            if bundle_id:
                bundle_ids_by_item[app_id].add(bundle_id)
            if bundle_name:
                bundle_names_by_item[app_id].add(bundle_name)

    stats: dict[str, dict[str, str | int]] = {}
    for app_id in sorted(set(bundle_ids_by_item) | set(bundle_names_by_item), key=sort_key):
        stats[app_id] = {
            "bundle_count": len(bundle_ids_by_item.get(app_id, set())),
            "bundle_ids": join_unique_sorted(bundle_ids_by_item.get(app_id, set())),
            "bundle_names": join_unique_sorted(bundle_names_by_item.get(app_id, set())),
        }
    return stats


def load_positive_library(
    path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, str | int]],
    dict[str, set[str]],
    Counter[str],
    dict[str, int],
]:
    pair_records: dict[tuple[str, str], dict[str, str | int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if compact_whitespace(row.get("has_positive_playtime")) != "true":
                continue
            user_id = compact_whitespace(row.get("steam_id") or row.get("source_user_id"))
            app_id = compact_whitespace(row.get("app_id"))
            if not user_id or not app_id:
                continue
            key = (user_id, app_id)
            playtime_forever = to_int(row.get("playtime_forever") or "")
            playtime_2weeks = to_int(row.get("playtime_2weeks") or "")
            current = pair_records.get(key)
            if current and int(current["playtime_forever"]) > playtime_forever:
                continue
            pair_records[key] = {
                "user_id": user_id,
                "app_id": app_id,
                "item_name": compact_whitespace(row.get("item_name")),
                "playtime_forever": playtime_forever,
                "playtime_2weeks": playtime_2weeks,
                "has_recent_playtime": compact_whitespace(row.get("has_recent_playtime")),
            }

    user_to_items: dict[str, set[str]] = defaultdict(set)
    item_support: Counter[str] = Counter()
    for user_id, app_id in pair_records:
        user_to_items[user_id].add(app_id)
        item_support[app_id] += 1
    user_support = {user_id: len(items) for user_id, items in user_to_items.items()}
    return pair_records, user_to_items, item_support, user_support


def new_review_stats() -> dict[str, str | int]:
    return {
        "total_review_count": 0,
        "australian_positive_review_count": 0,
        "australian_negative_review_count": 0,
        "steam_new_review_count": 0,
        "first_dated_review": "",
        "last_dated_review": "",
    }


def update_date_range(stats: dict[str, str | int], candidate: str) -> None:
    normalized = parse_iso_date(candidate)
    if not normalized:
        return
    current_min = str(stats["first_dated_review"] or "")
    current_max = str(stats["last_dated_review"] or "")
    if not current_min or normalized < current_min:
        stats["first_dated_review"] = normalized
    if not current_max or normalized > current_max:
        stats["last_dated_review"] = normalized


def load_review_context(
    path: Path,
    catalog_ids: set[str],
    library_users: set[str],
    positive_library_items: set[str],
) -> tuple[dict[str, dict[str, str | int]], dict[str, list[dict[str, str]]]]:
    review_stats: dict[str, dict[str, str | int]] = defaultdict(new_review_stats)
    cold_candidates_by_pair: dict[tuple[str, str], dict[str, str]] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            app_id = compact_whitespace(row.get("app_id"))
            if not app_id:
                continue

            stats = review_stats[app_id]
            stats["total_review_count"] = int(stats["total_review_count"]) + 1

            source = compact_whitespace(row.get("source"))
            recommend = compact_whitespace(row.get("recommend"))
            if source == "australian_user_reviews":
                if recommend == "true":
                    stats["australian_positive_review_count"] = int(stats["australian_positive_review_count"]) + 1
                elif recommend == "false":
                    stats["australian_negative_review_count"] = int(stats["australian_negative_review_count"]) + 1
            elif source == "steam_new":
                stats["steam_new_review_count"] = int(stats["steam_new_review_count"]) + 1

            update_date_range(stats, compact_whitespace(row.get("review_date")))

            if source != "australian_user_reviews" or recommend != "true":
                continue

            user_id = compact_whitespace(row.get("source_user_id") or row.get("steam_id"))
            if not user_id or user_id not in library_users:
                continue
            if app_id not in catalog_ids or app_id in positive_library_items:
                continue

            content = compact_whitespace(row.get("content"))
            key = (user_id, app_id)
            candidate = {
                "user_id": user_id,
                "app_id": app_id,
                "review_id": compact_whitespace(row.get("review_id")),
                "review_date": compact_whitespace(row.get("review_date")),
                "review_text": content,
                "review_text_length": str(len(content)),
                "source": source,
            }
            existing = cold_candidates_by_pair.get(key)
            if not existing or len(candidate["review_text"]) > len(existing["review_text"]):
                cold_candidates_by_pair[key] = candidate

    cold_candidates_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in cold_candidates_by_pair.values():
        cold_candidates_by_user[candidate["user_id"]].append(candidate)
    return review_stats, cold_candidates_by_user


def choose_warm_holdouts(
    user_to_items: dict[str, set[str]],
    pair_records: dict[tuple[str, str], dict[str, str | int]],
    item_support: Counter[str],
    *,
    min_train_interactions: int,
    warm_min_item_support: int,
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for user_id in sorted(user_to_items, key=sort_key):
        user_items = user_to_items[user_id]
        if len(user_items) - 1 < min_train_interactions:
            continue
        candidates = [
            item_id
            for item_id in user_items
            if item_support[item_id] >= warm_min_item_support
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item_id: (
                item_support[item_id],
                -int(pair_records[(user_id, item_id)]["playtime_forever"]),
                sort_key(item_id),
            )
        )
        selected[user_id] = candidates[0]
    return selected


def choose_long_tail_holdouts(
    user_to_items: dict[str, set[str]],
    pair_records: dict[tuple[str, str], dict[str, str | int]],
    item_support: Counter[str],
    *,
    min_train_interactions: int,
    long_tail_min_item_support: int,
    long_tail_max_item_support: int,
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for user_id in sorted(user_to_items, key=sort_key):
        user_items = user_to_items[user_id]
        if len(user_items) - 1 < min_train_interactions:
            continue
        candidates = [
            item_id
            for item_id in user_items
            if long_tail_min_item_support <= item_support[item_id] <= long_tail_max_item_support
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item_id: (
                item_support[item_id],
                -int(pair_records[(user_id, item_id)]["playtime_forever"]),
                sort_key(item_id),
            )
        )
        selected[user_id] = candidates[0]
    return selected


def choose_cold_rows(
    cold_candidates_by_user: dict[str, list[dict[str, str]]],
    user_support: dict[str, int],
    *,
    min_train_interactions: int,
    max_cold_evals_per_user: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for user_id in sorted(cold_candidates_by_user, key=sort_key):
        if user_support.get(user_id, 0) < min_train_interactions:
            continue
        candidates = sorted(
            cold_candidates_by_user[user_id],
            key=lambda row: (
                -int(row["review_text_length"]),
                sort_key(row["app_id"]),
                row["review_id"],
            ),
        )
        rows.extend(candidates[: max(1, max_cold_evals_per_user)])
    return rows


def build_train_rows(
    pair_records: dict[tuple[str, str], dict[str, str | int]],
    holdout_pairs: set[tuple[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for user_id, app_id in sorted(pair_records, key=lambda pair: (sort_key(pair[0]), sort_key(pair[1]))):
        if (user_id, app_id) in holdout_pairs:
            continue
        record = pair_records[(user_id, app_id)]
        rows.append(
            {
                "user_id": user_id,
                "app_id": app_id,
                "source": "user_library_positive_playtime",
                "playtime_forever": str(record["playtime_forever"]),
                "playtime_2weeks": str(record["playtime_2weeks"]),
                "has_recent_playtime": str(record["has_recent_playtime"]),
            }
        )
    return rows


def build_warm_eval_rows(
    selected_items_by_user: dict[str, str],
    pair_records: dict[tuple[str, str], dict[str, str | int]],
    item_support: Counter[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for user_id in sorted(selected_items_by_user, key=sort_key):
        app_id = selected_items_by_user[user_id]
        record = pair_records[(user_id, app_id)]
        rows.append(
            {
                "user_id": user_id,
                "app_id": app_id,
                "label": "1",
                "benchmark": "warm_start",
                "ground_truth_source": "user_library_positive_playtime",
                "source_support_before_holdout": str(item_support[app_id]),
                "source_support_after_holdout": str(item_support[app_id] - 1),
                "playtime_forever": str(record["playtime_forever"]),
                "playtime_2weeks": str(record["playtime_2weeks"]),
                "review_id": "",
                "review_date": "",
                "review_text": "",
            }
        )
    return rows


def build_long_tail_eval_rows(
    selected_items_by_user: dict[str, str],
    pair_records: dict[tuple[str, str], dict[str, str | int]],
    item_support: Counter[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for user_id in sorted(selected_items_by_user, key=sort_key):
        app_id = selected_items_by_user[user_id]
        record = pair_records[(user_id, app_id)]
        rows.append(
            {
                "user_id": user_id,
                "app_id": app_id,
                "label": "1",
                "benchmark": "long_tail",
                "ground_truth_source": "user_library_positive_playtime",
                "source_support_before_holdout": str(item_support[app_id]),
                "source_support_after_holdout": str(item_support[app_id] - 1),
                "playtime_forever": str(record["playtime_forever"]),
                "playtime_2weeks": str(record["playtime_2weeks"]),
                "review_id": "",
                "review_date": "",
                "review_text": "",
            }
        )
    return rows


def build_cold_eval_rows(cold_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in cold_rows:
        rows.append(
            {
                "user_id": row["user_id"],
                "app_id": row["app_id"],
                "label": "1",
                "benchmark": "cold_start",
                "ground_truth_source": "australian_positive_review",
                "source_support_before_holdout": "0",
                "source_support_after_holdout": "0",
                "playtime_forever": "",
                "playtime_2weeks": "",
                "review_id": row["review_id"],
                "review_date": row["review_date"],
                "review_text": row["review_text"],
            }
        )
    return rows


def count_items(rows: list[dict[str, str]], item_key: str = "app_id") -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        app_id = compact_whitespace(row.get(item_key))
        if app_id:
            counter[app_id] += 1
    return counter


def count_users(rows: list[dict[str, str]], user_key: str = "user_id") -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        user_id = compact_whitespace(row.get(user_key))
        if user_id:
            counter[user_id] += 1
    return counter


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_json_mappings(
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
) -> tuple[
    dict[int, list[int]],
    dict[int, list[int]],
    list[str],
    list[str],
    dict[str, int],
    dict[str, int],
]:
    user_ids = sorted(
        {
            compact_whitespace(row.get("user_id"))
            for row in train_rows + eval_rows
            if compact_whitespace(row.get("user_id"))
        },
        key=sort_key,
    )
    item_ids = sorted(
        {
            compact_whitespace(row.get("app_id"))
            for row in train_rows + eval_rows
            if compact_whitespace(row.get("app_id"))
        },
        key=sort_key,
    )

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_idx = {app_id: idx for idx, app_id in enumerate(item_ids)}

    train_mapping: dict[int, list[int]] = defaultdict(list)
    test_mapping: dict[int, list[int]] = defaultdict(list)

    for row in train_rows:
        train_mapping[user_id_to_idx[row["user_id"]]].append(item_id_to_idx[row["app_id"]])
    for row in eval_rows:
        test_mapping[user_id_to_idx[row["user_id"]]].append(item_id_to_idx[row["app_id"]])

    train_mapping = {
        user_idx: sorted(item_indices)
        for user_idx, item_indices in sorted(train_mapping.items())
        if item_indices
    }
    test_mapping = {
        user_idx: sorted(item_indices)
        for user_idx, item_indices in sorted(test_mapping.items())
        if item_indices
    }
    return train_mapping, test_mapping, user_ids, item_ids, user_id_to_idx, item_id_to_idx


def write_json_mapping(path: Path, mapping: dict[int, list[int]]) -> None:
    payload = {str(user_idx): item_indices for user_idx, item_indices in mapping.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_user_rows(
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    train_counts = count_users(train_rows)
    eval_counts = count_users(eval_rows)
    users = sorted(set(train_counts) | set(eval_counts), key=sort_key)
    rows: list[dict[str, str]] = []
    for user_id in users:
        rows.append(
            {
                "user_id": user_id,
                "train_interaction_count": str(train_counts.get(user_id, 0)),
                "eval_interaction_count": str(eval_counts.get(user_id, 0)),
                "participates_in_eval": "true" if eval_counts.get(user_id, 0) else "false",
            }
        )
    return rows


def build_item_rows(
    benchmark_name: str,
    item_ids: set[str],
    *,
    catalog: dict[str, dict[str, str]],
    bundle_stats: dict[str, dict[str, str | int]],
    review_stats: dict[str, dict[str, str | int]],
    train_item_counts: Counter[str],
    eval_item_counts: Counter[str],
    long_tail_max_item_support: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for app_id in sorted(item_ids, key=sort_key):
        metadata = catalog.get(app_id, {})
        bundle = bundle_stats.get(app_id, {})
        reviews = review_stats.get(app_id, new_review_stats())
        train_support = train_item_counts.get(app_id, 0)
        in_eval = eval_item_counts.get(app_id, 0) > 0
        if in_eval:
            benchmark_regime = benchmark_name
        elif train_support == 0:
            benchmark_regime = "metadata_only"
        elif train_support <= long_tail_max_item_support:
            benchmark_regime = "train_long_tail"
        else:
            benchmark_regime = "train_warm"
        rows.append(
            {
                "app_id": app_id,
                "app_name": metadata.get("app_name", ""),
                "title": metadata.get("title", ""),
                "genres": metadata.get("genres", ""),
                "tags": metadata.get("tags", ""),
                "specs": metadata.get("specs", ""),
                "developer": metadata.get("developer", ""),
                "publisher": metadata.get("publisher", ""),
                "release_date": metadata.get("release_date", ""),
                "price": metadata.get("price", ""),
                "discount_price": metadata.get("discount_price", ""),
                "sentiment": metadata.get("sentiment", ""),
                "early_access": metadata.get("early_access", ""),
                "url": metadata.get("url", ""),
                "reviews_url": metadata.get("reviews_url", ""),
                "bundle_count": str(bundle.get("bundle_count", 0)),
                "bundle_ids": str(bundle.get("bundle_ids", "")),
                "bundle_names": str(bundle.get("bundle_names", "")),
                "total_review_count": str(reviews["total_review_count"]),
                "australian_positive_review_count": str(reviews["australian_positive_review_count"]),
                "australian_negative_review_count": str(reviews["australian_negative_review_count"]),
                "steam_new_review_count": str(reviews["steam_new_review_count"]),
                "first_dated_review": str(reviews["first_dated_review"]),
                "last_dated_review": str(reviews["last_dated_review"]),
                "train_positive_support": str(train_support),
                "eval_positive_support": str(eval_item_counts.get(app_id, 0)),
                "in_train": "true" if train_support else "false",
                "in_eval": "true" if in_eval else "false",
                "benchmark_item_regime": benchmark_regime,
            }
        )
    return rows


def write_id_maps(
    benchmark_dir: Path,
    user_ids: list[str],
    item_ids: list[str],
) -> None:
    with (benchmark_dir / "user_id_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_idx", "user_id"])
        for idx, user_id in enumerate(user_ids):
            writer.writerow([idx, user_id])

    with (benchmark_dir / "item_id_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_idx", "app_id"])
        for idx, app_id in enumerate(item_ids):
            writer.writerow([idx, app_id])


def build_summary(
    benchmark_name: str,
    definition: str,
    benchmark_dir: Path,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    item_rows: list[dict[str, str]],
    *,
    long_tail_min_item_support: int,
    long_tail_max_item_support: int,
    warm_min_item_support: int,
    min_train_interactions: int,
) -> dict[str, Any]:
    train_users = count_users(train_rows)
    eval_users = count_users(eval_rows)
    train_items = count_items(train_rows)
    eval_items = count_items(eval_rows)
    eval_items_with_bundle = sum(1 for row in item_rows if row["in_eval"] == "true" and int(row["bundle_count"]) > 0)
    eval_items_with_reviews = sum(1 for row in item_rows if row["in_eval"] == "true" and int(row["total_review_count"]) > 0)

    summary = {
        "benchmark": benchmark_name,
        "definition": definition,
        "paths": {
            "directory": str(benchmark_dir),
            "train_interactions_csv": str(benchmark_dir / "train_interactions.csv"),
            "test_interactions_csv": str(benchmark_dir / "test_interactions.csv"),
            "items_csv": str(benchmark_dir / "items.csv"),
            "users_csv": str(benchmark_dir / "users.csv"),
            "train_json": str(benchmark_dir / "train.json"),
            "val_json": str(benchmark_dir / "val.json"),
            "test_json": str(benchmark_dir / "test.json"),
            "user_id_map_csv": str(benchmark_dir / "user_id_map.csv"),
            "item_id_map_csv": str(benchmark_dir / "item_id_map.csv"),
        },
        "selection": {
            "min_train_interactions": min_train_interactions,
            "long_tail_min_item_support": long_tail_min_item_support,
            "long_tail_max_item_support": long_tail_max_item_support,
            "warm_min_item_support": warm_min_item_support,
        },
        "train": {
            "user_count": len(train_users),
            "item_count": len(train_items),
            "interaction_count": len(train_rows),
        },
        "test": {
            "user_count": len(eval_users),
            "item_count": len(eval_items),
            "interaction_count": len(eval_rows),
        },
        "test_item_context": {
            "items_with_bundle_context": eval_items_with_bundle,
            "items_with_any_review_context": eval_items_with_reviews,
            "train_item_overlap_count": len(set(train_items) & set(eval_items)),
        },
    }
    return summary


def build_and_write_benchmark(
    benchmark_name: str,
    definition: str,
    benchmark_dir: Path,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    *,
    catalog: dict[str, dict[str, str]],
    bundle_stats: dict[str, dict[str, str | int]],
    review_stats: dict[str, dict[str, str | int]],
    long_tail_min_item_support: int,
    long_tail_max_item_support: int,
    warm_min_item_support: int,
    min_train_interactions: int,
) -> dict[str, Any]:
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    train_fieldnames = [
        "user_id",
        "app_id",
        "source",
        "playtime_forever",
        "playtime_2weeks",
        "has_recent_playtime",
    ]
    eval_fieldnames = [
        "user_id",
        "app_id",
        "label",
        "benchmark",
        "ground_truth_source",
        "source_support_before_holdout",
        "source_support_after_holdout",
        "playtime_forever",
        "playtime_2weeks",
        "review_id",
        "review_date",
        "review_text",
    ]

    train_item_counts = count_items(train_rows)
    eval_item_counts = count_items(eval_rows)
    item_rows = build_item_rows(
        benchmark_name,
        item_ids=set(train_item_counts) | set(eval_item_counts),
        catalog=catalog,
        bundle_stats=bundle_stats,
        review_stats=review_stats,
        train_item_counts=train_item_counts,
        eval_item_counts=eval_item_counts,
        long_tail_max_item_support=long_tail_max_item_support,
    )
    user_rows = build_user_rows(train_rows, eval_rows)

    write_csv(benchmark_dir / "train_interactions.csv", train_fieldnames, train_rows)
    write_csv(benchmark_dir / "test_interactions.csv", eval_fieldnames, eval_rows)
    write_csv(benchmark_dir / "items.csv", list(item_rows[0].keys()) if item_rows else [], item_rows)
    write_csv(benchmark_dir / "users.csv", list(user_rows[0].keys()) if user_rows else [], user_rows)

    train_mapping, test_mapping, user_ids, item_ids, _, _ = build_json_mappings(train_rows, eval_rows)
    write_json_mapping(benchmark_dir / "train.json", train_mapping)
    write_json_mapping(benchmark_dir / "val.json", {})
    write_json_mapping(benchmark_dir / "test.json", test_mapping)
    write_id_maps(benchmark_dir, user_ids, item_ids)

    summary = build_summary(
        benchmark_name,
        definition,
        benchmark_dir,
        train_rows,
        eval_rows,
        item_rows,
        long_tail_min_item_support=long_tail_min_item_support,
        long_tail_max_item_support=long_tail_max_item_support,
        warm_min_item_support=warm_min_item_support,
        min_train_interactions=min_train_interactions,
    )
    write_json(benchmark_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    item_catalog_path = input_dir / "item_catalog.csv"
    user_library_path = input_dir / "user_library.csv"
    reviews_path = input_dir / "reviews.csv"
    bundle_items_path = input_dir / "bundle_items.csv"

    required_paths = [item_catalog_path, user_library_path, reviews_path, bundle_items_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required input files: {', '.join(missing_paths)}")

    catalog = load_item_catalog(item_catalog_path)
    bundle_stats = load_bundle_stats(bundle_items_path)
    pair_records, user_to_items, item_support, user_support = load_positive_library(user_library_path)
    review_stats, cold_candidates_by_user = load_review_context(
        reviews_path,
        catalog_ids=set(catalog),
        library_users=set(user_to_items),
        positive_library_items=set(item_support),
    )

    item_support_values = sorted(item_support.values())
    long_tail_max_item_support = (
        int(args.long_tail_max_item_support)
        if args.long_tail_max_item_support is not None
        else quantile_support(item_support_values, args.long_tail_quantile)
    )
    long_tail_max_item_support = max(args.long_tail_min_item_support, long_tail_max_item_support)
    warm_min_item_support = (
        int(args.warm_min_item_support)
        if args.warm_min_item_support is not None
        else long_tail_max_item_support + 2
    )

    warm_holdouts = choose_warm_holdouts(
        user_to_items,
        pair_records,
        item_support,
        min_train_interactions=args.min_train_interactions,
        warm_min_item_support=warm_min_item_support,
    )
    long_tail_holdouts = choose_long_tail_holdouts(
        user_to_items,
        pair_records,
        item_support,
        min_train_interactions=args.min_train_interactions,
        long_tail_min_item_support=args.long_tail_min_item_support,
        long_tail_max_item_support=long_tail_max_item_support,
    )
    cold_rows = choose_cold_rows(
        cold_candidates_by_user,
        user_support,
        min_train_interactions=args.min_train_interactions,
        max_cold_evals_per_user=args.max_cold_evals_per_user,
    )

    warm_train_rows = build_train_rows(pair_records, {(user_id, app_id) for user_id, app_id in warm_holdouts.items()})
    warm_eval_rows = build_warm_eval_rows(warm_holdouts, pair_records, item_support)

    long_tail_train_rows = build_train_rows(
        pair_records,
        {(user_id, app_id) for user_id, app_id in long_tail_holdouts.items()},
    )
    long_tail_eval_rows = build_long_tail_eval_rows(long_tail_holdouts, pair_records, item_support)

    cold_train_rows = build_train_rows(pair_records, set())
    cold_eval_rows = build_cold_eval_rows(cold_rows)

    warm_summary = build_and_write_benchmark(
        "warm_start",
        (
            "Seen-user / seen-item evaluation. Test positives come from positive-library interactions "
            f"whose source support is at least {warm_min_item_support} before holdout, so they remain outside "
            "the long-tail bucket after removal."
        ),
        output_dir / "warm_start",
        warm_train_rows,
        warm_eval_rows,
        catalog=catalog,
        bundle_stats=bundle_stats,
        review_stats=review_stats,
        long_tail_min_item_support=args.long_tail_min_item_support,
        long_tail_max_item_support=long_tail_max_item_support,
        warm_min_item_support=warm_min_item_support,
        min_train_interactions=args.min_train_interactions,
    )
    cold_summary = build_and_write_benchmark(
        "cold_start",
        (
            "Seen-user / unseen-item evaluation. Test positives come from explicit positive "
            "australian_user_reviews rows whose items have zero positive-playtime support in the library."
        ),
        output_dir / "cold_start",
        cold_train_rows,
        cold_eval_rows,
        catalog=catalog,
        bundle_stats=bundle_stats,
        review_stats=review_stats,
        long_tail_min_item_support=args.long_tail_min_item_support,
        long_tail_max_item_support=long_tail_max_item_support,
        warm_min_item_support=warm_min_item_support,
        min_train_interactions=args.min_train_interactions,
    )
    long_tail_summary = build_and_write_benchmark(
        "long_tail",
        (
            "Seen-user / rare-item evaluation. Test positives come from positive-library interactions whose "
            f"source support is between {args.long_tail_min_item_support} and {long_tail_max_item_support} before holdout."
        ),
        output_dir / "long_tail",
        long_tail_train_rows,
        long_tail_eval_rows,
        catalog=catalog,
        bundle_stats=bundle_stats,
        review_stats=review_stats,
        long_tail_min_item_support=args.long_tail_min_item_support,
        long_tail_max_item_support=long_tail_max_item_support,
        warm_min_item_support=warm_min_item_support,
        min_train_interactions=args.min_train_interactions,
    )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_table_paths": {
            "item_catalog_csv": str(item_catalog_path),
            "user_library_csv": str(user_library_path),
            "reviews_csv": str(reviews_path),
            "bundle_items_csv": str(bundle_items_path),
        },
        "selection": {
            "min_train_interactions": args.min_train_interactions,
            "long_tail_quantile": args.long_tail_quantile,
            "long_tail_min_item_support": args.long_tail_min_item_support,
            "long_tail_max_item_support": long_tail_max_item_support,
            "warm_min_item_support": warm_min_item_support,
            "max_cold_evals_per_user": args.max_cold_evals_per_user,
        },
        "source_positive_library": {
            "user_count": len(user_to_items),
            "item_count": len(item_support),
            "interaction_count": len(pair_records),
        },
        "benchmarks": {
            "warm_start": warm_summary,
            "cold_start": cold_summary,
            "long_tail": long_tail_summary,
        },
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(
        "Thresholds: "
        f"long_tail_support=[{args.long_tail_min_item_support}, {long_tail_max_item_support}], "
        f"warm_min_support={warm_min_item_support}, "
        f"min_train_interactions={args.min_train_interactions}"
    )
    for benchmark_name, summary in [
        ("warm_start", warm_summary),
        ("cold_start", cold_summary),
        ("long_tail", long_tail_summary),
    ]:
        print(
            f"{benchmark_name}: "
            f"train={summary['train']['interaction_count']} interactions, "
            f"test={summary['test']['interaction_count']} positives, "
            f"test_users={summary['test']['user_count']}, "
            f"test_items={summary['test']['item_count']}"
        )
    print(f"Manifest: {output_dir / 'benchmark_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
