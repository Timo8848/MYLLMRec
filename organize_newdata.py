#!/usr/bin/env python3
"""
Organize the raw Steam dumps in NewData into normalized CSV tables.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


csv.field_size_limit(sys.maxsize)

STREAM_BUFFER_SIZE = 1024 * 1024
ITEM_FIELDS = [
    "app_id",
    "app_name",
    "title",
    "genres",
    "tags",
    "specs",
    "developer",
    "publisher",
    "release_date",
    "price",
    "discount_price",
    "sentiment",
    "early_access",
    "url",
    "reviews_url",
]
LIBRARY_FIELDS = [
    "source_user_id",
    "steam_id",
    "app_id",
    "item_name",
    "playtime_forever",
    "playtime_2weeks",
    "items_count",
    "has_positive_playtime",
    "has_recent_playtime",
    "user_url",
]
REVIEW_FIELDS = [
    "review_id",
    "source",
    "user_key",
    "source_user_id",
    "steam_id",
    "author_name",
    "app_id",
    "review_date",
    "recommend",
    "hours",
    "products",
    "page",
    "page_order",
    "early_access",
    "helpful",
    "funny",
    "found_funny",
    "compensation",
    "last_edited",
    "content",
]
BUNDLE_FIELDS = [
    "bundle_id",
    "bundle_name",
    "bundle_price",
    "bundle_final_price",
    "bundle_discount",
    "app_id",
    "item_name",
    "genre",
    "discounted_price",
    "item_url",
    "bundle_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Steam raw dumps under NewData.")
    parser.add_argument("--input-dir", default="./NewData")
    parser.add_argument("--output-dir", default="./NewData/processed")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="Print progress every N parsed lines for the large line-delimited files.",
    )
    return parser.parse_args()


def compact_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def join_multivalue(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (list, tuple, set)):
        return " | ".join(compact_whitespace(part) for part in value if compact_whitespace(part))
    return compact_whitespace(value)


def normalize_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def item_sort_key(app_id: str) -> tuple[int, int | str]:
    return (0, int(app_id)) if str(app_id).isdigit() else (1, str(app_id))


def parse_python_dict_lines(path: Path, progress_every: int) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="ignore", buffering=STREAM_BUFFER_SIZE) as handle:
        for line_no, line in enumerate(handle, start=1):
            if progress_every > 0 and line_no % progress_every == 0:
                print(f"[{path.name}] parsed {line_no:,} lines", flush=True)
            line = line.strip()
            if not line:
                continue
            yield line_no, ast.literal_eval(line)


def load_item_catalog(path: Path, progress_every: int) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    catalog: dict[str, dict[str, str]] = {}
    missing = Counter()
    duplicate_ids = 0
    for _, obj in parse_python_dict_lines(path, progress_every):
        app_id = compact_whitespace(obj.get("id"))
        if not app_id:
            missing["id"] += 1
            continue
        if app_id in catalog:
            duplicate_ids += 1
        row = {
            "app_id": app_id,
            "app_name": compact_whitespace(obj.get("app_name")),
            "title": compact_whitespace(obj.get("title")),
            "genres": join_multivalue(obj.get("genres")),
            "tags": join_multivalue(obj.get("tags")),
            "specs": join_multivalue(obj.get("specs")),
            "developer": compact_whitespace(obj.get("developer")),
            "publisher": compact_whitespace(obj.get("publisher")),
            "release_date": compact_whitespace(obj.get("release_date")),
            "price": compact_whitespace(obj.get("price")),
            "discount_price": compact_whitespace(obj.get("discount_price")),
            "sentiment": compact_whitespace(obj.get("sentiment")),
            "early_access": normalize_bool(obj.get("early_access")),
            "url": compact_whitespace(obj.get("url")),
            "reviews_url": compact_whitespace(obj.get("reviews_url")),
        }
        for key, value in row.items():
            if key != "app_id" and value == "":
                missing[key] += 1
        catalog[app_id] = row
    summary = {
        "unique_items": len(catalog),
        "duplicate_id_rows_overwritten": int(duplicate_ids),
        "missing_field_counts": dict(missing),
    }
    return catalog, summary


def write_item_catalog(path: Path, catalog: dict[str, dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITEM_FIELDS)
        writer.writeheader()
        for app_id in sorted(catalog, key=item_sort_key):
            writer.writerow(catalog[app_id])


def process_user_library(
    path: Path,
    output_path: Path,
    catalog_ids: set[str],
    progress_every: int,
) -> dict[str, Any]:
    unique_users: set[str] = set()
    unique_items: set[str] = set()
    interaction_rows = 0
    positive_playtime_rows = 0
    recent_playtime_rows = 0
    zero_playtime_rows = 0
    max_items_per_user = 0
    max_playtime_forever = 0
    max_playtime_2weeks = 0
    catalog_covered_items: set[str] = set()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIBRARY_FIELDS)
        writer.writeheader()
        for _, obj in parse_python_dict_lines(path, progress_every):
            source_user_id = compact_whitespace(obj.get("user_id"))
            if source_user_id:
                unique_users.add(source_user_id)
            items = obj.get("items") or []
            max_items_per_user = max(max_items_per_user, len(items))
            for item in items:
                app_id = compact_whitespace(item.get("item_id"))
                if not app_id:
                    continue
                playtime_forever = int(item.get("playtime_forever") or 0)
                playtime_2weeks = int(item.get("playtime_2weeks") or 0)
                unique_items.add(app_id)
                max_playtime_forever = max(max_playtime_forever, playtime_forever)
                max_playtime_2weeks = max(max_playtime_2weeks, playtime_2weeks)
                if playtime_forever > 0:
                    positive_playtime_rows += 1
                else:
                    zero_playtime_rows += 1
                if playtime_2weeks > 0:
                    recent_playtime_rows += 1
                if app_id in catalog_ids:
                    catalog_covered_items.add(app_id)
                writer.writerow(
                    {
                        "source_user_id": source_user_id,
                        "steam_id": compact_whitespace(obj.get("steam_id")),
                        "app_id": app_id,
                        "item_name": compact_whitespace(item.get("item_name")),
                        "playtime_forever": playtime_forever,
                        "playtime_2weeks": playtime_2weeks,
                        "items_count": int(obj.get("items_count") or 0),
                        "has_positive_playtime": normalize_bool(playtime_forever > 0),
                        "has_recent_playtime": normalize_bool(playtime_2weeks > 0),
                        "user_url": compact_whitespace(obj.get("user_url")),
                    }
                )
                interaction_rows += 1
    return {
        "user_rows": len(unique_users),
        "interaction_rows": int(interaction_rows),
        "unique_items": len(unique_items),
        "positive_playtime_rows": int(positive_playtime_rows),
        "recent_playtime_rows": int(recent_playtime_rows),
        "zero_playtime_rows": int(zero_playtime_rows),
        "catalog_item_overlap": len(catalog_covered_items),
        "catalog_item_overlap_ratio": round(len(catalog_covered_items) / len(unique_items), 6) if unique_items else 0.0,
        "max_items_per_user": int(max_items_per_user),
        "max_playtime_forever": int(max_playtime_forever),
        "max_playtime_2weeks": int(max_playtime_2weeks),
    }


def update_date_bounds(current_min: str | None, current_max: str | None, candidate: str) -> tuple[str | None, str | None]:
    if not candidate:
        return current_min, current_max
    try:
        parsed = datetime.strptime(candidate, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return current_min, current_max
    if current_min is None or parsed < current_min:
        current_min = parsed
    if current_max is None or parsed > current_max:
        current_max = parsed
    return current_min, current_max


def process_review_sources(
    australian_path: Path,
    steam_new_path: Path,
    output_path: Path,
    catalog_ids: set[str],
    progress_every: int,
) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {
        "australian_user_reviews": {
            "review_rows": 0,
            "user_rows": 0,
            "unique_items": 0,
            "positive_recommend_rows": 0,
            "negative_recommend_rows": 0,
            "empty_text_rows": 0,
            "catalog_item_overlap": 0,
        },
        "steam_new": {
            "review_rows": 0,
            "unique_items": 0,
            "empty_text_rows": 0,
            "catalog_item_overlap": 0,
            "dated_rows": 0,
            "min_review_date": None,
            "max_review_date": None,
        },
    }

    au_users: set[str] = set()
    au_items: set[str] = set()
    steam_items: set[str] = set()

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()

        for line_no, obj in parse_python_dict_lines(australian_path, progress_every):
            source_user_id = compact_whitespace(obj.get("user_id"))
            if source_user_id:
                au_users.add(source_user_id)
            for review_index, review in enumerate(obj.get("reviews") or []):
                app_id = compact_whitespace(review.get("item_id"))
                if not app_id:
                    continue
                au_items.add(app_id)
                recommend = review.get("recommend")
                content = compact_whitespace(review.get("review"))
                if app_id in catalog_ids:
                    summary["australian_user_reviews"]["catalog_item_overlap"] += 1 if app_id not in steam_items else 0
                writer.writerow(
                    {
                        "review_id": f"au:{line_no}:{review_index}",
                        "source": "australian_user_reviews",
                        "user_key": f"australian::{source_user_id}" if source_user_id else "",
                        "source_user_id": source_user_id,
                        "steam_id": "",
                        "author_name": "",
                        "app_id": app_id,
                        "review_date": compact_whitespace(review.get("posted")),
                        "recommend": normalize_bool(recommend),
                        "hours": "",
                        "products": "",
                        "page": "",
                        "page_order": "",
                        "early_access": "",
                        "helpful": compact_whitespace(review.get("helpful")),
                        "funny": compact_whitespace(review.get("funny")),
                        "found_funny": "",
                        "compensation": "",
                        "last_edited": compact_whitespace(review.get("last_edited")),
                        "content": content,
                    }
                )
                summary["australian_user_reviews"]["review_rows"] += 1
                if recommend is True:
                    summary["australian_user_reviews"]["positive_recommend_rows"] += 1
                elif recommend is False:
                    summary["australian_user_reviews"]["negative_recommend_rows"] += 1
                if not content:
                    summary["australian_user_reviews"]["empty_text_rows"] += 1

        for line_no, obj in parse_python_dict_lines(steam_new_path, progress_every):
            app_id = compact_whitespace(obj.get("product_id"))
            if not app_id:
                continue
            steam_items.add(app_id)
            content = compact_whitespace(obj.get("text"))
            review_date = compact_whitespace(obj.get("date"))
            if app_id in catalog_ids:
                summary["steam_new"]["catalog_item_overlap"] += 1 if app_id not in au_items else 0
            summary["steam_new"]["min_review_date"], summary["steam_new"]["max_review_date"] = update_date_bounds(
                summary["steam_new"]["min_review_date"],
                summary["steam_new"]["max_review_date"],
                review_date,
            )
            if review_date:
                summary["steam_new"]["dated_rows"] += 1
            writer.writerow(
                {
                    "review_id": f"steam_new:{line_no}",
                    "source": "steam_new",
                    "user_key": f"steam_new::{compact_whitespace(obj.get('username'))}" if compact_whitespace(obj.get("username")) else "",
                    "source_user_id": "",
                    "steam_id": "",
                    "author_name": compact_whitespace(obj.get("username")),
                    "app_id": app_id,
                    "review_date": review_date,
                    "recommend": "",
                    "hours": compact_whitespace(obj.get("hours")),
                    "products": compact_whitespace(obj.get("products")),
                    "page": compact_whitespace(obj.get("page")),
                    "page_order": compact_whitespace(obj.get("page_order")),
                    "early_access": normalize_bool(obj.get("early_access")),
                    "helpful": "",
                    "funny": "",
                    "found_funny": compact_whitespace(obj.get("found_funny")),
                    "compensation": compact_whitespace(obj.get("compensation")),
                    "last_edited": "",
                    "content": content,
                }
            )
            summary["steam_new"]["review_rows"] += 1
            if not content:
                summary["steam_new"]["empty_text_rows"] += 1

    summary["australian_user_reviews"]["user_rows"] = len(au_users)
    summary["australian_user_reviews"]["unique_items"] = len(au_items)
    summary["australian_user_reviews"]["catalog_item_overlap"] = sum(1 for item_id in au_items if item_id in catalog_ids)
    summary["australian_user_reviews"]["catalog_item_overlap_ratio"] = (
        round(summary["australian_user_reviews"]["catalog_item_overlap"] / len(au_items), 6) if au_items else 0.0
    )
    summary["steam_new"]["unique_items"] = len(steam_items)
    summary["steam_new"]["catalog_item_overlap"] = sum(1 for item_id in steam_items if item_id in catalog_ids)
    summary["steam_new"]["catalog_item_overlap_ratio"] = (
        round(summary["steam_new"]["catalog_item_overlap"] / len(steam_items), 6) if steam_items else 0.0
    )

    summary["combined"] = {
        "review_rows": int(
            summary["australian_user_reviews"]["review_rows"] + summary["steam_new"]["review_rows"]
        ),
        "unique_items": len(au_items | steam_items),
        "catalog_item_overlap": sum(1 for item_id in (au_items | steam_items) if item_id in catalog_ids),
    }
    return summary


def process_bundle_items(
    path: Path,
    output_path: Path,
    catalog_ids: set[str],
    progress_every: int,
) -> dict[str, Any]:
    bundle_ids: set[str] = set()
    bundled_items: set[str] = set()
    catalog_items: set[str] = set()
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BUNDLE_FIELDS)
        writer.writeheader()
        for _, obj in parse_python_dict_lines(path, progress_every):
            bundle_id = compact_whitespace(obj.get("bundle_id"))
            if bundle_id:
                bundle_ids.add(bundle_id)
            for item in obj.get("items") or []:
                app_id = compact_whitespace(item.get("item_id"))
                if not app_id:
                    continue
                bundled_items.add(app_id)
                if app_id in catalog_ids:
                    catalog_items.add(app_id)
                writer.writerow(
                    {
                        "bundle_id": bundle_id,
                        "bundle_name": compact_whitespace(obj.get("bundle_name")),
                        "bundle_price": compact_whitespace(obj.get("bundle_price")),
                        "bundle_final_price": compact_whitespace(obj.get("bundle_final_price")),
                        "bundle_discount": compact_whitespace(obj.get("bundle_discount")),
                        "app_id": app_id,
                        "item_name": compact_whitespace(item.get("item_name")),
                        "genre": compact_whitespace(item.get("genre")),
                        "discounted_price": compact_whitespace(item.get("discounted_price")),
                        "item_url": compact_whitespace(item.get("item_url")),
                        "bundle_url": compact_whitespace(obj.get("bundle_url")),
                    }
                )
                row_count += 1
    return {
        "bundle_rows": len(bundle_ids),
        "bundle_item_rows": int(row_count),
        "unique_items": len(bundled_items),
        "catalog_item_overlap": len(catalog_items),
        "catalog_item_overlap_ratio": round(len(catalog_items) / len(bundled_items), 6) if bundled_items else 0.0,
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    item_catalog_path = input_dir / "steam_games.json"
    users_items_path = input_dir / "australian_users_items.json"
    australian_reviews_path = input_dir / "australian_user_reviews.json"
    steam_new_reviews_path = input_dir / "steam_new.json"
    bundles_path = input_dir / "bundle_data.json"

    required = [
        item_catalog_path,
        users_items_path,
        australian_reviews_path,
        steam_new_reviews_path,
        bundles_path,
    ]
    missing_paths = [str(path) for path in required if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required input files: {', '.join(missing_paths)}")

    catalog, item_catalog_summary = load_item_catalog(item_catalog_path, args.progress_every)
    write_item_catalog(output_dir / "item_catalog.csv", catalog)

    catalog_ids = set(catalog)
    user_library_summary = process_user_library(
        users_items_path,
        output_dir / "user_library.csv",
        catalog_ids,
        args.progress_every,
    )
    review_summary = process_review_sources(
        australian_reviews_path,
        steam_new_reviews_path,
        output_dir / "reviews.csv",
        catalog_ids,
        args.progress_every,
    )
    bundle_summary = process_bundle_items(
        bundles_path,
        output_dir / "bundle_items.csv",
        catalog_ids,
        args.progress_every,
    )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "outputs": {
            "item_catalog_csv": str(output_dir / "item_catalog.csv"),
            "user_library_csv": str(output_dir / "user_library.csv"),
            "reviews_csv": str(output_dir / "reviews.csv"),
            "bundle_items_csv": str(output_dir / "bundle_items.csv"),
        },
        "item_catalog": item_catalog_summary,
        "user_library": user_library_summary,
        "reviews": review_summary,
        "bundle_items": bundle_summary,
    }
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Item catalog rows: {item_catalog_summary['unique_items']}")
    print(f"User library rows: {user_library_summary['interaction_rows']}")
    print(f"Review rows: {review_summary['combined']['review_rows']}")
    print(f"Bundle item rows: {bundle_summary['bundle_item_rows']}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
