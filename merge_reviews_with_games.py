#!/usr/bin/env python3
"""
Merge cleaned review data with Steam game metadata by app_id.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict


csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge review rows with game metadata on app_id.")
    parser.add_argument(
        "--reviews",
        default="data.csv",
        help="Cleaned review CSV path. Defaults to data.csv.",
    )
    parser.add_argument(
        "--games",
        default="games.csv",
        help="Game metadata CSV path. Defaults to games.csv.",
    )
    parser.add_argument(
        "--output",
        default="merged_data.csv",
        help="Merged CSV output path. Defaults to merged_data.csv.",
    )
    return parser.parse_args()


def load_games(games_path: Path) -> Dict[str, Dict[str, str]]:
    with games_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"app_id", "name", "short_description", "genres"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in games file: {', '.join(sorted(missing))}")

        games_by_app_id: Dict[str, Dict[str, str]] = {}
        for row in reader:
            app_id = (row.get("app_id") or "").strip()
            if not app_id:
                continue
            games_by_app_id[app_id] = {
                "name": row.get("name") or "",
                "short_description": row.get("short_description") or "",
                "genres": row.get("genres") or "",
            }
    return games_by_app_id


def merge_reviews(reviews_path: Path, games_by_app_id: Dict[str, Dict[str, str]], output_path: Path) -> tuple[int, int]:
    merged_rows = 0
    missing_metadata_rows = 0

    with reviews_path.open("r", encoding="utf-8", newline="") as review_handle:
        reader = csv.DictReader(review_handle)
        required = {"id", "app_id", "content", "author_id", "is_positive"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in reviews file: {', '.join(sorted(missing))}")

        fieldnames = [
            "id",
            "app_id",
            "name",
            "short_description",
            "genres",
            "content",
            "author_id",
            "is_positive",
        ]

        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                app_id = (row.get("app_id") or "").strip()
                game = games_by_app_id.get(app_id, {})
                if not game:
                    missing_metadata_rows += 1

                writer.writerow(
                    {
                        "id": row.get("id") or "",
                        "app_id": app_id,
                        "name": game.get("name", ""),
                        "short_description": game.get("short_description", ""),
                        "genres": game.get("genres", ""),
                        "content": row.get("content") or "",
                        "author_id": row.get("author_id") or "",
                        "is_positive": row.get("is_positive") or "",
                    }
                )
                merged_rows += 1

    return merged_rows, missing_metadata_rows


def main() -> int:
    args = parse_args()
    reviews_path = Path(args.reviews).expanduser().resolve()
    games_path = Path(args.games).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not reviews_path.exists():
        raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
    if not games_path.exists():
        raise FileNotFoundError(f"Games file not found: {games_path}")

    games_by_app_id = load_games(games_path)
    merged_rows, missing_metadata_rows = merge_reviews(reviews_path, games_by_app_id, output_path)

    print(f"Reviews file: {reviews_path}")
    print(f"Games file: {games_path}")
    print(f"Game metadata rows loaded: {len(games_by_app_id)}")
    print(f"Merged rows written: {merged_rows}")
    print(f"Rows missing metadata: {missing_metadata_rows}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
