#!/usr/bin/env python3
"""
Fetch Steam game metadata for app_ids found in a review dataset.

Input:
- A CSV file with an `app_id` column, typically the cleaned `data.csv`.

Output:
- A `games.csv` file with one row per unique app_id and the requested fields:
  `app_id`, `name`, `short_description`, `genres`

Design goals:
- Keep dependencies to the Python standard library.
- Cache successful responses to avoid refetching the same app_id.
- Retry transient failures with exponential backoff.
- Continue through partial failures and still emit a usable CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List


csv.field_size_limit(sys.maxsize)

STEAM_APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Steam metadata for unique app_ids.")
    parser.add_argument(
        "--input",
        default="data.csv",
        help="Input CSV containing an `app_id` column. Defaults to data.csv.",
    )
    parser.add_argument(
        "--output",
        default="games.csv",
        help="Output CSV path. Defaults to games.csv.",
    )
    parser.add_argument(
        "--cache",
        default="steam_appdetails_cache.json",
        help="JSON cache file for API responses. Defaults to steam_appdetails_cache.json.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.35,
        help="Base delay between successful requests. Defaults to 0.35 seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP request timeout in seconds. Defaults to 20.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per app_id for transient failures. Defaults to 5.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached entries and refetch everything.",
    )
    return parser.parse_args()


def load_unique_app_ids(input_path: Path) -> List[str]:
    app_ids = set()
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "app_id" not in (reader.fieldnames or []):
            raise ValueError(f"`app_id` column not found in {input_path}")
        for row in reader:
            app_id = (row.get("app_id") or "").strip()
            if app_id:
                app_ids.add(app_id)
    return sorted(app_ids, key=lambda value: int(value))


def load_cache(cache_path: Path) -> Dict[str, Dict[str, object]]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_cache(cache_path: Path, cache: Dict[str, Dict[str, object]]) -> None:
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_request_url(app_id: str) -> str:
    params = urllib.parse.urlencode({"appids": app_id})
    return f"{STEAM_APPDETAILS_URL}?{params}"


def create_ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_one(app_id: str, timeout: float, max_retries: int, ssl_context: ssl.SSLContext) -> Dict[str, object]:
    url = build_request_url(app_id)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SteamMetadataFetcher/1.0)",
        "Accept": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
                payload = json.loads(response.read().decode("utf-8"))
            app_payload = payload.get(app_id, {})
            if not isinstance(app_payload, dict):
                raise ValueError(f"Unexpected payload shape for app_id={app_id}")
            return app_payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == max_retries:
                raise
            sleep_seconds = min(20.0, (2 ** (attempt - 1)) + random.random())
            print(
                f"[warn] app_id={app_id} HTTP {exc.code}; retrying in {sleep_seconds:.2f}s "
                f"({attempt}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            if attempt == max_retries:
                raise
            sleep_seconds = min(20.0, (2 ** (attempt - 1)) + random.random())
            print(
                f"[warn] app_id={app_id} error={exc.__class__.__name__}; retrying in {sleep_seconds:.2f}s "
                f"({attempt}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Exhausted retries for app_id={app_id}")


def extract_game_row(app_id: str, app_payload: Dict[str, object]) -> Dict[str, str]:
    success = bool(app_payload.get("success"))
    data = app_payload.get("data") if success else {}
    if not isinstance(data, dict):
        data = {}

    genres = data.get("genres") or []
    genre_names = []
    if isinstance(genres, list):
        for genre in genres:
            if isinstance(genre, dict):
                description = str(genre.get("description") or "").strip()
                if description:
                    genre_names.append(description)

    return {
        "app_id": app_id,
        "name": str(data.get("name") or ""),
        "short_description": str(data.get("short_description") or ""),
        "genres": " | ".join(genre_names),
    }


def write_games_csv(rows: Iterable[Dict[str, str]], output_path: Path) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["app_id", "name", "short_description", "genres"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache_path = Path(args.cache).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    app_ids = load_unique_app_ids(input_path)
    cache = {} if args.force_refresh else load_cache(cache_path)
    ssl_context = create_ssl_context()
    rows: List[Dict[str, str]] = []
    fetched = 0
    cache_hits = 0

    print(f"Input dataset: {input_path}")
    print(f"Unique app_ids: {len(app_ids)}")

    for index, app_id in enumerate(app_ids, start=1):
        if not args.force_refresh and app_id in cache:
            app_payload = cache[app_id]
            cache_hits += 1
        else:
            app_payload = fetch_one(
                app_id,
                timeout=args.timeout,
                max_retries=args.max_retries,
                ssl_context=ssl_context,
            )
            cache[app_id] = app_payload
            fetched += 1
            save_cache(cache_path, cache)
            if index < len(app_ids):
                time.sleep(args.sleep_seconds)

        rows.append(extract_game_row(app_id, app_payload))
        if index % 10 == 0 or index == len(app_ids):
            print(f"Processed {index}/{len(app_ids)} app_ids")

    row_count = write_games_csv(rows, output_path)
    print(f"Cache hits: {cache_hits}")
    print(f"Fetched from API: {fetched}")
    print(f"Wrote {row_count} rows to {output_path}")
    print(f"Cache file: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
