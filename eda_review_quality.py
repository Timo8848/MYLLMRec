#!/usr/bin/env python3
"""
Lightweight EDA for review quality in output.csv.

The script focuses on finding reviews that are likely not useful for
downstream NLP/modeling tasks, such as blank texts, extremely short texts,
symbol-only content, noisy repeated characters, and duplicate reviews.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


csv.field_size_limit(sys.maxsize)

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
REPEATED_PUNCT_RE = re.compile(r"([!?.,~*#\-_=/\\|])\1{4,}")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


@dataclass
class ReviewRecord:
    row_number: int
    review_id: str
    app_id: str
    author_id: str
    is_positive: str
    content: str
    normalized_content: str
    char_count: int
    word_count: int
    alpha_ratio: float
    unique_word_ratio: float
    flags: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA and data quality scan for Steam review CSV files.",
    )
    parser.add_argument(
        "--input",
        default="output.csv",
        help="Path to the input CSV. Defaults to output.csv in the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="eda_output",
        help="Directory for generated reports. Defaults to ./eda_output.",
    )
    parser.add_argument(
        "--sample-per-flag",
        type=int,
        default=15,
        help="How many sample reviews to keep per flag in the markdown report.",
    )
    parser.add_argument(
        "--top-duplicates",
        type=int,
        default=50,
        help="How many duplicate texts to keep in the duplicate report.",
    )
    parser.add_argument(
        "--max-flagged-rows",
        type=int,
        default=50000,
        help="Cap the exported suspected-useless rows to keep output size manageable.",
    )
    parser.add_argument(
        "--clean-output",
        default="data.csv",
        help="Path to write the cleaned CSV after dropping suspected useless reviews.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percentile(sorted_values: Sequence[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def compute_flags(text: str, normalized_text: str) -> Dict[str, float | int | List[str]]:
    stripped = text.strip()
    words = WORD_RE.findall(stripped.lower())
    word_count = len(words)
    char_count = len(stripped)
    letter_count = sum(1 for char in stripped if char.isalpha())
    has_letters = letter_count > 0
    alpha_ratio = safe_ratio(letter_count, char_count)
    unique_word_ratio = safe_ratio(len(set(words)), word_count)
    compact = re.sub(r"\s+", "", stripped.lower())

    flags: List[str] = []

    if not stripped:
        flags.append("empty_or_whitespace")
    if word_count <= 3 and stripped:
        flags.append("very_short_text")
    if word_count <= 5 and stripped:
        flags.append("short_text")
    if stripped and not has_letters:
        flags.append("symbol_or_digit_only")
    if char_count >= 5 and alpha_ratio < 0.30:
        flags.append("low_alpha_ratio")
    if REPEATED_PUNCT_RE.search(stripped):
        flags.append("repeated_punctuation")
    if compact and len(compact) >= 4 and len(set(compact)) == 1:
        flags.append("single_repeated_char")
    if URL_RE.search(stripped):
        flags.append("contains_url")
    if word_count >= 6 and unique_word_ratio < 0.35:
        flags.append("low_unique_word_ratio")
    if len(stripped.splitlines()) >= 5 and alpha_ratio < 0.45:
        flags.append("ascii_or_unicode_art")

    return {
        "flags": flags,
        "word_count": word_count,
        "char_count": char_count,
        "alpha_ratio": round(alpha_ratio, 4),
        "unique_word_ratio": round(unique_word_ratio, 4),
    }


def load_reviews(input_path: Path) -> Dict[str, object]:
    records: List[ReviewRecord] = []
    flag_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    app_counts: Counter[str] = Counter()
    author_counts: Counter[str] = Counter()
    duplicate_ids: Counter[str] = Counter()
    normalized_counts: Counter[str] = Counter()
    char_lengths: List[int] = []
    word_lengths: List[int] = []

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"id", "app_id", "content", "author_id", "is_positive"}
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            missing_display = ", ".join(sorted(missing))
            raise ValueError(f"Missing expected columns: {missing_display}")

        for row_number, row in enumerate(reader, start=2):
            review_id = (row.get("id") or "").strip()
            app_id = (row.get("app_id") or "").strip()
            author_id = (row.get("author_id") or "").strip()
            is_positive = (row.get("is_positive") or "").strip()
            content = row.get("content") or ""
            normalized = normalize_text(content)
            metrics = compute_flags(content, normalized)

            label_counts[is_positive or "MISSING"] += 1
            app_counts[app_id or "MISSING"] += 1
            author_counts[author_id or "MISSING"] += 1
            duplicate_ids[review_id or "MISSING"] += 1
            normalized_counts[normalized] += 1
            char_lengths.append(metrics["char_count"])
            word_lengths.append(metrics["word_count"])

            record = ReviewRecord(
                row_number=row_number,
                review_id=review_id,
                app_id=app_id,
                author_id=author_id,
                is_positive=is_positive,
                content=content,
                normalized_content=normalized,
                char_count=metrics["char_count"],
                word_count=metrics["word_count"],
                alpha_ratio=metrics["alpha_ratio"],
                unique_word_ratio=metrics["unique_word_ratio"],
                flags=list(metrics["flags"]),
            )
            records.append(record)

            for flag in record.flags:
                flag_counts[flag] += 1

    for record in records:
        if normalized_counts[record.normalized_content] > 1:
            record.flags.append("duplicate_content")
            flag_counts["duplicate_content"] += 1

    return {
        "records": records,
        "flag_counts": flag_counts,
        "label_counts": label_counts,
        "app_counts": app_counts,
        "author_counts": author_counts,
        "duplicate_ids": duplicate_ids,
        "normalized_counts": normalized_counts,
        "char_lengths": char_lengths,
        "word_lengths": word_lengths,
    }


def is_suspected_useless(record: ReviewRecord) -> bool:
    strong_flags = {
        "empty_or_whitespace",
        "very_short_text",
        "symbol_or_digit_only",
        "low_alpha_ratio",
        "repeated_punctuation",
        "single_repeated_char",
        "ascii_or_unicode_art",
        "duplicate_content",
    }
    strong_match = any(flag in strong_flags for flag in record.flags)
    return strong_match or len(record.flags) >= 2


def build_summary(scan: Dict[str, object]) -> Dict[str, object]:
    records: List[ReviewRecord] = scan["records"]  # type: ignore[assignment]
    char_lengths = sorted(scan["char_lengths"])  # type: ignore[arg-type]
    word_lengths = sorted(scan["word_lengths"])  # type: ignore[arg-type]
    flag_counts: Counter[str] = scan["flag_counts"]  # type: ignore[assignment]
    label_counts: Counter[str] = scan["label_counts"]  # type: ignore[assignment]
    duplicate_ids: Counter[str] = scan["duplicate_ids"]  # type: ignore[assignment]
    normalized_counts: Counter[str] = scan["normalized_counts"]  # type: ignore[assignment]
    app_counts: Counter[str] = scan["app_counts"]  # type: ignore[assignment]
    author_counts: Counter[str] = scan["author_counts"]  # type: ignore[assignment]

    suspected = [record for record in records if is_suspected_useless(record)]

    duplicate_review_ids = sum(count - 1 for count in duplicate_ids.values() if count > 1)
    duplicate_contents = sum(count - 1 for count in normalized_counts.values() if count > 1)

    summary = {
        "row_count": len(records),
        "suspected_useless_count": len(suspected),
        "suspected_useless_ratio": round(safe_ratio(len(suspected), len(records)), 4),
        "label_distribution": dict(label_counts),
        "distinct_app_count": len(app_counts),
        "distinct_author_count": len(author_counts),
        "duplicate_review_id_rows": duplicate_review_ids,
        "duplicate_content_rows": duplicate_contents,
        "content_length_chars": {
            "min": min(char_lengths) if char_lengths else 0,
            "p25": round(percentile(char_lengths, 0.25), 2),
            "median": round(percentile(char_lengths, 0.50), 2),
            "p75": round(percentile(char_lengths, 0.75), 2),
            "p95": round(percentile(char_lengths, 0.95), 2),
            "max": max(char_lengths) if char_lengths else 0,
            "mean": round(statistics.mean(char_lengths), 2) if char_lengths else 0.0,
        },
        "content_length_words": {
            "min": min(word_lengths) if word_lengths else 0,
            "p25": round(percentile(word_lengths, 0.25), 2),
            "median": round(percentile(word_lengths, 0.50), 2),
            "p75": round(percentile(word_lengths, 0.75), 2),
            "p95": round(percentile(word_lengths, 0.95), 2),
            "max": max(word_lengths) if word_lengths else 0,
            "mean": round(statistics.mean(word_lengths), 2) if word_lengths else 0.0,
        },
        "flag_counts": dict(flag_counts.most_common()),
        "top_apps_by_review_count": app_counts.most_common(10),
        "top_authors_by_review_count": author_counts.most_common(10),
    }
    return summary


def export_flagged_rows(records: Iterable[ReviewRecord], output_path: Path, max_rows: int) -> int:
    exported = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row_number",
                "id",
                "app_id",
                "author_id",
                "is_positive",
                "char_count",
                "word_count",
                "alpha_ratio",
                "unique_word_ratio",
                "flags",
                "content",
            ]
        )
        for record in records:
            if not is_suspected_useless(record):
                continue
            writer.writerow(
                [
                    record.row_number,
                    record.review_id,
                    record.app_id,
                    record.author_id,
                    record.is_positive,
                    record.char_count,
                    record.word_count,
                    record.alpha_ratio,
                    record.unique_word_ratio,
                    "|".join(sorted(set(record.flags))),
                    record.content,
                ]
            )
            exported += 1
            if exported >= max_rows:
                break
    return exported


def export_duplicates(normalized_counts: Counter[str], output_path: Path, top_n: int) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["duplicate_count", "normalized_content"])
        for text, count in normalized_counts.most_common():
            if count <= 1:
                continue
            writer.writerow([count, text])
            top_n -= 1
            if top_n <= 0:
                break


def export_cleaned_rows(records: Iterable[ReviewRecord], output_path: Path) -> int:
    kept = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "app_id", "content", "author_id", "is_positive"])
        for record in records:
            if is_suspected_useless(record):
                continue
            writer.writerow(
                [
                    record.review_id,
                    record.app_id,
                    record.content,
                    record.author_id,
                    record.is_positive,
                ]
            )
            kept += 1
    return kept


def collect_samples(records: Sequence[ReviewRecord], per_flag: int) -> Dict[str, List[ReviewRecord]]:
    samples: Dict[str, List[ReviewRecord]] = defaultdict(list)
    for record in records:
        if not is_suspected_useless(record):
            continue
        for flag in sorted(set(record.flags)):
            if len(samples[flag]) < per_flag:
                samples[flag].append(record)
    return samples


def truncate_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def write_markdown_report(
    summary: Dict[str, object],
    records: Sequence[ReviewRecord],
    samples: Dict[str, List[ReviewRecord]],
    output_path: Path,
    flagged_export_count: int,
) -> None:
    suspected = [record for record in records if is_suspected_useless(record)]
    lines = [
        "# Review Quality EDA Report",
        "",
        "## Executive Summary",
        f"- Total rows: {summary['row_count']}",
        f"- Suspected useless reviews: {summary['suspected_useless_count']} ({summary['suspected_useless_ratio']:.2%})",
        f"- Duplicate review IDs: {summary['duplicate_review_id_rows']}",
        f"- Duplicate review texts: {summary['duplicate_content_rows']}",
        f"- Exported suspected-useless rows: {flagged_export_count}",
        "",
        "## Label Distribution",
    ]

    for label, count in summary["label_distribution"].items():  # type: ignore[union-attr]
        lines.append(f"- {label}: {count}")

    char_stats = summary["content_length_chars"]  # type: ignore[assignment]
    word_stats = summary["content_length_words"]  # type: ignore[assignment]
    lines.extend(
        [
            "",
            "## Content Length Stats",
            (
                "- Characters: "
                f"min={char_stats['min']}, p25={char_stats['p25']}, median={char_stats['median']}, "
                f"p75={char_stats['p75']}, p95={char_stats['p95']}, max={char_stats['max']}, mean={char_stats['mean']}"
            ),
            (
                "- Words: "
                f"min={word_stats['min']}, p25={word_stats['p25']}, median={word_stats['median']}, "
                f"p75={word_stats['p75']}, p95={word_stats['p95']}, max={word_stats['max']}, mean={word_stats['mean']}"
            ),
            "",
            "## Flag Counts",
        ]
    )

    for flag, count in summary["flag_counts"].items():  # type: ignore[union-attr]
        lines.append(f"- {flag}: {count}")

    lines.extend(
        [
            "",
            "## Sample Suspected Useless Reviews",
            "",
        ]
    )

    for flag, flag_samples in sorted(samples.items()):
        lines.append(f"### {flag}")
        for record in flag_samples:
            lines.append(
                (
                    f"- row={record.row_number}, id={record.review_id}, app_id={record.app_id}, "
                    f"label={record.is_positive}, words={record.word_count}, chars={record.char_count}: "
                    f"{truncate_text(record.content)}"
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "- `suspected useless` uses heuristics, not a gold label.",
            "- Very short reviews like `good` or `bad` may be valid sentiment labels but are low-information for richer text analysis.",
            "- Duplicate content can happen naturally, but heavy repetition is often a sign that deduplication will help before modeling.",
            "",
            "## Output Files",
            "- `review_quality_summary.json`: machine-readable summary.",
            "- `suspected_useless_reviews.csv`: rows flagged by heuristics.",
            "- `top_duplicate_reviews.csv`: most repeated normalized reviews.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(data: Dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    clean_output_path = Path(args.clean_output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    scan = load_reviews(input_path)
    records: List[ReviewRecord] = scan["records"]  # type: ignore[assignment]
    summary = build_summary(scan)
    samples = collect_samples(records, args.sample_per_flag)

    summary_path = output_dir / "review_quality_summary.json"
    flagged_path = output_dir / "suspected_useless_reviews.csv"
    duplicate_path = output_dir / "top_duplicate_reviews.csv"
    report_path = output_dir / "review_quality_report.md"

    flagged_export_count = export_flagged_rows(records, flagged_path, args.max_flagged_rows)
    cleaned_row_count = export_cleaned_rows(records, clean_output_path)
    export_duplicates(scan["normalized_counts"], duplicate_path, args.top_duplicates)  # type: ignore[arg-type]
    write_json(summary, summary_path)
    write_markdown_report(summary, records, samples, report_path, flagged_export_count)

    print(f"Input: {input_path}")
    print(f"Rows scanned: {summary['row_count']}")
    print(
        "Suspected useless reviews: "
        f"{summary['suspected_useless_count']} ({summary['suspected_useless_ratio']:.2%})"
    )
    print(f"Cleaned rows kept: {cleaned_row_count}")
    print(f"Cleaned CSV: {clean_output_path}")
    print(f"Duplicate content rows: {summary['duplicate_content_rows']}")
    print(f"Report directory: {output_dir}")
    print(f"- {summary_path.name}")
    print(f"- {flagged_path.name}")
    print(f"- {duplicate_path.name}")
    print(f"- {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
