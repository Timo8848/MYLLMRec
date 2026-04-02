#!/usr/bin/env python3
"""
Build LLMRec-ready feature packages from benchmark splits under NewData/processed/benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from prepare_steam_mvp import (
    build_candidate_indices,
    build_item_history_record,
    build_item_text,
    build_labeled_text,
    build_semantic_features,
    build_structured_user_profile_text,
    build_user_history_summary_text,
    compact_whitespace,
    normalize_rows,
    normalize_vector,
    sanitize_model_name,
    write_json_mapping,
)


BENCHMARK_ORDER = ("warm_start", "cold_start", "long_tail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build complete LLMRec feature packages for Steam benchmark splits."
    )
    parser.add_argument("--benchmark-root", default="./NewData/processed/benchmarks")
    parser.add_argument("--output-root", default="./LLMRec/data")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARK_ORDER),
        help="Benchmark split names to package.",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="steam",
        help="Output datasets are written as {dataset_prefix}_{benchmark}.",
    )
    parser.add_argument(
        "--text-feature-backend",
        choices=("encoder", "hash"),
        default="hash",
        help="How to build item text features stored in text_feat.npy.",
    )
    parser.add_argument(
        "--text-dim",
        type=int,
        default=256,
        help="Hash text feature dim when --text-feature-backend=hash.",
    )
    parser.add_argument(
        "--image-dim",
        type=int,
        default=256,
        help="Placeholder image feature dim.",
    )
    parser.add_argument(
        "--profile-dim",
        type=int,
        default=64,
        help="Hash dim used for item attributes and user profiles when --text-feature-backend=hash.",
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument(
        "--text-encoder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence encoder model used when --text-feature-backend=encoder.",
    )
    parser.add_argument(
        "--text-encoder-device",
        default="auto",
        help="Encoder device: auto/cpu/cuda/mps.",
    )
    parser.add_argument(
        "--text-encoder-batch-size",
        type=int,
        default=64,
        help="Batch size for encoder inference.",
    )
    parser.add_argument(
        "--text-max-length",
        type=int,
        default=256,
        help="Max sequence length for encoder tokenization when applicable.",
    )
    parser.add_argument(
        "--text-encoder-cache",
        default="",
        help="Optional shared cache file for encoder outputs.",
    )
    parser.add_argument(
        "--profile-history-max-items",
        type=int,
        default=10,
        help="Maximum number of train-history items serialized into each textual user profile.",
    )
    parser.add_argument(
        "--profile-description-max-chars",
        type=int,
        default=120,
        help="Maximum description characters kept per history item when building textual user profiles.",
    )
    parser.add_argument(
        "--profile-review-max-chars",
        type=int,
        default=0,
        help="Maximum review characters kept per history item when building textual user profiles.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return compact_whitespace(html.unescape(str(value or "")))


def first_non_empty(*values: str, fallback: str = "") -> str:
    for value in values:
        normalized = normalize_text(value)
        if normalized:
            return normalized
    return fallback


def load_json_mapping(path: Path) -> dict[int, list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(user_idx): [int(item_idx) for item_idx in item_indices]
        for user_idx, item_indices in raw.items()
    }


def load_ordered_ids(path: Path, index_field: str, value_field: str) -> list[str]:
    ordered: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idx = int(normalize_text(row.get(index_field)))
            value = normalize_text(row.get(value_field))
            ordered[idx] = value

    if not ordered:
        return []

    expected_indices = list(range(max(ordered) + 1))
    actual_indices = sorted(ordered)
    if actual_indices != expected_indices:
        raise ValueError(f"Non-contiguous indices in {path}: expected 0..{expected_indices[-1]}")
    return [ordered[idx] for idx in expected_indices]


def load_item_catalog(path: Path) -> dict[str, dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            app_id = normalize_text(row.get("app_id"))
            if not app_id:
                continue
            items[app_id] = {key: normalize_text(value) for key, value in row.items()}
    return items


def build_item_description(row: dict[str, str], title: str, genres: str) -> str:
    fields = []
    for label, key in (
        ("tags", "tags"),
        ("specs", "specs"),
        ("developer", "developer"),
        ("publisher", "publisher"),
        ("release_date", "release_date"),
        ("sentiment", "sentiment"),
        ("bundle_names", "bundle_names"),
    ):
        value = normalize_text(row.get(key, ""))
        if value:
            fields.append(f"{label}: {value}")

    if not fields:
        fallback_fields = []
        if title:
            fallback_fields.append(f"title: {title}")
        if genres:
            fallback_fields.append(f"genres: {genres}")
        fields = fallback_fields

    return "\n".join(fields)


def resolve_dataset_name(prefix: str, benchmark_name: str) -> str:
    normalized_prefix = prefix.strip("_")
    return f"{normalized_prefix}_{benchmark_name}" if normalized_prefix else benchmark_name


def resolve_shared_cache_path(args: argparse.Namespace) -> str | None:
    if args.text_feature_backend != "encoder":
        return None
    if args.text_encoder_cache:
        return str(Path(args.text_encoder_cache).expanduser().resolve())
    cache_name = f"{args.dataset_prefix}_benchmark_text_feature_cache_{sanitize_model_name(args.text_encoder_model)}.pkl"
    return str((Path(args.output_root).expanduser().resolve() / cache_name))


def count_non_empty_rows(mapping: dict[int, list[int]]) -> int:
    return sum(1 for item_indices in mapping.values() if item_indices)


def build_package(
    benchmark_name: str,
    benchmark_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    train = load_json_mapping(benchmark_dir / "train.json")
    val = load_json_mapping(benchmark_dir / "val.json")
    test = load_json_mapping(benchmark_dir / "test.json")

    kept_users = load_ordered_ids(benchmark_dir / "user_id_map.csv", "user_idx", "user_id")
    kept_items = load_ordered_ids(benchmark_dir / "item_id_map.csv", "item_idx", "app_id")
    benchmark_items = load_item_catalog(benchmark_dir / "items.csv")
    benchmark_summary = json.loads((benchmark_dir / "summary.json").read_text(encoding="utf-8"))

    n_users = len(kept_users)
    n_items = len(kept_items)
    if not n_users or not n_items:
        raise ValueError(f"{benchmark_name} has no users or items to package.")

    output_dir.mkdir(parents=True, exist_ok=True)

    item_metadata: dict[str, dict[str, str]] = {}
    item_rows: list[list[str | int]] = []
    title_texts: list[str] = []
    genre_texts: list[str] = []
    description_texts: list[str] = []
    combined_item_texts: list[str] = []

    for item_idx, app_id in enumerate(kept_items):
        row = benchmark_items.get(app_id, {})
        title = first_non_empty(
            row.get("title", ""),
            row.get("app_name", ""),
            fallback=f"steam_app_{app_id}",
        )
        genres = normalize_text(row.get("genres", ""))
        description = build_item_description(row, title=title, genres=genres)

        item_metadata[app_id] = {
            "name": title,
            "genres": genres,
            "short_description": description,
        }
        item_rows.append([item_idx, title, genres, description])
        title_texts.append(build_labeled_text("title", title))
        genre_texts.append(build_labeled_text("genres", genres))
        description_texts.append(build_labeled_text("description", description))
        combined_item_texts.append(build_item_text(title, genres, description))

    title_embeddings_np, attribute_cache_path = build_semantic_features(
        title_texts,
        args,
        output_dir,
        hash_dim=args.profile_dim,
        hash_seed=11,
    )
    genre_embeddings_np, genre_cache_path = build_semantic_features(
        genre_texts,
        args,
        output_dir,
        hash_dim=args.profile_dim,
        hash_seed=23,
    )
    description_embeddings_np, description_cache_path = build_semantic_features(
        description_texts,
        args,
        output_dir,
        hash_dim=args.profile_dim,
        hash_seed=37,
    )
    text_features_np, text_cache_path = build_semantic_features(
        combined_item_texts,
        args,
        output_dir,
        hash_dim=args.text_dim,
        hash_seed=101,
    )
    text_cache_path = text_cache_path or attribute_cache_path or genre_cache_path or description_cache_path

    pooled_profile_embeddings_np = normalize_rows(
        (title_embeddings_np + genre_embeddings_np + description_embeddings_np) / 3.0
    )
    image_features_np = np.zeros((n_items, args.image_dim), dtype=np.float32)

    pooled_user_init_embeddings = []
    history_summary_texts: list[str] = []
    structured_profile_texts: list[str] = []
    for user_idx in range(n_users):
        train_items = train.get(user_idx, [])
        if train_items:
            user_vec = pooled_profile_embeddings_np[train_items].mean(axis=0)
        else:
            user_vec = np.zeros(pooled_profile_embeddings_np.shape[1], dtype=np.float32)
        pooled_user_init_embeddings.append(normalize_vector(user_vec))

        history_records = []
        for item_idx in train_items[: max(0, args.profile_history_max_items)]:
            app_id = kept_items[item_idx]
            history_records.append(
                build_item_history_record(
                    item_id=app_id,
                    metadata=item_metadata[app_id],
                    review_text="",
                    description_max_chars=args.profile_description_max_chars,
                    review_max_chars=args.profile_review_max_chars,
                )
            )

        history_summary_texts.append(build_user_history_summary_text(history_records))
        structured_profile_texts.append(build_structured_user_profile_text(history_records))

    history_summary_embeddings_np, history_profile_cache_path = build_semantic_features(
        history_summary_texts,
        args,
        output_dir,
        hash_dim=args.profile_dim,
        hash_seed=53,
    )
    structured_profile_embeddings_np, structured_profile_cache_path = build_semantic_features(
        structured_profile_texts,
        args,
        output_dir,
        hash_dim=args.profile_dim,
        hash_seed=59,
    )
    history_summary_embeddings_np = normalize_rows(history_summary_embeddings_np)
    structured_profile_embeddings_np = normalize_rows(structured_profile_embeddings_np)

    train_rows = []
    train_cols = []
    for user_idx, item_ids in train.items():
        for item_idx in item_ids:
            train_rows.append(user_idx)
            train_cols.append(item_idx)
    train_values = np.ones(len(train_rows), dtype=np.float32)
    train_mat = sp.csr_matrix((train_values, (train_rows, train_cols)), shape=(n_users, n_items))

    popularity: Counter[int] = Counter()
    for item_indices in train.values():
        for item_idx in item_indices:
            popularity[item_idx] += 1
    candidate_indices = build_candidate_indices(
        n_users=n_users,
        n_items=n_items,
        train_by_user=train,
        popularity=popularity,
        k=args.candidate_k,
    )

    augmented_sample_dict = {}
    for user_idx in range(n_users):
        pos_source = test.get(user_idx) or val.get(user_idx) or train.get(user_idx, [])
        if not pos_source:
            continue
        seen = set(train.get(user_idx, [])) | set(val.get(user_idx, [])) | set(test.get(user_idx, []))
        neg_item = next((item_idx for item_idx in candidate_indices[user_idx] if item_idx not in seen), None)
        if neg_item is None:
            neg_item = next((item_idx for item_idx in range(n_items) if item_idx not in seen), 0)
        augmented_sample_dict[user_idx] = {0: int(pos_source[0]), 1: int(neg_item)}

    item_attribute_dict = {
        "title": [row for row in title_embeddings_np],
        "genre": [row for row in genre_embeddings_np],
        "description": [row for row in description_embeddings_np],
    }

    write_json_mapping(output_dir / "train.json", train)
    write_json_mapping(output_dir / "val.json", val)
    write_json_mapping(output_dir / "test.json", test)

    with (output_dir / "item_attribute.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(item_rows)

    with (output_dir / "user_id_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_idx", "user_id"])
        for user_idx, user_id in enumerate(kept_users):
            writer.writerow([user_idx, user_id])

    with (output_dir / "item_id_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_idx", "app_id"])
        for item_idx, app_id in enumerate(kept_items):
            writer.writerow([item_idx, app_id])

    np.save(output_dir / "text_feat.npy", text_features_np)
    np.save(output_dir / "image_feat.npy", image_features_np)
    pickle.dump(train_mat, open(output_dir / "train_mat", "wb"))
    pickle.dump(pooled_user_init_embeddings, open(output_dir / "augmented_user_init_embedding", "wb"))
    pickle.dump(pooled_user_init_embeddings, open(output_dir / "augmented_user_init_embedding_pooled", "wb"))
    pickle.dump(history_summary_embeddings_np, open(output_dir / "augmented_user_init_embedding_history_summary", "wb"))
    pickle.dump(structured_profile_embeddings_np, open(output_dir / "augmented_user_init_embedding_structured_profile", "wb"))
    pickle.dump(item_attribute_dict, open(output_dir / "augmented_atttribute_embedding_dict", "wb"))
    pickle.dump(augmented_sample_dict, open(output_dir / "augmented_sample_dict", "wb"))
    pickle.dump(candidate_indices, open(output_dir / "candidate_indices", "wb"))

    user_profile_text_artifact = {
        str(user_idx): {
            "history_summary": history_summary_texts[user_idx],
            "structured_profile": structured_profile_texts[user_idx],
        }
        for user_idx in range(n_users)
    }
    (output_dir / "user_profile_texts.json").write_text(
        json.dumps(user_profile_text_artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "benchmark_name": benchmark_name,
        "dataset_name": output_dir.name,
        "source_benchmark_directory": str(benchmark_dir),
        "source_benchmark_summary": {
            "definition": benchmark_summary.get("definition"),
            "train": benchmark_summary.get("train"),
            "test": benchmark_summary.get("test"),
            "test_item_context": benchmark_summary.get("test_item_context"),
        },
        "n_users": n_users,
        "n_items": n_items,
        "n_train": int(train_mat.nnz),
        "n_val": int(sum(len(v) for v in val.values())),
        "n_test": int(sum(len(v) for v in test.values())),
        "users_with_train_interactions": int(count_non_empty_rows(train)),
        "users_with_test_interactions": int(count_non_empty_rows(test)),
        "text_feature_backend": args.text_feature_backend,
        "text_encoder_model": args.text_encoder_model if args.text_feature_backend == "encoder" else None,
        "text_encoder_cache": str(text_cache_path) if text_cache_path else None,
        "text_dim": int(text_features_np.shape[1]),
        "image_dim": int(image_features_np.shape[1]),
        "profile_dim": int(len(pooled_user_init_embeddings[0])) if pooled_user_init_embeddings else 0,
        "item_attribute_dim": int(title_embeddings_np.shape[1]),
        "candidate_k": int(candidate_indices.shape[1]) if candidate_indices.size else 0,
        "profile_history_max_items": int(args.profile_history_max_items),
        "profile_description_max_chars": int(args.profile_description_max_chars),
        "profile_review_max_chars": int(args.profile_review_max_chars),
        "user_profile_variants": {
            "pooled": {
                "embedding_file": "augmented_user_init_embedding_pooled",
                "description": "Mean pooled train-item semantic embeddings.",
                "dim": int(len(pooled_user_init_embeddings[0])) if pooled_user_init_embeddings else 0,
            },
            "history_summary": {
                "embedding_file": "augmented_user_init_embedding_history_summary",
                "text_source": "user_profile_texts.json.history_summary",
                "description": "Serialized train history summary encoded into the same text space.",
                "dim": int(history_summary_embeddings_np.shape[1]),
            },
            "structured_profile": {
                "embedding_file": "augmented_user_init_embedding_structured_profile",
                "text_source": "user_profile_texts.json.structured_profile",
                "description": "Structured textual user profile encoded into the same text space.",
                "dim": int(structured_profile_embeddings_np.shape[1]),
            },
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    invalid = [name for name in args.benchmarks if name not in BENCHMARK_ORDER]
    if invalid:
        raise ValueError(f"Unsupported benchmark names: {', '.join(invalid)}")

    shared_cache_path = resolve_shared_cache_path(args)
    if shared_cache_path:
        args.text_encoder_cache = shared_cache_path

    manifest = {
        "benchmark_root": str(benchmark_root),
        "output_root": str(output_root),
        "text_feature_backend": args.text_feature_backend,
        "packages": {},
    }

    for benchmark_name in args.benchmarks:
        benchmark_dir = benchmark_root / benchmark_name
        dataset_name = resolve_dataset_name(args.dataset_prefix, benchmark_name)
        output_dir = output_root / dataset_name
        summary = build_package(benchmark_name, benchmark_dir, output_dir, args)
        manifest["packages"][benchmark_name] = {
            "dataset_name": dataset_name,
            "output_directory": str(output_dir),
            "summary_file": str(output_dir / "summary.json"),
            "n_users": summary["n_users"],
            "n_items": summary["n_items"],
            "n_train": summary["n_train"],
            "n_test": summary["n_test"],
        }
        print(
            f"{benchmark_name}: dataset={dataset_name}, users={summary['n_users']}, "
            f"items={summary['n_items']}, train={summary['n_train']}, test={summary['n_test']}"
        )

    manifest_path = output_root / "steam_benchmark_feature_packages.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    if shared_cache_path:
        print(f"Shared text encoder cache: {shared_cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
