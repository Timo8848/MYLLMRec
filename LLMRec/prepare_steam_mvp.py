#!/usr/bin/env python3
"""
Prepare a minimal Steam dataset package for LLMRec.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
PROFILE_KEYWORD_STOPWORDS = {
    "about",
    "after",
    "against",
    "all",
    "also",
    "and",
    "are",
    "around",
    "because",
    "been",
    "being",
    "between",
    "but",
    "can",
    "each",
    "features",
    "from",
    "game",
    "games",
    "has",
    "have",
    "into",
    "its",
    "just",
    "more",
    "most",
    "new",
    "not",
    "now",
    "off",
    "one",
    "online",
    "out",
    "over",
    "play",
    "player",
    "players",
    "playing",
    "same",
    "set",
    "steam",
    "team",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "this",
    "through",
    "time",
    "variety",
    "when",
    "which",
    "while",
    "with",
    "world",
    "your",
}
ITEM_COUNT_THRESHOLDS = (1, 2, 3, 5, 10, 20, 50, 100)
SOURCE_POPULARITY_BUCKETS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20-49", 20, 49),
    ("50+", 50, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a minimal Steam dataset for LLMRec.")
    parser.add_argument("--input", default="../merged_data.csv")
    parser.add_argument("--output-dir", default="./data/steam")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument("--min-item-interactions", type=int, default=3)
    parser.add_argument("--text-dim", type=int, default=256, help="Hash text feature dim when --text-feature-backend=hash.")
    parser.add_argument("--image-dim", type=int, default=256, help="Placeholder image feature dim.")
    parser.add_argument("--profile-dim", type=int, default=64, help="Hash dim used for item attributes and user profiles when --text-feature-backend=hash.")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument(
        "--text-feature-backend",
        choices=("encoder", "hash"),
        default="encoder",
        help="How to build item text features stored in text_feat.npy.",
    )
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
        default=None,
        help="Optional cache file for encoder outputs. Defaults to a model-specific cache inside the output dir.",
    )
    parser.add_argument(
        "--profile-history-max-items",
        type=int,
        default=20,
        help="Maximum number of train-history items to serialize into each textual user profile.",
    )
    parser.add_argument(
        "--profile-description-max-chars",
        type=int,
        default=180,
        help="Maximum description characters kept per history item when building textual user profiles.",
    )
    parser.add_argument(
        "--profile-review-max-chars",
        type=int,
        default=120,
        help="Maximum positive-review characters kept per history item when building textual user profiles.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def compact_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def truncate_text(text: str, max_chars: int) -> str:
    text = compact_whitespace(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def split_multivalue_field(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*\|\s*", (value or "").strip()) if part.strip()]


def format_ranked_terms(terms: list[str], empty_value: str = "") -> str:
    return ", ".join(terms) if terms else empty_value


def extract_top_keywords(texts: list[str], limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for token in tokenize(text):
            if len(token) < 4 or token in PROFILE_KEYWORD_STOPWORDS:
                continue
            counts[token] += 1
    return [token for token, _ in counts.most_common(limit)]


def build_item_history_record(
    item_id: str,
    metadata: dict[str, str],
    review_text: str,
    description_max_chars: int,
    review_max_chars: int,
) -> dict[str, str]:
    return {
        "item_id": item_id,
        "title": compact_whitespace(metadata.get("name", "")),
        "genres": compact_whitespace(metadata.get("genres", "")),
        "description": truncate_text(metadata.get("short_description", ""), description_max_chars),
        "review": truncate_text(review_text, review_max_chars),
    }


def build_user_history_summary_text(history_records: list[dict[str, str]]) -> str:
    if not history_records:
        return "User positive history summary: no positive training interactions available."

    genre_counts: Counter[str] = Counter()
    theme_keywords = extract_top_keywords(
        [record["description"] for record in history_records] + [record["review"] for record in history_records],
        limit=6,
    )
    for record in history_records:
        genre_counts.update(split_multivalue_field(record["genres"]))

    lines = [
        f"User positive history summary ({len(history_records)} training interactions):",
    ]
    for idx, record in enumerate(history_records, start=1):
        parts = []
        if record["genres"]:
            parts.append(f"genres: {record['genres']}")
        if record["description"]:
            parts.append(f"description: {record['description']}")
        if record["review"]:
            parts.append(f"positive review cue: {record['review']}")
        suffix = " | ".join(parts)
        if suffix:
            lines.append(f"{idx}. {record['title']} | {suffix}")
        else:
            lines.append(f"{idx}. {record['title']}")

    if genre_counts:
        lines.append(f"Recurring genres: {format_ranked_terms([genre for genre, _ in genre_counts.most_common(4)])}.")
    if theme_keywords:
        lines.append(f"Recurring themes: {format_ranked_terms(theme_keywords)}.")
    return "\n".join(lines)


def build_profile_statement(
    genre_terms: list[str],
    theme_terms: list[str],
    representative_titles: list[str],
) -> str:
    statements = []
    if genre_terms:
        statements.append(f"leans toward {format_ranked_terms(genre_terms[:3])} games")
    if theme_terms:
        statements.append(f"responds to themes like {format_ranked_terms(theme_terms[:4])}")
    if representative_titles:
        statements.append(f"representative titles include {format_ranked_terms(representative_titles[:3])}")
    if not statements:
        return "Preference signal is weak because the positive training history is sparse."
    return "This user " + "; ".join(statements) + "."


def build_structured_user_profile_text(history_records: list[dict[str, str]]) -> str:
    if not history_records:
        return "user_profile {\ninteraction_count: 0\npreference_summary: No positive training history available.\n}"

    genre_counts: Counter[str] = Counter()
    for record in history_records:
        genre_counts.update(split_multivalue_field(record["genres"]))

    favorite_genres = [genre for genre, _ in genre_counts.most_common(4)]
    representative_titles = [record["title"] for record in history_records[:5] if record["title"]]
    theme_terms = extract_top_keywords(
        [record["description"] for record in history_records] + [record["review"] for record in history_records],
        limit=6,
    )
    review_terms = extract_top_keywords([record["review"] for record in history_records], limit=5)
    lines = [
        "user_profile {",
        f"interaction_count: {len(history_records)}",
    ]
    if favorite_genres:
        lines.append(f"favorite_genres: {format_ranked_terms(favorite_genres)}")
    if representative_titles:
        lines.append(f"representative_games: {'; '.join(representative_titles)}")
    if theme_terms:
        lines.append(f"gameplay_themes: {format_ranked_terms(theme_terms)}")
    if review_terms:
        lines.append(f"review_cues: {format_ranked_terms(review_terms)}")
    lines.append(
        "preference_summary: "
        + build_profile_statement(favorite_genres, theme_terms, representative_titles)
    )
    lines.append("}")
    return "\n".join(lines)


def stable_hash(token: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{token}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def hashed_text_vector(text: str, dim: int, seed: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        base_hash = stable_hash(token, seed)
        bucket = base_hash % dim
        sign = 1.0 if ((base_hash >> 1) & 1) == 0 else -1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def build_item_text(title: str, genres: str, description: str) -> str:
    fields = []
    if title:
        fields.append(f"title: {title}")
    if genres:
        fields.append(f"genres: {genres}")
    if description:
        fields.append(f"description: {description}")
    return "\n".join(fields).strip()


def build_labeled_text(label: str, value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return f"{label}: {value}"


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def resolve_text_encoder_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text_feature_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    with cache_path.open("rb") as handle:
        raw_cache = pickle.load(handle)
    cache: dict[str, np.ndarray] = {}
    for key, value in raw_cache.items():
        cache[key] = np.asarray(value, dtype=np.float32)
    return cache


def save_text_feature_cache(cache_path: Path, cache: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle)


def mean_pool_embeddings(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return masked.sum(dim=1) / denom


def encode_texts_with_sentence_transformers(
    texts: list[str],
    model_name: str,
    device_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device_name)
    if max_length > 0:
        model.max_seq_length = max_length
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_texts_with_transformers(
    texts: list[str],
    model_name: str,
    device_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device_name)
    model.eval()

    batches = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded_inputs = {key: value.to(device_name) for key, value in encoded_inputs.items()}
        with torch.inference_mode():
            outputs = model(**encoded_inputs)
            pooled = mean_pool_embeddings(outputs.last_hidden_state, encoded_inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        batches.append(pooled.cpu().numpy().astype(np.float32))

    if not batches:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0)


def encode_texts_with_encoder(
    texts: list[str],
    model_name: str,
    device_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    failures = []
    try:
        return encode_texts_with_sentence_transformers(texts, model_name, device_name, batch_size, max_length)
    except Exception as exc:
        failures.append(f"sentence-transformers backend failed: {exc}")

    try:
        return encode_texts_with_transformers(texts, model_name, device_name, batch_size, max_length)
    except Exception as exc:
        failures.append(f"transformers backend failed: {exc}")

    raise RuntimeError(
        "Unable to build encoder-based text features. "
        "Install the text encoder dependencies and make sure the model is available locally or downloadable. "
        f"Details: {' | '.join(failures)}"
    )


def build_text_features(
    texts: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[np.ndarray, Path | None]:
    if not texts:
        dim = args.text_dim if args.text_feature_backend == "hash" else 0
        return np.zeros((0, dim), dtype=np.float32), None

    if args.text_feature_backend == "hash":
        text_features = [hashed_text_vector(text, args.text_dim, seed=101) for text in texts]
        return np.stack(text_features).astype(np.float32), None

    cache_path = (
        Path(args.text_encoder_cache).expanduser().resolve()
        if args.text_encoder_cache
        else output_dir / f"text_feature_cache_{sanitize_model_name(args.text_encoder_model)}.pkl"
    )
    cache = load_text_feature_cache(cache_path)
    text_keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

    missing_indices = [idx for idx, key in enumerate(text_keys) if key not in cache]
    if missing_indices:
        missing_texts = [texts[idx] for idx in missing_indices]
        encoded = encode_texts_with_encoder(
            missing_texts,
            model_name=args.text_encoder_model,
            device_name=resolve_text_encoder_device(args.text_encoder_device),
            batch_size=args.text_encoder_batch_size,
            max_length=args.text_max_length,
        )
        for idx, vector in zip(missing_indices, encoded):
            cache[text_keys[idx]] = np.asarray(vector, dtype=np.float32)
        save_text_feature_cache(cache_path, cache)

    ordered = [cache[key] for key in text_keys]
    return np.stack(ordered).astype(np.float32), cache_path


def build_semantic_features(
    texts: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    *,
    hash_dim: int,
    hash_seed: int,
) -> tuple[np.ndarray, Path | None]:
    if not texts:
        return np.zeros((0, hash_dim if args.text_feature_backend == "hash" else 0), dtype=np.float32), None

    if args.text_feature_backend == "hash":
        features = [hashed_text_vector(text, hash_dim, seed=hash_seed) for text in texts]
        return np.stack(features).astype(np.float32), None

    non_empty_indices = [idx for idx, text in enumerate(texts) if (text or "").strip()]
    if not non_empty_indices:
        placeholder_features, cache_path = build_text_features(["[missing]"], args, output_dir)
        return np.zeros((len(texts), placeholder_features.shape[1]), dtype=np.float32), cache_path

    non_empty_texts = [texts[idx] for idx in non_empty_indices]
    encoded_features, cache_path = build_text_features(non_empty_texts, args, output_dir)
    features = np.zeros((len(texts), encoded_features.shape[1]), dtype=np.float32)
    features[non_empty_indices] = encoded_features
    return features, cache_path


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (matrix / norms).astype(np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def iterative_k_core(interactions: set[tuple[str, str]], min_user: int, min_item: int) -> set[tuple[str, str]]:
    filtered = set(interactions)
    while True:
        user_counts = Counter(user_id for user_id, _ in filtered)
        item_counts = Counter(item_id for _, item_id in filtered)
        valid_users = {user_id for user_id, count in user_counts.items() if count >= min_user}
        valid_items = {item_id for item_id, count in item_counts.items() if count >= min_item}
        updated = {
            (user_id, item_id)
            for user_id, item_id in filtered
            if user_id in valid_users and item_id in valid_items
        }
        if len(updated) == len(filtered):
            return updated
        filtered = updated


def load_positive_interactions(
    input_path: Path,
) -> tuple[set[tuple[str, str]], dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    interactions: set[tuple[str, str]] = set()
    item_metadata: dict[str, dict[str, str]] = {}
    interaction_metadata: dict[tuple[str, str], dict[str, str]] = {}
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("is_positive") or "").strip() != "Positive":
                continue
            user_id = (row.get("author_id") or "").strip()
            item_id = (row.get("app_id") or "").strip()
            if not user_id or not item_id:
                continue
            interactions.add((user_id, item_id))
            if item_id not in item_metadata:
                item_metadata[item_id] = {
                    "name": (row.get("name") or "").strip(),
                    "genres": (row.get("genres") or "").strip(),
                    "short_description": (row.get("short_description") or "").strip(),
                }
            review_text = compact_whitespace(row.get("content") or "")
            interaction_key = (user_id, item_id)
            if interaction_key not in interaction_metadata or len(review_text) > len(interaction_metadata[interaction_key].get("review", "")):
                interaction_metadata[interaction_key] = {"review": review_text}
    return interactions, item_metadata, interaction_metadata


def split_user_items(
    user_to_items: dict[str, list[int]],
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    rng = random.Random(seed)
    train, val, test = {}, {}, {}
    for user_id, item_ids in user_to_items.items():
        items = sorted(item_ids)
        rng.shuffle(items)
        if len(items) >= 3:
            train[user_id] = items[:-2]
            val[user_id] = [items[-2]]
            test[user_id] = [items[-1]]
        elif len(items) == 2:
            train[user_id] = [items[0]]
            val[user_id] = []
            test[user_id] = [items[1]]
        elif len(items) == 1:
            train[user_id] = [items[0]]
            val[user_id] = []
            test[user_id] = []
        else:
            train[user_id] = []
            val[user_id] = []
            test[user_id] = []
    return train, val, test


def write_json_mapping(path: Path, mapping: dict[int, list[int]]) -> None:
    path.write_text(
        json.dumps({str(key): value for key, value in mapping.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_candidate_indices(
    n_users: int,
    n_items: int,
    train_by_user: dict[int, list[int]],
    popularity: Counter[int],
    k: int,
) -> np.ndarray:
    k = min(k, max(1, n_items - 1))
    popular_items = [item_idx for item_idx, _ in popularity.most_common()]
    candidates = np.zeros((n_users, k), dtype=np.int64)
    for user_idx in range(n_users):
        seen = set(train_by_user.get(user_idx, []))
        ranked = [item_idx for item_idx in popular_items if item_idx not in seen]
        if len(ranked) < k:
            ranked.extend(item_idx for item_idx in range(n_items) if item_idx not in seen and item_idx not in ranked)
        candidates[user_idx] = np.array(ranked[:k], dtype=np.int64)
    return candidates


def round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def item_sort_key(item_id: str):
    return (0, int(item_id)) if item_id.isdigit() else (1, item_id)


def compute_gini(counts: list[int]) -> float:
    if not counts:
        return 0.0
    values = np.sort(np.asarray(counts, dtype=np.float64))
    total = float(values.sum())
    if total <= 0:
        return 0.0
    n = values.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    gini = float(((2 * ranks - n - 1) * values).sum() / (n * total))
    return round_metric(gini)


def summarize_item_threshold_coverage(
    item_counts: Counter[str],
    catalog_size: int,
    thresholds: tuple[int, ...] = ITEM_COUNT_THRESHOLDS,
) -> dict[str, dict[str, float | int]]:
    values = list(item_counts.values())
    summary: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds:
        item_count = sum(1 for value in values if value >= threshold)
        summary[f"ge_{threshold}"] = {
            "item_count": int(item_count),
            "ratio_vs_catalog": round_metric(item_count / catalog_size) if catalog_size else 0.0,
        }
    return summary


def summarize_head_item_requirements(
    sorted_counts_desc: list[int],
    catalog_size: int,
    targets: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict[str, dict[str, float | int]]:
    total = sum(sorted_counts_desc)
    summary: dict[str, dict[str, float | int]] = {}
    for target in targets:
        if total <= 0 or not sorted_counts_desc:
            item_count = 0
            ratio_vs_active_items = 0.0
            ratio_vs_catalog = 0.0
        else:
            cumulative = 0
            item_count = len(sorted_counts_desc)
            for idx, count in enumerate(sorted_counts_desc, start=1):
                cumulative += count
                if cumulative / total >= target:
                    item_count = idx
                    break
            ratio_vs_active_items = item_count / len(sorted_counts_desc)
            ratio_vs_catalog = item_count / catalog_size if catalog_size else 0.0
        summary[f"reach_{int(target * 100)}pct_interactions"] = {
            "item_count": int(item_count),
            "ratio_vs_active_items": round_metric(ratio_vs_active_items),
            "ratio_vs_catalog": round_metric(ratio_vs_catalog),
        }
    return summary


def summarize_top_k_interaction_share(
    sorted_counts_desc: list[int],
    total_interactions: int,
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for top_k in top_ks:
        share = sum(sorted_counts_desc[:top_k]) / total_interactions if total_interactions else 0.0
        summary[f"top_{top_k}_share"] = round_metric(share)
    return summary


def summarize_top_items(
    item_counts: Counter[str],
    item_metadata: dict[str, dict[str, str]],
    item_id_to_idx: dict[str, int] | None,
    total_interactions: int,
    *,
    top_n: int = 10,
) -> list[dict[str, str | int | float | None]]:
    rows = []
    for rank, (item_id, count) in enumerate(item_counts.most_common(top_n), start=1):
        metadata = item_metadata.get(item_id, {})
        rows.append(
            {
                "rank": rank,
                "app_id": item_id,
                "item_idx": int(item_id_to_idx[item_id]) if item_id_to_idx and item_id in item_id_to_idx else None,
                "name": metadata.get("name", ""),
                "interaction_count": int(count),
                "interaction_share": round_metric(count / total_interactions) if total_interactions else 0.0,
            }
        )
    return rows


def build_split_item_counter(
    split_by_user: dict[int, list[int]],
    kept_items: list[str],
) -> Counter[str]:
    item_counts: Counter[str] = Counter()
    for item_indices in split_by_user.values():
        for item_idx in item_indices:
            item_counts[kept_items[item_idx]] += 1
    return item_counts


def build_split_item_coverage(
    split_item_counts: Counter[str],
    catalog_item_ids: set[str],
) -> dict[str, float | int]:
    distinct_items = len(split_item_counts)
    catalog_size = len(catalog_item_ids)
    missing_items = catalog_item_ids - set(split_item_counts)
    return {
        "distinct_items": int(distinct_items),
        "coverage_ratio_vs_post_k_core_catalog": round_metric(distinct_items / catalog_size) if catalog_size else 0.0,
        "missing_item_count": int(len(missing_items)),
    }


def build_eval_cold_start_summary(
    eval_item_counts: Counter[str],
    train_item_counts: Counter[str],
    catalog_item_ids: set[str],
    item_metadata: dict[str, dict[str, str]],
    item_id_to_idx: dict[str, int],
) -> dict[str, object]:
    unseen_item_ids = sorted(set(eval_item_counts) - set(train_item_counts), key=item_sort_key)
    return {
        "unseen_item_count": int(len(unseen_item_ids)),
        "unseen_item_ratio_vs_post_k_core_catalog": (
            round_metric(len(unseen_item_ids) / len(catalog_item_ids)) if catalog_item_ids else 0.0
        ),
        "unseen_items": summarize_top_items(
            Counter({item_id: eval_item_counts[item_id] for item_id in unseen_item_ids}),
            item_metadata,
            item_id_to_idx,
            total_interactions=sum(eval_item_counts.values()),
            top_n=min(20, len(unseen_item_ids)),
        ),
    }


def build_source_item_retention_by_popularity(
    source_item_counts: Counter[str],
    retained_item_ids: set[str],
) -> list[dict[str, float | int | str]]:
    rows = []
    for bucket_label, min_count, max_count in SOURCE_POPULARITY_BUCKETS:
        bucket_item_ids = [
            item_id
            for item_id, count in source_item_counts.items()
            if count >= min_count and (max_count is None or count <= max_count)
        ]
        if not bucket_item_ids:
            continue
        retained_count = sum(1 for item_id in bucket_item_ids if item_id in retained_item_ids)
        rows.append(
            {
                "source_positive_interaction_bucket": bucket_label,
                "source_item_count": int(len(bucket_item_ids)),
                "retained_item_count_after_k_core": int(retained_count),
                "retention_ratio": round_metric(retained_count / len(bucket_item_ids)),
            }
        )
    return rows


def build_item_popularity_skew_summary(
    item_counts: Counter[str],
    catalog_item_ids: set[str],
    item_metadata: dict[str, dict[str, str]],
    item_id_to_idx: dict[str, int] | None,
) -> dict[str, object]:
    total_interactions = int(sum(item_counts.values()))
    active_item_count = len(item_counts)
    counts_desc = [count for _, count in item_counts.most_common()]
    count_array = np.asarray(counts_desc, dtype=np.float64)
    if count_array.size:
        shares = count_array / total_interactions if total_interactions else np.zeros_like(count_array)
        hhi = float(np.square(shares).sum())
        effective_item_count = (1.0 / hhi) if hhi > 0 else 0.0
        count_stats = {
            "min": int(count_array.min()),
            "p25": round_metric(np.quantile(count_array, 0.25)),
            "median": round_metric(np.quantile(count_array, 0.5)),
            "p75": round_metric(np.quantile(count_array, 0.75)),
            "p90": round_metric(np.quantile(count_array, 0.9)),
            "p95": round_metric(np.quantile(count_array, 0.95)),
            "max": int(count_array.max()),
            "mean": round_metric(count_array.mean()),
        }
    else:
        hhi = 0.0
        effective_item_count = 0.0
        count_stats = {
            "min": 0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0,
            "mean": 0.0,
        }

    catalog_size = len(catalog_item_ids)
    return {
        "catalog_item_count": int(catalog_size),
        "active_item_count": int(active_item_count),
        "active_item_ratio_vs_catalog": round_metric(active_item_count / catalog_size) if catalog_size else 0.0,
        "total_interactions": total_interactions,
        "item_interaction_count_distribution": count_stats,
        "concentration": {
            "gini": compute_gini(counts_desc),
            "hhi": round_metric(hhi),
            "effective_item_count": round_metric(effective_item_count),
            "effective_item_ratio_vs_active_items": (
                round_metric(effective_item_count / active_item_count) if active_item_count else 0.0
            ),
            **summarize_top_k_interaction_share(counts_desc, total_interactions),
        },
        "head_item_requirements": summarize_head_item_requirements(counts_desc, catalog_size),
        "item_coverage_at_interaction_thresholds": summarize_item_threshold_coverage(item_counts, catalog_size),
        "top_items": summarize_top_items(
            item_counts,
            item_metadata,
            item_id_to_idx,
            total_interactions,
        ),
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    interactions, item_metadata, interaction_metadata = load_positive_interactions(input_path)
    source_users = {user_id for user_id, _ in interactions}
    source_items = {item_id for _, item_id in interactions}
    source_item_counts = Counter(item_id for _, item_id in interactions)
    filtered = iterative_k_core(
        interactions,
        min_user=args.min_user_interactions,
        min_item=args.min_item_interactions,
    )
    filtered_users = {user_id for user_id, _ in filtered}
    filtered_items = {item_id for _, item_id in filtered}
    filtered_item_counts = Counter(item_id for _, item_id in filtered)

    user_to_items_raw: dict[str, list[str]] = defaultdict(list)
    item_to_users_raw: dict[str, list[str]] = defaultdict(list)
    for user_id, item_id in sorted(filtered):
        user_to_items_raw[user_id].append(item_id)
        item_to_users_raw[item_id].append(user_id)

    kept_users = sorted(user_to_items_raw)
    kept_items = sorted(item_to_users_raw, key=int)
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(kept_users)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(kept_items)}

    indexed_user_to_items = {
        user_id: [item_id_to_idx[item_id] for item_id in item_ids]
        for user_id, item_ids in user_to_items_raw.items()
    }
    train_raw, val_raw, test_raw = split_user_items(indexed_user_to_items, args.seed)

    train = {user_id_to_idx[user_id]: item_ids for user_id, item_ids in train_raw.items() if item_ids}
    val = {user_id_to_idx[user_id]: item_ids for user_id, item_ids in val_raw.items()}
    test = {user_id_to_idx[user_id]: item_ids for user_id, item_ids in test_raw.items()}

    n_users = len(kept_users)
    n_items = len(kept_items)

    item_rows = []
    title_texts = []
    genre_texts = []
    description_texts = []
    combined_item_texts = []

    for item_id in kept_items:
        item_idx = item_id_to_idx[item_id]
        metadata = item_metadata[item_id]
        title = metadata["name"]
        genres = metadata["genres"]
        description = metadata["short_description"]

        combined_item_text = build_item_text(title, genres, description)

        item_rows.append([item_idx, title, genres, description])
        title_texts.append(build_labeled_text("title", title))
        genre_texts.append(build_labeled_text("genres", genres))
        description_texts.append(build_labeled_text("description", description))
        combined_item_texts.append(combined_item_text)

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
    history_summary_texts = []
    structured_profile_texts = []
    for user_idx in range(n_users):
        train_items = train.get(user_idx, [])
        if train_items:
            user_vec = pooled_profile_embeddings_np[train_items].mean(axis=0)
        else:
            user_vec = np.zeros(pooled_profile_embeddings_np.shape[1], dtype=np.float32)
        pooled_user_init_embeddings.append(normalize_vector(user_vec))

        user_id = kept_users[user_idx]
        history_records = []
        for item_idx in train_items[: max(0, args.profile_history_max_items)]:
            item_id = kept_items[item_idx]
            history_records.append(
                build_item_history_record(
                    item_id=item_id,
                    metadata=item_metadata[item_id],
                    review_text=interaction_metadata.get((user_id, item_id), {}).get("review", ""),
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

    popularity = Counter()
    for item_ids in train.values():
        for item_idx in item_ids:
            popularity[item_idx] += 1
    candidate_indices = build_candidate_indices(n_users, n_items, train, popularity, args.candidate_k)

    train_item_counts = build_split_item_counter(train, kept_items)
    val_item_counts = build_split_item_counter(val, kept_items)
    test_item_counts = build_split_item_counter(test, kept_items)

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
        writer.writerow(["user_idx", "author_id"])
        for user_id, user_idx in user_id_to_idx.items():
            writer.writerow([user_idx, user_id])

    with (output_dir / "item_id_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_idx", "app_id"])
        for item_id, item_idx in item_id_to_idx.items():
            writer.writerow([item_idx, item_id])

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

    dataset_diagnostics = {
        "item_coverage": {
            "source_positive_catalog": {
                "distinct_users": int(len(source_users)),
                "distinct_items": int(len(source_items)),
                "distinct_positive_interactions": int(len(interactions)),
            },
            "post_k_core_catalog": {
                "distinct_users": int(len(filtered_users)),
                "distinct_items": int(len(filtered_items)),
                "distinct_positive_interactions": int(len(filtered)),
                "retained_item_ratio_vs_source_catalog": (
                    round_metric(len(filtered_items) / len(source_items)) if source_items else 0.0
                ),
                "removed_item_count_vs_source_catalog": int(len(source_items - filtered_items)),
            },
            "source_item_retention_by_source_popularity": build_source_item_retention_by_popularity(
                source_item_counts,
                filtered_items,
            ),
            "removed_source_items_after_k_core": summarize_top_items(
                Counter({item_id: source_item_counts[item_id] for item_id in sorted(source_items - filtered_items, key=item_sort_key)}),
                item_metadata,
                item_id_to_idx=None,
                total_interactions=len(interactions),
                top_n=min(20, len(source_items - filtered_items)),
            ),
            "split_coverage_vs_post_k_core_catalog": {
                "train": build_split_item_coverage(train_item_counts, filtered_items),
                "val": build_split_item_coverage(val_item_counts, filtered_items),
                "test": build_split_item_coverage(test_item_counts, filtered_items),
            },
            "eval_unseen_items_vs_train": {
                "val": build_eval_cold_start_summary(
                    val_item_counts,
                    train_item_counts,
                    filtered_items,
                    item_metadata,
                    item_id_to_idx,
                ),
                "test": build_eval_cold_start_summary(
                    test_item_counts,
                    train_item_counts,
                    filtered_items,
                    item_metadata,
                    item_id_to_idx,
                ),
            },
        },
        "item_popularity_skew": {
            "source_positive_before_k_core": build_item_popularity_skew_summary(
                source_item_counts,
                source_items,
                item_metadata,
                item_id_to_idx=None,
            ),
            "post_k_core_all_positive": build_item_popularity_skew_summary(
                filtered_item_counts,
                filtered_items,
                item_metadata,
                item_id_to_idx,
            ),
            "train": build_item_popularity_skew_summary(
                train_item_counts,
                filtered_items,
                item_metadata,
                item_id_to_idx,
            ),
            "val": build_item_popularity_skew_summary(
                val_item_counts,
                filtered_items,
                item_metadata,
                item_id_to_idx,
            ),
            "test": build_item_popularity_skew_summary(
                test_item_counts,
                filtered_items,
                item_metadata,
                item_id_to_idx,
            ),
        },
    }
    dataset_diagnostics_path = output_dir / "dataset_diagnostics.json"
    dataset_diagnostics_path.write_text(
        json.dumps(dataset_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "source_file": str(input_path),
        "positive_interactions_before_k_core": len(interactions),
        "positive_interactions_after_k_core": len(filtered),
        "n_users_before_k_core": int(len(source_users)),
        "n_items_before_k_core": int(len(source_items)),
        "n_users": n_users,
        "n_items": n_items,
        "n_train": int(train_mat.nnz),
        "n_val": int(sum(len(v) for v in val.values())),
        "n_test": int(sum(len(v) for v in test.values())),
        "text_feature_backend": args.text_feature_backend,
        "text_encoder_model": args.text_encoder_model if args.text_feature_backend == "encoder" else None,
        "text_encoder_cache": str(text_cache_path) if text_cache_path else None,
        "text_dim": int(text_features_np.shape[1]),
        "image_dim": args.image_dim,
        "profile_dim": int(pooled_profile_embeddings_np.shape[1]),
        "item_attribute_dim": int(title_embeddings_np.shape[1]),
        "dataset_diagnostics_file": str(dataset_diagnostics_path),
        "item_diagnostics_snapshot": {
            "retained_item_ratio_vs_source_catalog": dataset_diagnostics["item_coverage"]["post_k_core_catalog"]["retained_item_ratio_vs_source_catalog"],
            "train_item_coverage_ratio_vs_post_k_core_catalog": dataset_diagnostics["item_coverage"]["split_coverage_vs_post_k_core_catalog"]["train"]["coverage_ratio_vs_post_k_core_catalog"],
            "post_k_core_item_popularity_gini": dataset_diagnostics["item_popularity_skew"]["post_k_core_all_positive"]["concentration"]["gini"],
            "post_k_core_top_10_interaction_share": dataset_diagnostics["item_popularity_skew"]["post_k_core_all_positive"]["concentration"]["top_10_share"],
            "train_item_popularity_gini": dataset_diagnostics["item_popularity_skew"]["train"]["concentration"]["gini"],
            "train_top_10_interaction_share": dataset_diagnostics["item_popularity_skew"]["train"]["concentration"]["top_10_share"],
        },
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
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Train interactions: {train_mat.nnz}")
    print(f"Val interactions: {sum(len(v) for v in val.values())}")
    print(f"Test interactions: {sum(len(v) for v in test.values())}")
    print(f"dataset diagnostics: {dataset_diagnostics_path}")
    print(
        "item coverage (source -> post-k-core): "
        f"{len(source_items)} -> {len(filtered_items)} "
        f"({dataset_diagnostics['item_coverage']['post_k_core_catalog']['retained_item_ratio_vs_source_catalog']:.4f})"
    )
    print(
        "train item coverage vs post-k-core catalog: "
        f"{dataset_diagnostics['item_coverage']['split_coverage_vs_post_k_core_catalog']['train']['distinct_items']}/{len(filtered_items)} "
        f"({dataset_diagnostics['item_coverage']['split_coverage_vs_post_k_core_catalog']['train']['coverage_ratio_vs_post_k_core_catalog']:.4f})"
    )
    print(
        "post-k-core popularity skew: "
        f"gini={dataset_diagnostics['item_popularity_skew']['post_k_core_all_positive']['concentration']['gini']:.4f}, "
        f"top10_share={dataset_diagnostics['item_popularity_skew']['post_k_core_all_positive']['concentration']['top_10_share']:.4f}"
    )
    print(
        "train popularity skew: "
        f"gini={dataset_diagnostics['item_popularity_skew']['train']['concentration']['gini']:.4f}, "
        f"top10_share={dataset_diagnostics['item_popularity_skew']['train']['concentration']['top_10_share']:.4f}"
    )
    print(f"text feature backend: {args.text_feature_backend}")
    if args.text_feature_backend == "encoder":
        print(f"text encoder model: {args.text_encoder_model}")
        if text_cache_path:
            print(f"text encoder cache: {text_cache_path}")
    print(f"text_feat.npy shape: {text_features_np.shape}")
    print(f"image_feat.npy shape: {image_features_np.shape}")
    print(f"user pooled profile shape: {(len(pooled_user_init_embeddings), len(pooled_user_init_embeddings[0]) if pooled_user_init_embeddings else 0)}")
    print(f"user history summary profile shape: {history_summary_embeddings_np.shape}")
    print(f"user structured profile shape: {structured_profile_embeddings_np.shape}")
    if history_profile_cache_path or structured_profile_cache_path:
        print(
            "user profile encoder cache: "
            f"{history_profile_cache_path or structured_profile_cache_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
