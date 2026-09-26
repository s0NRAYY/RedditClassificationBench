from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from sampling import assign_splits, build_manifests, split_communities

_SUBREDDIT_FIELDS = ("subreddit", "subreddit_name", "label")
_TITLE_FIELDS = ("title", "Title")
_BODY_FIELDS = ("selftext", "body", "text")
_ID_FIELDS = ("post_id", "id", "reddit_id", "name")
_PERMALINK_FIELDS = ("permalink", "perm_link", "url")
_CREATED_FIELDS = ("created_utc", "create_time", "created_at")
_DELETED = {"[deleted]", "[removed]"}
_POST_ID_RE = re.compile(r"/comments/([a-z0-9]+)/", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _first(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_title(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def clean_selftext(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").strip()
    if value.casefold() in _DELETED:
        return ""
    without_urls = _URL_RE.sub("", value)
    return value if any(char.isalnum() for char in without_urls) else ""


def post_text(post: dict, text_mode: str) -> str:
    if text_mode == "title":
        return post["title"]
    if text_mode == "title+selftext" and post["selftext"]:
        return f'{post["title"]}\n\n{post["selftext"]}'
    if text_mode == "title+selftext":
        raise ValueError("title+selftext requested for a title-only post")
    raise ValueError(f"Unknown text mode: {text_mode}")


def extract_post_id(row: dict[str, str], subreddit: str, title: str, created_utc: int) -> str:
    explicit = _first(row, _ID_FIELDS).removeprefix("t3_")
    if explicit and re.fullmatch(r"[a-z0-9]+", explicit, re.IGNORECASE):
        return explicit.lower()
    match = _POST_ID_RE.search(_first(row, _PERMALINK_FIELDS))
    if match:
        return match.group(1).lower()
    source = f"{subreddit.casefold()}\0{title.casefold()}\0{created_utc}"
    return hashlib.sha256(source.encode()).hexdigest()


def parse_created_utc(value: str) -> int:
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())


def discover_csvs(paths: Iterable[str | Path]) -> list[Path]:
    csvs: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            csvs.update(path.rglob("*.csv"))
        elif path.suffix.casefold() == ".csv":
            csvs.add(path)
        else:
            raise ValueError(f"Expected a CSV file or directory: {path}")
    if not csvs:
        raise ValueError("No post CSV files found")
    return sorted(csvs)


def load_paired_posts(paths: Iterable[str | Path]) -> tuple[list[dict], dict[str, int]]:
    by_id: dict[str, dict] = {}
    id_collisions: set[str] = set()
    read_rows = 0
    rejected_text = 0

    for path in discover_csvs(paths):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                read_rows += 1
                subreddit = _first(row, _SUBREDDIT_FIELDS)
                title = normalize_title(_first(row, _TITLE_FIELDS))
                body = clean_selftext(_first(row, _BODY_FIELDS))
                if not subreddit or not title or not body:
                    rejected_text += 1
                    continue
                created_utc = parse_created_utc(_first(row, _CREATED_FIELDS))
                post_id = extract_post_id(row, subreddit, title, created_utc)
                post = {
                    "post_id": post_id,
                    "subreddit": subreddit,
                    "title": title,
                    "selftext": body,
                    "created_utc": created_utc,
                }
                previous = by_id.get(post_id)
                if previous and previous["subreddit"].casefold() != subreddit.casefold():
                    id_collisions.add(post_id)
                elif previous is None or len(body) > len(previous["selftext"]):
                    by_id[post_id] = post

    for post_id in id_collisions:
        by_id.pop(post_id, None)

    title_counts = Counter(post["title"].casefold() for post in by_id.values())
    posts = [
        post for post in by_id.values() if title_counts[post["title"].casefold()] == 1
    ]
    posts.sort(key=lambda post: post["post_id"])
    report = {
        "rows_read": read_rows,
        "rejected_missing_or_unusable_text": rejected_text,
        "rejected_id_collisions": len(id_collisions),
        "rejected_duplicate_titles": len(by_id) - len(posts),
        "paired_posts": len(posts),
    }
    return posts, report


def load_title_posts(paths: Iterable[str | Path]) -> tuple[list[dict], dict[str, int]]:
    by_id: dict[str, dict] = {}
    read_rows = 0
    rejected_text = 0
    for path in discover_csvs(paths):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                read_rows += 1
                subreddit = _first(row, _SUBREDDIT_FIELDS)
                title = normalize_title(_first(row, (*_TITLE_FIELDS, "text")))
                if not subreddit or not title:
                    rejected_text += 1
                    continue
                created_utc = parse_created_utc(_first(row, _CREATED_FIELDS))
                post_id = extract_post_id(row, subreddit, title, created_utc)
                by_id.setdefault(
                    post_id,
                    {
                        "post_id": post_id,
                        "subreddit": subreddit,
                        "title": title,
                        "selftext": "",
                        "created_utc": created_utc,
                    },
                )

    title_counts = Counter(post["title"].casefold() for post in by_id.values())
    posts = [
        post for post in by_id.values() if title_counts[post["title"].casefold()] == 1
    ]
    posts.sort(key=lambda post: post["post_id"])
    return posts, {
        "rows_read": read_rows,
        "rejected_missing_title": rejected_text,
        "rejected_duplicate_titles": len(by_id) - len(posts),
        "title_posts": len(posts),
    }


def load_communities(path: str | Path) -> dict[str, dict]:
    communities: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            community_id = _first(row, ("community_id", "subreddit_name", "subreddit", "name"))
            description = _first(row, ("public_description", "description"))
            if not community_id or not description:
                continue
            communities[community_id.casefold()] = {
                "community_id": community_id,
                "name": f"r/{community_id}",
                "public_description": description,
            }
    return communities


def select_eligible(posts: list[dict], metadata: dict[str, dict], min_posts: int) -> tuple[list[dict], list[dict]]:
    counts = Counter(post["subreddit"].casefold() for post in posts)
    eligible = {
        key for key, count in counts.items() if count >= min_posts and key in metadata
    }
    selected_posts = [post for post in posts if post["subreddit"].casefold() in eligible]
    canonical_names = {key: metadata[key]["community_id"] for key in eligible}
    for post in selected_posts:
        post["subreddit"] = canonical_names[post["subreddit"].casefold()]
    communities = [metadata[key] for key in sorted(eligible)]
    return selected_posts, communities


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def build_dataset(
    args: argparse.Namespace,
    encode: Callable[[list[str]], object] | None = None,
) -> dict:
    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config["benchmark_seed"])
    max_k = max(config["candidate_counts"])

    source_mode = getattr(args, "source_mode", "paired")
    if source_mode == "paired":
        posts, report = load_paired_posts(args.posts)
        available_text_modes = ["title", "title+selftext"]
    else:
        posts, report = load_title_posts(args.posts)
        available_text_modes = ["title"]
    posts, communities = select_eligible(
        posts, load_communities(args.communities), args.min_posts
    )
    if len(communities) < 2 * max_k:
        raise ValueError(
            f"Need at least {2 * max_k} eligible communities for seen/unseen K={max_k}; "
            f"found {len(communities)}"
        )

    groups = split_communities(
        [community["community_id"] for community in communities],
        float(config["splits"]["unseen_fraction"]),
        max_k,
        seed,
    )
    for community in communities:
        community["split_group"] = groups[community["community_id"]]

    posts = assign_splits(
        posts,
        groups,
        float(config["splits"]["seen_train_fraction"]),
        float(config["splits"]["seen_dev_fraction"]),
        seed,
    )

    mining = config["negative_mining"]
    semantic_pool_size = int(mining["semantic_pool_size"])
    if semantic_pool_size < max_k - 1:
        raise ValueError("semantic_pool_size must contain at least max_k - 1 negatives")

    encoder = encode
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            mining["embedding_model"], revision=str(mining["revision"])
        )
        encoder = lambda texts: model.encode(
            texts,
            batch_size=args.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=bool(mining["normalize"]),
            show_progress_bar=True,
        )
    pools, instances = build_manifests(
        posts,
        communities,
        encode=encoder,
        semantic_pool_size=semantic_pool_size,
        seed=seed,
    )

    output = Path(args.output)
    write_parquet(output / "posts.parquet", posts)
    write_parquet(output / "communities.parquet", communities)
    write_parquet(output / "manifests" / "negative_pools.parquet", pools)
    write_parquet(output / "manifests" / "instances.parquet", instances)
    report.update(
        {
            "eligible_posts": len(posts),
            "eligible_communities": len(communities),
            "negative_pools": len(pools),
            "evaluation_instances": len(instances),
            "embedding_model": mining["embedding_model"],
            "embedding_revision": str(mining["revision"]),
            "benchmark_seed": seed,
            "available_text_modes": available_text_modes,
            "source_mode": source_mode,
        }
    )
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the dynamic subreddit benchmark")
    parser.add_argument("--posts", nargs="+", required=True, help="Raw post CSV files or directories")
    parser.add_argument("--communities", required=True, help="CSV containing public descriptions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="tasks/subreddit_dynamic/task.yaml")
    parser.add_argument(
        "--source-mode",
        choices=("paired", "title-only"),
        default="paired",
        help="Use title-only only when the frozen source has no post bodies",
    )
    parser.add_argument("--min-posts", type=int, default=20)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(build_dataset(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
