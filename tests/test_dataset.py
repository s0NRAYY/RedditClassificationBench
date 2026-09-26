import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dataset import build_dataset, load_paired_posts, load_title_posts, post_text


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_paired_filter_and_post_id_priority(tmp_path):
    rows = [
        {"id": "t3_abc", "permalink": "/r/x/comments/wrong/a/", "subreddit_name": "x", "title": "A title", "text": "Useful body", "create_time": "2024-01-01 00:00:00"},
        {"id": "", "permalink": "/r/x/comments/def/a/", "subreddit_name": "x", "title": "Another title", "text": "Another body", "create_time": "2024-01-02 00:00:00"},
        {"id": "gone", "permalink": "", "subreddit_name": "x", "title": "Gone", "text": "[deleted]", "create_time": "0"},
        {"id": "url", "permalink": "", "subreddit_name": "x", "title": "URL", "text": "https://example.com", "create_time": "0"},
        {"id": "dup1", "permalink": "", "subreddit_name": "x", "title": "Same   title", "text": "First body", "create_time": "0"},
        {"id": "dup2", "permalink": "", "subreddit_name": "y", "title": "same title", "text": "Second body", "create_time": "0"},
    ]
    source = tmp_path / "posts.csv"
    _write_csv(source, rows)

    posts, report = load_paired_posts([source])

    assert [post["post_id"] for post in posts] == ["abc", "def"]
    assert report["rejected_missing_or_unusable_text"] == 2
    assert report["rejected_duplicate_titles"] == 2
    assert post_text(posts[0], "title") == posts[0]["title"]
    assert post_text(posts[0], "title+selftext").endswith(posts[0]["selftext"])


def test_title_only_loader_keeps_explicit_provenance(tmp_path):
    source = tmp_path / "titles.csv"
    _write_csv(
        source,
        [
            {"label": "x", "text": "A useful title"},
            {"label": "y", "text": "Another useful title"},
        ],
    )

    posts, report = load_title_posts([source])

    assert len(posts) == 2
    assert all(post["selftext"] == "" for post in posts)
    assert report["title_posts"] == 2
    try:
        post_text(posts[0], "title+selftext")
    except ValueError:
        pass
    else:
        raise AssertionError("title-only rows must reject title+selftext")


def test_builder_writes_leak_free_paired_artifacts(tmp_path):
    post_rows = []
    community_rows = []
    for community_index in range(4):
        community = f"c{community_index}"
        community_rows.append(
            {"subreddit_name": community, "public_description": f"topic {community_index}"}
        )
        for post_index in range(3):
            post_rows.append(
                {
                    "id": f"{community}{post_index}",
                    "permalink": "",
                    "subreddit_name": community,
                    "title": f"Unique title {community_index} {post_index}",
                    "text": f"Body about topic {community_index} item {post_index}",
                    "create_time": str(1700000000 + post_index),
                }
            )

    posts_csv = tmp_path / "posts.csv"
    communities_csv = tmp_path / "communities.csv"
    _write_csv(posts_csv, post_rows)
    _write_csv(communities_csv, community_rows)
    config = tmp_path / "task.yaml"
    config.write_text(
        """candidate_counts: [2]
benchmark_seed: 7
splits:
  unseen_fraction: 0.5
  seen_train_fraction: 0.34
  seen_dev_fraction: 0.33
negative_mining:
  embedding_model: fake
  revision: frozen
  normalize: true
  semantic_pool_size: 2
"""
    )
    output = tmp_path / "output"

    def encode(texts):
        return np.array(
            [[1.0, float(sum(map(ord, text)) % 17 + 1)] for text in texts],
            dtype=np.float32,
        )

    report = build_dataset(
        argparse.Namespace(
            posts=[str(posts_csv)],
            communities=str(communities_csv),
            output=str(output),
            config=str(config),
            min_posts=3,
            embedding_batch_size=8,
        ),
        encode=encode,
    )

    posts = pq.read_table(output / "posts.parquet").to_pylist()
    communities = pq.read_table(output / "communities.parquet").to_pylist()
    pools = pq.read_table(output / "manifests/negative_pools.parquet").to_pylist()
    instances = pq.read_table(output / "manifests/instances.parquet").to_pylist()

    groups = {row["community_id"]: row["split_group"] for row in communities}
    assert report["eligible_posts"] == 12
    assert all(post["title"] and post["selftext"] for post in posts)
    assert all(post["split"] == "test" for post in posts if groups[post["subreddit"]] == "unseen")
    assert all(post["split"] != "train" for post in posts if groups[post["subreddit"]] == "unseen")
    assert all(row["target"] not in row["negative_pool"] for row in pools)
    assert all(row["community_track"] == groups[row["target"]] for row in instances)
