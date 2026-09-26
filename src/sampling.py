from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable

import numpy as np


def stable_uint(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(
        hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big"
    ) & ((1 << 63) - 1)


def split_communities(
    community_ids: list[str], unseen_fraction: float, max_k: int, seed: int
) -> dict[str, str]:
    if not 0 < unseen_fraction < 1:
        raise ValueError("unseen_fraction must be between 0 and 1")
    if len(community_ids) < 2 * max_k:
        raise ValueError("Each community track must contain at least max_k communities")
    ordered = sorted(community_ids, key=lambda value: stable_uint(seed, "community", value))
    unseen_count = round(len(ordered) * unseen_fraction)
    unseen_count = min(max(unseen_count, max_k), len(ordered) - max_k)
    unseen = set(ordered[:unseen_count])
    return {community_id: "unseen" if community_id in unseen else "seen" for community_id in community_ids}


def assign_splits(
    posts: list[dict],
    community_groups: dict[str, str],
    train_fraction: float,
    dev_fraction: float,
    seed: int,
) -> list[dict]:
    if train_fraction <= 0 or dev_fraction <= 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("Seen split fractions must leave non-empty train, dev, and test portions")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for post in posts:
        grouped[post["subreddit"]].append(post)

    output: list[dict] = []
    for community_id, community_posts in grouped.items():
        group = community_groups[community_id]
        ranked = sorted(
            community_posts,
            key=lambda post: stable_uint(seed, "post", post["post_id"]),
        )
        if group == "unseen":
            for post in ranked:
                output.append({**post, "split": "test"})
            continue
        if len(ranked) < 3:
            raise ValueError(f"Seen community {community_id} needs at least three posts")
        train_end = min(max(round(len(ranked) * train_fraction), 1), len(ranked) - 2)
        dev_count = min(max(round(len(ranked) * dev_fraction), 1), len(ranked) - train_end - 1)
        dev_end = train_end + dev_count
        for index, post in enumerate(ranked):
            split = "train" if index < train_end else "dev" if index < dev_end else "test"
            output.append({**post, "split": split})
    return sorted(output, key=lambda post: post["post_id"])


def _normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector")
    return matrix / norms


def _community_vectors(
    posts: list[dict],
    communities: list[dict],
    encode: Callable[[list[str]], np.ndarray],
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    dimensions: int | None = None
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    train_posts = [
        post
        for post in posts
        if post["split"] == "train"
    ]
    for start in range(0, len(train_posts), batch_size):
        batch = train_posts[start : start + batch_size]
        texts = [
            post["title"]
            if not post["selftext"]
            else f'{post["title"]}\n\n{post["selftext"]}'
            for post in batch
        ]
        vectors = np.asarray(encode(texts), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(batch):
            raise ValueError("Embedding model returned an unexpected shape")
        dimensions = vectors.shape[1]
        for post, vector in zip(batch, vectors, strict=True):
            community_id = post["subreddit"]
            sums[community_id] = sums.get(community_id, np.zeros(dimensions, dtype=np.float32)) + vector
            counts[community_id] += 1

    unseen = [community for community in communities if community["split_group"] == "unseen"]
    unseen_vectors = np.asarray(
        encode([community["public_description"] for community in unseen]),
        dtype=np.float32,
    )
    if unseen and (unseen_vectors.ndim != 2 or len(unseen_vectors) != len(unseen)):
        raise ValueError("Embedding model returned an unexpected shape")

    vectors = {community_id: vector / counts[community_id] for community_id, vector in sums.items()}
    vectors.update(
        {
            community["community_id"]: vector
            for community, vector in zip(unseen, unseen_vectors, strict=True)
        }
    )
    missing = {community["community_id"] for community in communities} - vectors.keys()
    if missing:
        raise ValueError(f"Missing train content for seen communities: {sorted(missing)}")
    return vectors


def build_negative_pools(
    communities: list[dict],
    vectors: dict[str, np.ndarray],
    semantic_pool_size: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for track in ("seen", "unseen"):
        track_ids = sorted(
            community["community_id"]
            for community in communities
            if community["split_group"] == track
        )
        matrix = _normalized(np.stack([vectors[community_id] for community_id in track_ids]))
        similarities = matrix @ matrix.T
        source = "content" if track == "seen" else "description"
        for index, target in enumerate(track_ids):
            ranked = [
                track_ids[candidate_index]
                for candidate_index in np.argsort(-similarities[index], kind="stable")
                if candidate_index != index
            ]
            semantic_source = ranked[: min(semantic_pool_size, len(ranked))]
            pools = {
                "hard": ranked,
                "semantic": sorted(
                    semantic_source,
                    key=lambda value: stable_uint(seed, "semantic", target, value),
                ),
                "random": sorted(
                    ranked,
                    key=lambda value: stable_uint(seed, "random", target, value),
                ),
            }
            for difficulty, negative_pool in pools.items():
                rows.append(
                    {
                        "pool_id": f"{track}:{target}:{difficulty}",
                        "community_track": track,
                        "target": target,
                        "difficulty": difficulty,
                        "mining_strategy": (
                            "random" if difficulty == "random" else f"{difficulty}_{source}"
                        ),
                        "negative_pool": negative_pool,
                    }
                )
    return rows



def build_manifests(
    posts: list[dict],
    communities: list[dict],
    encode: Callable[[list[str]], np.ndarray],
    semantic_pool_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    vectors = _community_vectors(posts, communities, encode)
    pools = build_negative_pools(communities, vectors, semantic_pool_size, seed)
    groups = {community["community_id"]: community["split_group"] for community in communities}
    instances = []
    for post in posts:
        if post["split"] == "train":
            continue
        track = groups[post["subreddit"]]
        for difficulty in ("random", "semantic", "hard"):
            instances.append(
                {
                    "post_id": post["post_id"],
                    "target": post["subreddit"],
                    "community_track": track,
                    "split": post["split"],
                    "difficulty": difficulty,
                    "pool_id": f'{track}:{post["subreddit"]}:{difficulty}',
                    "shuffle_seed": stable_uint(seed, post["post_id"], difficulty),
                }
            )
    return pools, instances


def candidates_for_k(
    target: str, negative_pool: list[str], k: int, shuffle_seed: int
) -> list[str]:
    if k < 2 or len(negative_pool) < k - 1:
        raise ValueError(
            f"Cannot construct K={k} from a pool of {len(negative_pool)} negatives"
        )
    candidates = [target, *negative_pool[: k - 1]]
    return sorted(
        candidates,
        key=lambda value: stable_uint(shuffle_seed, k, "candidate", value),
    )
