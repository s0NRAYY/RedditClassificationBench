#!/usr/bin/env python3
"""Supervised reference: logistic regression on title embeddings, trained on seen-train posts.

Unseen communities have no training posts by construction, so this baseline covers the seen
track only. It scores exactly the benchmark's `medium` candidate sets; the classifier ranks all
seen communities and the prediction is the best-scoring candidate. Candidate representation
does not matter to it (it knows community ids), so one number covers names and descriptions.

A different embedding model from the negative miner (all-MiniLM-L6-v2) is used on purpose:
hard negatives are nearest neighbours in MiniLM space, which would handicap a MiniLM classifier.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from evaluation import _load_rows, _select_posts
from sampling import candidates_for_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="data/subreddit_dynamic")
    parser.add_argument("--task-config", default="tasks/subreddit_dynamic/medium.yaml")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--train-per-community", type=int, default=200)
    parser.add_argument("--output", default="results/baselines/logreg-seen-medium.json")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    task = yaml.safe_load(Path(args.task_config).read_text())
    seed = int(task["benchmark_seed"])
    revision = HfApi().model_info(args.embedding_model).sha
    encoder = SentenceTransformer(args.embedding_model, revision=revision)

    communities = {row["community_id"]: row for row in _load_rows(dataset / "communities.parquet")}
    seen = sorted(c for c, row in communities.items() if row["split_group"] == "seen")
    instances = _load_rows(dataset / "manifests" / "instances.parquet")
    pools = {row["pool_id"]: row for row in _load_rows(dataset / "manifests" / "negative_pools.parquet")}
    selected = [
        row for row in _select_posts(instances, int(task["posts_per_community"]), seed)
        if row["community_track"] == "seen"
    ]
    test_ids = {row["post_id"] for row in selected}

    posts = pq.read_table(dataset / "posts.parquet", columns=["post_id", "subreddit", "title", "split"]).to_pylist()
    by_community: dict[str, list[dict]] = defaultdict(list)
    for post in posts:
        if post["split"] == "train" and post["subreddit"] in communities and post["post_id"] not in test_ids:
            by_community[post["subreddit"]].append(post)
    rng = random.Random(seed)
    train = [p for c in seen for p in rng.sample(by_community[c], min(args.train_per_community, len(by_community[c])))]
    test = {p["post_id"]: p for p in posts if p["post_id"] in test_ids}

    embed = lambda titles: encoder.encode(titles, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    classifier = LogisticRegression(max_iter=300, C=10.0)
    classifier.fit(embed([p["title"] for p in train]), [p["subreddit"] for p in train])
    column = {c: i for i, c in enumerate(classifier.classes_)}
    order = sorted(test)
    scores = dict(zip(order, classifier.predict_log_proba(embed([test[i]["title"] for i in order]))))

    instance = {(r["post_id"], r["difficulty"]): r for r in instances if r["post_id"] in test_ids}
    results = []
    for difficulty in task["difficulties"]:
        for k in task["candidate_counts"]:
            correct = []
            for row in selected:
                target = row["target"]
                inst = instance[(row["post_id"], difficulty)]
                candidates = candidates_for_k(target, pools[inst["pool_id"]]["negative_pool"], k, inst["shuffle_seed"])
                s = scores[row["post_id"]]
                ranked = sorted(candidates, key=lambda c: -s[column[c]])
                correct.append(ranked[0] == target)
            values = np.array(correct, dtype=float)
            boot = np.random.default_rng(seed).choice(values, (1000, len(values))).mean(axis=1)
            results.append({
                "community_track": "seen", "difficulty": difficulty, "k": k, "examples": len(values),
                "accuracy": float(values.mean()),
                "accuracy_ci95_low": float(np.quantile(boot, 0.025)),
                "accuracy_ci95_high": float(np.quantile(boot, 0.975)),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "baseline": "logistic-regression on title embeddings (supervised, seen track only)",
        "embedding_model": args.embedding_model, "embedding_revision": revision,
        "train_posts": len(train), "train_per_community": args.train_per_community,
        "seen_communities": len(seen), "task_config": args.task_config, "benchmark_seed": seed,
        "metrics": results,
    }, indent=2) + "\n")
    for r in results:
        print(r["difficulty"], r["k"], f"acc={r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
