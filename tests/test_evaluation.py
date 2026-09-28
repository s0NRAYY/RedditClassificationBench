import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import evaluation
from evaluation import evaluate


def _parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


class FakeAdapter:
    capabilities = {
        "protocol": 1,
        "probabilities": "native",
        "max_questions": None,
        "max_options": None,
    }
    metadata = {"fake": True}

    def warmup(self):
        pass
    def predict(self, state, questions):
        answers = {}
        tokens = 0
        for qid, question in questions.items():
            labels = list(question["criteria"])
            probability = 1 / len(labels)
            answers[qid] = {
                "choice": labels[0],
                "probabilities": {label: probability for label in labels},
            }
            tokens += len(state.split()) + len(labels)
        return {"answers": answers, "usage": {"input_tokens": tokens}}, 12.0

    def close(self):
        pass


def test_full_matrix_metrics_and_resume(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    communities = [
        {
            "community_id": community,
            "name": f"r/{community}",
            "public_description": f"Description for {community}",
            "split_group": track,
        }
        for community, track in (("a", "seen"), ("b", "seen"), ("c", "unseen"), ("d", "unseen"))
    ]
    posts = [
        {
            "post_id": community,
            "subreddit": community,
            "title": f"Post for {community}",
            "selftext": "Body",
            "created_utc": 0,
            "split": "test",
        }
        for community in ("a", "b", "c", "d")
    ]
    pools = []
    instances = []
    groups = {row["community_id"]: row["split_group"] for row in communities}
    peers = {"a": "b", "b": "a", "c": "d", "d": "c"}
    for community in peers:
        for difficulty in ("random", "semantic", "hard"):
            track = groups[community]
            pool_id = f"{track}:{community}:{difficulty}"
            pools.append(
                {
                    "pool_id": pool_id,
                    "community_track": track,
                    "target": community,
                    "difficulty": difficulty,
                    "mining_strategy": "fake",
                    "negative_pool": [peers[community]],
                }
            )
            instances.append(
                {
                    "post_id": community,
                    "target": community,
                    "community_track": track,
                    "split": "test",
                    "difficulty": difficulty,
                    "pool_id": pool_id,
                    "shuffle_seed": 7,
                }
            )
    _parquet(dataset / "posts.parquet", posts)
    _parquet(dataset / "communities.parquet", communities)
    _parquet(dataset / "manifests/negative_pools.parquet", pools)
    _parquet(dataset / "manifests/instances.parquet", instances)
    (dataset / "build.json").write_text(
        json.dumps({"available_text_modes": ["title", "title+selftext"]})
    )

    task = tmp_path / "task.yaml"
    task.write_text(
        """benchmark_seed: 7
candidate_counts: [2]
difficulties: [random, semantic, hard]
"""
    )
    config = tmp_path / "laya.yaml"
    config.write_text(
        """model: fake
revision: frozen
dtype: float16
device: gpu
batch_size: 32
compile: false
pad_to_multiple: null
posts_per_community: 1
bootstrap_samples: 10
representations: [name-only, anonymous-id+description]
instruction: Select one.
"""
    )
    output = tmp_path / "results"
    args = argparse.Namespace(
        dataset=str(dataset),
        output=str(output),
        config=str(config),
        task_config=str(task),
        posts_per_community=None,
        text_modes=None,
    )
    monkeypatch.setattr(
        evaluation,
        "_build_adapter",
        lambda args, config: (FakeAdapter(), "fake", {}),
    )

    metrics = evaluate(args)
    first_count = len((output / "predictions.jsonl").read_text().splitlines())
    resumed_metrics = evaluate(args)
    second_count = len((output / "predictions.jsonl").read_text().splitlines())

    assert first_count == 48
    assert second_count == first_count
    assert len(metrics) == 24
    assert resumed_metrics == metrics
    assert all(row["api_cost_usd"] == 0 for row in metrics)
