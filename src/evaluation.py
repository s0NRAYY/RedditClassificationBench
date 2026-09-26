from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from adapters import LayaAdapter
from dataset import post_text
from metrics import aggregate, read_jsonl, write_metrics
from sampling import candidates_for_k, stable_uint


def _load_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def _select_posts(instances: list[dict], per_community: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for instance in instances:
        if instance["difficulty"] == "random" and instance["split"] == "test":
            grouped[instance["target"]].append(instance)
    selected = []
    for target, rows in grouped.items():
        rows.sort(key=lambda row: stable_uint(seed, "evaluation", row["post_id"]))
        selected.extend(rows[:per_community])
    return sorted(selected, key=lambda row: (row["community_track"], row["target"], row["post_id"]))


def _questions_for_post(
    post: dict,
    track: str,
    community_by_id: dict[str, dict],
    pool_by_id: dict[str, dict],
    shuffle_seeds: dict[str, int],
    text_mode: str,
    representations: list[str],
    difficulties: list[str],
    candidate_counts: list[int],
    instruction: str,
) -> tuple[dict, dict]:
    questions: dict[str, dict] = {}
    metadata: dict[str, dict] = {}
    target = post["subreddit"]
    for difficulty in difficulties:
        pool = pool_by_id[f"{track}:{target}:{difficulty}"]["negative_pool"]
        for k in candidate_counts:
            candidates = candidates_for_k(target, pool, k, shuffle_seeds[difficulty])
            for representation in representations:
                qid = f"{text_mode}|{representation}|{track}|{difficulty}|{k}"
                if representation == "name-only":
                    labels = [community_by_id[candidate]["name"] for candidate in candidates]
                    criteria: list[str] | dict[str, str] = labels
                elif representation == "anonymous-id+description":
                    labels = [f"option_{index:02d}" for index in range(1, k + 1)]
                    criteria = {
                        label: community_by_id[candidate]["public_description"]
                        for label, candidate in zip(labels, candidates, strict=True)
                    }
                else:
                    raise ValueError(f"Unknown candidate representation: {representation}")
                questions[qid] = {
                    "type": "choice",
                    "instructions": instruction,
                    "criteria": criteria,
                }
                metadata[qid] = {
                    "candidates": candidates,
                    "labels": labels,
                    "representation": representation,
                    "difficulty": difficulty,
                    "k": k,
                }
    return questions, metadata


def _prediction_rows(
    post: dict,
    track: str,
    text_mode: str,
    result: dict,
    metadata: dict[str, dict],
    batch_latency_ms: float,
) -> list[dict]:
    count = len(metadata)
    input_tokens = result["usage"]["input_tokens"] / count
    rows = []
    for qid, meta in metadata.items():
        answer = result["answers"][qid]
        raw_probabilities = [float(answer["probabilities"][label]) for label in meta["labels"]]
        probability_sum = sum(raw_probabilities)
        if probability_sum <= 0:
            raise ValueError(f"Laya returned zero probability mass for {qid}")
        probabilities = [value / probability_sum for value in raw_probabilities]
        label_to_candidate = dict(zip(meta["labels"], meta["candidates"], strict=True))
        prediction = label_to_candidate[answer["choice"]]
        target_index = meta["candidates"].index(post["subreddit"])
        rows.append(
            {
                "prediction_id": f'{post["post_id"]}|{qid}',
                "post_id": post["post_id"],
                "target": post["subreddit"],
                "prediction": prediction,
                "correct": prediction == post["subreddit"],
                "text_mode": text_mode,
                "representation": meta["representation"],
                "community_track": track,
                "difficulty": meta["difficulty"],
                "k": meta["k"],
                "target_position": target_index,
                "true_probability": probabilities[target_index],
                "confidence": max(probabilities),
                "squared_probability_sum": sum(value * value for value in probabilities),
                "public_probability_sum": probability_sum,
                "candidate_fingerprint": hashlib.sha256(
                    "\0".join(meta["candidates"]).encode()
                ).hexdigest(),
                "batch_latency_ms": batch_latency_ms,
                "batch_questions": count,
                "amortized_latency_ms": batch_latency_ms / count,
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "api_cost_usd": 0.0,
            }
        )
    return rows


def _dataset_posts(path: Path, selected_ids: set[str]) -> dict[str, dict]:
    table = pq.read_table(path)
    mask = pc.is_in(table["post_id"], value_set=pa.array(sorted(selected_ids)))
    return {row["post_id"]: row for row in table.filter(mask).to_pylist()}


def evaluate(args: argparse.Namespace) -> list[dict]:
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = yaml.safe_load(Path(args.config).read_text())
    task_config = yaml.safe_load(Path(args.task_config).read_text())
    build = json.loads((dataset_dir / "build.json").read_text())

    text_modes = args.text_modes or build["available_text_modes"]
    unavailable = set(text_modes) - set(build["available_text_modes"])
    if unavailable:
        raise ValueError(f"Dataset does not provide text modes: {sorted(unavailable)}")

    instances = _load_rows(dataset_dir / "manifests" / "instances.parquet")
    selected = _select_posts(
        instances,
        args.posts_per_community or int(model_config["posts_per_community"]),
        int(task_config["benchmark_seed"]),
    )
    selected_ids = {row["post_id"] for row in selected}
    posts = _dataset_posts(dataset_dir / "posts.parquet", selected_ids)
    communities = _load_rows(dataset_dir / "communities.parquet")
    pools = _load_rows(dataset_dir / "manifests" / "negative_pools.parquet")
    community_by_id = {row["community_id"]: row for row in communities}
    pool_by_id = {row["pool_id"]: row for row in pools}
    instance_by_key = {
        (row["post_id"], row["difficulty"]): row
        for row in instances
        if row["post_id"] in selected_ids
    }

    try:
        laya_version = version("laya-mlx")
    except PackageNotFoundError:
        laya_version = "unavailable"

    adapter = LayaAdapter(
        model_config["model"],
        revision=model_config.get("revision"),
        dtype=model_config["dtype"],
        batch_size=int(model_config["batch_size"]),
        device=model_config["device"],
        compile=bool(model_config.get("compile", False)),
        pad_to_multiple=model_config.get("pad_to_multiple"),
    )

    run = {
        "model": model_config["model"],
        "revision": model_config.get("revision"),
        "dtype": model_config["dtype"],
        "laya_mlx_version": laya_version,
        "model_runtime": getattr(adapter, "metadata", {}),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "dataset_build": build,
        "text_modes": text_modes,
        "representations": model_config["representations"],
        "difficulties": task_config["difficulties"],
        "candidate_counts": task_config["candidate_counts"],
        "selected_posts": len(selected),
    }
    run_path = output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("Output directory belongs to a different evaluation configuration")
    run_path.write_text(json.dumps(run, indent=2) + "\n")

    predictions_path = output_dir / "predictions.jsonl"
    completed = set()
    if predictions_path.exists():
        completed = {row["prediction_id"] for row in read_jsonl(predictions_path)}

    adapter.warmup()

    with predictions_path.open("a", encoding="utf-8") as output:
        for index, selected_instance in enumerate(selected, start=1):
            post = posts[selected_instance["post_id"]]
            track = selected_instance["community_track"]
            shuffle_seeds = {
                difficulty: instance_by_key[(post["post_id"], difficulty)]["shuffle_seed"]
                for difficulty in task_config["difficulties"]
            }
            for text_mode in text_modes:
                questions, metadata = _questions_for_post(
                    post,
                    track,
                    community_by_id,
                    pool_by_id,
                    shuffle_seeds,
                    text_mode,
                    model_config["representations"],
                    task_config["difficulties"],
                    task_config["candidate_counts"],
                    model_config["instruction"],
                )
                metadata = {
                    qid: value
                    for qid, value in metadata.items()
                    if f'{post["post_id"]}|{qid}' not in completed
                }
                questions = {qid: questions[qid] for qid in metadata}
                batches: dict[tuple[str, int], list[str]] = defaultdict(list)
                for qid, item in metadata.items():
                    batches[(item["representation"], item["k"])].append(qid)
                state = post_text(post, text_mode)
                for qids in batches.values():
                    batch_questions = {qid: questions[qid] for qid in qids}
                    batch_metadata = {qid: metadata[qid] for qid in qids}
                    result, latency_ms = adapter.predict(state, batch_questions)
                    for row in _prediction_rows(
                        post, track, text_mode, result, batch_metadata, latency_ms
                    ):
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    output.flush()
            if index % 100 == 0 or index == len(selected):
                print(f"evaluated {index}/{len(selected)} posts", flush=True)

    rows = read_jsonl(predictions_path)
    metrics = aggregate(
        rows,
        bootstrap_samples=int(model_config["bootstrap_samples"]),
        seed=int(task_config["benchmark_seed"]),
    )
    write_metrics(metrics, output_dir)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dynamic subreddit routing")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/laya.yaml")
    parser.add_argument("--task-config", default="tasks/subreddit_dynamic/task.yaml")
    parser.add_argument("--posts-per-community", type=int)
    parser.add_argument("--text-modes", nargs="+", choices=("title", "title+selftext"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metrics = evaluate(parse_args(argv))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
