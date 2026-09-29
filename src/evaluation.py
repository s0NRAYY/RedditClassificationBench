from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from adapters import CommandAdapter
from credentials import model_credentials
from dataset import post_text
from metrics import aggregate, read_jsonl, write_metrics
from model_registry import resolve_model
from sampling import candidates_for_k, stable_uint

console = Console(stderr=True)


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
    usage = result.get("usage") or {}
    input_tokens = float(usage.get("input_tokens", 0)) / count
    output_tokens = float(usage.get("output_tokens", 0)) / count
    forward_count = float(usage.get("forward_count", 1)) / count
    probability_source = result.get("probability_source", "native")
    rows = []
    for qid, meta in metadata.items():
        answer = result["answers"][qid]
        raw = answer.get("probabilities")
        probabilities = None
        probability_sum = None
        if raw is not None:
            raw_probabilities = [float(raw[label]) for label in meta["labels"]]
            probability_sum = sum(raw_probabilities)
            if probability_sum <= 0:
                raise ValueError(f"Model returned zero probability mass for {qid}")
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
                "true_probability": (
                    probabilities[target_index] if probabilities is not None else None
                ),
                "confidence": max(probabilities) if probabilities is not None else None,
                "squared_probability_sum": (
                    sum(value * value for value in probabilities)
                    if probabilities is not None
                    else None
                ),
                "public_probability_sum": probability_sum,
                "probability_source": probability_source,
                "candidate_fingerprint": hashlib.sha256(
                    "\0".join(meta["candidates"]).encode()
                ).hexdigest(),
                "batch_latency_ms": batch_latency_ms,
                "batch_questions": count,
                "amortized_latency_ms": batch_latency_ms / count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "forward_count": forward_count,
                "api_cost_usd": float(usage.get("api_cost_usd", 0)) / count,
            }
        )
    return rows


def _dataset_posts(path: Path, selected_ids: set[str]) -> dict[str, dict]:
    table = pq.read_table(path)
    mask = pc.is_in(table["post_id"], value_set=pa.array(sorted(selected_ids)))
    return {row["post_id"]: row for row in table.filter(mask).to_pylist()}


def _build_adapter(args: argparse.Namespace, config: dict) -> tuple[CommandAdapter, str, dict]:
    requested_model = getattr(args, "model", None) or config["model"]
    model, registry = resolve_model(requested_model)
    adapter_name = (
        getattr(args, "adapter", None)
        or config.get("adapter")
        or registry.get("adapter")
    )
    command = getattr(args, "command_worker", None) or config.get("command")
    worker_env = None
    saved_credentials = model_credentials(model)
    effective_revision = config.get("revision") or registry.get("revision")
    if command is None:
        if adapter_name is None:
            raise ValueError(
                f"No registered adapter for {model}; pass --command-worker"
            )
        options = dict(registry.get("options", {}))
        options.update(config.get("adapter_options", {}))
        for key in ("probabilities", "max_questions", "max_options"):
            if key in registry:
                options.setdefault(key, registry[key])
        if effective_revision:
            options["revision"] = effective_revision
        for key in (
            "dtype",
            "batch_size",
            "device",
            "compile",
            "pad_to_multiple",
        ):
            if key in config:
                options.setdefault(key, config[key])
        endpoint = (
            getattr(args, "endpoint", None)
            or config.get("endpoint")
            or saved_credentials.get("endpoint")
        )
        if endpoint:
            options["endpoint"] = endpoint
        api_key = getattr(args, "api_key", None)
        if api_key is None:
            api_key = saved_credentials.get("api_key")
        if api_key:
            options["api_key_env"] = "SRB_MODEL_API_KEY"
            worker_env = os.environ.copy()
            worker_env["SRB_MODEL_API_KEY"] = api_key
        runtime_model = registry.get("runtime_model", model)
        command = [
            sys.executable,
            "-m",
            "model_worker",
            "--adapter",
            adapter_name,
            "--model",
            runtime_model,
            "--options",
            json.dumps(options, separators=(",", ":")),
        ]
    adapter = CommandAdapter(
        command,
        model=model,
        revision=effective_revision,
        env=worker_env,
    )
    return adapter, model, registry


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
    posts_per_community = int(
        args.posts_per_community
        or task_config.get("posts_per_community")
        or model_config["posts_per_community"]
    )
    selected = _select_posts(
        instances,
        posts_per_community,
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

    requested_model = getattr(args, "model", None) or model_config["model"]
    console.print(f"[bold cyan]Loading model[/] [white]{requested_model}[/]...")
    adapter, model, registry = _build_adapter(args, model_config)
    max_options = adapter.capabilities.get("max_options")
    candidate_counts = [
        k for k in task_config["candidate_counts"] if max_options is None or k <= max_options
    ]
    skipped_counts = [k for k in task_config["candidate_counts"] if k not in candidate_counts]
    if not candidate_counts:
        adapter.close()
        raise ValueError(
            f"Worker supports at most {max_options} options; "
            f"every requested candidate count {task_config['candidate_counts']} exceeds it"
        )
    if skipped_counts:
        console.print(
            f"[yellow]Skipping candidate counts {skipped_counts}: "
            f"worker supports at most {max_options} options[/]"
        )

    run = {
        "model": model,
        "revision": getattr(
            adapter,
            "revision",
            model_config.get("revision") or registry.get("revision"),
        ),
        "adapter": adapter.capabilities,
        "registry_entry": registry,
        "model_runtime": adapter.metadata,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "dataset_build": build,
        "text_modes": text_modes,
        "representations": model_config["representations"],
        "difficulties": task_config["difficulties"],
        "candidate_counts": candidate_counts,
        "selected_posts": len(selected),
    }
    if skipped_counts:
        run["skipped_candidate_counts"] = skipped_counts
    run_path = output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("Output directory belongs to a different evaluation configuration")
    run_path.write_text(json.dumps(run, indent=2) + "\n")

    predictions_path = output_dir / "predictions.jsonl"
    completed = set()
    if predictions_path.exists():
        completed = {row["prediction_id"] for row in read_jsonl(predictions_path)}

    expected_predictions = (
        len(selected)
        * len(text_modes)
        * len(model_config["representations"])
        * len(task_config["difficulties"])
        * len(candidate_counts)
    )
    configuration = (
        ("Model", model),
        ("Dataset", str(dataset_dir)),
        ("Task config", str(args.task_config)),
        ("Output", str(output_dir)),
        ("Posts / community", str(posts_per_community)),
        ("Text modes", ", ".join(text_modes)),
        ("Representations", ", ".join(model_config["representations"])),
        ("Difficulties", ", ".join(task_config["difficulties"])),
        ("Candidate counts", ", ".join(map(str, candidate_counts))),
        ("Selected posts", str(len(selected))),
        ("Predictions", f"{len(completed)} completed / {expected_predictions} total"),
    )
    details = Table.grid(padding=(0, 2))
    details.add_column(style="cyan", justify="right")
    details.add_column(style="white")
    for label, value in configuration:
        details.add_row(label, value)
    console.print(
        Panel(
            details,
            title="[bold cyan]⚙ Benchmark Configuration[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print("[cyan]◌[/] Warming up model...")
    adapter.warmup()

    written = 0
    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None, style="bright_black", complete_style="cyan"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("[magenta]{task.fields[predictions]} predictions"),
        console=console,
        expand=True,
    )
    task_id = progress.add_task(
        "Evaluating",
        total=len(selected),
        predictions=len(completed),
    )
    with progress, predictions_path.open("a", encoding="utf-8") as output:
        for selected_instance in selected:
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
                    candidate_counts,
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
                    prediction_rows = _prediction_rows(
                        post, track, text_mode, result, batch_metadata, latency_ms
                    )
                    for row in prediction_rows:
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    written += len(prediction_rows)
                    progress.update(
                        task_id,
                        predictions=len(completed) + written,
                    )
                    output.flush()
            progress.advance(task_id)
    console.print("[cyan]◌[/] Writing metrics...")
    adapter.close()

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
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--endpoint")
    parser.add_argument("--command-worker")
    parser.add_argument("--task-config", default="tasks/subreddit_dynamic/task.yaml")
    parser.add_argument("--posts-per-community", type=int)
    parser.add_argument("--text-modes", nargs="+", choices=("title", "title+selftext"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metrics = evaluate(parse_args(argv))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
