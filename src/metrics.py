from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

_GROUP_FIELDS = (
    "text_mode",
    "representation",
    "community_track",
    "difficulty",
    "k",
)
# Headline slice: zero-shot models have no training exposure to either track, so both are pooled
# ("all"); descriptions without names test understanding rather than subreddit-name recall.
HEADLINE = {"community_track": "all", "difficulty": "hard", "representation": "anonymous-id+description"}
HEADLINE_K = (16, 64)


def headline(metrics: list[dict]) -> list[dict]:
    return [
        row
        for row in metrics
        if row["k"] in HEADLINE_K and all(row[field] == value for field, value in HEADLINE.items())
    ]


def _ece(rows: list[dict], bins: int = 15) -> float:
    total = len(rows)
    value = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [
            row
            for row in rows
            if lower <= row["confidence"] < upper
            or (index == bins - 1 and row["confidence"] == 1.0)
        ]
        if bucket:
            accuracy = sum(row["correct"] for row in bucket) / len(bucket)
            confidence = sum(row["confidence"] for row in bucket) / len(bucket)
            value += len(bucket) / total * abs(accuracy - confidence)
    return value


def _aurc(rows: list[dict]) -> float:
    ordered = sorted(rows, key=lambda row: row["confidence"], reverse=True)
    errors = np.fromiter((not row["correct"] for row in ordered), dtype=np.float64)
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def _bootstrap_accuracy(rows: list[dict], samples: int, seed: int) -> tuple[float, float]:
    values = np.fromiter((row["correct"] for row in rows), dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = values[rng.integers(0, len(values), len(values))].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def aggregate(rows: list[dict], *, bootstrap_samples: int = 1000, seed: int = 0) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in _GROUP_FIELDS)
        grouped[key].append(row)
        grouped[key[:2] + ("all",) + key[3:]].append(row)

    output = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        per_community: dict[str, list[bool]] = defaultdict(list)
        for row in group:
            per_community[row["target"]].append(row["correct"])
        accuracy = sum(row["correct"] for row in group) / len(group)
        ci_low, ci_high = _bootstrap_accuracy(
            group, bootstrap_samples, seed + group_index
        )
        latencies = np.fromiter(
            (row["amortized_latency_ms"] for row in group), dtype=np.float64
        )
        probability_rows = [
            row for row in group if row.get("true_probability") is not None
        ]
        metric = dict(zip(_GROUP_FIELDS, key, strict=True))
        metric.update(
            examples=len(group),
            communities=len(per_community),
            accuracy=accuracy,
            accuracy_ci95_low=ci_low,
            accuracy_ci95_high=ci_high,
            macro_community_accuracy=float(
                np.mean([np.mean(values) for values in per_community.values()])
            ),
            probability_source=group[0].get("probability_source", "native"),
            nll=(
                float(
                    np.mean(
                        [
                            -math.log(max(row["true_probability"], 1e-12))
                            for row in probability_rows
                        ]
                    )
                )
                if probability_rows
                else None
            ),
            brier=(
                float(
                    np.mean(
                        [
                            row["squared_probability_sum"]
                            - 2 * row["true_probability"]
                            + 1
                            for row in probability_rows
                        ]
                    )
                )
                if probability_rows
                else None
            ),
            ece_15=_ece(probability_rows) if probability_rows else None,
            aurc=_aurc(probability_rows) if probability_rows else None,
            latency_ms_p50=float(np.quantile(latencies, 0.5)),
            latency_ms_p95=float(np.quantile(latencies, 0.95)),
            mean_input_tokens=float(np.mean([row["input_tokens"] for row in group])),
            mean_output_tokens=float(np.mean([row.get("output_tokens", 0) for row in group])),
            mean_forward_count=float(np.mean([row.get("forward_count", 1) for row in group])),
            api_cost_usd=float(sum(row.get("api_cost_usd", 0) for row in group)),
        )
        output.append(metric)
    return output


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_metrics(rows: list[dict], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=rows[0].keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
