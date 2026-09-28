from __future__ import annotations

import json
from pathlib import Path

from metrics import read_jsonl


COMPATIBILITY_FIELDS = (
    "dataset_build",
    "text_modes",
    "representations",
    "difficulties",
    "candidate_counts",
    "selected_posts",
)
SLICE_FIELDS = (
    "text_mode",
    "representation",
    "community_track",
    "difficulty",
    "k",
)


def compare_runs(paths: list[str | Path]) -> list[dict]:
    if len(paths) < 2:
        raise ValueError("Compare requires at least two result directories")

    runs = []
    for value in paths:
        path = Path(value)
        try:
            run = json.loads((path / "run.json").read_text())
            metrics = json.loads((path / "metrics.json").read_text())
            predictions = read_jsonl(path / "predictions.jsonl")
        except FileNotFoundError as error:
            raise ValueError(f"Incomplete result directory: {path}") from error
        runs.append((path, run, metrics, predictions))

    _, reference, reference_metrics, reference_predictions = runs[0]
    reference_ids = {row["prediction_id"] for row in reference_predictions}
    reference_posts = {row["post_id"] for row in reference_predictions}
    reference_slices = {
        tuple(row[field] for field in SLICE_FIELDS)
        for row in reference_metrics
    }
    for path, run, metrics, predictions in runs[1:]:
        for field in COMPATIBILITY_FIELDS:
            if run.get(field) != reference.get(field):
                raise ValueError(f"Incompatible {field}: {runs[0][0]} vs {path}")
        ids = {row["prediction_id"] for row in predictions}
        posts = {row["post_id"] for row in predictions}
        if posts != reference_posts or ids != reference_ids:
            raise ValueError(f"Runs contain different sampled predictions: {path}")
        slices = {tuple(row[field] for field in SLICE_FIELDS) for row in metrics}
        if slices != reference_slices:
            raise ValueError(f"Runs contain different metric slices: {path}")

    output = []
    for path, run, metrics, _ in runs:
        revision = run.get("revision")
        system = run["model"] + (f"@{revision[:8]}" if revision else "")
        for metric in metrics:
            output.append({"system": system, "result_dir": str(path), **metric})
    return output
