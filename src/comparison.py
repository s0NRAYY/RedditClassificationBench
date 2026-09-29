from __future__ import annotations

import json
from pathlib import Path

from metrics import read_jsonl


COMPATIBILITY_FIELDS = (
    "dataset_build",
    "text_modes",
    "representations",
    "difficulties",
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
    for path, run, metrics, predictions in runs[1:]:
        for field in COMPATIBILITY_FIELDS:
            if run.get(field) != reference.get(field):
                raise ValueError(f"Incompatible {field}: {runs[0][0]} vs {path}")
        if _requested_counts(run) != _requested_counts(reference):
            raise ValueError(f"Incompatible candidate_counts: {runs[0][0]} vs {path}")
        shared = set(run["candidate_counts"]) & set(reference["candidate_counts"])
        if _ids(predictions, shared) != _ids(reference_predictions, shared):
            raise ValueError(f"Runs contain different sampled predictions: {path}")
        if _slices(metrics, shared) != _slices(reference_metrics, shared):
            raise ValueError(f"Runs contain different metric slices: {path}")

    output = []
    for path, run, metrics, _ in runs:
        revision = run.get("revision")
        system = run["model"] + (f"@{revision[:8]}" if revision else "")
        for metric in metrics:
            output.append({"system": system, "result_dir": str(path), **metric})
    return output


def _requested_counts(run: dict) -> list[int]:
    """Candidate counts asked for, including those a worker's declared limits forced to skip."""
    return sorted(run["candidate_counts"] + run.get("skipped_candidate_counts", []))


def _ids(predictions: list[dict], counts: set[int]) -> set[str]:
    return {row["prediction_id"] for row in predictions if row["k"] in counts}


def _slices(metrics: list[dict], counts: set[int]) -> set[tuple]:
    return {tuple(row[field] for field in SLICE_FIELDS) for row in metrics if row["k"] in counts}
