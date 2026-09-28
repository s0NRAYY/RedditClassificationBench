import json
from pathlib import Path

import pytest

from comparison import compare_runs


def _result(path: Path, model: str, prediction_id: str = "p|q") -> None:
    path.mkdir()
    run = {
        "model": model,
        "revision": "a" * 40,
        "dataset_build": {"fingerprint": "dataset"},
        "text_modes": ["title"],
        "representations": ["name-only"],
        "difficulties": ["random"],
        "candidate_counts": [2],
        "selected_posts": 1,
    }
    metric = {
        "text_mode": "title",
        "representation": "name-only",
        "community_track": "seen",
        "difficulty": "random",
        "k": 2,
        "accuracy": 1.0,
        "nll": None,
        "ece_15": None,
        "latency_ms_p50": 1.0,
        "mean_forward_count": 1.0,
    }
    (path / "run.json").write_text(json.dumps(run))
    (path / "metrics.json").write_text(json.dumps([metric]))
    (path / "predictions.jsonl").write_text(
        json.dumps({"prediction_id": prediction_id, "post_id": "p"}) + "\n"
    )


def test_compare_preserves_probability_free_metrics(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _result(first, "model/one")
    _result(second, "model/two")

    rows = compare_runs([first, second])

    assert [row["system"] for row in rows] == [
        "model/one@aaaaaaaa",
        "model/two@aaaaaaaa",
    ]
    assert all(row["nll"] is None for row in rows)


def test_compare_rejects_different_samples(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _result(first, "model/one")
    _result(second, "model/two", prediction_id="p|different")

    with pytest.raises(ValueError, match="different sampled predictions"):
        compare_runs([first, second])
