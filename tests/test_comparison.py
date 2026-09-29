import json
from pathlib import Path

import pytest

from comparison import compare_runs


def _result(path: Path, model: str, prediction_id: str = "p|q", counts=(2,), skipped=()) -> None:
    path.mkdir()
    run = {
        "model": model,
        "revision": "a" * 40,
        "dataset_build": {"fingerprint": "dataset"},
        "text_modes": ["title"],
        "representations": ["name-only"],
        "difficulties": ["random"],
        "candidate_counts": list(counts),
        "selected_posts": 1,
    }
    if skipped:
        run["skipped_candidate_counts"] = list(skipped)
    metrics = [
        {
            "text_mode": "title",
            "representation": "name-only",
            "community_track": "seen",
            "difficulty": "random",
            "k": k,
            "accuracy": 1.0,
            "nll": None,
            "ece_15": None,
            "latency_ms_p50": 1.0,
            "mean_forward_count": 1.0,
        }
        for k in counts
    ]
    (path / "run.json").write_text(json.dumps(run))
    (path / "metrics.json").write_text(json.dumps(metrics))
    (path / "predictions.jsonl").write_text(
        "".join(
            json.dumps({"prediction_id": f"{prediction_id}|{k}", "post_id": "p", "k": k}) + "\n"
            for k in counts
        )
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


def test_compare_accepts_counts_skipped_by_worker_limits(tmp_path):
    full, limited = tmp_path / "full", tmp_path / "limited"
    _result(full, "model/one", counts=(2, 4))
    _result(limited, "model/two", counts=(2,), skipped=(4,))

    rows = compare_runs([full, limited])

    assert sorted((row["system"], row["k"]) for row in rows) == [
        ("model/one@aaaaaaaa", 2),
        ("model/one@aaaaaaaa", 4),
        ("model/two@aaaaaaaa", 2),
    ]


def test_compare_rejects_different_requested_counts(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _result(first, "model/one", counts=(2, 4))
    _result(second, "model/two", counts=(2,))

    with pytest.raises(ValueError, match="candidate_counts"):
        compare_runs([first, second])
