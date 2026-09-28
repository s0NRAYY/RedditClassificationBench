import io
import json
import sys
from pathlib import Path

import pytest

from adapters import CommandAdapter
from evaluation import _prediction_rows
from metrics import aggregate
from model_registry import MODEL_REGISTRY, doctor
import model_worker
from model_worker import HttpBackend, TransformersPredictBackend


def test_command_protocol_and_probability_free_metrics():
    worker = Path(__file__).with_name("support_worker.py")
    adapter = CommandAdapter([sys.executable, str(worker)], model="fake")
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": ["a", "b"],
        }
    }
    result, latency = adapter.predict("post", questions)
    adapter.close()

    rows = _prediction_rows(
        {"post_id": "p", "subreddit": "a"},
        "seen",
        "title",
        result,
        {
            "route": {
                "candidates": ["a", "b"],
                "labels": ["a", "b"],
                "representation": "name-only",
                "difficulty": "random",
                "k": 2,
            }
        },
        latency,
    )
    metrics = aggregate(rows, bootstrap_samples=4)

    assert rows[0]["probability_source"] == "none"
    assert rows[0]["true_probability"] is None
    assert metrics[0]["nll"] is None
    assert metrics[0]["ece_15"] is None
    assert metrics[0]["mean_forward_count"] == 1


def test_command_protocol_enforces_capabilities():
    worker = Path(__file__).with_name("support_worker.py")
    adapter = CommandAdapter([sys.executable, str(worker)], model="fake")
    with pytest.raises(ValueError, match="at most 2 options"):
        adapter.predict(
            "post",
            {
                "route": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": ["a", "b", "c"],
                }
            },
        )
    adapter.close()


def test_registry_declares_runtime_and_probability_contracts():
    assert MODEL_REGISTRY
    for model, entry in MODEL_REGISTRY.items():
        assert entry["adapter"]
        assert entry["probabilities"] in {"native", "self_reported", "none"}
        assert len(entry["revision"]) == 40
        int(entry["revision"], 16)
        checks = doctor(model, "http://localhost:8000")
        assert checks[0] == ("registry", True, entry["adapter"])


def test_transformers_predict_maps_candidate_aligned_probabilities():
    class Model:
        def predict(self, **kwargs):
            return {"probabilities": [0.25, 0.75]}

    backend = TransformersPredictBackend.__new__(TransformersPredictBackend)
    backend.model = Model()

    result = backend.predict(
        "post",
        {
            "route": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": ["a", "b"],
            }
        },
    )

    assert result["answers"]["route"] == {
        "choice": "b",
        "probabilities": {"a": 0.25, "b": 0.75},
    }



def test_http_backend_sends_systemone_choice_criteria_as_object(monkeypatch):
    captured = {}

    def urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                {
                    "answers": {
                        "route": {
                            "choice": "b",
                            "probabilities": {"a": 0.25, "b": 0.75},
                        }
                    }
                }
            ).encode()
        )

    monkeypatch.setattr(model_worker, "urlopen", urlopen)
    backend = HttpBackend("kev", {"endpoint": "http://localhost"})
    result = backend.predict(
        "post",
        {
            "route": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": ["a", "b"],
            }
        },
    )

    assert captured["questions"]["route"]["criteria"] == {"a": "a", "b": "b"}
    assert result["answers"]["route"]["choice"] == "b"