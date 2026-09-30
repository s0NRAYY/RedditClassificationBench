import io
import json
import sys
from pathlib import Path
from http.client import RemoteDisconnected
from urllib.error import HTTPError

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
        if entry["revision"] is None:
            assert entry.get("requires_endpoint"), f"{model}: only hosted APIs may omit a revision"
        else:
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



def test_http_backend_retries_and_reports_cost_and_served_model(monkeypatch):
    captured = []

    def urlopen(request, timeout):
        captured.append(json.loads(request.data))
        if len(captured) == 1:
            raise HTTPError(request.full_url, 503, "busy", {}, io.BytesIO(b"busy"))
        return io.BytesIO(
            json.dumps(
                {
                    "answers": {
                        "route": {
                            "choice": "b",
                            "probabilities": {"a": 0.25, "b": 0.75},
                        }
                    },
                    "model": "typesafe/jev-1.13-20260917",
                    "usage": {"cost": 0.00002, "input_tokens": 40},
                }
            ).encode()
        )

    monkeypatch.setattr(model_worker, "urlopen", urlopen)
    monkeypatch.setattr(model_worker, "sleep", lambda seconds: None)
    backend = HttpBackend("typesafe/jev-1.13", {"endpoint": "https://openrouter.ai/api"})
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

    assert backend.url == "https://openrouter.ai/api/v1/systemone"
    assert len(captured) == 2
    assert captured[1]["questions"]["route"]["criteria"] == {"a": "a", "b": "b"}
    assert result["answers"]["route"]["choice"] == "b"
    assert result["usage"]["api_cost_usd"] == 0.00002
    assert result["served_model"] == "typesafe/jev-1.13-20260917"


def test_http_backend_does_not_retry_client_errors(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        raise HTTPError(request.full_url, 402, "credits", {}, io.BytesIO(b"no credits"))

    monkeypatch.setattr(model_worker, "urlopen", urlopen)
    backend = HttpBackend("typesafe/jev-1.13", {"endpoint": "https://openrouter.ai/api"})
    with pytest.raises(RuntimeError, match="HTTP 402"):
        backend.predict("post", {"route": {"type": "choice", "instructions": "Choose.", "criteria": ["a", "b"]}})
    assert len(calls) == 1


def test_http_backend_retries_dropped_connections(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise RemoteDisconnected("Remote end closed connection without response")
        return io.BytesIO(json.dumps({"answers": {"route": {"choice": "a"}}}).encode())

    monkeypatch.setattr(model_worker, "urlopen", urlopen)
    monkeypatch.setattr(model_worker, "sleep", lambda seconds: None)
    backend = HttpBackend("typesafe/jev-1.13", {"endpoint": "https://openrouter.ai/api"})
    result = backend.predict("post", {"route": {"type": "choice", "instructions": "Choose.", "criteria": ["a", "b"]}})
    assert result["answers"]["route"]["choice"] == "a"
    assert len(calls) == 2