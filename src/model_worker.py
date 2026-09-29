from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _labels(question: dict) -> tuple[list[str], dict[str, str]]:
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        return list(criteria), {str(key): str(value) for key, value in criteria.items()}
    if isinstance(criteria, list):
        return [str(value) for value in criteria], {str(value): str(value) for value in criteria}
    return ["no", "yes"], {"no": "No", "yes": "Yes"}


def _normalize_result(result: dict, questions: dict) -> dict:
    answers = result.get("answers", result)
    normalized = {}
    for qid, question in questions.items():
        answer = answers[qid]
        labels, _ = _labels(question)
        probabilities = answer.get("probabilities")
        choice = answer.get("choice", answer.get("decision"))
        if choice is None and probabilities:
            choice = max(probabilities, key=probabilities.get)
        if choice not in labels:
            raise ValueError(f"{qid}: worker selected unknown option {choice!r}")
        normalized[qid] = {"choice": choice}
        if probabilities is not None:
            missing = set(labels) - set(probabilities)
            if missing:
                raise ValueError(f"{qid}: probabilities missing options {sorted(missing)}")
            normalized[qid]["probabilities"] = {label: float(probabilities[label]) for label in labels}
    usage = result.get("usage") or {}
    return {"answers": normalized, "usage": usage}


class HttpBackend:
    probabilities = "native"
    max_questions = None
    max_options = None

    def __init__(self, model: str, options: dict, *, surogate: bool = False) -> None:
        endpoint = options.get("endpoint")
        if not endpoint:
            raise ValueError("HTTP adapters require endpoint")
        suffix = "/api/alpha/decisions" if surogate else "/v1/systemone"
        self.url = (
            endpoint.rstrip("/")
            if endpoint.rstrip("/").endswith(suffix)
            else endpoint.rstrip("/") + suffix
        )
        self.model = model
        self.options = options
        self.probabilities = options.get("probabilities", "native")
        self.max_questions = options.get("max_questions")
        self.max_options = options.get("max_options")
        self.api_key_env = options.get(
            "api_key_env",
            "SUROGATE_API_KEY" if surogate else "SYSTEMONE_API_KEY",
        )

    def predict(self, state: str, questions: dict) -> dict:
        wire_questions = {
            qid: (
                {
                    **question,
                    "criteria": {
                        str(label): str(label) for label in question["criteria"]
                    },
                }
                if isinstance(question.get("criteria"), list)
                else question
            )
            for qid, question in questions.items()
        }
        payload = {"model": self.model, "state": state, "questions": wire_questions}
        if "thinking" in self.options:
            payload["thinking"] = bool(self.options["thinking"])
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(self.url, json.dumps(payload).encode(), headers=headers, method="POST")
        attempts = int(self.options.get("attempts", 6))
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=float(self.options.get("timeout", 120))) as response:
                    result = json.load(response)
                break
            except HTTPError as error:
                detail = error.read().decode(errors="replace")
                if (error.code == 429 or error.code >= 500) and attempt < attempts - 1:
                    sleep(min(2**attempt, 30))
                    continue
                raise RuntimeError(f"HTTP {error.code} from {self.url}: {detail}") from error
            except (URLError, TimeoutError):
                if attempt == attempts - 1:
                    raise
                sleep(min(2**attempt, 30))
        usage = result.get("usage") or {}
        if "cost" in usage:
            usage.setdefault("api_cost_usd", usage["cost"])
        normalized = _normalize_result(result, questions)
        if result.get("model"):
            normalized["served_model"] = result["model"]
        return normalized


class LayaBackend:
    probabilities = "native"
    max_questions = None
    max_options = None

    def __init__(self, model: str, options: dict) -> None:
        import laya_mlx

        allowed = {key: options[key] for key in ("revision", "dtype", "batch_size", "device", "compile", "pad_to_multiple") if key in options}
        self.agent = laya_mlx.load(model, **allowed)

    def predict(self, state: str, questions: dict) -> dict:
        return _normalize_result(self.agent.predict(state, questions), questions)


class OpenJevBackend:
    probabilities = "native"
    max_questions = None
    max_options = None

    def __init__(self, model: str, options: dict) -> None:
        from huggingface_hub import hf_hub_download

        code = Path(hf_hub_download(model, "code/openjev_decide.py", revision=options.get("revision"))).parent
        hf_hub_download(model, "code/modeling_openjev.py", revision=options.get("revision"))
        sys.path.insert(0, str(code))
        OpenJev = importlib.import_module("openjev_decide").OpenJev
        self.model = OpenJev.from_pretrained(
            model,
            subfolder=options.get("subfolder", "qwen3.5-4b-nli-v5"),
            device=options.get("device"),
        )

    def predict(self, state: str, questions: dict) -> dict:
        requests = []
        for question in questions.values():
            labels, descriptions = _labels(question)
            instruction = question["instructions"]
            if descriptions != {label: label for label in labels}:
                instruction += "\nAllowed answers and rubric: " + json.dumps(descriptions, ensure_ascii=False)
            requests.append({"type": question["type"], "instructions": instruction, "options": labels})
        raw = self.model.decide(state, requests)
        answers = {}
        for (qid, question), answer in zip(questions.items(), raw, strict=True):
            labels, _ = _labels(question)
            probabilities = answer.get("probabilities")
            if probabilities is None:
                yes = float(answer["noul"])
                probabilities = {labels[0]: 1 - yes, labels[-1]: yes}
            answers[qid] = {"probabilities": probabilities}
        return _normalize_result({"answers": answers, "usage": {"forward_count": sum(len(_labels(q)[0]) for q in questions.values())}}, questions)


class InternDecisionBackend:
    probabilities = "native"
    max_questions = 16
    max_options = 62

    def __init__(self, model: str, options: dict) -> None:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model, revision=options.get("revision")))
        sys.path.insert(0, str(path))
        DecisionEngine = importlib.import_module("inference").DecisionEngine
        kwargs = {"checkpoint": str(path)}
        for key in ("device", "temperature", "max_length"):
            if key in options:
                kwargs[key] = options[key]
        self.engine = DecisionEngine(**kwargs)

    def predict(self, state: str, questions: dict) -> dict:
        return _normalize_result(self.engine.predict({"state": state, "questions": questions}), questions)


class TransformersPredictBackend:
    probabilities = "native"
    max_questions = None
    max_options = 255

    def __init__(self, model: str, options: dict) -> None:
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            revision=options.get("revision"),
            trust_remote_code=True,
            dtype=options.get("dtype", "auto"),
            device_map=options.get("device_map", "auto"),
        )

    def predict(self, state: str, questions: dict) -> dict:
        answers = {}
        for qid, question in questions.items():
            labels, descriptions = _labels(question)
            candidates = [
                {"id": label, "label": label, "description": descriptions[label]}
                for label in labels
            ]
            raw = self.model.predict(
                state={"text": state},
                instructions=question["instructions"],
                decision_type=question["type"],
                row_id=qid,
                candidates=candidates,
            )
            answers[qid] = {
                "probabilities": dict(zip(labels, raw["probabilities"], strict=True))
            }
        return _normalize_result({"answers": answers, "usage": {"forward_count": len(questions)}}, questions)


class GLiNERBackend:
    probabilities = "none"
    max_questions = None
    max_options = None

    def __init__(self, model: str, options: dict) -> None:
        from gliner2 import AutoExtractor

        self.model = AutoExtractor.from_pretrained(
            model,
            revision=options.get("revision"),
        )

    def predict(self, state: str, questions: dict) -> dict:
        schema = {}
        reverse = {}
        for qid, question in questions.items():
            labels, descriptions = _labels(question)
            rendered = [label if descriptions[label] == label else f"{label}: {descriptions[label]}" for label in labels]
            schema[qid] = rendered
            reverse[qid] = dict(zip(rendered, labels, strict=True))
        selected = self.model.classify_text(state, schema)
        answers = {qid: {"choice": reverse[qid][selected[qid]]} for qid in questions}
        return {"answers": answers, "usage": {"forward_count": 1}}


BACKENDS = {
    "laya": LayaBackend,
    "openjev": OpenJevBackend,
    "intern-decision": InternDecisionBackend,
    "transformers-predict": TransformersPredictBackend,
    "gliner2": GLiNERBackend,
}


def build_backend(adapter: str, model: str, options: dict):
    if adapter == "systemone-http":
        backend = HttpBackend(model, options)
    elif adapter == "surogate-http":
        backend = HttpBackend(model, options, surogate=True)
    else:
        try:
            backend = BACKENDS[adapter](model, options)
        except KeyError as error:
            raise ValueError(f"Unknown worker adapter: {adapter}") from error
    backend.runtime_model = model
    backend.reference_revision = options.get("revision")
    if isinstance(backend, HttpBackend):
        backend.resolved_revision = None
        backend.revision_status = "server-unavailable"
    else:
        backend.resolved_revision = options.get("revision")
        backend.revision_status = (
            "pinned" if backend.resolved_revision else "unresolved"
        )
    return backend


def serve(backend) -> None:
    capabilities = {
        "type": "capabilities",
        "protocol": 1,
        "probabilities": backend.probabilities,
        "max_questions": backend.max_questions,
        "max_options": backend.max_options,
        "modalities": ["text"],
        "batching": True,
        "runtime_model": backend.runtime_model,
        "resolved_revision": backend.resolved_revision,
        "revision_status": backend.revision_status,
        "reference_revision": backend.reference_revision,
    }
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("type") == "hello":
                response = capabilities
            elif message.get("type") == "close":
                print(json.dumps({"type": "closed"}), flush=True)
                return
            elif message.get("type") == "predict":
                started = perf_counter()
                with redirect_stdout(sys.stderr):
                    result = backend.predict(message["state"], message["questions"])
                response = {
                    "type": "result",
                    "id": message["id"],
                    "result": result,
                    "timing_ms": (perf_counter() - started) * 1000,
                }
            else:
                raise ValueError("Unknown protocol message")
        except Exception as error:
            response = {"type": "error", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Social Routing Bench model worker")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            backend = build_backend(args.adapter, args.model, json.loads(args.options))
    except Exception as error:
        sys.stdin.readline()
        extra = {
            "gliner2": "gliner2",
            "intern-decision": "intern-decision",
            "laya": "laya",
            "openjev": "openjev",
            "transformers-predict": "transformers-predict",
        }.get(args.adapter)
        hint = (
            f" Install with: python -m pip install -e '.[{extra}]'."
            if extra and isinstance(error, ModuleNotFoundError)
            else ""
        )
        print(
            json.dumps(
                {
                    "type": "error",
                    "error": f"Could not start {args.adapter}: {error}.{hint}",
                }
            ),
            flush=True,
        )
        return
    serve(backend)


if __name__ == "__main__":
    main()
