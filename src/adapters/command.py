from __future__ import annotations

import atexit
from collections.abc import Mapping
import json
import shlex
import subprocess
from time import perf_counter
from uuid import uuid4


class CommandAdapter:
    """One persistent JSONL worker process per evaluation run."""

    def __init__(
        self,
        command: str | list[str],
        *,
        model: str,
        revision: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        argv = shlex.split(command) if isinstance(command, str) else command
        if not argv:
            raise ValueError("Worker command cannot be empty")
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self.model = model
        self.revision = revision
        atexit.register(self.close)
        hello = self._exchange({"type": "hello", "protocol": 1})
        if hello.get("type") != "capabilities" or hello.get("protocol") != 1:
            self.close()
            raise RuntimeError(f"Worker returned an invalid handshake: {hello}")
        probability_source = hello.get("probabilities")
        if probability_source not in {"native", "self_reported", "none"}:
            self.close()
            raise RuntimeError("Worker must declare probabilities as native, self_reported, or none")
        self.capabilities = hello

    @property
    def metadata(self) -> dict:
        return {
            "worker_command": self.process.args,
            "worker_protocol": 1,
            "capabilities": self.capabilities,
        }

    def _exchange(self, message: dict) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError(f"Worker exited with status {self.process.returncode}")
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Worker closed stdout (status {self.process.poll()})")
        response = json.loads(line)
        if response.get("type") == "error":
            raise RuntimeError(response.get("error", "Worker prediction failed"))
        return response

    def _validate(self, questions: dict) -> None:
        maximum = self.capabilities.get("max_questions")
        if maximum is not None and len(questions) > maximum:
            raise ValueError(f"Worker supports at most {maximum} questions, got {len(questions)}")
        maximum = self.capabilities.get("max_options")
        if maximum is not None:
            for qid, question in questions.items():
                count = len(question.get("criteria") or ("no", "yes"))
                if count > maximum:
                    raise ValueError(f"Worker supports at most {maximum} options; {qid} has {count}")

    def predict(self, state: str, questions: dict) -> tuple[dict, float]:
        self._validate(questions)
        request_id = uuid4().hex
        started = perf_counter()
        response = self._exchange(
            {"type": "predict", "id": request_id, "state": state, "questions": questions}
        )
        elapsed_ms = (perf_counter() - started) * 1000
        if response.get("type") != "result" or response.get("id") != request_id:
            raise RuntimeError(f"Worker returned an invalid result envelope: {response}")
        result = response.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise RuntimeError("Worker result must contain an answers object")
        result.setdefault("usage", {})
        result["usage"].setdefault("input_tokens", 0)
        result["usage"].setdefault("output_tokens", 0)
        result["probability_source"] = self.capabilities["probabilities"]
        return result, float(response.get("timing_ms", elapsed_ms))

    def warmup(self) -> None:
        self.predict(
            "A short benchmark warmup post.",
            {"warmup": {"type": "choice", "instructions": "Select one.", "criteria": ["r/example", "r/other"]}},
        )

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            self._exchange({"type": "close"})
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError):
            process.terminate()
        finally:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
