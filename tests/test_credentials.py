import argparse
import stat

from rich.console import Console

import cli
import evaluation
from credentials import credentials_path, model_credentials, save_model_credentials


MODEL = "jaredpalmer/kev-0.8b"


def test_credentials_are_saved_per_model_with_private_permissions(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    path = save_model_credentials(MODEL, "https://systemone.example", "secret")

    assert path == credentials_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert model_credentials(MODEL) == {
        "endpoint": "https://systemone.example",
        "api_key": "secret",
    }
    assert model_credentials("jaredpalmer/kev-4b") == {}


def test_interactive_configuration_prompts_and_saves(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "console", Console(force_terminal=True))
    answers = iter(("https://systemone.example", "secret"))
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(answers))
    args = argparse.Namespace(model=MODEL, endpoint=None)

    cli._prepare_http_credentials(args)

    assert args.endpoint == "https://systemone.example"
    assert args.api_key == "secret"
    assert model_credentials(MODEL)["api_key"] == "secret"


def test_api_key_is_in_worker_environment_not_command(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    captured = {}

    class FakeAdapter:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]

    monkeypatch.setattr(evaluation, "CommandAdapter", FakeAdapter)
    args = argparse.Namespace(
        model=MODEL,
        adapter=None,
        command_worker=None,
        endpoint="https://systemone.example",
        api_key="secret",
    )

    evaluation._build_adapter(args, {})

    assert "secret" not in " ".join(captured["command"])
    assert captured["env"]["SRB_MODEL_API_KEY"] == "secret"
