from __future__ import annotations

import json
import os
from pathlib import Path

from model_registry import canonical_model


def credentials_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "social-routing-bench" / "credentials.json"


def load_credentials() -> dict[str, dict[str, str]]:
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read credential store {path}: {error}"
        ) from error
    if not isinstance(data, dict) or not all(
        isinstance(model, str) and isinstance(values, dict)
        for model, values in data.items()
    ):
        raise RuntimeError(f"Invalid credential store: {path}")
    return data


def model_credentials(model: str) -> dict[str, str]:
    return dict(load_credentials().get(canonical_model(model), {}))


def save_model_credentials(model: str, endpoint: str, api_key: str) -> Path:
    path = credentials_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    data = load_credentials()
    data[canonical_model(model)] = {"endpoint": endpoint, "api_key": api_key}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    return path
