from __future__ import annotations

from importlib.util import find_spec

SOURCE = "https://huggingface.co/spaces/multimodalart/jev-decision-index"

# Only entries whose inference contract is known are automatic. Unknown Decision
# Index models remain usable through the command adapter.
MODEL_REGISTRY: dict[str, dict] = {
    "convaiinnovations/laya": {
        "adapter": "laya",
        "runtime_model": "aac6fef/laya-multilingual-mlx",
        "revision": "f2b4faf51023039425946074e2cf1361d2db11d5",
        "extra": "laya",
        "probabilities": "native",
        "max_questions": None,
        "max_options": None,
    },
    "AlexWortega/openjev": {
        "adapter": "openjev",
        "revision": "a298f274886c4676c42f1a4262401b6aa9653e6d",
        "extra": "openjev",
        "probabilities": "native",
        "max_questions": None,
        "max_options": None,
        "options": {"subfolder": "qwen3.5-4b-nli-v5"},
    },
    "fastino/GLiNER2.5-Decide": {
        "adapter": "gliner2",
        "revision": "5a7adf72a23b4d311abae6ce050d7f0012bb3416",
        "extra": "gliner2",
        "probabilities": "none",
        "max_questions": None,
        "max_options": None,
    },
    "fastino/GLiNER2.5-multi-Decide": {
        "adapter": "gliner2",
        "revision": "a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f",
        "extra": "gliner2",
        "probabilities": "none",
        "max_questions": None,
        "max_options": None,
    },
    **{
        model: {
            "adapter": "intern-decision",
            "revision": revision,
            "extra": "intern-decision",
            "probabilities": "native",
            "max_questions": 16,
            "max_options": 62,
        }
        for model, revision in {
            "internlm/Intern-Decision-0.8B": "85a0cc5a99d67ea8d56dfe98115689212867171d",
            "internlm/Intern-Decision-2B": "8797836c65fc91a2435b1fb6850b5f0aabd75cc3",
            "internlm/Intern-Decision-4B": "0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd",
        }.items()
    },
    **{
        model: {
            "adapter": "transformers-predict",
            "revision": revision,
            "extra": "transformers-predict",
            "probabilities": "native",
            "max_questions": None,
            "max_options": 255,
        }
        for model, revision in {
            "Hanno-Labs/bosun-v3.1-0.6b": "1d8b6f9611f9b64b514ce8b57cd86398fbc31a3b",
            "Hanno-Labs/bosun-v3.1-1.7b": "1d8dc82a20e4a32ed60927a47272d6efff48eed2",
        }.items()
    },
    **{
        model: {
            "adapter": "systemone-http",
            "revision": revision,
            "extra": None,
            "probabilities": "native",
            "max_questions": None,
            "max_options": None,
            "requires_endpoint": True,
        }
        for model, revision in {
            "jaredpalmer/kev-0.8b": "9a45d25eb2ab761841196625383fa1dff0e56c1e",
            "jaredpalmer/kev-4b": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
            "jaredpalmer/kev-9b": "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c",
        }.items()
    },
    "surogate/rune-26b-a4b-GGUF": {
        "adapter": "surogate-http",
        "revision": "c6b360d47895bb77bdf3805a13ee5a5557ab1921",
        "extra": None,
        "probabilities": "native",
        "max_questions": None,
        "max_options": None,
        "requires_endpoint": True,
    },
    **{
        model: {
            "adapter": "systemone-http",
            "revision": None,
            "extra": None,
            "probabilities": "native",
            "max_questions": None,
            "max_options": None,
            "requires_endpoint": True,
            "default_endpoint": "https://openrouter.ai/api",
        }
        for model in ("typesafe/jev-1.13", "~typesafe/jev-latest")
    },
}

EXTRA_IMPORTS = {
    "laya": "laya_mlx",
    "openjev": "huggingface_hub",
    "intern-decision": "huggingface_hub",
    "transformers-predict": "transformers",
    "gliner2": "gliner2",
}


def canonical_model(value: str) -> str:
    return value.removeprefix("hf://")


def resolve_model(value: str) -> tuple[str, dict]:
    model = canonical_model(value)
    return model, dict(MODEL_REGISTRY.get(model, {}))


def doctor(value: str, endpoint: str | None = None) -> list[tuple[str, bool, str]]:
    model, entry = resolve_model(value)
    checks = [("registry", bool(entry), entry.get("adapter", "unknown model; use --command"))]
    if not entry:
        return checks
    module = EXTRA_IMPORTS.get(entry["adapter"])
    if module:
        checks.append(("runtime", find_spec(module) is not None, module))
    if entry.get("requires_endpoint"):
        endpoint = endpoint or entry.get("default_endpoint")
        checks.append(("endpoint", bool(endpoint), endpoint or "pass --endpoint"))
    checks.append(("model", True, model))
    return checks
