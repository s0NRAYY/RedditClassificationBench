from __future__ import annotations

from time import perf_counter


class LayaAdapter:
    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        dtype: str = "float16",
        batch_size: int = 32,
        device: str = "gpu",
        compile: bool = False,
        pad_to_multiple: int | None = None,
    ) -> None:
        try:
            import laya_mlx
        except ImportError as error:
            raise RuntimeError("Install the Laya runtime with: pip install -e '.[laya]'") from error

        self.agent = laya_mlx.load(
            model,
            revision=revision,
            dtype=dtype,
            batch_size=batch_size,
            device=device,
            compile=compile,
            pad_to_multiple=pad_to_multiple,
        )
        self.model = model
        self.revision = revision
        self.dtype = dtype

    @property
    def metadata(self) -> dict:
        return {
            "max_len": self.agent.cfg["max_len"],
            "head_max_len": self.agent.cfg["head_max_len"],
            "temperature_raw": self.agent.temperature_raw,
            "temperature_effective": self.agent.temperature,
            "temperature_by_options_raw": self.agent.temperature_by_options_raw,
            "temperature_by_options_effective": self.agent.temperature_by_options,
        }

    def predict(self, state: str, questions: dict) -> tuple[dict, float]:
        started = perf_counter()
        result = self.agent.predict(state, questions)
        return result, (perf_counter() - started) * 1000

    def warmup(self) -> None:
        self.predict(
            "A short benchmark warmup post.",
            {
                "warmup": {
                    "type": "choice",
                    "instructions": "Select the best community.",
                    "criteria": ["r/example", "r/other"],
                }
            },
        )
