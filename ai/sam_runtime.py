"""Process-wide SAM 2 predictor: one spec on GPU at a time."""

from __future__ import annotations

from ai.contracts import FAST_TRACK_STRIDE, TrackMode
from ai.model_manager import ModelSpec, load_sam2_predictor, spec_for_mode


class SamRuntime:
    """Lazy-load Tiny or Small. Call predictor_for from the SAM worker thread."""

    _instance: "SamRuntime | None" = None

    def __init__(self) -> None:
        self._predictor = None
        self._spec: ModelSpec | None = None

    @classmethod
    def instance(cls) -> "SamRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        if cls._instance is not None:
            cls._instance.release()
        cls._instance = None

    def predictor_for(self, spec: ModelSpec):
        if self._predictor is not None and self._spec is spec:
            return self._predictor
        if self._predictor is not None and self._spec is not None:
            if self._spec.model_id == spec.model_id:
                return self._predictor
        self.release()
        self._predictor = load_sam2_predictor(spec, download=False)
        self._spec = spec
        return self._predictor

    def release(self) -> None:
        predictor = self._predictor
        self._predictor = None
        self._spec = None
        del predictor
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def settings_for_mode(mode: TrackMode) -> tuple[ModelSpec, int, int | None]:
    spec = spec_for_mode(mode)
    if mode is TrackMode.FAST:
        return spec, FAST_TRACK_STRIDE, None
    return spec, 1, None
