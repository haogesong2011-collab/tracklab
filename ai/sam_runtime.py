"""Process-wide SAM 2 predictor: one spec on GPU at a time."""

from __future__ import annotations

import os
import sys

from ai.contracts import FAST_TRACK_STRIDE, TrackMode
from ai.model_manager import ModelSpec, load_sam2_predictor, spec_for_mode


def inference_thread_budget(cores: int | None = None) -> int:
    """Leave two cores for the UI and the rest of the system."""
    count = os.cpu_count() if cores is None else cores
    return max(1, int(count or 4) - 2)


def reserve_system_headroom() -> None:
    """Leave CPU for the UI.

    PyTorch's default MPS low watermark is 1.4. A high watermark below that
    aborts with "invalid low watermark ratio 1.4", so a too-low cap is removed.
    """
    high = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if high is not None:
        try:
            too_low = float(high) <= 1.4
        except ValueError:
            too_low = True
        if too_low:
            os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    # Torch 2.2 on Intel Macs has no MPS bicubic upsample. Missing ops then
    # run on CPU instead of aborting the track. Native ops stay on MPS.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    budget = str(inference_thread_budget())
    os.environ.setdefault("OMP_NUM_THREADS", budget)
    os.environ.setdefault("MKL_NUM_THREADS", budget)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", budget)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", budget)


class _BFloat16AsFloat32:
    """SAM 2 stores mask memory as bfloat16. Intel MPS cannot."""

    def __init__(self, torch_mod: object) -> None:
        self._torch = torch_mod

    def __getattr__(self, name: str):
        if name == "bfloat16":
            return getattr(self._torch, "float32")
        return getattr(self._torch, name)


def mps_supports_bfloat16(torch_mod: object) -> bool:
    """False only when MPS is present and rejects bfloat16."""
    backends = getattr(torch_mod, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not mps.is_available():
        return True
    try:
        getattr(torch_mod, "zeros")(
            1, device="mps", dtype=getattr(torch_mod, "bfloat16")
        )
    except Exception:  # noqa: BLE001
        return False
    return True


def install_float32_mask_memory(modules: list[object], torch_mod: object) -> None:
    proxy = _BFloat16AsFloat32(torch_mod)
    for module in modules:
        if not isinstance(getattr(module, "torch", None), _BFloat16AsFloat32):
            module.torch = proxy


def use_float32_mask_memory() -> None:
    """Keep SAM 2 mask memory in float32 when MPS has no bfloat16."""
    try:
        import torch
    except ImportError:
        return
    if mps_supports_bfloat16(torch):
        return
    import sam2.sam2_video_predictor as predictor

    modules: list[object] = [predictor]
    legacy = sys.modules.get("sam2.sam2_video_predictor_legacy")
    if legacy is not None:
        modules.append(legacy)
    install_float32_mask_memory(modules, torch)


def mps_supports_complex(torch_mod: object) -> bool:
    """False only when MPS is present and rejects complex tensors."""
    backends = getattr(torch_mod, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not mps.is_available():
        return True
    try:
        ones = torch_mod.ones(1)
        torch_mod.polar(ones, torch_mod.zeros(1)).to("mps")
    except Exception:  # noqa: BLE001
        return False
    return True


def compute_axial_cis_real(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """Same axial frequencies as SAM 2, stored as interleaved cos/sin."""
    import torch

    step = torch.arange(0, dim, 4)[: (dim // 4)].float() / dim
    freqs = 1.0 / (theta ** step)
    length = end_x * end_y
    t = torch.arange(length, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    ang_x = torch.outer(t_x, freqs)
    ang_y = torch.outer(t_y, freqs)
    x_pairs = torch.stack((torch.cos(ang_x), torch.sin(ang_x)), dim=-1)
    y_pairs = torch.stack((torch.cos(ang_y), torch.sin(ang_y)), dim=-1)
    return torch.cat((x_pairs, y_pairs), dim=1).flatten(-2)


def _rotate_real(x, freqs):
    import torch

    x32 = x.float()
    freqs32 = freqs.float()
    x1 = x32[..., 0::2]
    x2 = x32[..., 1::2]
    cos = freqs32[..., 0::2]
    sin = freqs32[..., 1::2]
    while cos.ndim < x1.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x).to(x.device)


def apply_rotary_enc_real(xq, xk, freqs_cis, repeat_freqs_k: bool = False):
    """Rotary encoding without complex tensors. Matches SAM 2 on CPU."""
    xq_out = _rotate_real(xq, freqs_cis)
    if xk.shape[-2] == 0:
        return xq_out, xk
    freqs_k = freqs_cis
    if repeat_freqs_k:
        repeats = xk.shape[-2] // xq.shape[-2]
        if repeats != 1:
            freqs_k = freqs_cis.repeat(repeats, 1)
    return xq_out, _rotate_real(xk, freqs_k)


def install_real_rope(modules: list[object]) -> None:
    for module in modules:
        if getattr(module, "compute_axial_cis", None) is compute_axial_cis_real:
            continue
        if hasattr(module, "compute_axial_cis"):
            module.compute_axial_cis = compute_axial_cis_real
        if hasattr(module, "apply_rotary_enc"):
            module.apply_rotary_enc = apply_rotary_enc_real


def mps_needs_real_rope(torch_mod: object, machine: str | None = None) -> bool:
    """Intel MPS has no complex support. Apple Silicon keeps SAM 2's complex path."""
    import platform

    backends = getattr(torch_mod, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not mps.is_available():
        return False
    kind = platform.machine() if machine is None else machine
    if kind.lower() in {"x86_64", "amd64", "i386"}:
        return True
    return not mps_supports_complex(torch_mod)


def use_real_rope_on_mps() -> None:
    """Intel MPS cannot hold ComplexFloat. Keep the original path when it can."""
    try:
        import torch
    except ImportError:
        return
    if not mps_needs_real_rope(torch):
        return
    import sam2.modeling.position_encoding as position_encoding
    import sam2.modeling.sam.transformer as transformer

    install_real_rope([position_encoding, transformer])


def prepare_inference_thread() -> None:
    """Run on the SAM worker. Lowers its priority and caps torch's thread pool."""
    _set_utility_qos()
    reserve_system_headroom()
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(inference_thread_budget())
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    use_float32_mask_memory()
    use_real_rope_on_mps()


def _set_utility_qos() -> None:
    if sys.platform != "darwin":
        return
    import ctypes

    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        fn = lib.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        fn(0x11, 0)  # QOS_CLASS_UTILITY
    except Exception:  # noqa: BLE001
        return


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
