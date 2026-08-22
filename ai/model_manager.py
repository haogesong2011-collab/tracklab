"""Checkpoint download, checksum, and device selection for SAM 2.1 Tiny."""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path


class ModelNotAvailable(RuntimeError):
    """Raised when SAM 2 cannot be loaded. Never fall back to color-blob tracking."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    filename: str
    url: str
    sha256: str
    config: str
    license: str
    version: str
    hf_id: str


SAM21_TINY = ModelSpec(
    model_id="sam2.1_hiera_tiny",
    filename="sam2.1_hiera_tiny.pt",
    url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    sha256="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    config="configs/sam2.1/sam2.1_hiera_t.yaml",
    license="Apache-2.0",
    version="2.1.0",
    hf_id="facebook/sam2.1-hiera-tiny",
)

DEFAULT_SPEC = SAM21_TINY


def source_urls(spec: ModelSpec = DEFAULT_SPEC) -> list[str]:
    """Official Meta CDN first, Hugging Face as a certificate-friendly mirror."""
    urls = [spec.url]
    hf = f"https://huggingface.co/{spec.hf_id}/resolve/main/{spec.filename}"
    if hf not in urls:
        urls.append(hf)
    return urls


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi: python.org / venv builds on macOS often lack a CA bundle."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download_file(url: str, dest: Path) -> None:
    errors: list[str] = []
    request = urllib.request.Request(url, headers={"User-Agent": "TrackLab/0.1"})
    try:
        with urllib.request.urlopen(request, context=_ssl_context(), timeout=120) as resp:
            with dest.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        return
    except Exception as exc:  # noqa: BLE001
        errors.append(f"urllib ({url}): {exc}")
        dest.unlink(missing_ok=True)

    curl = shutil.which("curl")
    if curl:
        completed = subprocess.run(
            [curl, "-L", "--fail", "--retry", "3", "-o", str(dest), url],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
            return
        dest.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or str(completed.returncode)).strip()
        errors.append(f"curl ({url}): {detail}")
    raise OSError("；".join(errors))


def cache_dir() -> Path:
    override = os.environ.get("TRACKLAB_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "tracklab" / "models"


def checkpoint_path(spec: ModelSpec = DEFAULT_SPEC) -> Path:
    return cache_dir() / spec.filename


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, spec: ModelSpec = DEFAULT_SPEC) -> None:
    if not path.is_file():
        raise ModelNotAvailable(f"找不到权重文件：{path}")
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ModelNotAvailable(
            f"权重校验失败：{path.name} SHA-256={actual}，期望 {spec.sha256}"
        )


def ensure_checkpoint(
    spec: ModelSpec = DEFAULT_SPEC, *, download: bool = False
) -> Path:
    """Return a verified checkpoint path. Never downloads unless download=True."""
    path = checkpoint_path(spec)
    if path.is_file():
        verify_checkpoint(path, spec)
        return path
    if not download:
        raise ModelNotAvailable(
            "未找到 SAM 2.1 Tiny 权重。"
            f"请运行 TrackLab 并确认下载，或手动保存到 {path}。"
            f"来源：{spec.url}（{spec.license}）"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    errors: list[str] = []
    for url in source_urls(spec):
        try:
            _download_file(url, tmp)
            verify_checkpoint(tmp, spec)
            tmp.replace(path)
            return path
        except ModelNotAvailable as exc:
            tmp.unlink(missing_ok=True)
            errors.append(str(exc))
            continue
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            errors.append(str(exc))
    raise ModelNotAvailable(
        "下载 SAM 2.1 Tiny 失败："
        + "；".join(errors)
        + f"。也可手动下载到 {path}（{spec.license}）。"
    )


def select_device() -> str:
    try:
        import torch
    except ImportError as exc:
        raise ModelNotAvailable(
            "未安装 PyTorch。请执行：pip install -r requirements-ai.txt"
        ) from exc
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_sam2_predictor(spec: ModelSpec = DEFAULT_SPEC, *, download: bool = False):
    """Build a SAM 2 video predictor. Imports torch/sam2 lazily."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise ModelNotAvailable(
            "未安装 SAM 2。请执行：pip install -r requirements-ai.txt"
        ) from exc
    ckpt = ensure_checkpoint(spec, download=download)
    device = select_device()
    try:
        # vos_optimized uses torch.compile; keep eager on all devices, especially MPS.
        predictor = build_sam2_video_predictor(
            spec.config,
            str(ckpt),
            device=device,
            vos_optimized=False,
        )
    except TypeError:
        predictor = build_sam2_video_predictor(spec.config, str(ckpt), device=device)
    predictor._tracklab_device = device  # noqa: SLF001
    predictor._tracklab_spec = spec  # noqa: SLF001
    return predictor
