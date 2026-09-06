"""Checkpoint download progress and cancel — no network, no real weights."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import CancelToken  # noqa: E402
from ai.model_manager import (  # noqa: E402
    DownloadCancelled,
    ModelSpec,
    ensure_checkpoint,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args) -> bool:
        return False


def _spec_for(payload: bytes, filename: str = "fake.pt") -> ModelSpec:
    return ModelSpec(
        model_id="sam2.1_hiera_tiny",
        filename=filename,
        url="https://example.test/fake.pt",
        sha256=hashlib.sha256(payload).hexdigest(),
        config="configs/sam2.1/sam2.1_hiera_t.yaml",
        license="Apache-2.0",
        version="2.1.0",
        hf_id="facebook/sam2.1-hiera-tiny",
    )


class ModelDownloadTests(unittest.TestCase):
    def test_progress_callback_and_checksum(self) -> None:
        payload = b"ckpt" * (1 << 18)  # 1 MiB
        spec = _spec_for(payload)
        calls: list[tuple[int, int]] = []
        stages: list[str] = []

        def fake_urlopen(_request, context=None, timeout=None):  # noqa: ANN001
            return _FakeResponse(payload)

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRACKLAB_MODEL_DIR"] = tmp
            try:
                with patch("ai.model_manager.urllib.request.urlopen", fake_urlopen):
                    path = ensure_checkpoint(
                        spec,
                        download=True,
                        progress=lambda rec, tot: calls.append((rec, tot)),
                        stage=stages.append,
                    )
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_bytes(), payload)
                self.assertGreaterEqual(len(calls), 2)
                self.assertEqual(calls[0], (0, len(payload)))
                self.assertEqual(calls[-1][0], len(payload))
                self.assertIn("download", stages)
                self.assertIn("verify", stages)
            finally:
                os.environ.pop("TRACKLAB_MODEL_DIR", None)

    def test_cancel_raises_without_leaving_part_file(self) -> None:
        payload = (b"abcdefgh" * (1 << 18))  # 2 MiB so there is a second chunk
        spec = _spec_for(payload)
        token = CancelToken()

        def fake_urlopen(_request, context=None, timeout=None):  # noqa: ANN001
            return _FakeResponse(payload)

        def on_progress(received: int, _total: int) -> None:
            if received >= 1 << 20:
                token.cancel()

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRACKLAB_MODEL_DIR"] = tmp
            try:
                with patch("ai.model_manager.urllib.request.urlopen", fake_urlopen):
                    with self.assertRaises(DownloadCancelled):
                        ensure_checkpoint(
                            spec, download=True, progress=on_progress, cancel=token
                        )
                leftovers = list(Path(tmp).glob("*"))
                self.assertEqual(leftovers, [])
            finally:
                os.environ.pop("TRACKLAB_MODEL_DIR", None)
