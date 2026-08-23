"""DeepSeek client mock tests. CI must never call the real API."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from io import BytesIO

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import CancelToken  # noqa: E402
from ai.deepseek_client import (  # noqa: E402
    DeepSeekClient,
    DeepSeekError,
    DEFAULT_CHAT_MODEL,
)


class _HTTPError(HTTPError):
    def __init__(self, code: int, body: str) -> None:
        super().__init__(
            url="https://api.deepseek.com/chat/completions",
            code=code,
            msg="err",
            hdrs=None,
            fp=BytesIO(body.encode("utf-8")),
        )


class DeepSeekClientTests(unittest.TestCase):
    def test_stream_chunks_and_usage(self) -> None:
        calls = {"n": 0}

        def transport(url, headers, payload, timeout, stream):
            calls["n"] += 1
            self.assertTrue(stream)
            self.assertIn("Authorization", headers)
            self.assertNotIn("sk-secret", url)
            body = json.loads(payload.decode())
            self.assertEqual(body["model"], DEFAULT_CHAT_MODEL)
            return [
                'data: {"choices":[{"delta":{"content":"你"}}]}',
                'data: {"choices":[{"delta":{"content":"好"}}]}',
                'data: {"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5},"choices":[]}',
                "data: [DONE]",
            ]

        chunks: list[str] = []
        client = DeepSeekClient("sk-test", transport=transport, sleep=lambda _s: None)
        response = client.complete(
            [{"role": "user", "content": "hi"}],
            stream=True,
            on_chunk=chunks.append,
        )
        self.assertEqual(response.text, "你好")
        self.assertEqual(chunks, ["你", "好"])
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(calls["n"], 1)

    def test_stream_reasoning_then_content(self) -> None:
        thoughts: list[str] = []
        chunks: list[str] = []

        def transport(url, headers, payload, timeout, stream):
            return [
                'data: {"choices":[{"delta":{"reasoning_content":"先看轨迹"}}]}',
                'data: {"choices":[{"delta":{"reasoning_content":"再写结论"}}]}',
                'data: {"choices":[{"delta":{"content":"这是斜抛"}}]}',
                "data: [DONE]",
            ]

        client = DeepSeekClient("sk-test", transport=transport, sleep=lambda _s: None)
        response = client.complete(
            [{"role": "user", "content": "分析"}],
            stream=True,
            on_chunk=chunks.append,
            on_reasoning=thoughts.append,
        )
        self.assertEqual(response.reasoning, "先看轨迹再写结论")
        self.assertEqual(response.text, "这是斜抛")
        self.assertEqual(thoughts, ["先看轨迹", "再写结论"])
        self.assertEqual(chunks, ["这是斜抛"])

    def test_json_mode_and_empty_retry(self) -> None:
        payloads = iter(["", '{"purpose":"测g","equipment":"尺","principle":"自由落体","errors":"","conclusion":"ok"}'])

        def transport(url, headers, payload, timeout, stream):
            body = json.loads(payload.decode())
            self.assertEqual(body["response_format"], {"type": "json_object"})
            text = next(payloads)
            return json.dumps({"choices": [{"message": {"content": text}}]})

        client = DeepSeekClient("sk-test", transport=transport, sleep=lambda _s: None)
        response = client.complete(
            [{"role": "user", "content": "return json"}],
            json_mode=True,
        )
        self.assertEqual(response.json_data["purpose"], "测g")

    def test_cancel_stops_stream(self) -> None:
        token = CancelToken()

        def transport(url, headers, payload, timeout, stream):
            token.cancel()
            return [
                'data: {"choices":[{"delta":{"content":"半"}}]}',
                'data: {"choices":[{"delta":{"content":"段"}}]}',
                "data: [DONE]",
            ]

        client = DeepSeekClient("sk-test", transport=transport, sleep=lambda _s: None)
        response = client.complete(
            [{"role": "user", "content": "x"}],
            stream=True,
            cancel=token,
        )
        self.assertTrue(response.cancelled)

    def test_ssl_error_is_chinese(self) -> None:
        import ssl

        def transport(url, headers, payload, timeout, stream):
            raise URLError(ssl.SSLCertVerificationError("certificate verify failed"))

        client = DeepSeekClient("sk-test", transport=transport, sleep=lambda _s: None)
        with self.assertRaises(DeepSeekError) as ctx:
            client.complete([{"role": "user", "content": "x"}])
        self.assertIn("证书", str(ctx.exception))
        self.assertEqual(ctx.exception.code, "ssl")
        self.assertFalse(ctx.exception.retryable)

    def test_timeout_is_retryable_chinese(self) -> None:
        def transport(url, headers, payload, timeout, stream):
            raise TimeoutError("timed out")

        client = DeepSeekClient("sk-test", transport=transport, sleep=lambda _s: None)
        with self.assertRaises(DeepSeekError) as ctx:
            client.complete([{"role": "user", "content": "x"}])
        self.assertIn("超时", str(ctx.exception))
        self.assertTrue(ctx.exception.retryable)

    def test_401_not_retried(self) -> None:
        n = {"c": 0}

        def transport(url, headers, payload, timeout, stream):
            n["c"] += 1
            raise _HTTPError(401, '{"error":{"message":"invalid"}}')

        client = DeepSeekClient("sk-test", transport=transport, sleep=lambda _s: None)
        with self.assertRaises(DeepSeekError) as ctx:
            client.complete([{"role": "user", "content": "x"}])
        self.assertEqual(n["c"], 1)
        self.assertIn("API Key", str(ctx.exception))
        self.assertFalse(ctx.exception.retryable)

    def test_429_and_5xx_retry(self) -> None:
        n = {"c": 0}

        def transport(url, headers, payload, timeout, stream):
            n["c"] += 1
            if n["c"] < 3:
                raise _HTTPError(429 if n["c"] == 1 else 503, "{}")
            return json.dumps({"choices": [{"message": {"content": "ok"}}]})

        client = DeepSeekClient("sk-test", transport=transport, sleep=lambda _s: None)
        response = client.complete([{"role": "user", "content": "x"}])
        self.assertEqual(response.text, "ok")
        self.assertEqual(n["c"], 3)


if __name__ == "__main__":
    unittest.main()
