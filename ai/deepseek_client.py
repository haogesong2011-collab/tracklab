"""OpenAI-compatible DeepSeek chat client with streaming, JSON mode, and retries."""

from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai.contracts import CancelToken

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
DEFAULT_REPORT_MODEL = "deepseek-v4-pro"
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 60.0
MAX_RETRIES = 3
EMPTY_JSON_RETRIES = 1

Transport = Callable[[str, dict[str, str], bytes, float, bool], Any]
ChunkHandler = Callable[[str], None]


class DeepSeekError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        code: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.code = code


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class DeepSeekResponse:
    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    json_data: dict[str, Any] | None = None
    cancelled: bool = False
    reasoning: str = ""


def chinese_error(status: int | None, body: str = "", exc: BaseException | None = None) -> DeepSeekError:
    lowered = (body or "").lower()
    if status in {401, 403}:
        return DeepSeekError(
            "API Key 无效或没有访问权限，请在「DeepSeek 设置」中重新填写。",
            status=status,
            code="auth",
        )
    if status == 402 or "insufficient" in lowered or "balance" in lowered:
        return DeepSeekError(
            "DeepSeek 账户余额不足，请到平台充值后再试。",
            status=status or 402,
            code="balance",
        )
    if status in {400, 422}:
        return DeepSeekError(
            "请求参数不被接口接受。请检查模型 ID，或稍后再试。",
            status=status,
            code="bad_request",
        )
    if status == 429:
        return DeepSeekError(
            "请求过于频繁，请稍后再试。",
            status=429,
            retryable=True,
            code="rate_limit",
        )
    if "content" in lowered and ("filter" in lowered or "policy" in lowered):
        return DeepSeekError(
            "内容被安全策略拦截，请改写问题后再试。",
            status=status,
            code="filtered",
        )
    if status is not None and status >= 500:
        return DeepSeekError(
            "DeepSeek 服务暂时不可用，请稍后重试。",
            status=status,
            retryable=True,
            code="server",
        )
    if exc is not None:
        extra = f"{exc} {getattr(exc, 'reason', '')}".lower()
        if "timed out" in extra:
            return DeepSeekError("连接超时，请检查网络后重试。", retryable=True, code="timeout")
        if "certificate" in extra or "ssl" in extra:
            return DeepSeekError(
                "HTTPS 证书校验失败。请在虚拟环境执行 python -m pip install certifi 后重启 TrackLab。",
                retryable=False,
                code="ssl",
            )
    if status is None:
        return DeepSeekError("无法连接 DeepSeek，请检查网络后重试。", retryable=True, code="network")
    return DeepSeekError("DeepSeek 请求失败，请稍后重试。", status=status, retryable=True, code="unknown")


def ssl_context() -> ssl.SSLContext:
    """Prefer certifi: python.org / PlatformIO builds on macOS often lack a CA bundle."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        report_model: str = DEFAULT_REPORT_MODEL,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        from ai.api_credentials import normalize_api_key

        cleaned = normalize_api_key(api_key)
        if not cleaned:
            raise DeepSeekError("尚未配置 API Key。", code="auth")
        self.api_key = cleaned
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.report_model = report_model
        self._transport = transport
        self._sleep = sleep

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        json_mode: bool = False,
        stream: bool = False,
        on_chunk: ChunkHandler | None = None,
        on_reasoning: ChunkHandler | None = None,
        cancel: CancelToken | None = None,
    ) -> DeepSeekResponse:
        chosen = model or (self.report_model if json_mode else self.chat_model)
        empty_tries = 0
        while True:
            response = self._send(
                messages,
                model=chosen,
                json_mode=json_mode,
                stream=stream,
                on_chunk=on_chunk,
                on_reasoning=on_reasoning,
                cancel=cancel,
            )
            if response.cancelled:
                return response
            if json_mode:
                parsed = _parse_json_object(response.text)
                if parsed is None:
                    empty_tries += 1
                    if empty_tries <= EMPTY_JSON_RETRIES:
                        continue
                    raise DeepSeekError("模型没有返回有效 JSON，请重试。", code="empty_json")
                response.json_data = parsed
            return response

    def _send(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        json_mode: bool,
        stream: bool,
        on_chunk: ChunkHandler | None,
        on_reasoning: ChunkHandler | None,
        cancel: CancelToken | None,
    ) -> DeepSeekResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream_options"] = {"include_usage": True}
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        last_error: DeepSeekError | None = None
        for attempt in range(MAX_RETRIES):
            if cancel is not None and cancel.cancelled:
                return DeepSeekResponse(text="", model=model, cancelled=True)
            try:
                raw = self._do_request(url, headers, payload, stream)
                if stream:
                    return self._read_stream(
                        raw,
                        model=model,
                        on_chunk=on_chunk,
                        on_reasoning=on_reasoning,
                        cancel=cancel,
                    )
                return self._read_json(raw, model=model)
            except DeepSeekError as exc:
                last_error = exc
                if not exc.retryable or attempt >= MAX_RETRIES - 1:
                    raise
                self._sleep(0.4 * (2**attempt))
            except TimeoutError as exc:
                last_error = chinese_error(None, exc=exc)
                if attempt >= MAX_RETRIES - 1:
                    raise last_error from exc
                self._sleep(0.4 * (2**attempt))
            except UnicodeEncodeError as exc:
                raise DeepSeekError(
                    "请求头里出现了中文或特殊字符。请重新只粘贴 sk- 开头的 API Key，不要带说明文字。",
                    code="encoding",
                ) from exc
            except URLError as exc:
                last_error = chinese_error(None, exc=exc)
                if not last_error.retryable or attempt >= MAX_RETRIES - 1:
                    raise last_error from exc
                self._sleep(0.4 * (2**attempt))
        assert last_error is not None
        raise last_error

    def _do_request(
        self,
        url: str,
        headers: dict[str, str],
        payload: bytes,
        stream: bool,
    ) -> Any:
        timeout = READ_TIMEOUT_S
        if self._transport is not None:
            try:
                return self._transport(url, headers, payload, timeout, stream)
            except HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                raise chinese_error(exc.code, body) from exc
        request = Request(url, data=payload, headers=headers, method="POST")
        try:
            return urlopen(request, timeout=timeout, context=ssl_context())
        except UnicodeEncodeError as exc:
            raise DeepSeekError(
                "请求头里出现了中文或特殊字符。请重新只粘贴 sk- 开头的 API Key，不要带说明文字。",
                code="encoding",
            ) from exc
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise chinese_error(exc.code, body) from exc

    def _read_json(self, raw: Any, *, model: str) -> DeepSeekResponse:
        text = _response_text(raw)
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise DeepSeekError("无法解析 DeepSeek 响应。", code="bad_json") from exc
        content = ""
        reasoning = ""
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")
            reasoning = _reasoning_from(message)
        return DeepSeekResponse(
            text=content,
            model=str(data.get("model") or model),
            usage=_usage_from(data.get("usage")),
            reasoning=reasoning,
        )

    def _read_stream(
        self,
        raw: Any,
        *,
        model: str,
        on_chunk: ChunkHandler | None,
        on_reasoning: ChunkHandler | None,
        cancel: CancelToken | None,
    ) -> DeepSeekResponse:
        parts: list[str] = []
        thoughts: list[str] = []
        usage = TokenUsage()
        used_model = model
        for line in _iter_sse_lines(raw):
            if cancel is not None and cancel.cancelled:
                return DeepSeekResponse(
                    text="".join(parts),
                    model=used_model,
                    usage=usage,
                    cancelled=True,
                    reasoning="".join(thoughts),
                )
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                used_model = str(payload.get("model") or used_model)
                if payload.get("usage"):
                    usage = _usage_from(payload.get("usage"))
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                thought = _reasoning_from(delta)
                if thought:
                    thoughts.append(thought)
                    if on_reasoning is not None:
                        on_reasoning(thought)
                piece = delta.get("content") or ""
                if piece:
                    parts.append(str(piece))
                    if on_chunk is not None:
                        on_chunk(str(piece))
        return DeepSeekResponse(
            text="".join(parts),
            model=used_model,
            usage=usage,
            reasoning="".join(thoughts),
        )


def _reasoning_from(payload: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _usage_from(raw: Any) -> TokenUsage:
    data = raw or {}
    return TokenUsage(
        prompt_tokens=int(data.get("prompt_tokens") or 0),
        completion_tokens=int(data.get("completion_tokens") or 0),
        total_tokens=int(data.get("total_tokens") or 0),
    )


def _response_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw
    read = getattr(raw, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)
    return str(raw)


def _iter_sse_lines(raw: Any) -> Iterator[str]:
    if isinstance(raw, list):
        for line in raw:
            yield str(line).rstrip("\n")
        return
    iterator = getattr(raw, "__iter__", None)
    if callable(iterator) and not isinstance(raw, (bytes, bytearray, str)):
        for line in raw:
            if isinstance(line, bytes):
                yield line.decode("utf-8", errors="replace").rstrip("\n")
            else:
                yield str(line).rstrip("\n")
        return
    text = _response_text(raw)
    for line in text.splitlines():
        yield line


def _parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return data
