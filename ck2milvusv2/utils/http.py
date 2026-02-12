"""HTTP 请求封装（标准库 urllib）。

尽量减少三方依赖：
- 发送 JSON
- 返回 JSON 或 raw text
- 检测 429 限流并抛出专用异常
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class RateLimitError(Exception):
    """API 限流（HTTP 429）。调用方可据此做更长退避。"""


def http_post_raw(url: str, *, headers: dict[str, str], payload: dict, timeout: int) -> str:
    """POST JSON 并返回响应文本。

    Args:
        url: URL。
        headers: 请求头。
        payload: JSON payload。
        timeout: 超时秒。

    Returns:
        响应文本。

    Raises:
        RateLimitError: 服务端返回 HTTP 429。
    """

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            b = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RateLimitError(f"HTTP 429 rate-limited: {url}") from exc
        raise
    return b.decode("utf-8", errors="replace")


def http_post_json(url: str, *, headers: dict[str, str], payload: dict, timeout: int) -> dict:
    """POST JSON 并解析 JSON 响应。

    Args:
        url: URL。
        headers: 请求头。
        payload: JSON payload。
        timeout: 超时秒。

    Returns:
        dict。
    """

    raw = http_post_raw(url, headers=headers, payload=payload, timeout=timeout)
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError("http_post_json response is not dict")
    return obj
