"""通用重试工具。

用于：
- 模型 HTTP 调用
- ClickHouse/Milvus 短暂抖动

特殊处理：
- RateLimitError → 使用更长退避（rate_limit_backoff_seconds）
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from ck2milvusv2.utils.http import RateLimitError

T = TypeVar("T")

logger = logging.getLogger(__name__)


def retry(
    fn: Callable[[], T],
    *,
    retries: int,
    backoff_seconds: float = 0.5,
    rate_limit_backoff_seconds: float = 5.0,
) -> T:
    """执行函数并在失败时重试。

    Args:
        fn: 待执行函数。
        retries: 重试次数（不含首次）。
        backoff_seconds: 退避基准秒数（指数增长）。
        rate_limit_backoff_seconds: 限流退避基准秒数（指数增长）。

    Returns:
        fn 的返回值。

    Raises:
        最后一次异常会被抛出。
    """

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                raise
            if isinstance(exc, RateLimitError):
                sleep = rate_limit_backoff_seconds * (2 ** attempt)
                logger.warning("rate-limited, retry %d/%d  backoff=%.1fs", attempt + 1, retries, sleep)
            else:
                sleep = backoff_seconds * (2 ** attempt)
                logger.warning("retry %d/%d  err=%s  backoff=%.1fs", attempt + 1, retries, type(exc).__name__, sleep)
            time.sleep(sleep)
            attempt += 1
