"""日志初始化。

要求：关键节点日志清晰、准确、详细。
"""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """初始化全局 logging。

    Args:
        level: 日志级别。

    Returns:
        无。
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
