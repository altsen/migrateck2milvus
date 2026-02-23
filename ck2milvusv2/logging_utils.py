"""日志初始化。

要求：关键节点日志清晰、准确、详细。
支持同时输出到 console 和文件，异常自动记录完整堆栈。
"""

from __future__ import annotations

import logging
import os
import sys


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] - %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """初始化全局 logging（console + 可选文件）。

    - 日志格式包含源文件名和行号，方便定位。
    - 当环境变量 LOG_FILE 非空时，同时写入该文件。
    - 未捕获异常也会被 logging 记录。

    Args:
        level: 日志级别。

    Returns:
        无。
    """

    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # 避免重复添加 handler（如 setup_logging 被多次调用）
    if root.handlers:
        return

    fmt = logging.Formatter(_LOG_FORMAT)

    # ── console handler ──
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # ── 可选：文件 handler ──
    log_file = os.environ.get("LOG_FILE", "").strip()
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # ── 未捕获异常→ logging ──
    def _excepthook(exc_type, exc_value, exc_tb):
        """将未捕获异常写入 logging。"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.getLogger("ck2milvusv2").critical(
            "未捕获异常", exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _excepthook
