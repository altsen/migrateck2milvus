"""环境变量加载（.env）。

优先级规则：
- 已存在的环境变量（export/set）优先
- 再读取 .env（只填补缺失，不覆盖已有）
"""

from __future__ import annotations

from pathlib import Path


def load_dotenv(path: str | Path, *, override: bool = False) -> None:
    """加载 `.env` 文件到环境变量。

    Args:
        path: `.env` 路径。
        override: 是否覆盖已存在的环境变量。

    Returns:
        无。
    """

    p = Path(path)
    if not p.exists():
        return

    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("\"'")
        if not k:
            continue
        if (not override) and (k in __import__("os").environ):
            continue
        __import__("os").environ[k] = v
