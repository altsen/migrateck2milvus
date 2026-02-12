"""时区工具：统一使用东8区（+08:00）。

全项目所有涉及时间的逻辑，均使用本工具提供的东8区时间，不使用 UTC。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

TZ_EAST8 = timezone(timedelta(hours=8))
"""东8区时区对象。"""


def now_east8() -> datetime:
    """返回当前东8区时间。"""
    return datetime.now(TZ_EAST8)


def now_east8_str() -> str:
    """返回当前东8区时间字符串（yyyy-mm-dd HH:MM:SS）。"""
    return now_east8().strftime("%Y-%m-%d %H:%M:%S")
