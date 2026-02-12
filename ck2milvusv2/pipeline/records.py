"""流程内部数据结构（避免模块循环依赖）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExpandedRow:
    """特殊 Nested 展开后的单条记录。"""

    milvus_pk: str
    source_pk: str
    cursor_value: str
    time_ts: int
    space_key: str
    content: str
    scalar: dict
    vector: list[float] | None = None
    dup_count: int = 1
