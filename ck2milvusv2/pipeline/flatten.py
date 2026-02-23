"""特殊 Nested flatten。

实现要点：
- 排空筛选与值集合过滤已通过 SQL arrayFilter 下推到 ClickHouse 层
- 本模块仅负责将已过滤的 Nested 数组展开为 ExpandedRow 列表
"""

from __future__ import annotations

import json
from datetime import datetime

from ck2milvusv2.pipeline.records import ExpandedRow
from ck2milvusv2.types import TableConfig
from ck2milvusv2.utils.tz import TZ_EAST8


def flatten_special_nested(row: dict, cfg: TableConfig) -> list[ExpandedRow]:
    """把一条 ClickHouse 行按特殊 Nested 规则展开为多条。

    说明：排空筛选和值集合过滤已在 SQL 层通过 arrayFilter 完成，
    本函数仅负责将已过滤的数组展开为 ExpandedRow。

    Args:
        row: ClickHouse 行（dict），Nested 子列已预过滤。
        cfg: 表配置。

    Returns:
        ExpandedRow 列表。
    """

    rule = cfg.special_nested
    prefix = rule.prefix

    def _col(sub: str):
        """读取 Nested 子列值，兼容 arrayFilter 别名（_af_{sub}）与原始列名。"""
        af_key = f"_af_{sub}"
        if af_key in row:
            return row[af_key]
        return row.get(f"{prefix}.{sub}")

    # 使用 align_by 列确定展开长度（已通过 SQL arrayFilter 预过滤）
    align_vals = _ensure_list(_col(rule.align_by))
    n = len(align_vals)

    if n == 0:
        return []

    # 收集各子列数据
    sliced: dict[str, list] = {}
    for sub in rule.fields.keys():
        sliced[sub] = _ensure_list(_col(sub))

    source_pk = _compact_source_pk(row, cfg)
    cursor_val = row.get(cfg.cursor_field)
    cursor_s = "" if cursor_val is None else str(cursor_val)
    time_ts = _cursor_to_ts_seconds(cursor_val)
    content = "" if row.get(cfg.vector_source_field) is None else str(row.get(cfg.vector_source_field))

    out: list[ExpandedRow] = []
    for expand_index in range(n):
        scalar: dict = {}
        for ck_name in cfg.scalar_fields:
            milvus_name = (cfg.scalar_field_mappings or {}).get(ck_name, ck_name)
            scalar[milvus_name] = _serialize_scalar_value(row.get(ck_name))

        for sub, target_name in rule.fields.items():
            vals = sliced.get(sub) or []
            scalar[target_name] = None if expand_index >= len(vals) else vals[expand_index]

        scalar[rule.expand_index_field] = expand_index

        space_key = _build_space_key(scalar, rule.space_keys)
        milvus_pk = f"{source_pk}:{expand_index}"

        out.append(
            ExpandedRow(
                milvus_pk=milvus_pk,
                source_pk=source_pk,
                cursor_value=cursor_s,
                time_ts=int(time_ts),
                space_key=space_key,
                content=content,
                scalar=scalar,
            )
        )
    return out


def _serialize_scalar_value(v):
    """把 ClickHouse 标量字段值序列化为可入库的值。

    约定：
    - Array/Nested（Python 侧通常是 list/tuple/dict）统一转成 JSON 字符串
    - JSON 序列化必须使用 ensure_ascii=False，避免中文被转义

    Args:
        v: ClickHouse 返回的原始值。

    Returns:
        处理后的值；None 保持为 None（上游插入时会转成空串）。
    """

    if v is None:
        return None
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _build_space_key(scalar: dict, space_keys: list[str]) -> str:
    """拼接去重空间键。"""

    parts: list[str] = []
    for k in space_keys:
        v = scalar.get(k)
        parts.append("" if v is None else str(v))
    return "|".join(parts)


def _compact_source_pk(row: dict, cfg: TableConfig) -> str:
    """拼接 source pk（支持复合主键）。"""

    parts: list[str] = []
    for f in cfg.pk_fields:
        v = row.get(f)
        parts.append("" if v is None else str(v))
    return "|".join(parts)


def _ensure_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _is_empty(v) -> bool:
    if v is None:
        return True
    s = str(v)
    return s.strip() == "" or s.strip().lower() == "null"


def _cursor_to_ts_seconds(v) -> int:
    """将 cursor 值转换为 epoch seconds。

    Args:
        v: cursor 值。

    Returns:
        seconds。
    """

    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=TZ_EAST8)
        return int(v.timestamp())
    # 尝试解析常见字符串格式
    s = str(v).strip().replace("T", " ")
    # 端点可能包含时区后缀（例如 +8 / +08:00），简单按需求截断
    if len(s) >= 19:
        s = s[:19]
    try:
        # 兼容 `YYYY-mm-dd HH:MM:SS`，使用东8区
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return int(dt.replace(tzinfo=TZ_EAST8).timestamp())
    except Exception:  # noqa: BLE001
        try:
            dt2 = datetime.fromisoformat(s)
            if dt2.tzinfo is None:
                dt2 = dt2.replace(tzinfo=TZ_EAST8)
            return int(dt2.timestamp())
        except Exception:  # noqa: BLE001
            return 0
