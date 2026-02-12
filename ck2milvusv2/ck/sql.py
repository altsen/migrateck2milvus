"""ClickHouse SQL 拼接辅助。

目的：
- 统一处理标识符与表名引用，避免注入与转义错误
"""

from __future__ import annotations


def qident(name: str) -> str:
    """对 ClickHouse 标识符做反引号转义。

    Args:
        name: 标识符（db/table/column）。

    Returns:
        转义后的标识符。
    """

    n = name.replace("`", "``")
    return f"`{n}`"


def qtable(full_table: str) -> str:
    """对完整表名做转义。

    Args:
        full_table: 形如 `db.table`。

    Returns:
        转义后的引用。
    """

    if "." not in full_table:
        return qident(full_table)
    db, tb = full_table.split(".", 1)
    return f"{qident(db)}.{qident(tb)}"
