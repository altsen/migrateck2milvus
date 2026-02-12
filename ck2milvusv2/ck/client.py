"""ClickHouse 客户端封装。

基于 `clickhouse-connect`：
- command：执行 DDL/DML
- query：返回 dict 行列表
- describe_table：读取表结构

并发注意：clickhouse-connect client（session）不支持并发查询。
本封装采用“每线程一个 client”，确保多 worker 与心跳线程并存时不报错。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import clickhouse_connect

from ck2milvusv2.types import ClickHouseConfig
from ck2milvusv2.utils.retry import retry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnInfo:
    """表字段元信息（DESCRIBE TABLE 输出子集）。"""

    name: str
    ch_type: str


class ClickHouse:
    """ClickHouse HTTP 客户端（线程隔离 session）。"""

    def __init__(self, cfg: ClickHouseConfig):
        """创建客户端封装。

        Args:
            cfg: ClickHouse 连接配置。
        """

        self._cfg = cfg
        self._local = threading.local()

    def _get_client(self):
        """获取当前线程的 clickhouse-connect client。

        Returns:
            clickhouse-connect client。
        """

        c = getattr(self._local, "client", None)
        if c is not None:
            return c

        cfg = self._cfg
        c = clickhouse_connect.get_client(
            host=cfg.host,
            port=cfg.port,
            username=cfg.username,
            password=cfg.password,
            database=cfg.database,
            secure=cfg.secure,
        )
        self._local.client = c
        return c

    def command(self, sql: str, parameters: dict | None = None) -> None:
        """执行无返回 SQL。

        Args:
            sql: SQL 文本。
            parameters: 可选参数。

        Returns:
            无。
        """

        def _run() -> None:
            self._get_client().command(sql, parameters=parameters)

        retry(_run, retries=3)

    def query(self, sql: str, parameters: dict | None = None) -> list[dict]:
        """执行查询并返回 dict 行列表。

        Args:
            sql: SQL 文本。
            parameters: 可选参数。

        Returns:
            行列表。
        """

        def _run() -> list[dict]:
            res = self._get_client().query(sql, parameters=parameters)
            out: list[dict] = []
            for row in res.result_rows:
                out.append(dict(zip(res.column_names, row)))
            return out

        return retry(_run, retries=3)

    def query_value(self, sql: str, parameters: dict | None = None):
        """执行查询并返回第一行第一列。

        Args:
            sql: SQL 文本。
            parameters: 参数。

        Returns:
            值或 None。
        """

        rows = self.query(sql, parameters=parameters)
        if not rows:
            return None
        return next(iter(rows[0].values()))

    def describe_table(self, full_table: str) -> list[ColumnInfo]:
        """读取表结构（字段名 + 类型）。

        Args:
            full_table: 完整表名（调用方需保证已正确引用/转义）。

        Returns:
            字段信息列表。
        """

        rows = self.query(f"DESCRIBE TABLE {full_table}")
        cols: list[ColumnInfo] = []
        for r in rows:
            cols.append(ColumnInfo(name=str(r["name"]), ch_type=str(r["type"])))
        return cols

    def server_now_string(self) -> str:
        """获取服务器端 now() 字符串。

        Returns:
            时间字符串。
        """

        return str(self.query_value("SELECT toString(now()) AS now"))

    def insert_rows(self, full_table: str, *, rows: list[dict], columns: list[str]) -> None:
        """批量插入多行。

        Args:
            full_table: 完整表名（db.table）。
            rows: 行列表（dict）。
            columns: 列名顺序。

        Returns:
            无。
        """

        if not rows:
            return

        # clickhouse-connect insert 接受 list[list] 或 list[tuple]
        data: list[list] = []
        for r in rows:
            data.append([r.get(c) for c in columns])

        def _run() -> None:
            self._get_client().insert(full_table, data, column_names=columns)

        retry(_run, retries=3)
