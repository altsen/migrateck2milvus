"""ClickHouse 元数据表（v2）。

用途：
- `task_lock`：作业级锁（TTL + 心跳），避免 cron/systemd timer 重入
- `checkpoints`：每表游标 checkpoint（full/incremental/llm 均可用）
- `dedup_eliminated`：淘汰项与重复关系记录
- `dedup_batch_stats`：批次统计
- `error_rows`：行级错误记录（用于审计/重试）

说明：
- v2 不包含 v1 的 llm_todo 队列（不做超长分流）
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime
from typing import Any

from ck2milvusv2.ck.client import ClickHouse
from ck2milvusv2.utils.tz import TZ_EAST8, now_east8
from ck2milvusv2.ck.sql import qident


logger = logging.getLogger(__name__)


META_TABLE_LOCK = "task_lock"
META_TABLE_RUNS = "task_runs"
META_TABLE_CHECKPOINTS = "checkpoints"
META_TABLE_DEDUP_ELIMINATED = "dedup_eliminated"
META_TABLE_DEDUP_STATS = "dedup_batch_stats"
META_TABLE_ERROR_ROWS = "error_rows"


def ensure_meta_db_and_tables(ck: ClickHouse, meta_db: str) -> None:
    """初始化 meta 数据库与表。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库名。

    Returns:
        无。
    """

    ck.command(f"CREATE DATABASE IF NOT EXISTS {qident(meta_db)}")

    ck.command(
        f"""
        CREATE TABLE IF NOT EXISTS {qident(meta_db)}.{qident(META_TABLE_LOCK)}
        (
            job_name String,
            holder String,
            heartbeat DateTime,
            expires_at DateTime,
            version UInt64
        )
        ENGINE = ReplacingMergeTree(version)
        ORDER BY (job_name)
        """
    )

    ck.command(
        f"""
        CREATE TABLE IF NOT EXISTS {qident(meta_db)}.{qident(META_TABLE_RUNS)}
        (
            run_id String,
            job_name String,
            mode String,
            status String,
            start_time DateTime,
            end_time DateTime,
            cursor_field String,
            cursor_start String,
            cursor_end String,
            table_name String,
            stats String,
            error String
        )
        ENGINE = MergeTree
        ORDER BY (job_name, start_time, run_id)
        """
    )

    ck.command(
        f"""
        CREATE TABLE IF NOT EXISTS {qident(meta_db)}.{qident(META_TABLE_CHECKPOINTS)}
        (
            job_name String,
            table_name String,
            mode String,
            cursor_field String,
            last_cursor String,
            last_pk_values Array(String),
            updated_at DateTime
        )
        ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (job_name, table_name, mode, cursor_field)
        """
    )

    ck.command(
        f"""
        CREATE TABLE IF NOT EXISTS {qident(meta_db)}.{qident(META_TABLE_DEDUP_ELIMINATED)}
        (
            batch_id String,
            job_name String,
            table_name String,
            stage String,
            space_key String,
            source_pk String,
            duplicate_of_source_pk String,
            duplicate_of_milvus_pk String,
            similarity Float64,
            dup_count UInt64,
            created_at DateTime
        )
        ENGINE = MergeTree
        ORDER BY (job_name, table_name, created_at, batch_id)
        """
    )

    ck.command(
        f"""
        CREATE TABLE IF NOT EXISTS {qident(meta_db)}.{qident(META_TABLE_DEDUP_STATS)}
        (
            batch_id String,
            job_name String,
            table_name String,
            input_rows UInt64,
            after_batch_dedup UInt64,
            after_milvus_dedup UInt64,
            inserted_rows UInt64,
            eliminated_batch_rows UInt64,
            eliminated_milvus_rows UInt64,
            timing String,
            ts DateTime
        )
        ENGINE = MergeTree
        ORDER BY (job_name, table_name, ts, batch_id)
        """
    )

    ck.command(
        f"""
        CREATE TABLE IF NOT EXISTS {qident(meta_db)}.{qident(META_TABLE_ERROR_ROWS)}
        (
            job_name String,
            table_name String,
            pk String,
            error String,
            created_at DateTime
        )
        ENGINE = MergeTree
        ORDER BY (job_name, table_name, created_at, pk)
        """
    )


_ALL_META_TABLES = [
    META_TABLE_LOCK,
    META_TABLE_RUNS,
    META_TABLE_CHECKPOINTS,
    META_TABLE_DEDUP_ELIMINATED,
    META_TABLE_DEDUP_STATS,
    META_TABLE_ERROR_ROWS,
]


def drop_meta_tables(ck: ClickHouse, meta_db: str) -> None:
    """删除所有 meta 表（不删除数据库本身）。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库名。
    """
    for tbl in _ALL_META_TABLES:
        ck.command(f"DROP TABLE IF EXISTS {qident(meta_db)}.{qident(tbl)}")
    logger.info("dropped all meta tables in %s", meta_db)


class MetaLock:
    """用 ClickHouse 表实现作业锁（TTL + 心跳）。"""

    _MAX_HEARTBEAT_FAILURES = 5

    def __init__(
        self,
        *,
        ck: ClickHouse,
        meta_db: str,
        job_name: str,
        ttl_seconds: int,
        heartbeat_seconds: int,
        max_hold_seconds: int = 0,
    ):
        """初始化锁。

        Args:
            ck: ClickHouse 客户端。
            meta_db: meta 数据库名。
            job_name: 作业名。
            ttl_seconds: TTL 秒数。
            heartbeat_seconds: 心跳间隔秒。
            max_hold_seconds: 最大持锁时长（0=不限制）。
        """

        self._ck = ck
        self._meta_db = meta_db
        self._job_name = job_name
        self._ttl_seconds = int(ttl_seconds)
        self._heartbeat_seconds = int(heartbeat_seconds)
        self._max_hold_seconds = int(max_hold_seconds)
        self._holder = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._version = int(time.time() * 1000)
        self._acquired_at: float = 0.0

    @property
    def holder(self) -> str:
        """当前 holder 标识。"""

        return self._holder

    def try_acquire(self) -> bool:
        """尝试获取锁。

        Returns:
            成功 True，否则 False。
        """

        now = self._ck.server_now_string()
        latest = self._ck.query(
            f"""
            SELECT holder, expires_at, version
            FROM {qident(self._meta_db)}.{qident(META_TABLE_LOCK)} FINAL
            WHERE job_name = {{job_name:String}}
            LIMIT 1
            """,
            parameters={"job_name": self._job_name},
        )
        if latest:
            expires_at = str(latest[0]["expires_at"])
            holder = str(latest[0]["holder"])
            if expires_at > now:
                logger.warning("lock busy: holder=%s expires_at=%s", holder, expires_at)
                return False

        self._insert_lock_row()
        confirm = self._ck.query(
            f"""
            SELECT holder, expires_at, version
            FROM {qident(self._meta_db)}.{qident(META_TABLE_LOCK)} FINAL
            WHERE job_name = {{job_name:String}}
            LIMIT 1
            """,
            parameters={"job_name": self._job_name},
        )
        if not confirm:
            return False

        if str(confirm[0]["holder"]) != self._holder:
            logger.warning("lock lost after acquire, current holder=%s", str(confirm[0]["holder"]))
            return False

        self._acquired_at = time.monotonic()
        self._start_heartbeat()
        return True

    def release(self) -> None:
        """释放锁（停止心跳并写入过期记录）。"""

        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

        try:
            self._insert_lock_row(expires_in_seconds=0)
        except Exception:  # noqa: BLE001
            logger.warning("failed to release lock")

    def _insert_lock_row(self, *, expires_in_seconds: int | None = None) -> None:
        """写入锁记录。

        Args:
            expires_in_seconds: 过期秒数；None 表示使用 ttl_seconds。

        Returns:
            无。
        """

        ttl = self._ttl_seconds if expires_in_seconds is None else int(expires_in_seconds)
        self._version = int(time.time() * 1000)
        self._ck.command(
            f"""
            INSERT INTO {qident(self._meta_db)}.{qident(META_TABLE_LOCK)}
            (job_name, holder, heartbeat, expires_at, version)
            VALUES
            ({{job_name:String}}, {{holder:String}}, now(), now() + INTERVAL {{ttl:Int32}} SECOND, {{version:UInt64}})
            """,
            parameters={
                "job_name": self._job_name,
                "holder": self._holder,
                "ttl": ttl,
                "version": self._version,
            },
        )

    def _start_heartbeat(self) -> None:
        """启动心跳线程。"""

        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._heartbeat_loop, name="MetaLockHeartbeat", daemon=True)
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        """心跳循环。

        Returns:
            无。
        """

        failures = 0
        while not self._stop.is_set():
            if self._max_hold_seconds > 0 and self._acquired_at > 0:
                if (time.monotonic() - self._acquired_at) > self._max_hold_seconds:
                    logger.warning("max_hold_seconds reached, stop renew")
                    return

            try:
                self._insert_lock_row()
                failures = 0
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.warning("heartbeat failed %d/%d err=%s", failures, self._MAX_HEARTBEAT_FAILURES, exc)
                if failures >= self._MAX_HEARTBEAT_FAILURES:
                    logger.warning("too many heartbeat failures, stop renew")
                    return

            self._stop.wait(timeout=self._heartbeat_seconds)



def get_checkpoint(
    ck: ClickHouse,
    meta_db: str,
    *,
    job_name: str,
    table_name: str,
    mode: str,
    cursor_field: str,
) -> tuple[str, list[str]] | None:
    """读取 checkpoint。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库。
        job_name: 作业名。
        table_name: 表标识。
        mode: 模式（full/incremental/llm）。
        cursor_field: 游标字段。

    Returns:
        (last_cursor, last_pk_values) 或 None。
    """

    rows = ck.query(
        f"""
        SELECT last_cursor, last_pk_values
        FROM {qident(meta_db)}.{qident(META_TABLE_CHECKPOINTS)} FINAL
        WHERE job_name={{job:String}} AND table_name={{tb:String}} AND mode={{mode:String}} AND cursor_field={{cf:String}}
        LIMIT 1
        """,
        parameters={"job": job_name, "tb": table_name, "mode": mode, "cf": cursor_field},
    )
    if not rows:
        return None
    return str(rows[0]["last_cursor"]), list(rows[0]["last_pk_values"] or [])


def set_checkpoint(
    ck: ClickHouse,
    meta_db: str,
    *,
    job_name: str,
    table_name: str,
    mode: str,
    cursor_field: str,
    last_cursor: str,
    last_pk_values: list[str],
) -> None:
    """写入 checkpoint。

    兼容性说明：
    ClickHouse < 22.8 不支持 `{param:Array(String)}` 参数化绑定，
    因此 `last_pk_values` 使用客户端拼接的 Array 字面量替代。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库。
        job_name: 作业名。
        table_name: 表标识。
        mode: 模式。
        cursor_field: 游标字段。
        last_cursor: 最新游标。
        last_pk_values: 最新 pk 列表（字符串化）。

    Returns:
        无。
    """

    # ── 客户端拼接 Array(String) 字面量，兼容 CK 22.6 ──
    escaped = ", ".join(f"'{_escape_ck_str(v)}'" for v in last_pk_values)
    array_literal = f"[{escaped}]"

    ck.command(
        f"""
        INSERT INTO {qident(meta_db)}.{qident(META_TABLE_CHECKPOINTS)}
        (job_name, table_name, mode, cursor_field, last_cursor, last_pk_values, updated_at)
        VALUES
        ({{job:String}}, {{tb:String}}, {{mode:String}}, {{cf:String}}, {{lc:String}}, {array_literal}, now())
        """,
        parameters={
            "job": job_name,
            "tb": table_name,
            "mode": mode,
            "cf": cursor_field,
            "lc": last_cursor,
        },
    )


def _escape_ck_str(s: str) -> str:
    """转义 ClickHouse 字符串字面量中的特殊字符。

    Args:
        s: 原始字符串。

    Returns:
        转义后的字符串。
    """
    return s.replace("\\", "\\\\").replace("'", "\\'")


def clear_checkpoint(
    ck: ClickHouse,
    meta_db: str,
    *,
    job_name: str,
    table_name: str,
    mode: str,
    cursor_field: str,
) -> None:
    """清空 checkpoint（通过写入空值覆盖）。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库。
        job_name: 作业名。
        table_name: 表标识。
        mode: 模式。
        cursor_field: 游标字段。

    Returns:
        无。
    """

    set_checkpoint(
        ck,
        meta_db,
        job_name=job_name,
        table_name=table_name,
        mode=mode,
        cursor_field=cursor_field,
        last_cursor="",
        last_pk_values=[],
    )


def record_dedup_eliminated(
    ck: ClickHouse,
    meta_db: str,
    *,
    batch_id: str,
    job_name: str,
    table_name: str,
    stage: str,
    space_key: str,
    source_pk: str,
    duplicate_of_source_pk: str,
    duplicate_of_milvus_pk: str,
    similarity: float,
    dup_count: int,
) -> None:
    """记录去重淘汰关系。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库。
        batch_id: 批次 id。
        job_name: 作业名。
        table_name: 表标识。
        stage: batch/milvus。
        space_key: 去重空间。
        source_pk: 淘汰项 pk。
        duplicate_of_source_pk: 批内 keeper pk（可空）。
        duplicate_of_milvus_pk: Milvus 持有者 pk（可空）。
        similarity: 相似度。
        dup_count: 淘汰项代表数量（通常为 1 或批内聚合后的值）。

    Returns:
        无。
    """

    ck.command(
        f"""
        INSERT INTO {qident(meta_db)}.{qident(META_TABLE_DEDUP_ELIMINATED)}
        (batch_id, job_name, table_name, stage, space_key, source_pk,
         duplicate_of_source_pk, duplicate_of_milvus_pk, similarity, dup_count, created_at)
        VALUES
        ({{bid:String}}, {{job:String}}, {{tb:String}}, {{stage:String}}, {{sk:String}}, {{spk:String}},
         {{dspk:String}}, {{dmpk:String}}, {{sim:Float64}}, {{dc:UInt64}}, now())
        """,
        parameters={
            "bid": batch_id,
            "job": job_name,
            "tb": table_name,
            "stage": stage,
            "sk": space_key,
            "spk": source_pk,
            "dspk": duplicate_of_source_pk,
            "dmpk": duplicate_of_milvus_pk,
            "sim": float(similarity),
            "dc": int(dup_count),
        },
    )


def record_dedup_eliminated_many(
    ck: ClickHouse,
    meta_db: str,
    *,
    rows: list[dict[str, object]],
) -> None:
    """批量写入去重淘汰关系。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库。
        rows: 行列表，每行包含 record_dedup_eliminated 所需字段。

    Returns:
        无。
    """

    if not rows:
        return

    def _coerce_datetime(v: object) -> datetime:
        if isinstance(v, datetime):
            return v if v.tzinfo is not None else v.replace(tzinfo=TZ_EAST8)
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v), tz=TZ_EAST8)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return now_east8()
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=TZ_EAST8)
            except ValueError:
                pass
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    return datetime.strptime(s, fmt).replace(tzinfo=TZ_EAST8)
                except ValueError:
                    continue
        return now_east8()

    cols = [
        "batch_id",
        "job_name",
        "table_name",
        "stage",
        "space_key",
        "source_pk",
        "duplicate_of_source_pk",
        "duplicate_of_milvus_pk",
        "similarity",
        "dup_count",
        "created_at",
    ]

    now = now_east8()
    to_insert: list[dict] = []
    for r in rows:
        x = dict(r)
        created_at = x.get("created_at")
        x["created_at"] = now if created_at is None else _coerce_datetime(created_at)
        to_insert.append(x)

    ck.insert_rows(
        f"{meta_db}.{META_TABLE_DEDUP_ELIMINATED}",
        rows=to_insert,
        columns=cols,
    )


def record_dedup_stats(
    ck: ClickHouse,
    meta_db: str,
    *,
    batch_id: str,
    job_name: str,
    table_name: str,
    input_rows: int,
    after_batch_dedup: int,
    after_milvus_dedup: int,
    inserted_rows: int,
    eliminated_batch_rows: int,
    eliminated_milvus_rows: int,
    timing: str,
) -> None:
    """记录批次统计。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库。
        batch_id: 批次 id。
        job_name: 作业名。
        table_name: 表标识。
        input_rows: 输入行数。
        after_batch_dedup: 批内去重后。
        after_milvus_dedup: 库内去重后。
        inserted_rows: 插入条数。
        eliminated_batch_rows: 批内淘汰条数。
        eliminated_milvus_rows: 库内淘汰条数。
        timing: 时延统计（JSON 字符串）。

    Returns:
        无。
    """

    ck.command(
        f"""
        INSERT INTO {qident(meta_db)}.{qident(META_TABLE_DEDUP_STATS)}
        (batch_id, job_name, table_name, input_rows, after_batch_dedup, after_milvus_dedup,
         inserted_rows, eliminated_batch_rows, eliminated_milvus_rows, timing, ts)
        VALUES
        ({{bid:String}}, {{job:String}}, {{tb:String}}, {{in:UInt64}}, {{abd:UInt64}}, {{amd:UInt64}},
         {{ins:UInt64}}, {{eb:UInt64}}, {{em:UInt64}}, {{tim:String}}, now())
        """,
        parameters={
            "bid": batch_id,
            "job": job_name,
            "tb": table_name,
            "in": int(input_rows),
            "abd": int(after_batch_dedup),
            "amd": int(after_milvus_dedup),
            "ins": int(inserted_rows),
            "eb": int(eliminated_batch_rows),
            "em": int(eliminated_milvus_rows),
            "tim": timing,
        },
    )


def record_error_row(ck: ClickHouse, meta_db: str, *, job_name: str, table_name: str, pk: str, error: str) -> None:
    """记录行级错误。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库。
        job_name: 作业名。
        table_name: 表标识。
        pk: 行 pk。
        error: 错误信息。

    Returns:
        无。
    """

    ck.command(
        f"""
        INSERT INTO {qident(meta_db)}.{qident(META_TABLE_ERROR_ROWS)}
        (job_name, table_name, pk, error, created_at)
        VALUES
        ({{job:String}}, {{tb:String}}, {{pk:String}}, {{err:String}}, now())
        """,
        parameters={"job": job_name, "tb": table_name, "pk": pk, "err": (error or "")[:500]},
    )


def record_task_run(
    ck: ClickHouse,
    meta_db: str,
    *,
    run_id: str,
    job_name: str,
    mode: str,
    status: str,
    start_time: str,
    end_time: str = "",
    cursor_field: str = "",
    cursor_start: str = "",
    cursor_end: str = "",
    table_name: str = "",
    stats: str = "",
    error: str = "",
) -> None:
    """记录一次任务运行。

    Args:
        ck: ClickHouse 客户端。
        meta_db: meta 数据库。
        run_id: 唯一 id。
        job_name: 作业名。
        mode: 模式。
        status: running/success/failed。
        start_time: 开始时间字符串（UTC）。
        end_time: 结束时间。
        cursor_field: 游标字段。
        cursor_start: 窗口开始。
        cursor_end: 窗口结束。
        table_name: 表标识。
        stats: 统计。
        error: 错误。

    Returns:
        无。
    """

    def _to_dt(s: str) -> str:
        if not s:
            return "1970-01-01 00:00:00"
        return s

    now_local = now_east8().strftime("%Y-%m-%d %H:%M:%S")
    ck.command(
        f"""
        INSERT INTO {qident(meta_db)}.{qident(META_TABLE_RUNS)}
        (run_id, job_name, mode, status, start_time, end_time,
         cursor_field, cursor_start, cursor_end, table_name, stats, error)
        VALUES
        ({{rid:String}}, {{job:String}}, {{mode:String}}, {{st:String}},
         toDateTime({{start:String}}), toDateTime({{end:String}}),
         {{cf:String}}, {{cs:String}}, {{ce:String}}, {{tb:String}}, {{stats:String}}, {{err:String}})
        """,
        parameters={
            "rid": run_id,
            "job": job_name,
            "mode": mode,
            "st": status,
            "start": _to_dt(start_time),
            "end": _to_dt(end_time or now_local),
            "cf": cursor_field,
            "cs": cursor_start,
            "ce": cursor_end,
            "tb": table_name,
            "stats": stats,
            "err": (error or "")[:500],
        },
    )
