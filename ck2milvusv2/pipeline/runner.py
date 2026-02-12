"""任务入口（init / run）。

职责：
- 初始化 meta 表
- 获取作业锁（防重入）
- 按 mode 调度流程A / 流程B
- 支持流程级多 worker 并发（每个 worker 处理不重叠的数据段）

并发模型说明：
- 并发 = 多个 flow worker 线程，各自处理一段不重叠的数据
- 每个 worker 内部顺序分批处理，模型调用按 batch_size 分批，无线程池
- FLOW_A_WORKERS / FLOW_B_WORKERS 控制流程级并发度
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
import uuid
from datetime import datetime as dt

import config

from ck2milvusv2.ck.client import ClickHouse
from ck2milvusv2.ck.meta import MetaLock, clear_checkpoint, drop_meta_tables, ensure_meta_db_and_tables, record_task_run
from ck2milvusv2.ck.sql import qtable
from ck2milvusv2.models.factory import build_models
from ck2milvusv2.pipeline.flow_a import run_flow_a
from ck2milvusv2.pipeline.flow_b import run_flow_b
from ck2milvusv2.types import JobConfig, TableConfig
from ck2milvusv2.utils.tz import TZ_EAST8, now_east8_str

logger = logging.getLogger(__name__)


# ── 公开接口 ──────────────────────────────────────────────────


def init_meta(job: JobConfig | None = None, *, drop: bool = False) -> None:
    """初始化 meta 数据库和所有元数据表。

    Args:
        job: 可选的外部配置；默认使用 config.JOB。
        drop: 若为 True，先删除所有 meta 表再创建。
    """
    job2 = job or config.JOB
    ck = ClickHouse(job2.clickhouse)
    if drop:
        drop_meta_tables(ck, job2.meta.database)
    ensure_meta_db_and_tables(ck, job2.meta.database)


def run_mode(*, mode: str, table_filter: list[str] | None, checkpoint_strategy: str) -> None:
    """执行指定 mode 的迁移任务（带锁 + 状态记录）。"""
    job = config.JOB
    ck = ClickHouse(job.clickhouse)
    ensure_meta_db_and_tables(ck, job.meta.database)

    lock = MetaLock(
        ck=ck, meta_db=job.meta.database, job_name=job.job_name,
        ttl_seconds=job.runtime.lock_ttl_seconds,
        heartbeat_seconds=job.runtime.lock_heartbeat_seconds,
        max_hold_seconds=job.runtime.lock_max_hold_seconds,
    )
    if not lock.try_acquire():
        raise RuntimeError("job already running (lock busy)")

    run_id = uuid.uuid4().hex
    start_time = now_east8_str()
    record_task_run(
        ck, job.meta.database,
        run_id=run_id, job_name=job.job_name, mode=mode,
        status="running", start_time=start_time,
        stats=json.dumps({"tables": len(job.tables)}),
    )

    try:
        tables = _filter_tables(job.tables, table_filter)
        if checkpoint_strategy == "restart":
            _clear_all_checkpoints(ck, job, tables, mode)

        embedder, summarizer = build_models(job.model)
        t0 = time.monotonic()

        for idx, t in enumerate(tables, 1):
            t_start = time.monotonic()
            if mode in {"full", "incremental"}:
                _dispatch_flow_a(job, t, mode, embedder)
            elif mode == "llm":
                _dispatch_flow_b(job, t, mode, embedder, summarizer)
            else:
                raise ValueError(f"unknown mode: {mode}")
            logger.info(
                "table %d/%d [%s] done  elapsed=%.1fs",
                idx, len(tables), t.target_collection, time.monotonic() - t_start,
            )

        elapsed = round(time.monotonic() - t0, 3)
        logger.info("===== run_mode(%s) FINISHED  tables=%d  total_elapsed=%.1fs =====", mode, len(tables), elapsed)

        record_task_run(
            ck, job.meta.database,
            run_id=run_id, job_name=job.job_name, mode=mode,
            status="success", start_time=start_time, end_time=now_east8_str(),
            stats=json.dumps({
                "elapsed_seconds": elapsed,
                "tables": len(tables),
            }),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run failed")
        record_task_run(
            ck, job.meta.database,
            run_id=run_id, job_name=job.job_name, mode=mode,
            status="failed", start_time=start_time, end_time=now_east8_str(),
            error=str(exc)[:500],
        )
        raise
    finally:
        lock.release()


# ── 流程级并发调度 ────────────────────────────────────────────


def _dispatch_flow_a(job: JobConfig, t: TableConfig, mode: str, embedder) -> None:
    """调度 Flow A：支持多 worker 并发处理不同 cursor 数据段。"""
    n = max(1, job.runtime.flow_a_workers)
    if n == 1:
        run_flow_a(job=job, table_cfg=t, mode=mode, embedder=embedder)
        return

    segments = _split_cursor_range(job, t, n)
    logger.info("flow_a dispatch %d workers, segments=%s", n, segments)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = {
            pool.submit(
                run_flow_a, job=job, table_cfg=t, mode=mode, embedder=embedder,
                worker_id=i, segment_start=seg[0], segment_end=seg[1],
            ): i
            for i, seg in enumerate(segments)
        }
        for fut in concurrent.futures.as_completed(futs):
            wid = futs[fut]
            try:
                fut.result()
                logger.info("flow_a worker=%d done", wid)
            except Exception:
                logger.exception("flow_a worker=%d failed", wid)
                raise


def _dispatch_flow_b(job: JobConfig, t: TableConfig, mode: str, embedder, summarizer) -> None:
    """调度 Flow B：支持多 worker 并发处理不同 time_ts 数据段。"""
    n = max(1, job.runtime.flow_b_workers)
    if n == 1:
        run_flow_b(job=job, table_cfg=t, mode=mode, embedder=embedder, summarizer=summarizer)
        return

    start_ts = int(t.llm_start_ts or 0)
    end_ts = int(t.llm_end_ts or 0) or int(time.time())
    segments = _split_ts_range(start_ts, end_ts, n)
    logger.info("flow_b dispatch %d workers, segments=%s", n, segments)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = {
            pool.submit(
                run_flow_b, job=job, table_cfg=t, mode=mode,
                embedder=embedder, summarizer=summarizer,
                worker_id=i, segment_start_ts=seg[0], segment_end_ts=seg[1],
            ): i
            for i, seg in enumerate(segments)
        }
        for fut in concurrent.futures.as_completed(futs):
            wid = futs[fut]
            try:
                fut.result()
                logger.info("flow_b worker=%d done", wid)
            except Exception:
                logger.exception("flow_b worker=%d failed", wid)
                raise


# ── 数据段切分 ────────────────────────────────────────────────


def _split_cursor_range(job: JobConfig, t: TableConfig, n: int) -> list[tuple[str, str]]:
    """查询源表 cursor 范围并均分为 n 段。"""
    ck = ClickHouse(job.clickhouse)
    cursor_type = (t.cursor_field_type or "datetime").strip().lower()

    where = ""
    if t.full_start:
        if cursor_type == "datetime":
            where = f"WHERE {t.cursor_field} >= toDateTime('{t.full_start[:19]}')"
        else:
            where = f"WHERE {t.cursor_field} >= {float(t.full_start)}"

    sql = f"SELECT min({t.cursor_field}) AS mn, max({t.cursor_field}) AS mx FROM {qtable(t.source_table)} {where}"
    rows = ck.query(sql)
    if not rows or rows[0]["mn"] is None:
        return [("", "")]

    mn, mx = rows[0]["mn"], rows[0]["mx"]

    if cursor_type == "datetime":
        mn_ts = _to_ts(mn)
        mx_ts = _to_ts(mx) + 1  # 包含 max
        if mn_ts >= mx_ts:
            return [("", "")]
        step = (mx_ts - mn_ts) / n
        segs = []
        for i in range(n):
            s = int(mn_ts + i * step)
            e = int(mn_ts + (i + 1) * step) if i < n - 1 else mx_ts
            segs.append((
                dt.fromtimestamp(s, tz=TZ_EAST8).strftime("%Y-%m-%d %H:%M:%S"),
                dt.fromtimestamp(e, tz=TZ_EAST8).strftime("%Y-%m-%d %H:%M:%S"),
            ))
        if t.full_end:
            segs[-1] = (segs[-1][0], t.full_end)
        return segs
    else:
        mn_f, mx_f = float(mn), float(mx) + 1
        step = (mx_f - mn_f) / n
        segs = []
        for i in range(n):
            s = mn_f + i * step
            e = mn_f + (i + 1) * step if i < n - 1 else mx_f
            segs.append((str(s), str(e)))
        if t.full_end:
            segs[-1] = (segs[-1][0], t.full_end)
        return segs


def _split_ts_range(start_ts: int, end_ts: int, n: int) -> list[tuple[int, int]]:
    """将时间戳范围均分为 n 段。"""
    if start_ts >= end_ts or n <= 1:
        return [(start_ts, end_ts)]
    step = (end_ts - start_ts) / n
    segs = []
    for i in range(n):
        s = int(start_ts + i * step)
        e = int(start_ts + (i + 1) * step) if i < n - 1 else end_ts
        segs.append((s, e))
    return segs


def _to_ts(v) -> int:
    """将 cursor 值转为 unix timestamp（东8区）。"""
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, dt):
        if v.tzinfo is None:
            v = v.replace(tzinfo=TZ_EAST8)
        return int(v.timestamp())
    s = str(v).strip().replace("T", " ")
    if len(s) >= 19:
        s = s[:19]
    try:
        naive = dt.strptime(s, "%Y-%m-%d %H:%M:%S")
        return int(naive.replace(tzinfo=TZ_EAST8).timestamp())
    except Exception:  # noqa: BLE001
        return 0


# ── 工具 ──────────────────────────────────────────────────────


def _filter_tables(tables: list[TableConfig], table_filter: list[str] | None) -> list[TableConfig]:
    if not table_filter:
        return tables
    allowed = {x.strip() for x in table_filter if x.strip()}
    return [t for t in tables if t.source_table in allowed or t.target_collection in allowed]


def _clear_all_checkpoints(ck: ClickHouse, job: JobConfig, tables: list[TableConfig], mode: str) -> None:
    """清除所有表和所有 worker 的 checkpoint。"""
    for t in tables:
        cursor_field = t.milvus_time_field if mode == "llm" else t.cursor_field
        # 主 worker（worker_id=0）
        clear_checkpoint(
            ck, job.meta.database, job_name=job.job_name,
            table_name=t.target_collection, mode=mode, cursor_field=cursor_field,
        )
        # 额外 worker (worker_id >= 1)
        n = job.runtime.flow_a_workers if mode != "llm" else job.runtime.flow_b_workers
        for i in range(1, max(1, n)):
            clear_checkpoint(
                ck, job.meta.database, job_name=job.job_name,
                table_name=f"{t.target_collection}:w{i}",
                mode=mode, cursor_field=cursor_field,
            )
