"""流程A：ClickHouse → Milvus 去重入库（full / incremental）。

每个 worker 处理一个不重叠的 cursor 数据段，顺序执行：
  读取 CK → 展开 Nested → 向量化 → 批内去重 → 库内去重 → 入库 → 记录淘汰
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field

from pymilvus import Collection

from ck2milvusv2.ck.client import ClickHouse
from ck2milvusv2.ck.meta import (
    get_checkpoint,
    record_dedup_eliminated_many,
    record_dedup_stats,
    record_error_row,
    set_checkpoint,
)
from ck2milvusv2.ck.sql import qtable
from ck2milvusv2.milvus.client import Milvus
from ck2milvusv2.milvus.io import adaptive_insert, find_duplicate_in_milvus, upsert_entity
from ck2milvusv2.milvus.schema import ensure_collection
from ck2milvusv2.models.base import Embedder
from ck2milvusv2.pipeline.flatten import flatten_special_nested
from ck2milvusv2.pipeline.records import ExpandedRow
from ck2milvusv2.pipeline.vector import embed_texts
from ck2milvusv2.types import DedupConfig, JobConfig, SpecialNestedFlattenRule, TableConfig

logger = logging.getLogger(__name__)


# ── 入口 ──────────────────────────────────────────────────────


def run_flow_a(
    *,
    job: JobConfig,
    table_cfg: TableConfig,
    mode: str,
    embedder: Embedder,
    worker_id: int = 0,
    segment_start: str = "",
    segment_end: str = "",
) -> None:
    """单个 worker 执行流程A。

    Args:
        worker_id: 当前 worker 编号（用于 checkpoint 隔离）。
        segment_start/segment_end: 本 worker 负责的 cursor 范围。
    """
    ck = ClickHouse(job.clickhouse)
    col = ensure_collection(
        collection_name=table_cfg.target_collection,
        cfg=table_cfg, dim=job.model.embedding_dim,
        using=Milvus(job.milvus).alias,
    )

    ckpt_key = _ckpt_table_name(table_cfg.target_collection, worker_id)
    last = get_checkpoint(
        ck, job.meta.database, job_name=job.job_name,
        table_name=ckpt_key, mode=mode, cursor_field=table_cfg.cursor_field,
    )
    last_cursor, last_pk_values = last or ("", [])

    eff_start = segment_start or table_cfg.full_start or ""
    eff_end = segment_end or table_cfg.full_end or ""

    logger.info(
        "flow_a start worker=%d mode=%s table=%s cursor=[%s, %s) last=%s",
        worker_id, mode, table_cfg.source_table, eff_start, eff_end, last_cursor,
    )

    while True:
        rows = _read_batch(
            ck, table_cfg, job.runtime.ck_batch_size,
            last_cursor, last_pk_values, eff_start, eff_end,
        )
        if not rows:
            _safe_flush(col)
            logger.info("flow_a done worker=%d table=%s", worker_id, table_cfg.source_table)
            return

        result = _process_batch(col, table_cfg, job, embedder, rows)
        if result:
            _save_eliminated(ck, job, table_cfg, result)
            _save_stats(ck, job, table_cfg, result)

        last_cursor, last_pk_values = _extract_cursor(rows, table_cfg)
        set_checkpoint(
            ck, job.meta.database, job_name=job.job_name,
            table_name=ckpt_key, mode=mode,
            cursor_field=table_cfg.cursor_field,
            last_cursor=last_cursor, last_pk_values=last_pk_values,
        )


# ── 批处理核心 ────────────────────────────────────────────────


@dataclass
class BatchResult:
    """一批数据的处理结果。"""

    batch_id: str
    expanded: list[ExpandedRow]
    kept_batch: list[ExpandedRow]
    kept_milvus: list[ExpandedRow]
    inserted: int
    elim_batch: list[dict]
    elim_milvus: list[dict]
    updated: int
    timing: dict = field(default_factory=dict)


def _process_batch(
    col: Collection, cfg: TableConfig, job: JobConfig,
    embedder: Embedder, rows: list[dict],
) -> BatchResult | None:
    """处理一批 CK 数据：展开 → 向量化 → 去重 → 入库。"""
    batch_id = uuid.uuid4().hex
    t0 = time.monotonic()

    # 1. 展开
    expanded = _expand_rows(cfg, rows)
    if not expanded:
        return None
    t_expand = time.monotonic() - t0

    # 2. 向量化（顺序分批，无线程池）
    t1 = time.monotonic()
    vectors = embed_texts(
        embedder=embedder,
        texts=[x.content for x in expanded],
        batch_size=job.model.embedding_batch_size,
    )
    for i, v in enumerate(vectors):
        expanded[i].vector = _normalize(v)
    t_embed = time.monotonic() - t1

    # 3. 批内去重
    t2 = time.monotonic()
    kept, elim_batch = _dedup_in_batch(expanded, job.dedup)
    t_dedup_batch = time.monotonic() - t2

    # 4. 库内去重
    t3 = time.monotonic()
    kept2, elim_milvus, updated = _dedup_against_milvus(col, cfg, job, kept)
    t_dedup_milvus = time.monotonic() - t3

    # 5. 入库
    t4 = time.monotonic()
    inserted = _insert_kept(col, cfg, job, kept2)
    if inserted > 0:
        _safe_flush(col)
    t_insert = time.monotonic() - t4

    timing = {
        "expand": round(t_expand, 3), "embed": round(t_embed, 3),
        "dedup_batch": round(t_dedup_batch, 3), "dedup_milvus": round(t_dedup_milvus, 3),
        "insert": round(t_insert, 3),
    }

    elapsed = max(0.0001, time.monotonic() - t0)
    logger.info(
        "flow_a batch id=%s read=%d expanded=%d kept1=%d kept2=%d ins=%d "
        "elim_b=%d elim_m=%d qps=%.1f timing=%s",
        batch_id, len(rows), len(expanded), len(kept), len(kept2),
        inserted, len(elim_batch), len(elim_milvus),
        len(expanded) / elapsed, json.dumps(timing),
    )

    return BatchResult(
        batch_id=batch_id, expanded=expanded,
        kept_batch=kept, kept_milvus=kept2,
        inserted=inserted, elim_batch=elim_batch,
        elim_milvus=elim_milvus, updated=updated, timing=timing,
    )


# ── CK 读取 ──────────────────────────────────────────────────


def _read_batch(
    ck: ClickHouse, cfg: TableConfig, batch_size: int,
    last_cursor: str, last_pk_values: list[str],
    start_cursor: str, end_cursor: str,
) -> list[dict]:
    """读取 ClickHouse 一批数据（稳定游标分页）。"""
    cols = _select_columns(cfg)
    order_by = ", ".join([cfg.cursor_field] + cfg.pk_fields)
    where_parts, params = _build_where(cfg, last_cursor, last_pk_values, start_cursor, end_cursor)
    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    sql = (
        f"SELECT {', '.join(cols)} FROM {qtable(cfg.source_table)} "
        f"{where} ORDER BY {order_by} LIMIT {{lim:UInt64}}"
    )
    params["lim"] = int(batch_size)
    return ck.query(sql, parameters=params)


def _select_columns(cfg: TableConfig) -> list[str]:
    """构建 SELECT 列列表（Nested 子列使用 arrayFilter 做 SQL 层排空/值过滤）。

    对特殊 Nested 子列，利用 CK arrayFilter 将排空筛选和值集合过滤
    下推到 SQL 层执行，减少传输数据量，提升大数据量场景性能。
    """
    cols = list(cfg.pk_fields) + [cfg.cursor_field, cfg.vector_source_field] + list(cfg.scalar_fields)
    af_exprs = _build_nested_array_filter(cfg.special_nested)
    for k in cfg.special_nested.fields:
        if k in af_exprs:
            cols.append(af_exprs[k])
        else:
            cols.append(f"{cfg.special_nested.prefix}.{k}")
    return cols


def _build_nested_array_filter(rule: SpecialNestedFlattenRule) -> dict[str, str]:
    """为 Nested 子列构建 arrayFilter SQL（排空 + 值集合过滤下推到 CK）。

    利用 ClickHouse arrayFilter 高阶函数，在 SQL 层完成排空和值过滤，
    避免将大量无效数据传到 Python 侧。

    CK arrayFilter 签名：
        arrayFilter(func(x, y1, ..., yN), source_arr, cond1_arr, ..., condN_arr)

    对每个 Nested 子列，使用相同的过滤条件（其它子列作为条件数组），
    确保所有子列索引对齐。

    Args:
        rule: SpecialNestedFlattenRule 配置。

    Returns:
        {sub_key: "arrayFilter(...) AS `prefix.key`"} 映射；
        无需过滤的子列不出现在返回值中。
    """
    prefix = rule.prefix

    # ── 收集条件列（去重、有序） ──
    cond_cols: list[str] = []
    seen: set[str] = set()
    for col_name in (rule.align_by, rule.empty_filter_field):
        if col_name and col_name not in seen:
            cond_cols.append(col_name)
            seen.add(col_name)

    vf_active: dict[str, list[str]] = {}
    for sub, allow in (rule.value_filters or {}).items():
        if allow:
            vf_active[sub] = [str(a) for a in allow]
            if sub not in seen:
                cond_cols.append(sub)
                seen.add(sub)

    if not cond_cols:
        return {}

    # ── 构建 lambda 参数与映射 ──
    param_map: dict[str, str] = {}
    for i, c in enumerate(cond_cols):
        param_map[c] = f"_c{i}"
    params = ["_v"] + [param_map[c] for c in cond_cols]

    # ── 构建过滤条件 ──
    conditions: list[str] = []

    def _not_empty(param: str) -> str:
        """生成非空判断 SQL 片段。"""
        return f"(toString({param}) != '' AND lower(trim(toString({param}))) != 'null')"

    if rule.align_by and rule.align_by in param_map:
        conditions.append(_not_empty(param_map[rule.align_by]))

    if (rule.empty_filter_field
            and rule.empty_filter_field != rule.align_by
            and rule.empty_filter_field in param_map):
        conditions.append(_not_empty(param_map[rule.empty_filter_field]))

    for sub, allow_list in vf_active.items():
        p = param_map[sub]
        escaped = ", ".join(f"'{_escape_sql_str(a)}'" for a in allow_list)
        conditions.append(f"(toString({p}) IN ({escaped}))")

    if not conditions:
        return {}

    cond_str = " AND ".join(conditions)
    lambda_str = f"({', '.join(params)}) -> ({cond_str})"
    cond_refs = [f"{prefix}.{c}" for c in cond_cols]

    # ── 为每个子列生成 arrayFilter 表达式 ──
    result: dict[str, str] = {}
    for k in rule.fields:
        source = f"{prefix}.{k}"
        all_arrays = ", ".join([source] + cond_refs)
        result[k] = f"arrayFilter({lambda_str}, {all_arrays}) AS `{prefix}.{k}`"

    return result


def _escape_sql_str(s: str) -> str:
    """转义 SQL 字符串中的单引号和反斜杠。"""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _build_where(
    cfg: TableConfig, last_cursor: str, last_pk_values: list[str],
    start_cursor: str, end_cursor: str,
) -> tuple[list[str], dict]:
    """构建 WHERE 子句（cursor + pk 分页）。"""
    cursor_type = (cfg.cursor_field_type or "datetime").strip().lower()
    if cursor_type not in {"datetime", "number"}:
        raise ValueError(f"unsupported CURSOR_FIELD_TYPE={cfg.cursor_field_type!r}")

    def _expr(param: str) -> str:
        return f"toDateTime({{{param}:String}})" if cursor_type == "datetime" else f"{{{param}:Float64}}"

    def _val(v: str):
        return _normalize_datetime(v) if cursor_type == "datetime" else float(v)

    parts: list[str] = []
    params: dict = {}

    if end_cursor:
        params["ec"] = _val(end_cursor)
        parts.append(f"{cfg.cursor_field} < {_expr('ec')}")

    if last_cursor:
        params["lc"] = _val(last_cursor)
        for i in range(len(cfg.pk_fields)):
            params[f"pk{i}"] = last_pk_values[i] if i < len(last_pk_values) else ""

        if len(cfg.pk_fields) == 1:
            parts.append(
                f"(({cfg.cursor_field} > {_expr('lc')}) "
                f"OR ({cfg.cursor_field} = {_expr('lc')} AND {cfg.pk_fields[0]} > {{pk0:String}}))"
            )
        else:
            left = ", ".join(cfg.pk_fields)
            right = ", ".join(f"{{pk{i}:String}}" for i in range(len(cfg.pk_fields)))
            parts.append(
                f"(({cfg.cursor_field} > {_expr('lc')}) "
                f"OR ({cfg.cursor_field} = {_expr('lc')} AND ({left}) > ({right})))"
            )
    elif start_cursor:
        params["sc"] = _val(start_cursor)
        parts.append(f"{cfg.cursor_field} >= {_expr('sc')}")

    return parts, params


# ── 展开 & 向量 ──────────────────────────────────────────────


def _expand_rows(cfg: TableConfig, rows: list[dict]) -> list[ExpandedRow]:
    out: list[ExpandedRow] = []
    for r in rows:
        try:
            out.extend(flatten_special_nested(r, cfg))
        except Exception as exc:  # noqa: BLE001
            pk = "|".join("" if r.get(f) is None else str(r.get(f)) for f in cfg.pk_fields)
            logger.warning("flatten failed pk=%s err=%s", pk, exc)
    return out


def _normalize(v: list[float]) -> list[float]:
    if not v:
        return []
    s = sum(float(x) * float(x) for x in v)
    if s <= 0:
        return [0.0] * len(v)
    inv = 1.0 / math.sqrt(s)
    return [float(x) * inv for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    return sum(float(ai) * float(bi) for ai, bi in zip(a, b))


# ── 批内去重 ─────────────────────────────────────────────────


def _dedup_in_batch(
    expanded: list[ExpandedRow], dedup: DedupConfig,
) -> tuple[list[ExpandedRow], list[dict]]:
    th = float(dedup.batch_threshold)

    groups: dict[str, list[ExpandedRow]] = {}
    for x in expanded:
        groups.setdefault(x.space_key, []).append(x)

    kept: list[ExpandedRow] = []
    eliminated: list[dict] = []

    for sk, items in groups.items():
        if dedup.keep_strategy == "earliest_by_cursor":
            items = sorted(items, key=lambda x: x.cursor_value)

        keep_list: list[ExpandedRow] = []
        keep_vecs: list[list[float]] = []

        for it in items:
            vn = _normalize(it.vector or [])
            best_sim, best_idx = -1.0, -1
            for j, kv in enumerate(keep_vecs):
                sim = _dot(vn, kv)
                if sim > best_sim:
                    best_sim, best_idx = sim, j

            if best_sim >= th and best_idx >= 0:
                keeper = keep_list[best_idx]
                keeper.dup_count += 1
                eliminated.append({
                    "space_key": sk, "source_pk": it.milvus_pk,
                    "duplicate_of_source_pk": keeper.milvus_pk,
                    "duplicate_of_milvus_pk": "", "similarity": best_sim, "dup_count": 1,
                })
                continue

            keep_list.append(it)
            keep_vecs.append(vn)

        kept.extend(keep_list)

    return kept, eliminated


# ── 库内去重 ─────────────────────────────────────────────────


def _dedup_against_milvus(
    col: Collection, cfg: TableConfig, job: JobConfig,
    kept: list[ExpandedRow],
) -> tuple[list[ExpandedRow], list[dict], int]:
    if not kept:
        return [], [], 0

    batch_min_ts = min(x.time_ts for x in kept)
    window_start = batch_min_ts - int(job.runtime.lookback_hours) * 3600
    window_end = batch_min_ts

    survivors: list[ExpandedRow] = []
    eliminated: list[dict] = []
    updated = 0

    for r in kept:
        space_vals = {
            k: ("" if r.scalar.get(k) is None else str(r.scalar.get(k)))
            for k in cfg.special_nested.space_keys
        }
        hit = find_duplicate_in_milvus(
            col=col, cfg=cfg, vector=r.vector or [],
            space_values=space_vals,
            window_start=window_start, window_end=window_end,
            topk=job.dedup.milvus_topk,
        )
        if hit is None or float(hit["similarity"]) < float(job.dedup.milvus_threshold):
            survivors.append(r)
            continue

        holder_pk = str(hit["pk"])
        holder_dup = int(hit.get("dup_count") or 1)
        if _update_holder_dup(col, cfg, holder_pk, holder_dup + r.dup_count):
            updated += 1

        eliminated.append({
            "space_key": r.space_key, "source_pk": r.milvus_pk,
            "duplicate_of_source_pk": "", "duplicate_of_milvus_pk": holder_pk,
            "similarity": float(hit["similarity"]), "dup_count": r.dup_count,
        })

    return survivors, eliminated, updated


def _update_holder_dup(col: Collection, cfg: TableConfig, pk: str, new_dup: int) -> bool:
    fields = [f.name for f in col.schema.fields]
    rows = col.query(expr=f'{cfg.milvus_pk_field} == "{pk}"', output_fields=fields, limit=1)
    if not rows:
        return False
    entity = dict(rows[0])
    entity[cfg.milvus_dup_count_field] = int(new_dup)
    try:
        upsert_entity(col=col, entity=entity)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("update dup_count failed pk=%s err=%s", pk, exc)
        return False


# ── 入库 ─────────────────────────────────────────────────────


def _insert_kept(
    col: Collection, cfg: TableConfig, job: JobConfig, rows: list[ExpandedRow],
) -> int:
    if not rows:
        return 0

    entities = [_build_entity(r, cfg) for r in rows]
    adaptive_insert(
        col=col, entities=entities,
        batch_size=job.runtime.milvus_insert_batch_size,
        min_batch_size=job.runtime.milvus_insert_min_batch_size,
        max_retries=job.runtime.milvus_insert_max_retries,
        backoff_seconds=job.runtime.milvus_insert_retry_backoff_seconds,
        backoff_max_seconds=job.runtime.milvus_insert_retry_backoff_max_seconds,
        flush_on_retry=job.runtime.milvus_insert_flush_on_retry,
    )
    return len(entities)


def _build_entity(r: ExpandedRow, cfg: TableConfig) -> dict:
    """从 ExpandedRow 构建 Milvus entity dict。"""
    skip = {
        cfg.milvus_pk_field, cfg.milvus_content_field, cfg.milvus_llm_field,
        cfg.milvus_vector_field, cfg.milvus_dup_count_field, cfg.milvus_time_field,
    }
    ent = {
        cfg.milvus_pk_field: r.milvus_pk,
        cfg.milvus_content_field: r.content,
        cfg.milvus_llm_field: r.scalar.get(cfg.milvus_llm_field, ""),
        cfg.milvus_dup_count_field: int(r.dup_count),
        cfg.milvus_time_field: int(r.time_ts),
    }
    for k, v in (r.scalar or {}).items():
        if k in skip:
            continue
        if k == cfg.special_nested.expand_index_field:
            ent[k] = 0 if v is None else int(v)
        else:
            ent[k] = "" if v is None else str(v)
    ent[cfg.milvus_vector_field] = r.vector or []
    return ent


# ── 记录 ─────────────────────────────────────────────────────


def _save_eliminated(ck: ClickHouse, job: JobConfig, cfg: TableConfig, r: BatchResult) -> None:
    for stage, elim in [("batch", r.elim_batch), ("milvus", r.elim_milvus)]:
        if not elim:
            continue
        rows = [{
            "batch_id": r.batch_id, "job_name": job.job_name,
            "table_name": cfg.target_collection, "stage": stage,
            "space_key": str(e.get("space_key", "")),
            "source_pk": str(e.get("source_pk", "")),
            "duplicate_of_source_pk": str(e.get("duplicate_of_source_pk", "")),
            "duplicate_of_milvus_pk": str(e.get("duplicate_of_milvus_pk", "")),
            "similarity": float(e.get("similarity", 0)),
            "dup_count": int(e.get("dup_count", 1)),
        } for e in elim]
        try:
            record_dedup_eliminated_many(ck, job.meta.database, rows=rows)
        except Exception as exc:  # noqa: BLE001
            for row in rows[:10]:
                record_error_row(
                    ck, job.meta.database, job_name=job.job_name,
                    table_name=cfg.target_collection,
                    pk=str(row.get("source_pk", "")),
                    error=f"record_eliminated failed: {exc}",
                )


def _save_stats(ck: ClickHouse, job: JobConfig, cfg: TableConfig, r: BatchResult) -> None:
    record_dedup_stats(
        ck, job.meta.database,
        batch_id=r.batch_id, job_name=job.job_name,
        table_name=cfg.target_collection,
        input_rows=len(r.expanded), after_batch_dedup=len(r.kept_batch),
        after_milvus_dedup=len(r.kept_milvus), inserted_rows=r.inserted,
        eliminated_batch_rows=len(r.elim_batch),
        eliminated_milvus_rows=len(r.elim_milvus),
        timing=json.dumps(r.timing),
    )


# ── 工具 ─────────────────────────────────────────────────────


def _extract_cursor(rows: list[dict], cfg: TableConfig) -> tuple[str, list[str]]:
    last = rows[-1]
    cursor = last.get(cfg.cursor_field)
    return (
        "" if cursor is None else str(cursor),
        ["" if last.get(f) is None else str(last.get(f)) for f in cfg.pk_fields],
    )


def _ckpt_table_name(collection: str, worker_id: int) -> str:
    """生成 checkpoint 用的 table_name（多 worker 隔离）。"""
    return collection if worker_id == 0 else f"{collection}:w{worker_id}"


def _normalize_datetime(raw: str) -> str:
    s = (raw or "").strip().replace("T", " ")
    return s[:19] if len(s) >= 19 else s


def _safe_flush(col: Collection) -> None:
    try:
        col.flush()
    except Exception:  # noqa: BLE001
        pass
