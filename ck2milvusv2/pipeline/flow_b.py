"""流程B：Milvus 原文 → LLM 规范化 → 覆盖写回。

每个 worker 处理一个不重叠的 time_ts 数据段：
  查询 Milvus → 按 selector 路由提示词 → LLM 规范化 → 重新向量化 → upsert
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter

from pymilvus import Collection

from ck2milvusv2.ck.client import ClickHouse
from ck2milvusv2.ck.meta import get_checkpoint, set_checkpoint
from ck2milvusv2.milvus.client import Milvus
from ck2milvusv2.milvus.io import upsert_entity
from ck2milvusv2.milvus.schema import ensure_collection
from ck2milvusv2.models.base import Embedder, Summarizer
from ck2milvusv2.pipeline.vector import embed_texts
from ck2milvusv2.types import JobConfig, TableConfig

logger = logging.getLogger(__name__)


# ── 入口 ──────────────────────────────────────────────────────


def run_flow_b(
    *,
    job: JobConfig,
    table_cfg: TableConfig,
    mode: str,
    embedder: Embedder,
    summarizer: Summarizer,
    worker_id: int = 0,
    segment_start_ts: int = 0,
    segment_end_ts: int = 0,
) -> None:
    """单个 worker 执行流程B。"""
    if mode != "llm":
        raise ValueError("flow_b only supports mode=llm")

    col = ensure_collection(
        collection_name=table_cfg.target_collection,
        cfg=table_cfg, dim=job.model.embedding_dim,
        using=Milvus(job.milvus).alias,
    )
    ck = ClickHouse(job.clickhouse)
    ckpt_key = _ckpt_table_name(table_cfg.target_collection, worker_id)

    # 确定时间范围
    start_ts = segment_start_ts or int(table_cfg.llm_start_ts or 0)
    end_ts = segment_end_ts or int(table_cfg.llm_end_ts or 0) or int(time.time())

    # checkpoint 续跑
    last = get_checkpoint(
        ck, job.meta.database, job_name=job.job_name,
        table_name=ckpt_key, mode="llm", cursor_field=table_cfg.milvus_time_field,
    )
    if last and last[0]:
        try:
            start_ts = int(str(last[0]))
        except Exception:  # noqa: BLE001
            pass

    logger.info(
        "flow_b start worker=%d table=%s ts=[%d, %d)",
        worker_id, table_cfg.target_collection, start_ts, end_ts,
    )

    fields = [f.name for f in col.schema.fields]
    limit = max(1, int(job.model.llm_batch_size))
    selector_field = table_cfg.prompt_routing.selector_field

    while True:
        expr = (
            f"{table_cfg.milvus_time_field} >= {start_ts} && "
            f"{table_cfg.milvus_time_field} < {end_ts} && "
            f'{table_cfg.milvus_llm_field} == ""'
        )
        rows = col.query(expr=expr, output_fields=fields, limit=limit, offset=0)
        if not rows:
            break

        _process_llm_batch(col, table_cfg, job, embedder, summarizer, rows, selector_field)

    _safe_flush(col)
    set_checkpoint(
        ck, job.meta.database, job_name=job.job_name,
        table_name=ckpt_key, mode="llm",
        cursor_field=table_cfg.milvus_time_field,
        last_cursor=str(end_ts), last_pk_values=[],
    )


# ── 批处理核心 ────────────────────────────────────────────────


def _process_llm_batch(
    col: Collection, cfg: TableConfig, job: JobConfig,
    embedder: Embedder, summarizer: Summarizer,
    rows: list[dict], selector_field: str,
) -> None:
    """处理一批 LLM 规范化：路由 → 总结 → 向量化 → upsert。"""
    t0 = time.monotonic()

    docs = [str(r.get(cfg.milvus_content_field) or "") for r in rows]
    selector_vals = [str(r.get(selector_field) or "") for r in rows]

    logger.info(
        "flow_b batch pick=%d selector_counts=%s",
        len(rows), dict(Counter(selector_vals)),
    )

    # 1. LLM 规范化（按 selector 路由，顺序处理，无线程池）
    t1 = time.monotonic()
    llm_out = _summarize_routed(
        summarizer, docs, selector_vals,
        cfg.prompt_routing.templates,
        cfg.prompt_routing.default_template,
        int(cfg.prompt_routing.max_chars),
    )
    t_llm = time.monotonic() - t1

    # 2. 后处理：空结果回退到原文，截断
    max_chars = int(cfg.prompt_routing.max_chars)
    llm_out = [
        (str(t or "").strip() or str(d or "").strip())[:max_chars]
        for t, d in zip(llm_out, docs)
    ]

    # 3. 重新向量化
    t2 = time.monotonic()
    vecs = embed_texts(embedder=embedder, texts=llm_out, batch_size=job.model.embedding_batch_size)
    t_emb = time.monotonic() - t2

    # 4. 逐条 upsert
    t3 = time.monotonic()
    ok, fail = 0, 0
    for i, r in enumerate(rows):
        pk = str(r.get(cfg.milvus_pk_field) or "")
        if not pk:
            continue
        entity = dict(r)
        entity[cfg.milvus_llm_field] = llm_out[i]
        entity[cfg.milvus_vector_field] = _normalize(vecs[i])
        try:
            upsert_entity(col=col, entity=entity)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            logger.warning("flow_b upsert failed pk=%s err=%s", pk, exc)
    t_up = time.monotonic() - t3

    elapsed = max(0.0001, time.monotonic() - t0)
    logger.info(
        "flow_b batch done size=%d ok=%d fail=%d qps=%.1f "
        "timing={llm:%.3f, emb:%.3f, up:%.3f}",
        len(rows), ok, fail, ok / elapsed, t_llm, t_emb, t_up,
    )


# ── LLM 路由 ─────────────────────────────────────────────────


def _summarize_routed(
    summarizer: Summarizer, docs: list[str], selector_vals: list[str],
    templates: dict[str, str], default_template: str, max_chars: int,
) -> list[str]:
    """按 selector 路由提示词，顺序分组调用 LLM（无线程池）。"""
    if not docs:
        return []

    groups: dict[str, list[int]] = {}
    for i, sv in enumerate(selector_vals):
        tpl = templates.get(sv) or default_template
        groups.setdefault(tpl, []).append(i)

    out = [""] * len(docs)
    for tpl, idxs in groups.items():
        sub_docs = [docs[i] for i in idxs]
        res = summarizer.summarize_batch(sub_docs, max_chars=max_chars, prompt_template=tpl)
        for j, i in enumerate(idxs):
            out[i] = res[j]

    return out


# ── 工具 ─────────────────────────────────────────────────────


def _normalize(v: list[float]) -> list[float]:
    if not v:
        return []
    s = sum(float(x) * float(x) for x in v)
    if s <= 0:
        return [0.0] * len(v)
    inv = 1.0 / math.sqrt(s)
    return [float(x) * inv for x in v]


def _ckpt_table_name(collection: str, worker_id: int) -> str:
    return collection if worker_id == 0 else f"{collection}:w{worker_id}"


def _safe_flush(col: Collection) -> None:
    try:
        col.flush()
    except Exception:  # noqa: BLE001
        pass
