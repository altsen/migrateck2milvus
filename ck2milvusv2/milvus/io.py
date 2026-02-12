"""Milvus 常用 IO：search / insert / upsert（含降批重试）。"""

from __future__ import annotations

import logging
import time

from pymilvus import Collection

from ck2milvusv2.types import TableConfig


logger = logging.getLogger(__name__)


def _filter_entities_to_schema(*, col: Collection, entities: list[dict]) -> list[dict]:
    """将待写入实体裁剪为 collection schema 中存在的字段。

    Milvus（默认关闭 dynamic field）在插入包含未知字段的 dict 时会直接报错。
    为保证流程鲁棒性，这里对 entity 做一次字段裁剪。

    Args:
        col: collection。
        entities: entity dict 列表。

    Returns:
        裁剪后的 entity dict 列表（顺序不变）。
    """

    if not entities:
        return entities

    allowed = {f.name for f in col.schema.fields}
    dropped: set[str] = set()
    out: list[dict] = []
    for ent in entities:
        extra = [k for k in ent.keys() if k not in allowed]
        if extra:
            dropped.update(extra)
            out.append({k: v for k, v in ent.items() if k in allowed})
        else:
            out.append(ent)

    if dropped:
        sample = ",".join(sorted(dropped)[:12])
        logger.warning("milvus insert dropped unknown fields: %s", sample)
    return out


def find_duplicate_in_milvus(
    *,
    col: Collection,
    cfg: TableConfig,
    vector: list[float],
    space_values: dict[str, str],
    window_start: int,
    window_end: int,
    topk: int,
) -> dict | None:
    """在同 space + 时间窗口内检索最相似候选。

    Args:
        col: collection。
        cfg: 表配置。
        vector: 向量。
        space_values: 去重空间键字段值（字段名为已 flatten 后写入 Milvus 的标量字段名）。
        window_start: 窗口开始 ts。
        window_end: 窗口结束 ts。
        topk: topK。

    Returns:
        命中 dict 或 None。dict 至少包含：pk, similarity, dup_count。
    """

    if not vector:
        return None

    expr_parts: list[str] = []
    for k in cfg.special_nested.space_keys:
        v = space_values.get(k, "")
        vv = str(v).replace('\\', '\\\\').replace('"', '\\"')
        expr_parts.append(f"{k} == \"{vv}\"")

    expr_parts.append(f"{cfg.milvus_time_field} >= {int(window_start)}")
    expr_parts.append(f"{cfg.milvus_time_field} <= {int(window_end)}")
    expr = " && ".join(expr_parts)

    params = {"metric_type": (cfg.milvus_vector_metric or "IP").upper(), "params": {"ef": 64}}
    res = col.search(
        data=[vector],
        anns_field=cfg.milvus_vector_field,
        param=params,
        limit=max(1, int(topk)),
        expr=expr,
        output_fields=[cfg.milvus_pk_field, cfg.milvus_dup_count_field],
    )
    if not res or not res[0]:
        return None

    hit = res[0][0]
    ent = hit.entity
    pk = ent.get(cfg.milvus_pk_field)
    dup = ent.get(cfg.milvus_dup_count_field)
    return {"pk": pk, "dup_count": dup, "similarity": float(hit.distance)}


def adaptive_insert(
    *,
    col: Collection,
    entities: list[dict],
    batch_size: int,
    min_batch_size: int,
    max_retries: int,
    backoff_seconds: int,
    backoff_max_seconds: int,
    flush_on_retry: bool,
) -> None:
    """批量插入并在失败时降批重试。

    Args:
        col: collection。
        entities: entity dict 列表。
        batch_size: 初始批大小。
        min_batch_size: 最小批大小。
        max_retries: 最大重试次数。
        backoff_seconds: 退避基准。
        backoff_max_seconds: 最大退避。
        flush_on_retry: 是否在重试前 flush。

    Returns:
        无。
    """

    if not entities:
        return

    entities = _filter_entities_to_schema(col=col, entities=entities)

    bs = max(1, int(batch_size))
    min_bs = max(1, int(min_batch_size))
    i = 0
    while i < len(entities):
        chunk = entities[i : i + bs]
        attempt = 0
        while True:
            try:
                col.insert(chunk)
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                attempt += 1
                if flush_on_retry:
                    try:
                        col.flush()
                    except Exception:  # noqa: BLE001
                        pass

                if ("quota" in msg or "memory" in msg or "exceed" in msg or "too large" in msg) and bs > min_bs:
                    bs = max(min_bs, bs // 2)
                    logger.warning("milvus insert reduce batch to %d due to %s", bs, msg[:120])

                if attempt > int(max_retries):
                    raise

                sleep = min(float(backoff_max_seconds), float(backoff_seconds) * (2 ** (attempt - 1)))
                time.sleep(sleep)

        i += len(chunk)


def upsert_entity(*, col: Collection, entity: dict) -> None:
    """upsert 单条实体。

    Args:
        col: collection。
        entity: entity dict。

    Returns:
        无。
    """

    fn = getattr(col, "upsert", None)
    if callable(fn):
        fn([entity])
        return

    # 兜底：delete + insert
    filtered = _filter_entities_to_schema(col=col, entities=[entity])[0]
    pk_field = col.schema.primary_field.name
    pk = filtered.get(pk_field)
    if pk is None:
        raise RuntimeError("upsert fallback missing pk")
    try:
        col.delete(expr=f"{pk_field} == \"{pk}\"")
    except Exception:  # noqa: BLE001
        pass
    col.insert([filtered])
