"""Milvus schema 创建与索引策略。

原则：
- 默认不为标量字段创建索引（避免影响大规模写入性能）
- 仅对显式配置的标量字段创建索引（用于过滤）
"""

from __future__ import annotations

import logging

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

from ck2milvusv2.types import TableConfig


logger = logging.getLogger(__name__)


def ensure_collection(
    *,
    collection_name: str,
    cfg: TableConfig,
    dim: int,
    using: str,
) -> Collection:
    """确保 collection 存在。

    Args:
        collection_name: collection 名。
        cfg: 表配置。
        dim: 向量维度。
        using: pymilvus alias。

    Returns:
        Collection。
    """

    from pymilvus import utility

    if utility.has_collection(collection_name, using=using):
        return Collection(collection_name, using=using)

    fields: list[FieldSchema] = []
    fields.append(
        FieldSchema(
            name=cfg.milvus_pk_field,
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=512,
        )
    )

    fields.append(FieldSchema(name=cfg.milvus_content_field, dtype=DataType.VARCHAR, max_length=65535))
    fields.append(FieldSchema(name=cfg.milvus_llm_field, dtype=DataType.VARCHAR, max_length=65535))
    fields.append(FieldSchema(name=cfg.milvus_dup_count_field, dtype=DataType.INT64))
    fields.append(FieldSchema(name=cfg.milvus_time_field, dtype=DataType.INT64))

    defined_names = {f.name for f in fields}

    # 特殊 Nested flatten 后的标量字段（如 argument_id/argument_type/argument_name 等）
    # 这些字段会参与入库与审计，因此必须出现在 schema 中。
    for f in (cfg.special_nested.fields or {}).values():
        if not f:
            continue
        if f in {cfg.milvus_pk_field, cfg.milvus_content_field, cfg.milvus_llm_field}:
            continue
        if f in defined_names:
            continue
        fields.append(FieldSchema(name=f, dtype=DataType.VARCHAR, max_length=65535))
        defined_names.add(f)

    # 去重空间字段（来自 flatten 后）
    for sk in cfg.special_nested.space_keys:
        if sk in defined_names:
            continue
        fields.append(FieldSchema(name=sk, dtype=DataType.VARCHAR, max_length=512))
        defined_names.add(sk)

    # 展开序号字段（用于审计/排查）
    ex_idx = cfg.special_nested.expand_index_field
    if ex_idx and ex_idx not in defined_names:
        fields.append(FieldSchema(name=ex_idx, dtype=DataType.INT64))
        defined_names.add(ex_idx)

    # 标量字段（统一用 VARCHAR，避免 ClickHouse→Milvus 类型错配）
    # 说明：Nested/Array 字段在入库前会被序列化为 JSON 字符串。
    for ck_name in cfg.scalar_fields:
        f = (cfg.scalar_field_mappings or {}).get(ck_name, ck_name)
        if "." in f:
            raise ValueError(
                f"invalid milvus field name '{f}' (from '{ck_name}'); "
                "please set SCALAR_FIELD_MAPPINGS_JSON to map it to a name without '.'"
            )

        if f in {cfg.milvus_pk_field, cfg.milvus_content_field, cfg.milvus_llm_field}:
            continue
        if f in defined_names:
            continue
        fields.append(FieldSchema(name=f, dtype=DataType.VARCHAR, max_length=65535))
        defined_names.add(f)

    fields.append(FieldSchema(name=cfg.milvus_vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim))

    schema = CollectionSchema(fields, description=f"ck2milvusv2 {collection_name}")
    col = Collection(collection_name, schema=schema, using=using)

    # 向量索引
    metric = (cfg.milvus_vector_metric or "IP").upper()
    index_params = {
        "metric_type": metric,
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    col.create_index(cfg.milvus_vector_field, index_params=index_params)

    # 标量索引（可选）
    for f in cfg.scalar_index_fields:
        if f == cfg.milvus_vector_field:
            continue
        try:
            col.create_index(f, index_params={"index_type": "INVERTED"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("create scalar index failed field=%s err=%s", f, exc)

    col.load()
    logger.info("created collection %s", collection_name)
    return col
