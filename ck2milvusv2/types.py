"""类型与配置定义（纯数据）。

约束：本文件只包含 dataclass，不包含业务逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClickHouseConfig:
    """ClickHouse 连接配置。"""

    host: str
    port: int
    database: str
    username: str
    password: str
    secure: bool = False


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus 连接配置。"""

    host: str
    port: int
    user: str | None = None
    password: str | None = None
    db_name: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """模型调用配置（embedding + LLM）。"""

    mode: str
    base_url: str
    embedding_mode: str
    llm_mode: str
    embedding_base_url: str
    llm_base_url: str
    api_key: str
    llm_model: str
    embedding_model: str
    embedding_dim: int
    timeout_seconds: int
    embedding_batch_size: int
    llm_batch_size: int
    llm_prompt_template: str
    llm_max_chars: int
    temperature: float = 0.2


@dataclass(frozen=True)
class SpecialNestedFlattenRule:
    """特殊 Nested 字段 flatten 规则。

    说明：
    - prefix 对应 ClickHouse 列名前缀（如 arguments）
    - fields 为二级字段映射：id/type/name → milvus 标量字段名
    - align_by/empty_filter_field 用于“排空筛选”
    - value_filters 支持值集合过滤（仅内容范围过滤）
    - space_keys 为去重空间键（必须是 flatten 后的标量字段名）
    """

    prefix: str
    fields: dict[str, str]
    align_by: str
    empty_filter_field: str
    value_filters: dict[str, list[str]]
    space_keys: list[str]
    expand_index_field: str = "expand_index"


@dataclass(frozen=True)
class PromptRoutingRule:
    """LLM 动态提示词路由规则。"""

    selector_field: str
    templates: dict[str, str]
    default_template: str
    max_chars: int


@dataclass(frozen=True)
class TableConfig:
    """单表迁移配置。"""

    source_table: str
    target_collection: str
    pk_fields: list[str]
    cursor_field: str
    vector_source_field: str
    scalar_fields: list[str]
    special_nested: SpecialNestedFlattenRule
    prompt_routing: PromptRoutingRule

    # Milvus 字段名（可配置映射）
    milvus_pk_field: str
    milvus_content_field: str
    milvus_llm_field: str
    milvus_vector_field: str
    milvus_dup_count_field: str
    milvus_time_field: str

    # cursor 字段类型：
    # - datetime: ClickHouse DateTime/DateTime64，where 条件使用 toDateTime()
    # - number: Int/Float 等数值类型，where 条件直接数值比较
    cursor_field_type: str = "datetime"

    # ClickHouse 标量字段名 -> Milvus 字段名映射。
    # 典型场景：Nested 子列名形如 `entity_list.id`，Milvus 不建议使用带 `.` 的字段名。
    scalar_field_mappings: dict[str, str] = field(default_factory=dict)
    milvus_vector_metric: str = "IP"  # IP / COSINE / L2

    # 写入性能：默认不加标量索引；仅对显式指定字段建索引
    scalar_index_fields: list[str] = field(default_factory=list)

    # full 模式支持手工指定起止（空字符串表示不限制）
    full_start: str = ""
    full_end: str = ""

    # llm 模式支持手工指定起止时间戳（0 表示不限制）
    llm_start_ts: int = 0
    llm_end_ts: int = 0


@dataclass(frozen=True)
class DedupConfig:
    """去重配置（批内 + 库内）。"""

    batch_threshold: float
    milvus_topk: int
    milvus_threshold: float
    keep_strategy: str = "first_seen"  # first_seen / earliest_by_cursor


@dataclass(frozen=True)
class RuntimeConfig:
    """运行时批量/并发/锁/重试参数。"""

    ck_batch_size: int
    flow_a_workers: int
    flow_b_workers: int
    milvus_search_batch_size: int
    milvus_insert_batch_size: int
    milvus_insert_min_batch_size: int
    milvus_insert_max_retries: int
    milvus_insert_retry_backoff_seconds: int
    milvus_insert_retry_backoff_max_seconds: int
    milvus_insert_flush_on_retry: bool
    lock_ttl_seconds: int
    lock_heartbeat_seconds: int
    lock_max_hold_seconds: int
    llm_fail_threshold: int
    lookback_hours: int


@dataclass(frozen=True)
class MetaDBConfig:
    """元数据库配置（ClickHouse）。"""

    database: str
    auto_create_tables: bool = True


@dataclass(frozen=True)
class JobConfig:
    """任务总配置。"""

    job_name: str
    clickhouse: ClickHouseConfig
    milvus: MilvusConfig
    model: ModelConfig
    meta: MetaDBConfig
    tables: list[TableConfig]
    dedup: DedupConfig
    runtime: RuntimeConfig
