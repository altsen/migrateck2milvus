"""项目配置（唯一入口）。

约束：
- 所有影响程序流程的配置必须在本文件出现
- 配置优先级：环境变量 > .env > config.py 默认

说明：
- v2 不处理“超长文本分流/待办队列”，LLM 阶段直接处理 Milvus 中已入库记录
- v2 Milvus 仅保留单一向量字段：LLM 规范化后会覆盖写入该向量字段
"""

from __future__ import annotations

import os
import json
from pathlib import Path

from ck2milvusv2.env_loader import load_dotenv
from ck2milvusv2.types import (
    ClickHouseConfig,
    DedupConfig,
    JobConfig,
    MetaDBConfig,
    MilvusConfig,
    ModelConfig,
    PromptRoutingRule,
    RuntimeConfig,
    SpecialNestedFlattenRule,
    TableConfig,
)


_REPO_ROOT = Path(__file__).resolve().parent

_DOTENV_PATH = os.environ.get("CK2MILVUSV2_DOTENV")
load_dotenv(_DOTENV_PATH or (_REPO_ROOT / ".env"), override=False)


def _env_str(key: str, default: str) -> str:
    """读取字符串环境变量。

    Args:
        key: 环境变量名。
        default: 缺省值。

    Returns:
        环境变量值或缺省值。
    """

    # 统一做 strip：避免 .env 的 CRLF 或误写空格导致字段名/枚举值出现隐形字符。
    return (os.environ.get(key, default) or "").strip()


def _env_int(key: str, default: int) -> int:
    """读取整型环境变量。

    Args:
        key: 环境变量名。
        default: 缺省值。

    Returns:
        整型值。
    """

    raw = os.environ.get(key)
    return default if raw is None or raw == "" else int(raw)


def _env_bool(key: str, default: bool) -> bool:
    """读取布尔环境变量。

    Args:
        key: 环境变量名。
        default: 缺省值。

    Returns:
        布尔值。
    """

    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_csv_list(key: str, default: str = "") -> list[str]:
    """读取逗号分隔列表。

    Args:
        key: 环境变量名。
        default: 缺省字符串。

    Returns:
        列表。
    """

    raw = os.environ.get(key, default)
    if raw is None:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _env_json_dict(key: str) -> dict[str, str]:
    """读取 JSON 字典环境变量。

    Args:
        key: 环境变量名，内容应为 JSON object（key/value 均为字符串）。

    Returns:
        dict。
    """

    raw = os.environ.get(key) or ""
    if not raw.strip():
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"{key} must be a JSON object")
    out: dict[str, str] = {}
    for k, v in obj.items():
        out[str(k)] = str(v)
    return out


def _read_text_file(path: str) -> str:
    """读取文本文件。

    Args:
        path: 文件路径。

    Returns:
        内容。
    """

    p = Path(path)
    return p.read_text(encoding="utf-8")


def _build_clickhouse() -> ClickHouseConfig:
    """构建 ClickHouse 连接配置。"""

    return ClickHouseConfig(
        host=_env_str("CK_HOST", "localhost"),
        port=_env_int("CK_PORT", 8123),
        database=_env_str("CK_DATABASE", "default"),
        username=_env_str("CK_USERNAME", "default"),
        password=_env_str("CK_PASSWORD", ""),
        secure=_env_bool("CK_SECURE", False),
    )


def _build_milvus() -> MilvusConfig:
    """构建 Milvus 连接配置。"""

    return MilvusConfig(
        host=_env_str("MILVUS_HOST", "localhost"),
        port=_env_int("MILVUS_PORT", 19530),
        user=os.environ.get("MILVUS_USER") or None,
        password=os.environ.get("MILVUS_PASSWORD") or None,
        db_name=os.environ.get("MILVUS_DB") or None,
    )


PROMPT_TEMPLATE_SUMMARY_BATCH_V1 = (
    "你是文本压缩器。将长文本压缩为短文本，保留关键信息。\n"
    "\n"
    "要求：\n"
    "1) 只基于原文，不编造信息\n"
    "2) 保留：核心观点、人物/组织/时间/地点/数字/事件\n"
    "3) 每条总结控制在 {MAX_CHARS} 字以内\n"
    "4) 输出纯文本，不要JSON/XML/代码块\n"
    "\n"
    "输入有多条语料，用 ===N=== 分隔。输出必须一一对应，也用 ===N=== 分隔。\n"
    "\n"
    "输入：\n"
    "{BATCH_DOCS}\n"
    "\n"
    "输出格式示例（3条语料）：\n"
    "===1===\n"
    "第一条的压缩结果...\n"
    "===2===\n"
    "第二条的压缩结果...\n"
    "===3===\n"
    "第三条的压缩结果...\n"
)


def _build_model() -> ModelConfig:
    """构建模型配置（仅保留 glm/local 入口）。"""

    legacy_mode = _env_str("MODEL_MODE", "glm")
    legacy_base_url = _env_str("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    return ModelConfig(
        mode=legacy_mode,
        base_url=legacy_base_url,
        embedding_mode=_env_str("EMBEDDING_MODE", legacy_mode),
        llm_mode=_env_str("LLM_MODE", legacy_mode),
        embedding_base_url=_env_str("EMBEDDING_BASE_URL", legacy_base_url),
        llm_base_url=_env_str("LLM_BASE_URL", legacy_base_url),
        api_key=(os.environ.get("GLM_API_KEY") or os.environ.get("API_KEY") or ""),
        llm_model=_env_str("LLM_MODEL", "glm-4-flashx"),
        embedding_model=_env_str("EMBEDDING_MODEL", "embedding-3"),
        embedding_dim=_env_int("EMBEDDING_DIM", 2048),
        timeout_seconds=_env_int("MODEL_TIMEOUT_SECONDS", 60),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
        llm_batch_size=_env_int("LLM_BATCH_SIZE", 8),
        llm_prompt_template=(os.environ.get("LLM_PROMPT_TEMPLATE") or PROMPT_TEMPLATE_SUMMARY_BATCH_V1),
        llm_max_chars=_env_int("LLM_MAX_CHARS", 800),
        temperature=float(_env_str("MODEL_TEMPERATURE", "0.2")),
    )


def _build_runtime() -> RuntimeConfig:
    """构建运行时配置（批量/并发/锁/重试）。"""

    return RuntimeConfig(
        ck_batch_size=_env_int("CK_BATCH_SIZE", 200),
        flow_a_workers=_env_int("FLOW_A_WORKERS", 1),
        flow_b_workers=_env_int("FLOW_B_WORKERS", 1),
        milvus_search_batch_size=_env_int("MILVUS_SEARCH_BATCH_SIZE", 64),
        milvus_insert_batch_size=_env_int("MILVUS_INSERT_BATCH_SIZE", 200),
        milvus_insert_min_batch_size=_env_int("MILVUS_INSERT_MIN_BATCH_SIZE", 1),
        milvus_insert_max_retries=_env_int("MILVUS_INSERT_MAX_RETRIES", 8),
        milvus_insert_retry_backoff_seconds=_env_int("MILVUS_INSERT_RETRY_BACKOFF_SECONDS", 1),
        milvus_insert_retry_backoff_max_seconds=_env_int("MILVUS_INSERT_RETRY_BACKOFF_MAX_SECONDS", 30),
        milvus_insert_flush_on_retry=_env_bool("MILVUS_INSERT_FLUSH_ON_RETRY", True),
        lock_ttl_seconds=_env_int("LOCK_TTL_SECONDS", 300),
        lock_heartbeat_seconds=_env_int("LOCK_HEARTBEAT_SECONDS", 15),
        lock_max_hold_seconds=_env_int("LOCK_MAX_HOLD_SECONDS", 0),
        llm_fail_threshold=_env_int("LLM_FAIL_THRESHOLD", 3),
        lookback_hours=_env_int("DEDUP_LOOKBACK_HOURS", 24),
    )


def _build_dedup() -> DedupConfig:
    """构建去重配置。"""

    return DedupConfig(
        batch_threshold=float(_env_str("DEDUP_BATCH_THRESHOLD", "0.98")),
        milvus_topk=_env_int("DEDUP_MILVUS_TOPK", 3),
        milvus_threshold=float(_env_str("DEDUP_MILVUS_THRESHOLD", "0.98")),
        keep_strategy=_env_str("DEDUP_KEEP_STRATEGY", "first_seen"),
    )


def _build_tables() -> list[TableConfig]:
    """构建表配置列表。

    说明：这里仅提供一个示例表。现场使用时建议直接在本函数中按需增减。
    """

    special = SpecialNestedFlattenRule(
        prefix=_env_str("SPECIAL_NESTED_PREFIX", "arguments"),
        fields={
            "id": _env_str("ARG_ID_FIELD", "argument_id"),
            "type": _env_str("ARG_TYPE_FIELD", "argument_type"),
            "name": _env_str("ARG_NAME_FIELD", "argument_name"),
        },
        align_by=_env_str("ARG_ALIGN_BY", "id"),
        empty_filter_field=_env_str("ARG_EMPTY_FILTER", "id"),
        value_filters={
            "type": _env_csv_list("ARG_TYPE_ALLOW", ""),
        },
        space_keys=[_env_str("ARG_ID_FIELD", "argument_id"), _env_str("ARG_TYPE_FIELD", "argument_type")],
        expand_index_field=_env_str("EXPAND_INDEX_FIELD", "expand_index"),
    )

    prompt_routing = PromptRoutingRule(
        selector_field=_env_str("LLM_SELECTOR_FIELD", "argument_type"),
        templates=_env_json_dict("LLM_TEMPLATES_JSON")
        if not (os.environ.get("LLM_TEMPLATES_FILE") or "").strip()
        else _env_json_dict_from_file(os.environ.get("LLM_TEMPLATES_FILE") or ""),
        default_template=os.environ.get("LLM_PROMPT_TEMPLATE") or PROMPT_TEMPLATE_SUMMARY_BATCH_V1,
        max_chars=_env_int("LLM_MAX_CHARS", 800),
    )

    t = TableConfig(
        source_table=_env_str("SOURCE_TABLE", "gtb_v1.event_share_data"),
        target_collection=_env_str("TARGET_COLLECTION", "event_share_data"),
        pk_fields=_env_csv_list("PK_FIELDS", "mongo_id") or ["mongo_id"],
        cursor_field=_env_str("CURSOR_FIELD", "create_time"),
        cursor_field_type=_env_str("CURSOR_FIELD_TYPE", "datetime"),
        vector_source_field=_env_str("VECTOR_FIELD", "content"),
        # 标量字段：覆盖示例表中的 Array/Nested 子列。
        # - `geo` 为 Array(Float64)
        # - `entity_list.id` 为 Nested 子列（物理类型为 Array(String)）
        scalar_fields=_env_csv_list(
            "SCALAR_FIELDS",
            "doc_id,type,subtype,title,publish_time,label,tone,geo,entity_list.id",
        ),
        # Nested/Array 字段名映射：ClickHouse Nested 子列通常形如 `entity_list.id`。
        # Milvus 字段名建议不包含 `.`，因此需要映射到 `entity_list_id`。
        scalar_field_mappings=(
            {"entity_list.id": "entity_list_id"}
            | _env_json_dict("SCALAR_FIELD_MAPPINGS_JSON")
        ),
        scalar_index_fields=_env_csv_list("SCALAR_INDEX_FIELDS", ""),
        special_nested=special,
        prompt_routing=prompt_routing,
        milvus_pk_field=_env_str("MILVUS_PK_FIELD", "pk"),
        milvus_content_field=_env_str("MILVUS_CONTENT_FIELD", "content"),
        milvus_llm_field=_env_str("MILVUS_LLM_FIELD", "llmcontent"),
        milvus_vector_field=_env_str("MILVUS_VECTOR_FIELD", "vector_content"),
        milvus_dup_count_field=_env_str("MILVUS_DUP_COUNT_FIELD", "dup_count"),
        milvus_time_field=_env_str("MILVUS_TIME_FIELD", "create_time_ts"),
        milvus_vector_metric=_env_str("MILVUS_VECTOR_METRIC", "IP"),
        full_start=_env_str("FULL_START", ""),
        full_end=_env_str("FULL_END", ""),
        llm_start_ts=_env_int("LLM_START_TS", 0),
        llm_end_ts=_env_int("LLM_END_TS", 0),
    )
    return [t]


def _env_json_dict_from_file(path: str) -> dict[str, str]:
    """从文件读取 JSON 字典。

    Args:
        path: 文件路径。

    Returns:
        dict。
    """

    if not path.strip():
        return {}
    raw = _read_text_file(path)
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("LLM_TEMPLATES_FILE must contain a JSON object")
    return {str(k): str(v) for k, v in obj.items()}


JOB = JobConfig(
    job_name=_env_str("JOB_NAME", "ck2milvusv2"),
    clickhouse=_build_clickhouse(),
    milvus=_build_milvus(),
    model=_build_model(),
    meta=MetaDBConfig(database=_env_str("META_DB", "ck2milvusv2_meta"), auto_create_tables=True),
    tables=_build_tables(),
    dedup=_build_dedup(),
    runtime=_build_runtime(),
)
