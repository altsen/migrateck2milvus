"""模型工厂：按配置构造 Embedder + Summarizer。"""

from __future__ import annotations

from ck2milvusv2.models.base import Embedder, Summarizer
from ck2milvusv2.models.glm import GLMEmbedder, GLMSummarizer
from ck2milvusv2.models.local_embedder import LocalEmbedder
from ck2milvusv2.models.local_llm import LocalLLMSummarizer
from ck2milvusv2.types import ModelConfig


def build_models(cfg: ModelConfig) -> tuple[Embedder, Summarizer]:
    """构建模型实例。

    Args:
        cfg: 模型配置。

    Returns:
        (embedder, summarizer)。
    """

    embed_mode = (cfg.embedding_mode or cfg.mode or "glm").strip().lower()
    llm_mode = (cfg.llm_mode or cfg.mode or "glm").strip().lower()

    if embed_mode == "glm":
        embedder: Embedder = GLMEmbedder(cfg)
    elif embed_mode == "local":
        embedder = LocalEmbedder(cfg)
    else:
        raise ValueError(f"unsupported embedding_mode: {embed_mode}")

    if llm_mode == "glm":
        summarizer: Summarizer = GLMSummarizer(cfg)
    elif llm_mode == "local":
        summarizer = LocalLLMSummarizer(cfg)
    else:
        raise ValueError(f"unsupported llm_mode: {llm_mode}")

    return embedder, summarizer
