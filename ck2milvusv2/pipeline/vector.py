"""embedding 批量封装（顺序分批，无线程池）。

并发说明：
- 本模块仅负责将文本列表按 batch_size 分批调用 embedder
- 不使用线程池——真正的并发在流程级（多 worker 并行处理不重叠数据段）
"""

from __future__ import annotations

from ck2milvusv2.models.base import Embedder


def embed_texts(*, embedder: Embedder, texts: list[str], batch_size: int) -> list[list[float]]:
    """批量向量化（顺序分批）。

    Args:
        embedder: embedding 模型实例。
        texts: 文本列表。
        batch_size: 单次 API 请求批大小。

    Returns:
        向量列表，与 texts 一一对应。
    """
    if not texts:
        return []
    bs = max(1, int(batch_size))
    out: list[list[float]] = []
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        vecs = embedder.embed(chunk)
        out.extend(vecs)
    return out
