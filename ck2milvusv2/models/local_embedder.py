"""本地 embedding 适配器骨架（非标准协议，可改造）。"""

from __future__ import annotations

import json
import logging

from ck2milvusv2.models.base import Embedder
from ck2milvusv2.types import ModelConfig
from ck2milvusv2.utils.http import http_post_raw
from ck2milvusv2.utils.retry import retry


logger = logging.getLogger(__name__)


class LocalEmbedder(Embedder):
    """本地 embedding 适配器（可改造骨架）。"""

    def __init__(self, cfg: ModelConfig):
        """初始化。

        Args:
            cfg: 模型配置。
        """

        self._cfg = cfg

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。

        Args:
            texts: 文本列表。

        Returns:
            向量列表。
        """

        if not texts:
            return []
        return retry(lambda: self._call_local_embedding(texts), retries=3)

    def _build_embedding_url(self) -> str:
        """构造 embedding URL。

        Returns:
            URL。
        """

        return self._cfg.embedding_base_url.rstrip("/") + "/embed"

    def _call_local_embedding(self, texts: list[str]) -> list[list[float]]:
        """调用本地 embedding 服务。

        Args:
            texts: 文本列表。

        Returns:
            向量列表。
        """

        url = self._build_embedding_url()
        headers: dict[str, str] = {}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"

        payload = {"texts": texts, "model": self._cfg.embedding_model}
        raw = http_post_raw(url, headers=headers, payload=payload, timeout=self._cfg.timeout_seconds)

        try:
            obj = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"local embedding response is not json: {raw[:200]}") from e

        if isinstance(obj, dict) and isinstance(obj.get("embeddings"), list):
            embs = obj["embeddings"]
            if embs and isinstance(embs[0], list):
                return [[float(x) for x in v] for v in embs]

        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            out: list[list[float]] = []
            for d in obj["data"]:
                if isinstance(d, dict) and isinstance(d.get("embedding"), list):
                    out.append([float(x) for x in d["embedding"]])
            if out:
                return out

        logger.warning("local embedding response json not recognized")
        raise RuntimeError("local embedding response json not recognized")
