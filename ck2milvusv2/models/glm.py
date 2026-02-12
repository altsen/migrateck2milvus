"""GLM（BigModel）适配器：embedding + 批量总结。"""

from __future__ import annotations

import logging

from ck2milvusv2.models.base import Embedder
from ck2milvusv2.models.batch_summarizer import BatchPromptSummarizer
from ck2milvusv2.types import ModelConfig
from ck2milvusv2.utils.http import http_post_json
from ck2milvusv2.utils.retry import retry


logger = logging.getLogger(__name__)


class GLMEmbedder(Embedder):
    """GLM embedding 适配器（类 OpenAI `/embeddings`）。"""

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

        if not self._cfg.api_key:
            raise RuntimeError("GLM api_key missing (set GLM_API_KEY or API_KEY)")

        url = self._cfg.embedding_base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}
        payload = {"model": self._cfg.embedding_model, "input": texts}

        def _call() -> list[list[float]]:
            obj = http_post_json(url, headers=headers, payload=payload, timeout=self._cfg.timeout_seconds)
            data = obj.get("data") or []
            data_sorted = sorted(data, key=lambda x: x.get("index", 0))
            return [d["embedding"] for d in data_sorted]

        return retry(_call, retries=3)


class GLMSummarizer(BatchPromptSummarizer):
    """GLM 批量总结器（`/chat/completions`）。"""

    def __init__(self, cfg: ModelConfig):
        """初始化。

        Args:
            cfg: 模型配置。
        """

        super().__init__(cfg)

    def _call_llm(self, prompt: str) -> str:
        """调用 GLM chat/completions。

        Args:
            prompt: prompt 文本。

        Returns:
            输出文本。
        """

        if not self._cfg.api_key:
            raise RuntimeError("GLM api_key missing (set GLM_API_KEY or API_KEY)")

        url = self._cfg.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}
        payload = {
            "model": self._cfg.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._cfg.temperature,
        }

        def _call() -> str:
            obj = http_post_json(url, headers=headers, payload=payload, timeout=self._cfg.timeout_seconds)
            choices = obj.get("choices") or []
            if not choices:
                raise RuntimeError(f"GLM chat empty choices: {obj}")
            msg = choices[0].get("message") or {}
            return str(msg.get("content") or "")

        return retry(_call, retries=3)
