"""本地 LLM 适配器骨架（非标准协议，可改造）。"""

from __future__ import annotations

import json
import logging

from ck2milvusv2.models.batch_summarizer import BatchPromptSummarizer
from ck2milvusv2.types import ModelConfig
from ck2milvusv2.utils.http import http_post_raw
from ck2milvusv2.utils.retry import retry


logger = logging.getLogger(__name__)


class LocalLLMSummarizer(BatchPromptSummarizer):
    """本地 LLM 总结器（可改造骨架）。"""

    def __init__(self, cfg: ModelConfig):
        """初始化。

        Args:
            cfg: 模型配置。
        """

        super().__init__(cfg)

    def _build_llm_url(self) -> str:
        """构造 LLM URL。

        Returns:
            URL。
        """

        return self._cfg.llm_base_url.rstrip("/") + "/chat"

    def _call_llm(self, prompt: str) -> str:
        """调用本地 LLM 并返回纯文本。

        Args:
            prompt: prompt 文本。

        Returns:
            输出文本。
        """

        url = self._build_llm_url()
        headers: dict[str, str] = {}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"

        payload = {
            "prompt": prompt,
            "max_chars": int(self._cfg.llm_max_chars),
            "temperature": self._cfg.temperature,
        }

        def _call() -> str:
            raw = http_post_raw(url, headers=headers, payload=payload, timeout=self._cfg.timeout_seconds)
            try:
                obj = json.loads(raw)
            except Exception:  # noqa: BLE001
                return raw

            if isinstance(obj, dict):
                if isinstance(obj.get("choices"), list) and obj["choices"]:
                    c0 = obj["choices"][0]
                    if isinstance(c0, dict):
                        msg = c0.get("message")
                        if isinstance(msg, dict) and msg.get("content"):
                            return str(msg["content"])
                        if c0.get("text"):
                            return str(c0["text"])
                if obj.get("content"):
                    return str(obj["content"])
                if obj.get("text"):
                    return str(obj["text"])

            logger.warning("local llm response json not recognized, fallback to raw")
            return raw

        return retry(_call, retries=3)
