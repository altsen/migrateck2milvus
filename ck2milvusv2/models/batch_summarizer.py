"""批量总结器基类（统一 prompt 构造与 ===N=== 解析）。"""

from __future__ import annotations

import logging
import re

from ck2milvusv2.models.base import Summarizer
from ck2milvusv2.types import ModelConfig


logger = logging.getLogger(__name__)


_SPLIT_RE = re.compile(r"^===([0-9]+)===$", re.MULTILINE)


class BatchPromptSummarizer(Summarizer):
    """统一的批量 prompt + 解析实现。

子类只需实现 `_call_llm(prompt) -> str`。
"""

    def __init__(self, cfg: ModelConfig):
        """初始化。

        Args:
            cfg: 模型配置。
        """

        self._cfg = cfg

    def summarize_batch(self, docs: list[str], *, max_chars: int, prompt_template: str) -> list[str]:
        """批量总结/规范化。

        Args:
            docs: 文本列表。
            max_chars: 单条输出最大字符数。
            prompt_template: prompt 模板。

        Returns:
            输出列表。
        """

        if not docs:
            return []

        prompt = self._build_prompt(docs, max_chars=max_chars, prompt_template=prompt_template)
        raw = self._call_llm(prompt)
        out = self._parse_batched_output(raw, n=len(docs))
        return out

    def _build_prompt(self, docs: list[str], *, max_chars: int, prompt_template: str) -> str:
        """构建批量 prompt。

        Args:
            docs: 文本列表。
            max_chars: 单条输出目标最大字符数。
            prompt_template: prompt 模板。

        Returns:
            prompt 文本。
        """

        parts: list[str] = []
        for i, d in enumerate(docs, start=1):
            parts.append(f"==={i}===\n{d}")
        batch_docs = "\n".join(parts)
        return (
            prompt_template.replace("{MAX_CHARS}", str(int(max_chars))).replace("{BATCH_DOCS}", batch_docs)
        )

    def _parse_batched_output(self, text: str, *, n: int) -> list[str]:
        """解析 `===N===` 批量输出。

        Args:
            text: 模型输出文本。
            n: 期望条数。

        Returns:
            解析后的输出列表。
        """

        t = (text or "").strip()
        if not t:
            return [""] * n

        # 使用分隔符定位段落
        matches = list(_SPLIT_RE.finditer(t))
        if not matches:
            # 兜底：单段输出，复制/截断到 n 条
            logger.warning("llm output has no separators, fallback to single")
            return [t] + ([""] * (n - 1))

        items: dict[int, str] = {}
        for idx, m in enumerate(matches):
            try:
                k = int(m.group(1))
            except Exception:  # noqa: BLE001
                continue
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(t)
            items[k] = t[start:end].strip()

        out: list[str] = []
        for i in range(1, n + 1):
            out.append(items.get(i, ""))
        return out

    def _call_llm(self, prompt: str) -> str:
        """子类实现：调用 LLM 并返回文本。

        Args:
            prompt: prompt 文本。

        Returns:
            模型输出文本。
        """

        raise NotImplementedError
