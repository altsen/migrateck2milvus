"""模型抽象接口。

本模块只定义最小接口，便于在 pipeline 中替换不同模型实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """向量化模型接口。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。

        Args:
            texts: 文本列表。

        Returns:
            向量列表（与输入一一对应）。
        """


class Summarizer(ABC):
    """LLM 总结/规范化接口。"""

    @abstractmethod
    def summarize_batch(self, docs: list[str], *, max_chars: int, prompt_template: str) -> list[str]:
        """批量总结/规范化。

        Args:
            docs: 原文列表。
            max_chars: 单条输出目标最大字符数。
            prompt_template: 提示词模板（必须含 {MAX_CHARS} 与 {BATCH_DOCS} 占位符）。

        Returns:
            输出文本列表（与输入一一对应）。
        """
