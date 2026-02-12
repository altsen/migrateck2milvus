"""Milvus 客户端封装。"""

from __future__ import annotations

import logging

from pymilvus import Collection, connections, utility

from ck2milvusv2.types import MilvusConfig


logger = logging.getLogger(__name__)


class Milvus:
    """Milvus 连接管理器。

    目标：
    - 统一管理 pymilvus 连接 alias
    - 提供最小常用操作（has/get/drop）
    """

    def __init__(self, cfg: MilvusConfig, *, alias: str = "default"):
        """建立连接。

        Args:
            cfg: Milvus 配置。
            alias: pymilvus alias。
        """

        self._cfg = cfg
        self._alias = alias
        connections.connect(
            alias=alias,
            host=cfg.host,
            port=str(cfg.port),
            user=cfg.user,
            password=cfg.password,
            db_name=cfg.db_name,
        )

    @property
    def alias(self) -> str:
        """连接 alias。"""

        return self._alias

    def has_collection(self, name: str) -> bool:
        """判断 collection 是否存在。

        Args:
            name: collection 名。

        Returns:
            存在返回 True，否则 False。
        """

        return utility.has_collection(name, using=self._alias)

    def get_collection(self, name: str) -> Collection:
        """获取 collection（要求已存在）。

        Args:
            name: collection 名。

        Returns:
            Collection 对象。
        """

        return Collection(name, using=self._alias)

    def drop_collection(self, name: str) -> None:
        """删除 collection（若存在）。

        Args:
            name: collection 名。

        Returns:
            无。
        """

        if utility.has_collection(name, using=self._alias):
            utility.drop_collection(name, using=self._alias)
            logger.info("dropped collection %s", name)
