# ck2milvusv2

ClickHouse → Milvus 迁移工具（v2）。

核心能力：
- 流程A：去重入库（批内去重 + Milvus 回查去重 + 重复关系落业务表）
- 流程B：Milvus 原文整理（LLM 规范化 + 覆盖写回向量）

功能设计与实现口径以 [docs/ck2milvusv2_功能迭代与实现方案.md](docs/ck2milvusv2_功能迭代与实现方案.md) 为准。

更多说明：
- 使用说明：[docs/usage.md](docs/usage.md)
- 部署建议：[docs/deploy.md](docs/deploy.md)
- 校验清单：[docs/validation.md](docs/validation.md)

## 快速开始（本地）

1) Python 环境：
- `workon ck2milvus`
- `pip install -r requirements.txt`

2) 初始化 meta 表：
- `python -m ck2milvusv2 init`

3) 执行：
- 全量：`python -m ck2milvusv2 run --mode full`
- 增量：`python -m ck2milvusv2 run --mode incremental`
- LLM： `python -m ck2milvusv2 run --mode llm`

说明：生产可用 cron/systemd timer 定时触发 one-shot 入口命令。

## e2e（GLM）

前置：根目录 `.env` 内提供 `GLM_API_KEY` 或 `API_KEY`。

运行：
- `cd e2e && chmod +x run_e2e_glm.sh && ./run_e2e_glm.sh`
