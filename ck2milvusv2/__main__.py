"""ck2milvusv2 命令行入口。

提供：
- `init`：初始化 ClickHouse meta 表
- `run`：执行 one-shot（full/incremental/llm）
- `model-test`：独立验证 embedding/llm 适配
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback

from ck2milvusv2.logging_utils import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。

    Returns:
        argparse parser。
    """

    p = argparse.ArgumentParser(prog="ck2milvusv2", description="ClickHouse -> Milvus 迁移工具（v2）")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="初始化 ClickHouse meta 数据库/表").add_argument(
        "--no-drop",
        action="store_true",
        default=False,
        help="保留已有 meta 表（默认会先 DROP 再 CREATE）",
    )

    run = sub.add_parser("run", help="执行一次性任务（one-shot）：full/incremental/llm")
    run.add_argument("--mode", choices=["full", "incremental", "llm"], required=True)
    run.add_argument("--tables", default="", help="逗号分隔的表名/collection；为空表示全部")
    run.add_argument(
        "--checkpoint",
        choices=["resume", "restart"],
        default="resume",
        help="checkpoint 策略：resume=接着上次位置继续；restart=忽略并从头重跑（仅清空 checkpoint）",
    )

    mt = sub.add_parser("model-test", help="独立验证模型适配（不依赖迁移流程）")
    mt.add_argument("--kind", choices=["embed", "llm", "both"], default="both")
    mt.add_argument("--n", type=int, default=4, help="批量条数")
    mt.add_argument("--batch", type=int, default=4, help="单次请求批大小")
    mt.add_argument("--max-chars", type=int, default=200, help="LLM 单条摘要目标最大字符数")

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。

    Args:
        argv: 参数列表（不含程序名）。

    Returns:
        进程退出码。
    """

    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger("ck2milvusv2")

    # ── 延迟导入，确保 logging 就绪后才加载配置/模块 ──
    try:
        from ck2milvusv2.models.factory import build_models
        from ck2milvusv2.pipeline.runner import init_meta, run_mode
    except Exception:
        logger.error("模块导入失败，完整堆栈:\n%s", traceback.format_exc())
        return 1

    if args.cmd == "init":
        try:
            init_meta(drop=not args.no_drop)
            logger.info("meta init done (drop=%s)", not args.no_drop)
            return 0
        except Exception:
            logger.error("init 失败，完整堆栈:\n%s", traceback.format_exc())
            return 1

    if args.cmd == "run":
        try:
            table_filter = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None
            run_mode(mode=args.mode, table_filter=table_filter, checkpoint_strategy=str(args.checkpoint))
            return 0
        except Exception:
            logger.error("run 失败，完整堆栈:\n%s", traceback.format_exc())
            return 1

    if args.cmd == "model-test":
        try:
            import config

            embedder, summarizer = build_models(config.JOB.model)

            n = max(1, int(args.n))
            batch = max(1, int(args.batch))

            def _chunks(items: list[str], size: int):
                """按 size 分批切分列表。

                Args:
                    items: 输入列表。
                    size: 每批大小。

                Yields:
                    子列表分块。
                """

                for i in range(0, len(items), size):
                    yield items[i : i + size]

            if args.kind in {"embed", "both"}:
                texts = [f"样本{i+1}：用于embedding验证，含数字{i}与实体张三/李四。" for i in range(n)]
                out = []
                for ch in _chunks(texts, batch):
                    out.extend(embedder.embed(ch))
                dim = len(out[0]) if out else 0
                logger.info("model-test embed ok n=%d batch=%d dim=%d", n, batch, dim)

            if args.kind in {"llm", "both"}:
                docs = [
                    (
                        f"样本{i+1}：2026年2月6日，某地发生事件，涉及预算{i*10}万、周期{i+1}个月、负责人王五。"
                        + " 细节" * (30 + i)
                    )
                    for i in range(n)
                ]
                out2 = []
                for ch in _chunks(docs, batch):
                    out2.extend(summarizer.summarize_batch(ch, max_chars=int(args.max_chars)))
                non_empty = sum(1 for x in out2 if (x or "").strip())
                logger.info("model-test llm ok n=%d batch=%d non_empty=%d", n, batch, non_empty)

            return 0
        except Exception:
            logger.error("model-test 失败，完整堆栈:\n%s", traceback.format_exc())
            return 1

    logger.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
