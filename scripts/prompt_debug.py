#!/usr/bin/env python3
"""提示词效果调试脚本 —— 部署前验证 LLM 摘要质量。

功能：
- 从文件读取语料（支持批量，指定分隔符分割多条）
- 调用项目配置中的模型（glm / local），执行摘要
- 对比展示：原文 vs 生成摘要
- 支持自定义提示词模板（文件 / 命令行）
- 支持自定义 max_chars、batch_size

用法示例：
  # 单文件，默认分隔符 ===
  python scripts/prompt_debug.py -f samples.txt

  # 指定分隔符为 ---
  python scripts/prompt_debug.py -f samples.txt --sep '---'

  # 指定自定义提示词模板文件
  python scripts/prompt_debug.py -f samples.txt --template my_prompt.txt

  # 直接传入文本（多条用分隔符分隔）
  python scripts/prompt_debug.py -t "第一条文本===第二条文本===第三条文本"

  # 调整 max_chars 和 batch_size
  python scripts/prompt_debug.py -f samples.txt --max-chars 400 --batch-size 4

  # 使用 stdin
  cat samples.txt | python scripts/prompt_debug.py --sep '---'

注意：脚本依赖项目 config.py 和 .env 配置。请在项目根目录下运行。
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time

# 确保项目根目录在 sys.path 中
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ck2milvusv2.logging_utils import setup_logging  # noqa: E402
from ck2milvusv2.models.factory import build_models  # noqa: E402


# ── 常量 ──────────────────────────────────────────────────────

DEFAULT_SEP = "==="
DISPLAY_WIDTH = 80
DIVIDER = "─" * DISPLAY_WIDTH


# ── CLI ───────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        argparse parser。
    """

    p = argparse.ArgumentParser(
        prog="prompt_debug",
        description="提示词效果调试 —— 对比原文与 LLM 生成摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例：
              python scripts/prompt_debug.py -f samples.txt
              python scripts/prompt_debug.py -t "文本1===文本2" --sep "==="
              cat data.txt | python scripts/prompt_debug.py --sep "---"
        """),
    )
    p.add_argument("-f", "--file", help="语料文件路径（多条用分隔符分割）")
    p.add_argument("-t", "--text", help="直接传入文本（多条用分隔符分割）")
    p.add_argument("--sep", default=DEFAULT_SEP, help=f"多条语料的分隔符（默认 '{DEFAULT_SEP}'）")
    p.add_argument("--template", help="自定义提示词模板文件路径（须含 {{MAX_CHARS}} 和 {{BATCH_DOCS}} 占位符）")
    p.add_argument("--max-chars", type=int, default=0, help="单条摘要目标最大字符数（0=使用配置默认值）")
    p.add_argument("--batch-size", type=int, default=0, help="每批条数（0=使用配置默认值）")
    p.add_argument("--log-level", default="WARNING", help="日志级别（默认 WARNING，静默模式）")
    p.add_argument("--verbose", "-v", action="store_true", help="显示详细信息（等同 --log-level=INFO）")
    return p


# ── 语料加载 ──────────────────────────────────────────────────


def _load_docs(args: argparse.Namespace) -> list[str]:
    """从文件 / 命令行 / stdin 加载语料。

    Args:
        args: 已解析的命令行参数。

    Returns:
        非空语料文本列表。
    """

    raw = ""
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            raw = fh.read()
    elif args.text:
        raw = args.text
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        print("错误：请指定 -f/--file 或 -t/--text，或从 stdin 输入", file=sys.stderr)
        sys.exit(2)

    sep = args.sep
    docs = [d.strip() for d in raw.split(sep) if d.strip()]
    if not docs:
        print("错误：未提取到有效语料", file=sys.stderr)
        sys.exit(2)
    return docs


def _load_template(path: str) -> str:
    """从文件加载提示词模板。

    Args:
        path: 模板文件路径。

    Returns:
        提示词模板文本。
    """

    with open(path, encoding="utf-8") as fh:
        tpl = fh.read()
    if "{BATCH_DOCS}" not in tpl:
        print(f"警告：模板文件缺少 {{BATCH_DOCS}} 占位符: {path}", file=sys.stderr)
    if "{MAX_CHARS}" not in tpl:
        print(f"警告：模板文件缺少 {{MAX_CHARS}} 占位符: {path}", file=sys.stderr)
    return tpl


# ── 对比展示 ──────────────────────────────────────────────────


def _display_comparison(docs: list[str], results: list[str], elapsed: float) -> None:
    """打印原文与摘要的对比展示。

    Args:
        docs: 原文列表。
        results: 摘要结果列表。
        elapsed: 总耗时（秒）。
    """

    n = len(docs)
    print(f"\n{'═' * DISPLAY_WIDTH}")
    print(f"  提示词调试结果  |  共 {n} 条  |  耗时 {elapsed:.1f}s")
    print(f"{'═' * DISPLAY_WIDTH}\n")

    for i, (orig, summ) in enumerate(zip(docs, results), 1):
        print(f"[{i}/{n}] {DIVIDER}")

        print(f"  【原文】({len(orig)} 字)")
        for line in orig.splitlines():
            print(f"    {line}")

        print()
        summ_text = (summ or "").strip()
        if summ_text:
            ratio = len(summ_text) / max(len(orig), 1) * 100
            print(f"  【摘要】({len(summ_text)} 字, 压缩率 {ratio:.0f}%)")
            for line in summ_text.splitlines():
                print(f"    {line}")
        else:
            print("  【摘要】(空)")

        print()

    print(DIVIDER)
    non_empty = sum(1 for s in results if (s or "").strip())
    avg_ratio = 0.0
    if non_empty:
        ratios = [
            len((s or "").strip()) / max(len(d), 1)
            for d, s in zip(docs, results)
            if (s or "").strip()
        ]
        avg_ratio = sum(ratios) / len(ratios) * 100 if ratios else 0.0

    print(f"  汇总: 总计={n}  成功={non_empty}  失败={n - non_empty}  平均压缩率={avg_ratio:.0f}%  耗时={elapsed:.1f}s")
    print(DIVIDER)


# ── 主流程 ────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。

    Args:
        argv: 命令行参数（不含程序名）。

    Returns:
        进程退出码。
    """

    args = _build_parser().parse_args(argv)

    log_level = "INFO" if args.verbose else args.log_level
    setup_logging(log_level)

    # 延迟 import config，确保 .env 已加载
    import config

    docs = _load_docs(args)
    print(f"已加载 {len(docs)} 条语料")

    # 构建模型
    _, summarizer = build_models(config.JOB.model)

    # 确定参数
    max_chars = args.max_chars or config.JOB.model.llm_max_chars
    batch_size = args.batch_size or config.JOB.model.llm_batch_size

    # 提示词模板
    if args.template:
        prompt_template = _load_template(args.template)
    else:
        prompt_template = config.JOB.model.llm_prompt_template
    print(f"模型模式: {config.JOB.model.mode}  max_chars={max_chars}  batch_size={batch_size}")

    # 分批调用
    results: list[str] = []
    t0 = time.monotonic()
    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        print(f"  处理批次 {i // batch_size + 1} ({len(chunk)} 条)...", end="", flush=True)
        bt = time.monotonic()
        out = summarizer.summarize_batch(chunk, max_chars=max_chars, prompt_template=prompt_template)
        print(f" {time.monotonic() - bt:.1f}s")
        results.extend(out)
    elapsed = time.monotonic() - t0

    # 展示对比
    _display_comparison(docs, results, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
