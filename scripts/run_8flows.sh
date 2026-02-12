#!/usr/bin/env bash
set -euo pipefail

# 一键运行 8 个流程（A/B × full/incr × endpoint on/off）。
# 约定：端点 on/off 通过 .env 中的 FULL_START/FULL_END 与 LLM_START_TS/LLM_END_TS 控制。
# 如需在同一份 .env 上自动切换端点，本脚本会在运行期间临时覆盖环境变量（不会改写 .env 文件）。

VENV_DIR="${1:-}"
if [[ -z "$VENV_DIR" ]]; then
  echo "Usage: $0 /path/to/venv" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

run_py() {
  "$ROOT_DIR/scripts/venv_run.sh" "$VENV_DIR" python -m ck2milvusv2 "$@"
}

# 端点 OFF：清空窗口变量
endpoint_off() {
  export FULL_START=""
  export FULL_END=""
  export LLM_START_TS="0"
  export LLM_END_TS="0"
}

# 端点 ON：使用 .env 中的窗口变量（不在这里写死）
endpoint_on() {
  : "${FULL_START:?FULL_START must be set in .env for endpoint-on cases}"
  : "${FULL_END:?FULL_END must be set in .env for endpoint-on cases}"
  : "${LLM_START_TS:?LLM_START_TS must be set in .env for endpoint-on cases}"
  : "${LLM_END_TS:?LLM_END_TS must be set in .env for endpoint-on cases}"
}

echo "== init meta =="
run_py init

echo "== CASE 1: A_full endpoint=off =="
endpoint_off
run_py run --mode full --checkpoint restart

echo "== CASE 2: A_full endpoint=on =="
endpoint_on
run_py run --mode full --checkpoint restart

echo "== CASE 3: A_incr endpoint=off =="
endpoint_off
run_py run --mode incremental --checkpoint resume

echo "== CASE 4: A_incr endpoint=on =="
# incremental 要强制使用 FULL_START/FULL_END，需清空 checkpoint
endpoint_on
run_py run --mode incremental --checkpoint restart

echo "== CASE 5: B_full endpoint=off =="
endpoint_off
run_py run --mode llm --checkpoint restart

echo "== CASE 6: B_full endpoint=on =="
endpoint_on
run_py run --mode llm --checkpoint restart

echo "== CASE 7: B_incr endpoint=off =="
endpoint_off
run_py run --mode llm --checkpoint resume

echo "== CASE 8: B_incr endpoint=on =="
# 为避免 checkpoint 覆盖手工窗口，这里同样使用 restart
endpoint_on
run_py run --mode llm --checkpoint restart

echo "ALL 8 FLOWS DONE"
