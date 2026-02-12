#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-}"
shift || true

if [[ -z "$VENV_DIR" ]]; then
  echo "Usage: $0 /path/to/venv <command...>" >&2
  exit 2
fi

ACTIVATE="$VENV_DIR/bin/activate"
if [[ ! -f "$ACTIVATE" ]]; then
  echo "Invalid venv dir: $VENV_DIR (missing $ACTIVATE)" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$ACTIVATE"

exec "$@"
