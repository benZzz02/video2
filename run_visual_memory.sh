#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${EVAL_PYTHON:-python}" -u \
  "$PACKAGE_DIR/Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py" \
  --recent-frames 16 \
  --history-compression progressive_pool \
  --history-pool-stride 2 \
  --history-pool-age-seconds 60 \
  --history-read-mode fill \
  --context-tokens 32768 \
  "$@"
