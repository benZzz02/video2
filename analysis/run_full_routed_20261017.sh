#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangnuohua/medhorizon/video2-fill-full-context-20261015
exec /home/zhangnuohua/miniconda3/envs/vllm/bin/python -u \
  tools/run_visual_ovo_local.py \
  --history-read-mode routed \
  --route-threshold 0.8 \
  --route-escalation SHORT \
  --memory-layout full_context \
  --output results/ovo-routed-vcheck-2gpu-20261017 "$@"
