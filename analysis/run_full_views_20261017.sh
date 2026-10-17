#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangnuohua/medhorizon/video2-fill-full-context-20261015
exec /home/zhangnuohua/miniconda3/envs/vllm/bin/python -u \
  tools/run_visual_ovo_local.py \
  --history-read-mode views \
  --memory-layout full_context \
  --output results/ovo-two-views-2gpu-20261017 "$@"
