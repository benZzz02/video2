#!/usr/bin/env bash
# V-CAST storage under the fixed main method (no threshold). Only questions the router labels
# SHORT or LONG (1,894) are run; NOW questions answer with the official recent-16 s path and are
# identical across storage modes. Labels come from the deterministic text-only router.
set -euo pipefail

cd /home/zhangnuohua/medhorizon/video2-vcast-20261020
exec /home/zhangnuohua/miniconda3/envs/vllm/bin/python -u \
  tools/run_visual_ovo_local.py \
  --manifest analysis/vcast-20261020/manifest_short_long.json \
  --history-compression vcast --vcast-temperature 0.7 \
  --history-read-mode routed \
  --route-threshold 0 \
  --route-escalation SHORT \
  --memory-layout full_context \
  --output results/ovo-vcast-succ-short-long-2gpu-20261020 "$@"
