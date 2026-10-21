#!/usr/bin/env bash
# Qwen3-VL-8B-Instruct, same settings as OneStreamer (no retuning): main method without threshold,
# V-CAST storage. Only questions Qwen's own router labels SHORT or LONG (1,533) are run; NOW questions
# answer with Qwen's official recent-16 s path (analysis/qwen3vl8b-20261019/recent16-*.jsonl in the
# fill-full-context worktree), which is identical across storage modes.
set -euo pipefail

cd /home/zhangnuohua/medhorizon/video2-vcast-20261020
exec /home/zhangnuohua/miniconda3/envs/vllm/bin/python -u \
  tools/run_visual_ovo_local.py \
  --manifest analysis/vcast-20261020/manifest_qwen3vl8b_short_long.json \
  --model-path /mnt/truenas_nfs/dataset/znh1302/qwen3-VL-8B \
  --history-compression vcast --vcast-temperature 0.7 \
  --history-read-mode routed \
  --route-threshold 0 \
  --route-escalation SHORT \
  --memory-layout full_context \
  --output results/ovo-vcast-qwen3vl8b-short-long-2gpu-20261021 "$@"
