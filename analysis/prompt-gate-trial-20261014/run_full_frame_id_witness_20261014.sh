#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangnuohua/medhorizon/video2-action-readout-20261013
exec /home/zhangnuohua/miniconda3/envs/vllm/bin/python -u \
  tools/run_visual_ovo_local.py \
  --history-read-mode fill \
  --answer-prompt frame_id_witness \
  --output results/ovo-frame-id-witness-fill-2gpu-20261014
