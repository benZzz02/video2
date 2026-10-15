#!/usr/bin/env bash
# Usage: run_shard.sh <a|b> <GPU-UUID> [--dry-run]
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TAG=$1; GPU=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/home/zhangnuohua/medhorizon/OneStreamer/.runtime/ovo \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
exec /home/zhangnuohua/miniconda3/envs/vllm/bin/python -u \
  "$ROOT/Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py" \
  --manifest "$HERE/samples-$TAG.json" --model-path /mnt/truenas_nfs/dataset/znh1302/OneStreamer \
  --output "$HERE/shard-$TAG" --recent-frames 16 --history-compression budget_pool \
  --history-store-budget context --history-store-tokens 131072 --history-capacity 131072 \
  --history-store-mib 4096 --stream-include-end --history-read-mode fill --context-tokens 131072 \
  --attn-implementation sdpa "$@"
