#!/usr/bin/env bash
set -euo pipefail

ENV=/media/home/johnson/llm/qwen35-env
EXPORT=/media/home/johnson/workspace/LLM_PostProcess/experiments/exp-006-20260903-private-holdout/export_fsdp_hf_fixed.py
DEST=/media/home/johnson/llm/exports

unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE || true
export PATH="$ENV/bin:/home/johnson/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/home/johnson/miniforge3/lib:$ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

mkdir -p "$DEST"

export_one () {
  local name=$1 ckpt=$2
  if [[ -f "$DEST/$name/config.json" ]]; then
    echo "SKIP $name already exported"
    return
  fi
  echo "EXPORTING $name from $ckpt"
  "$ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 \
    "$EXPORT" --checkpoint "$ckpt" --output "$DEST/$name"
  echo "EXPORTED $name"
}

export_one exp002-step75 /media/home/johnson/llm/checkpoints/exp-002-20260828/repair-mix-v0.1/step-75
export_one exp004-step170 /media/home/johnson/llm/checkpoints/exp-004-20260901/aie-v04/step-170

echo "ALL EXPORTS COMPLETE"
du -sh "$DEST"/*
