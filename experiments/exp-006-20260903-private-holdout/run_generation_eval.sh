#!/usr/bin/env bash
set -euo pipefail
ENV=/media/home/johnson/llm/qwen35-env
EXP=/media/home/johnson/workspace/LLM_PostProcess/experiments/exp-006-20260903-private-holdout
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE || true
export PATH="$ENV/bin:/home/johnson/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/home/johnson/miniforge3/lib:$ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/media/home/johnson/llm/hf-cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
arm () {
  local label=$1 path=$2
  echo "=== ARM $label ==="
  "$ENV/bin/python" "$EXP/generate_and_score.py" --model "$path" --label "$label" \
    --tp 8 --out "$EXP/results/${label}.json"
}
arm base           /media/home/johnson/llm/models/Qwen3.5-9B
arm exp002_step75  /media/home/johnson/llm/exports/exp002-step75
arm exp004_step170 /media/home/johnson/llm/exports/exp004-step170
echo "ALL GENERATION ARMS COMPLETE"
