#!/usr/bin/env bash
set -euo pipefail

ENV=/media/home/johnson/llm/qwen35-env
REPO=/media/home/johnson/workspace/LLM_PostProcess
SCRIPT=/media/home/johnson/llm/scripts/qwen35-9b/fine-tune-9b.py
MODEL=/media/home/johnson/llm/models/Qwen3.5-9B
EXP=$REPO/experiments/exp-001-20260827
TRAIN=$EXP/data/teacher_b_native_train.jsonl
EVAL=$EXP/data/teacher_b_native_heldout.jsonl
OUT=/media/home/johnson/llm/checkpoints/exp-001-20260827/stage1-train
LOG=$EXP/artifacts/train_stage1.log
PREFLIGHT=$EXP/artifacts/train_stage1_preflight.txt

mkdir -p "$OUT" "$EXP/artifacts"

unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE || true
export PATH="$ENV/bin:/home/johnson/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/home/johnson/miniforge3/lib:$ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

{
  echo '# exp-001-20260827 stage1 preflight'
  date -Is
  echo
  echo '[gpu]'
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
  echo
  echo '[compute]'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  echo
  echo '[command]'
  printf '%q ' "$ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 "$SCRIPT" --model "$MODEL" --dataset "$TRAIN" --eval-dataset "$EVAL" --eval-examples 100 --output "$OUT" --max-steps 75 --save-steps 25 --seq-len 768 --grad-accum 4 --lr 5e-6
  echo
} > "$PREFLIGHT"

cd "$REPO"

"$ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 \
  "$SCRIPT" \
  --model "$MODEL" \
  --dataset "$TRAIN" \
  --eval-dataset "$EVAL" \
  --eval-examples 100 \
  --output "$OUT" \
  --max-steps 75 \
  --save-steps 25 \
  --seq-len 768 \
  --grad-accum 4 \
  --lr 5e-6 2>&1 | tee "$LOG"
