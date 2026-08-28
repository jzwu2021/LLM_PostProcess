#!/usr/bin/env bash
set -euo pipefail

ENV=/media/home/johnson/llm/qwen35-env
REPO=/media/home/johnson/workspace/LLM_PostProcess
SCRIPT=/media/home/johnson/llm/scripts/qwen35-9b/fine-tune-9b.py
MODEL=/media/home/johnson/llm/models/Qwen3.5-9B
EXP=$REPO/experiments/exp-002-20260828
TRAIN=$EXP/data/teacher_b_plus_repair_v0.1_train.jsonl
EVAL=$REPO/experiments/exp-001-20260827/data/teacher_b_native_heldout.jsonl
OUT=/media/home/johnson/llm/checkpoints/exp-002-20260828/repair-mix-v0.1
LOG=$EXP/artifacts/train_repair_mix_v0.1.log
PREFLIGHT=$EXP/artifacts/train_repair_mix_v0.1_preflight.txt

mkdir -p "$OUT" "$EXP/artifacts"
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE || true
export PATH="$ENV/bin:/home/johnson/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/home/johnson/miniforge3/lib:$ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

{
  echo '# exp-002-20260828 repair-mix-v0.1 preflight'
  date -Is
  echo '[git]'; git -C "$REPO" rev-parse HEAD
  echo '[dataset]'; sha256sum "$TRAIN" "$EVAL"
  echo '[gpu]'; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
  echo '[command]'
  printf '%q ' "$ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 "$SCRIPT" --model "$MODEL" --dataset "$TRAIN" --eval-dataset "$EVAL" --eval-examples 100 --output "$OUT" --max-steps 75 --save-steps 25 --seq-len 768 --grad-accum 4 --lr 2e-6
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
  --lr 2e-6 2>&1 | tee "$LOG"