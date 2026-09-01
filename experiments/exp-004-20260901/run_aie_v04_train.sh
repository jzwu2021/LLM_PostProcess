#!/usr/bin/env bash
set -euo pipefail

ENV=/media/home/johnson/llm/qwen35-env
REPO=/media/home/johnson/workspace/LLM_PostProcess
SCRIPT=/media/home/johnson/llm/scripts/qwen35-9b/fine-tune-9b.py
MODEL=/media/home/johnson/llm/models/Qwen3.5-9B
EXP=$REPO/experiments/exp-004-20260901
TRAIN=$EXP/data/aie_v04_train.jsonl
EVAL=$EXP/data/aie_v04_heldout.jsonl
OUT=/media/home/johnson/llm/checkpoints/exp-004-20260901/aie-v04
LOG=$EXP/artifacts/train_aie_v04.log
PREFLIGHT=$EXP/artifacts/train_aie_v04_preflight.txt

# 2700 records / (world 8 * grad_accum 4) = 85 steps per epoch; 170 = 2 epochs.
# seq-len 768 verified against the measured maximum of 510 tokens: 0 truncated.
MAX_STEPS=170
SAVE_STEPS=34
SEQ_LEN=768
GRAD_ACCUM=4
LR=2e-6

mkdir -p "$OUT" "$EXP/artifacts"
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE || true
export PATH="$ENV/bin:/home/johnson/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/home/johnson/miniforge3/lib:$ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

{
  echo '# exp-004-20260901 aie-v04 preflight'
  date -Is
  echo '[git]'; git -C "$REPO" rev-parse HEAD
  echo '[dataset]'; sha256sum "$TRAIN" "$EVAL"
  echo '[split]'; echo 'mechanism-disjoint: 90 train mechanisms, 10 held-out, 0 shared questions'
  echo '[seq-len check]'; echo 'measured max 510 tokens, seq-len 768, 0 records truncated'
  echo '[coverage]'; echo "2700 records / 32 per step = 85 steps per epoch; max-steps $MAX_STEPS = 2 epochs"
  echo '[gpu]'; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
  echo '[disk]'; df -h /media/home/johnson/llm | tail -1
  echo '[command]'
  printf '%q ' "$ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 "$SCRIPT" \
    --model "$MODEL" --dataset "$TRAIN" --eval-dataset "$EVAL" --eval-examples 100 \
    --output "$OUT" --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" \
    --seq-len "$SEQ_LEN" --grad-accum "$GRAD_ACCUM" --lr "$LR"
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
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --seq-len "$SEQ_LEN" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" 2>&1 | tee "$LOG"
