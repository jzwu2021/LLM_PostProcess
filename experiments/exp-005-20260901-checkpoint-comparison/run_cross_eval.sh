#!/usr/bin/env bash
set -euo pipefail

ENV=/media/home/johnson/llm/qwen35-env
REPO=/media/home/johnson/workspace/LLM_PostProcess
EXP=$REPO/experiments/exp-005-20260901-checkpoint-comparison

E2_HELDOUT=$REPO/experiments/exp-001-20260827/data/teacher_b_native_heldout.jsonl
E4_HELDOUT=$REPO/experiments/exp-004-20260901/data/aie_v04_heldout.jsonl
BENCH=$EXP/data/benchmark_as_messages.jsonl

CKPT_E2=/media/home/johnson/llm/checkpoints/exp-002-20260828/repair-mix-v0.1/step-75
CKPT_E4=/media/home/johnson/llm/checkpoints/exp-004-20260901/aie-v04/step-170

mkdir -p "$EXP/results" "$EXP/artifacts"
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE || true
export PATH="$ENV/bin:/home/johnson/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/home/johnson/miniforge3/lib:$ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

run_arm () {
  local label=$1 ckpt=$2
  local ckpt_arg=()
  [[ -n "$ckpt" ]] && ckpt_arg=(--checkpoint "$ckpt")
  "$ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 \
    "$EXP/eval_cross_loss.py" \
    "${ckpt_arg[@]}" \
    --label "$label" \
    --datasets "exp002_heldout=$E2_HELDOUT" "exp004_heldout=$E4_HELDOUT" "benchmark=$BENCH" \
    --examples 100 --seq-len 768 \
    --out "$EXP/results/${label}.json"
}

{
  echo "# exp-005 cross-evaluation preflight"; date -Is
  echo '[git]'; git -C "$REPO" rev-parse HEAD
  echo '[datasets]'; sha256sum "$E2_HELDOUT" "$E4_HELDOUT" "$BENCH"
  echo '[checkpoints]'; ls -d "$CKPT_E2" "$CKPT_E4"
} > "$EXP/artifacts/preflight.txt"

run_arm base ""
run_arm exp002_step75 "$CKPT_E2"
run_arm exp004_step170 "$CKPT_E4"

echo "ALL ARMS COMPLETE"
