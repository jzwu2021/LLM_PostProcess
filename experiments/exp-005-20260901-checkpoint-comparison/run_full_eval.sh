#!/usr/bin/env bash
set -euo pipefail
ENV=/media/home/johnson/llm/qwen35-env
REPO=/media/home/johnson/workspace/LLM_PostProcess
EXP=$REPO/experiments/exp-005-20260901-checkpoint-comparison
E2=$REPO/experiments/exp-001-20260827/data/teacher_b_native_heldout.jsonl
E4=$REPO/experiments/exp-004-20260901/data/aie_v04_heldout.jsonl
BENCH=$EXP/data/benchmark_as_messages.jsonl
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE || true
export PATH="$ENV/bin:/home/johnson/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/home/johnson/miniforge3/lib:$ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
run () {
  local label=$1 ckpt=$2; local arg=()
  [[ -n "$ckpt" ]] && arg=(--checkpoint "$ckpt")
  "$ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 "$EXP/eval_cross_loss.py" \
    "${arg[@]}" --label "$label" \
    --datasets "exp002_heldout=$E2" "exp004_heldout=$E4" "benchmark=$BENCH" \
    --examples 0 --seq-len 768 --out "$EXP/results_full/${label}.json"
}
run base ""
run exp002_step75 /media/home/johnson/llm/checkpoints/exp-002-20260828/repair-mix-v0.1/step-75
run exp004_step170 /media/home/johnson/llm/checkpoints/exp-004-20260901/aie-v04/step-170
echo "ALL ARMS COMPLETE"
