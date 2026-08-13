# Experiment 2026-08-13: Qwen3.5-9B Base Tool-Call Baseline

## Status

`COMPLETE`. The final result is written to `results/base-toolcall-103.json` and the stdout/stderr trace is written to `logs/evaluate.log`.

This is Phase 1 baseline work. It is a generation-based tool-call smoke/baseline, not the full 500-record AI/LLM Infrastructure Model Domain Capability benchmark and not evidence of domain expertise.

## Objective

Measure the untrained local Qwen3.5-9B Base model on the fixed 103-case native tool-call suite before any new domain post-training. The result is a behavior baseline for later native-template masked SFT, preference optimization, and tool-use experiments.

Primary metrics:

- exact suite pass count;
- tool-name selection correctness;
- JSON/schema validity;
- no-tool behavior;
- multi-tool behavior;
- multi-round/error-recovery behavior;
- raw per-case response for audit.

The evaluation uses `temperature=0.0`, `max_tokens=256`, the native Qwen3.5 vLLM parser, and a localhost-only OpenAI-compatible endpoint.

## Fixed inputs

- Base model: `/media/home/johnson/llm/models/Qwen3.5-9B`
- Model name exposed to the evaluator: `qwen35-9b-base`
- Suite: `/media/home/johnson/llm/datasets/qwen35-native-fc/eval-suite.json`
- Suite size: 103 cases
- Evaluator: `/media/home/johnson/llm/scripts/qwen35-9b/evaluate-toolcall-suite.py`
- Repository commit before experiment: `b4c4d7717d2ecc87a6aac2df63c2655943621117`

## Hardware and environment

Recorded on `2026-08-13T17:05:23+00:00` before service startup:

- Host: Linux 5.15.0-125-generic
- GPUs: 8 x NVIDIA A30, 24576 MiB each
- NVIDIA driver: `580.173.02`
- Python: `3.12.9`
- PyTorch: `2.13.0+cu129`
- Transformers: `5.15.0`
- vLLM: `0.26.1rc1.dev659+g23f25aac1`
- Accelerate: `1.14.0`
- Datasets: `5.0.1`
- TRL: `1.9.2`
- PEFT: `0.20.0`
- Safetensors: `0.8.0`
- CUDA visible devices: `0,1,2,3,4,5,6,7`
- Model files: config and tokenizer present; 4 safetensors shards; 19,329,393,661 bytes across model directory files
- Disk at preflight: `/media` 73T total, 26T used, 44T available

## Service command

```bash
env -u CONDA_EXE -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL bash -lc '
  export PATH=/media/home/johnson/llm/qwen35-env/bin:$PATH
  export LD_LIBRARY_PATH=/home/johnson/miniforge3/lib:/media/home/johnson/llm/qwen35-env/lib/python3.12/site-packages/nvidia/cu13/lib:/media/home/johnson/llm/qwen35-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  exec /media/home/johnson/llm/qwen35-env/bin/vllm serve /media/home/johnson/llm/models/Qwen3.5-9B \
    --served-model-name qwen35-9b-base \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.82 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --host 127.0.0.1 \
    --port 8001
'
```

The `/home/johnson/miniforge3/lib` entry was required because the host Python environment's ICU dependency requires `CXXABI_1.3.15`, which is absent from the system `libstdc++`. The CUDA 13 paths were required for `libcudart.so.13`.

## Failed startup attempts and fixes

### Attempt 1: missing CUDA runtime path

Command used the vLLM executable without an explicit CUDA library path. Result:

```text
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

This attempt did not reach model loading and was not counted as an evaluation result.

### Attempt 2: CUDA paths only

After adding the virtual environment CUDA paths, startup reached Python imports but failed with:

```text
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

The fix was to add `/home/johnson/miniforge3/lib` and remove inherited Conda activation variables from the launch shell.

### Successful startup

The fixed command passed `/health` after approximately 230 seconds and returned:

```text
model=qwen35-9b-base
max_model_len=4096
```

## Evaluation command

```bash
mkdir -p experiments/2026-08-13-base-toolcall-baseline/results experiments/2026-08-13-base-toolcall-baseline/logs
export PATH=/media/home/johnson/llm/qwen35-env/bin:$PATH
export LD_LIBRARY_PATH=/home/johnson/miniforge3/lib:/media/home/johnson/llm/qwen35-env/lib/python3.12/site-packages/nvidia/cu13/lib:/media/home/johnson/llm/qwen35-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/media/home/johnson/llm/qwen35-env/bin/python /media/home/johnson/llm/scripts/qwen35-9b/evaluate-toolcall-suite.py \
  --suite /media/home/johnson/llm/datasets/qwen35-native-fc/eval-suite.json \
  --base http://127.0.0.1:8001/v1 \
  --model qwen35-9b-base \
  --output experiments/2026-08-13-base-toolcall-baseline/results/base-toolcall-103.json \
  2>&1 | tee experiments/2026-08-13-base-toolcall-baseline/logs/evaluate.log
```

## Result

The fresh evaluator completed successfully with exit code `0`:

```text
TOOLCALL_EVAL_DONE passed=53/103 output=experiments/2026-08-13-base-toolcall-baseline/results/base-toolcall-103.json
```

- Cases: `103`
- Passed: `53`
- Failed: `50`
- Exact pass rate: `51.4563%`
- Result JSON SHA-256: `9be661461e2e0767b2558254a995f7b03dc7582d645eedb32b6e1377c581eff9`
- Evaluation log SHA-256: `4b06daf074c5c28b5855a5191f10f41ba82b90873db9b563b504a4b1f68fb1b4`
- Suite SHA-256: `3d9fde756def503fe4901246602e84550939ef4f2eeb0b3b2d7acfbf1f73c020`
- Model config SHA-256: `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`

The evaluator did not emit wall-clock duration or per-request latency. Therefore runtime metrics are intentionally marked as not measured rather than reconstructed from timestamps. GPU memory sampled after the run was approximately 20.0 GiB per GPU, but this is an idle/post-evaluation snapshot and is not a peak-memory measurement.

## Interpretation rules

- This is a generation/parser/schema baseline, not a tool execution benchmark.
- Passing a tool-call case does not prove that an external tool was executed correctly.
- The score must not be used as evidence of AI/LLM Infrastructure domain knowledge.
- Later domain SFT experiments must use an independent held-out domain benchmark and report Model Domain Capability separately from Runtime/System Capability.
- Any training run must first pass the checkpoint save/load/export round-trip audit.
