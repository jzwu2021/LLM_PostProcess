import json

d = json.load(open('/tmp/tb_next.json'))
items = d['items']
assert d['lane'] == 'train' and d['batch'] == '0109'

COMMON = """0. Scope and non-claims
This is a serving-system measurement protocol only. It says nothing about model
quality or domain capability. No number transfers across engine commit, weights
hash, quantization scheme, scheduler/batching policy, sequence-length regime,
tenancy, or GPU SKU. Every reported number must carry its full config tuple.

1. Frozen setup (if unrecorded, the run is void)
- Engine + commit, launch flags (max_num_seqs, max_num_batched_tokens,
  gpu_memory_utilization, chunked-prefill on/off, TP/PP degree), weights hash,
  quantization, tokenizer version, CUDA/driver, container digest.
- Hardware: GPU SKU and count (e.g. 8x A30 24GB), NVLink/PCIe topology from
  `nvidia-smi topo -m`, host NUMA pinning, NIC/RDMA config if multi-node.
- Client: closed-loop vs open-loop. Use OPEN-LOOP (Poisson arrivals) for latency
  SLO claims; closed-loop concurrency sweeps only for saturation throughput.
  Closed-loop hides queueing delay and will silently understate P99.

2. Workload definition (must be fixed and versioned)
- Two-mode mixture: short-prompt (e.g. 128 +/- 32 in-tok, 512 out-tok) and
  long-generation (e.g. 2048 in-tok, 2048 out-tok), mixing ratio r fixed per run
  (sweep r in {0.9/0.1, 0.7/0.3, 0.5/0.5}).
- Token counts measured with the SERVING tokenizer, not word counts.
- Fixed RNG seed for the request trace; replay the identical trace across arms.
- Deterministic decode (temperature 0 / fixed seed) so output length is stable;
  otherwise output length is a confounder that dominates TPOT and throughput.

3. Metric definitions (ambiguity here is the #1 source of bogus results)
- TTFT = first-token-received timestamp - request-send timestamp, client-side,
  streaming enabled. Includes queueing. Also record server-side scheduler
  admission time so TTFT can be decomposed into queue_wait + prefill_compute.
- TPOT = (last_token_ts - first_token_ts) / (out_tokens - 1), per request.
  Report per-request distribution, never a global token/time ratio.
- Throughput: report output tok/s AND total (in+out) tok/s AND req/s separately;
  a single "tokens/s" number is not comparable across prompt/generation mixes.
- Queueing: scheduler queue depth and time-in-queue sampled at >=1 Hz, plus
  running/waiting/swapped sequence counts and KV-cache utilization.
- P99: computed over the steady-state window only, per request class (short vs
  long reported separately AND pooled). Need >= 2000 requests per arm per class
  for a usable P99 CI; report bootstrap 95% CI, not a bare point estimate.

4. Procedure
- Warmup: discard the first 120 s or first 200 requests, whichever is larger;
  record the warmup cut explicitly. Verify steady state via flat KV-utilization
  and flat queue depth before starting the measurement window.
- Load sweep: ramp arrival rate lambda from ~10% to ~110% of the measured
  saturation point in >= 8 steps; each step >= 5 min of steady state.
- Repeat each arm 3-5 times on different process launches (not just different
  windows) to capture launch-to-launch variance; report median and IQR.
- Randomize arm order across repeats to defuse thermal/clock drift.

5. Instrumentation
- Per-request trace: arrival, admission, first token, last token, in/out tokens,
  request class, preemption/recompute events.
- GPU: nvidia-smi dmon or DCGM at 1 Hz for SM util, memory used, SM clock,
  power, and throttle reasons (thermal/power cap). Clock throttling is a common
  hidden confounder on dense 8-GPU nodes.
- Engine counters: KV blocks used/free, preemptions, cache hit rate for prefix
  caching, batch size histogram, prefill vs decode step counts.
- Multi-GPU: NCCL collective time if TP > 1; per-step allreduce latency.

6. Expected confounders (control or report each)
- Closed-loop clients masking queue delay; unbounded client-side concurrency.
- Prefix caching inflating TTFT wins when synthetic prompts share prefixes -
  randomize prompt prefixes unless caching is the variable under test.
- Output-length drift from sampling; tokenizer mismatch between client and server.
- Chunked prefill changing the prefill/decode interference profile.
- Preemption/swap storms near KV exhaustion causing bimodal TPOT.
- Noisy neighbours, other tenants, background compaction, NUMA misbinding.
- Thermal/power throttling in long runs; cold page cache on first weight load.

7. Analysis
- Plot TTFT-P50/P99 and TPOT-P50/P99 vs offered load per class; identify the
  knee where queueing time exceeds compute time - that is the usable capacity,
  not the max-throughput point.
- Decompose TTFT into queue_wait + prefill; if queue_wait dominates at target
  load, the fix is admission control/replicas, not kernel tuning.
- Use Little's Law (L = lambda * W) as a consistency check against measured
  in-flight sequence counts; a mismatch means the trace or clocks are wrong.

8. Rollback / stop criteria
- Abort an arm if throttle reasons are non-zero > 1% of samples, if error rate
  > 0.5%, if measured output-length distribution deviates > 2% from the frozen
  trace, or if run-to-run median TTFT spread exceeds 10%.
- Promote a config only if it improves the target metric beyond the measured
  run-to-run noise band (non-overlapping bootstrap CIs) with no regression in
  the other class; otherwise roll back to the pinned baseline image.
"""

TROUBLE = """Diagnostic framing (Troubleshooting variant {n})
Symptom triage first: P99 latency regressions in mixed traffic almost always
resolve to one of four mechanisms. Test them in this order, cheapest first.
H-T1 Queueing-bound: queue_wait/TTFT > 0.5 at target load. Evidence: scheduler
  queue depth > 0 persistently while SM util < 70%. Fix: admission control,
  more replicas, lower max_num_seqs to shorten head-of-line blocking.
H-T2 KV-exhaustion / preemption: nonzero preemption counters, bimodal TPOT,
  KV utilization pinned near 100%. Fix: cap max_num_seqs, reduce max context,
  enable/tune swapping, raise gpu_memory_utilization only after OOM headroom
  is measured.
H-T3 Prefill/decode interference: long prompts stall decode steps; decode-step
  interval histogram shows spikes aligned with prefill admissions. Fix: enable
  chunked prefill and tune chunk size; or split prefill/decode across replicas
  (disaggregated serving) and re-measure the KV transfer cost explicitly.
H-T4 Hardware/collective: throttle reasons nonzero, or NCCL allreduce time
  variance high on TP>1. Fix: check `nvidia-smi topo -m`, power caps, and NIC
  path before touching engine flags.
Each hypothesis is falsified by a single-variable experiment against the frozen
baseline trace; if the predicted counter does not move, discard and move on.
"""

PERF = """Analysis framing (Performance Analysis variant {n})
Build a roofline-style capacity model before tuning, then falsify it.
- Prefill is compute-bound: expected time ~ (2 * P_active * in_tokens) / achieved
  FLOP/s. Measure achieved FLOP/s at a fixed prompt length; do not assume peak.
- Decode is memory-bandwidth-bound: expected per-step time ~ (bytes of weights +
  KV read) / achieved HBM BW. TPOT should be near-flat in batch size until the
  batch is large enough to become compute-bound - the batch size at which TPOT
  starts rising is the measured crossover point.
- KV bytes/token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes; compute it
  from the config and validate against observed KV block consumption. A > 10%
  mismatch means the engine is doing something you have not modelled.
Predicted vs measured gap > 20% on either phase means the model is wrong or a
confounder is active; investigate before reporting any speedup.
"""

DESIGN = """Design framing (System Design variant {n})
Deliverables of the evaluation harness, not just numbers:
- A versioned trace generator + replayer (open-loop, seeded) checked into the
  repo, so any result is reproducible from a commit hash.
- A results schema: one row per request, one row per 1 Hz sample, one manifest
  row per arm carrying the full config tuple; sha256-manifested outputs.
- Separate reporting for short vs long classes and an explicit SLO statement,
  e.g. "TTFT P99 <= 500 ms and TPOT P99 <= 40 ms at lambda = X req/s with mix r".
  Capacity is defined as the max lambda meeting the SLO, not max throughput.
- A decision gate: config promoted only on non-overlapping CIs plus a canary at
  5% live traffic with automatic rollback on SLO breach for 5 consecutive minutes.
- Explicit non-goal: this harness does not evaluate model accuracy; keep
  evaluation records isolated from any training corpus to avoid contamination.
"""

recs = []
for it in items:
    cat = it['category']
    n = it['id'].split('-')[-1].lstrip('0')
    if cat == 'Troubleshooting':
        head, conf = TROUBLE.format(n=n), 0.71
        risks = ["源答案仅是评分要点清单，未给出可执行的诊断顺序与判据",
                 "未定义 TTFT/TPOT 的测量端点，易把排队时间算入或漏算",
                 "未提及抢占/KV 耗尽导致的双峰 TPOT 误判为模型问题"]
    elif cat == 'Performance Analysis':
        head, conf = PERF.format(n=n), 0.7
        risks = ["源答案未给出 prefill 计算受限 / decode 带宽受限的量化模型",
                 "缺少 KV bytes/token 的可核对计算，容易误判显存容量",
                 "未要求报告置信区间，单点数值不可比较"]
    else:
        head, conf = DESIGN.format(n=n), 0.7
        risks = ["源答案未定义容量口径（SLO 下最大到达率 vs 最大吞吐）",
                 "缺少可复现的 trace 生成器与结果 schema，结论不可审计",
                 "未规定灰度与自动回滚门槛"]
    ans = (f"Evaluation plan for mixed short-prompt / long-generation serving ({it['id']}).\n\n"
           + head + "\n" + COMMON)
    recs.append({
        "source_id": it['id'],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": it['messages'][1]['content'],
        "source_assistant": it['messages'][2]['content'],
        "corrected_answer": ans,
        "quality_dimensions": {"technical_correctness": 3,
                               "instruction_coverage": 2,
                               "operational_safety": 2},
        "risks": risks,
        "evidence_required": [
            "engine commit / launch flags / weights hash / quantization / tokenizer version",
            "GPU SKU+count, nvidia-smi topo -m, driver+CUDA, container digest",
            "open-loop seeded request trace with fixed short/long mixing ratio r",
            "per-request trace: arrival, admission, first token, last token, token counts, preemptions",
            "1 Hz DCGM/nvidia-smi samples incl. throttle reasons, SM util, HBM used, clocks",
            "engine counters: KV utilization, queue depth, batch-size histogram, prefix-cache hit rate",
            "steady-state window definition + warmup cut, >=2000 req/class/arm, bootstrap 95% CI",
            "3-5 independent launches per arm with randomized arm order"
        ],
        "confidence": conf,
    })

out = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0109.jsonl'
with open(out, 'w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print('wrote', out, len(recs))
