#!/usr/bin/env python3
"""Build a deterministic, auditable seed corpus for AI/LLM Infrastructure SFT.

This is an authored seed corpus, not a dump of external documents. Every item
must pass expert review before being used for a publishable training run.
"""
import hashlib, json, math, random
from pathlib import Path

ROOT=Path(__file__).parent
OUT=ROOT/'corpus'
OUT.mkdir(exist_ok=True)


def sha(s): return hashlib.sha256(s.encode()).hexdigest()[:16]
def item(i, category, task, q, a, difficulty, concepts, verifier='rubric'):
    return {
      'id': f'corpus-{i:05d}', 'category': category, 'task_type': task,
      'difficulty': difficulty, 'messages': [
        {'role':'system','content':'You are an AI/LLM Infrastructure engineer. State assumptions, use units, distinguish measured facts from estimates, and do not invent platform-specific facts.'},
        {'role':'user','content':q}, {'role':'assistant','content':a}],
      'concepts': concepts, 'verifier': verifier,
      'provenance': 'authored_synthetic_seed_v0.1',
      'review_status': 'needs_domain_expert_review',
      'contamination_policy': 'not copied from benchmark; keep evaluation records isolated from training'
    }

rows=[]; i=1
# Knowledge/concept seed: 10 concepts x 10 formulations.
knowledge=[
 ('KV cache','During autoregressive decoding, cached keys and values avoid recomputing prior tokens; memory grows with layers, sequence length, KV heads, head dimension, and bytes per value.','kv_cache'),
 ('prefill','Prefill processes the prompt and is generally parallel across prompt tokens; its cost is dominated by prompt processing and often compute utilization.','prefill'),
 ('decode','Decode generates one or a few tokens per step and repeatedly reads model weights and KV state; it is often sensitive to memory bandwidth and scheduling.','decode'),
 ('continuous batching','Continuous batching admits and retires requests at iteration boundaries so completed sequences do not hold a static batch slot.','batching'),
 ('tensor parallelism','Tensor parallelism shards computation within layers and introduces collective communication; the best degree depends on memory, topology, batch, and communication cost.','tensor_parallel'),
 ('pipeline parallelism','Pipeline parallelism partitions layers across stages and can require microbatching to reduce pipeline bubbles.','pipeline_parallel'),
 ('MoE','Mixture-of-Experts routes tokens to selected experts, reducing active compute per token but introducing routing, capacity, and all-to-all concerns.','moe'),
 ('quantization','Quantization reduces representation precision and often memory traffic, but accuracy, kernel support, calibration, and outlier handling must be measured.','quantization'),
 ('NCCL','NCCL implements GPU collectives; diagnosis should separate topology, transport, process-group, rank, timeout, and workload causes.','nccl'),
 ('speculative decoding','Speculative decoding uses a draft model to propose tokens and a target model to verify them; speedup depends on acceptance rate and verification cost.','spec_decode'),
]
forms=[
 'Define {name} and explain one reason it matters in LLM infrastructure.',
 'Contrast {name} with a naive implementation that does not use it.',
 'Give two failure modes or trade-offs associated with {name}.',
 'Explain how {name} interacts with latency, throughput, or memory.',
 'Give a measurement plan for validating whether {name} helps a serving workload.',
 'What assumptions must be stated before making a performance claim about {name}?',
 'Explain how {name} changes between training and inference.',
 'Give one misleading intuition about {name} and correct it.',
 'Design a small controlled experiment for {name}.',
 'Write a concise runbook entry for an engineer investigating {name}.'
]
for name,answer,concept in knowledge:
  for f,form in enumerate(forms):
    rows.append(item(i,'Knowledge/Concept','explanation',form.format(name=name),answer,'easy' if f<3 else 'medium',[concept],'rubric')); i+=1

# Calculation tasks: varied values held out from the benchmark generator.
for n in range(100):
  layers=24+(n%5)*8; seq=1024+(n%7)*512; heads=2+(n%4)*2; dim=64+(n%3)*32; b=2 if n%3 else 1
  bytes_total=2*layers*seq*heads*dim*b
  gib=bytes_total/(1024**3)
  q=f'A model has {layers} layers, {heads} KV heads, head dimension {dim}, sequence length {seq}, and {"BF16/FP16" if b==2 else "INT8"} KV values. Estimate bytes for one request\'s K/V cache. Show the formula and report GiB.'
  a=f'Use 2 × layers × sequence_length × KV_heads × head_dim × bytes_per_value. Here this is 2 × {layers} × {seq} × {heads} × {dim} × {b} = {bytes_total} bytes = {gib:.6f} GiB. This excludes allocator metadata and other runtime memory.'
  rows.append(item(i,'Calculation','numeric',q,a,'medium',['kv_cache','memory'],'exact_numeric')); i+=1

# Design / diagnosis / optimization examples.
scenarios=[
 ('serving capacity','A service receives a mix of short prompts and long generations. Design an evaluation plan that reports TTFT, TPOT, throughput, queueing, and P99 latency.','separate prefill and decode behavior; fix workload distribution; report warmup; use repeated trials; capture GPU memory and request-level traces.'),
 ('long context','A long-context workload intermittently hits OOM after several concurrent requests. Give a prioritized diagnosis and mitigations.','measure per-request KV growth and allocator fragmentation; check max sequence and batch; inspect active sequences; test paged/prefix cache, admission control, quantization, and context limits.'),
 ('NCCL startup','A multi-GPU job hangs during collective initialization. Provide a diagnosis plan.','record ranks and environment; verify process-group rendezvous; inspect topology and GPU visibility; run a minimal all-reduce; test interface selection and timeout; compare single-node and reduced-world runs.'),
 ('quantization','A team wants lower serving cost through weight-only quantization. Define a fair comparison.','fix model and prompts; compare quality, memory, TTFT, TPOT, throughput, concurrency, kernel support, calibration set, and failure cases; report confidence intervals.'),
 ('parallelism','Choose between tensor and pipeline parallelism for a latency-sensitive multi-GPU service. State the assumptions needed.','compare per-GPU memory, communication topology, layer partitioning, batch and microbatch behavior, synchronization, latency target, and failure/reconfiguration costs.'),
 ('agent runtime','An agent repeatedly calls a calculator when the answer is already known. Design metrics and an intervention.','measure unnecessary calls, tool success, final correctness, trajectory length, tool latency, and recovery; add a stop/no-tool evaluation and preference or reward signal.'),
 ('moe serving','A sparse MoE service has uneven expert load and tail latency. Analyze likely causes and experiments.','measure routing distribution, expert capacity overflow, all-to-all time, token dropping/padding, placement, batch composition, and compare routing/capacity policies.'),
 ('speculation','A speculative decoder is slower than the target model alone. Identify what to measure.','measure draft latency, target verification latency, acceptance rate by position, accepted tokens per step, synchronization overhead, and workload dependence.'),
 ('memory hierarchy','GPU compute utilization is low while memory bandwidth is near saturation during decode. Explain the diagnosis and next experiments.','separate weight/KV reads from launch overhead; vary batch, sequence, precision, cache reuse, and kernel; compare roofline-like limits to measured counters.'),
 ('benchmarking','Two serving systems claim different tokens/s. Design a reproducible comparison.','freeze model revision, tokenizer, precision, hardware, topology, prompt/generation distributions, concurrency, warmup, stopping rules, and telemetry; publish raw request traces.'),
]
for category, prompt, keypoints in scenarios:
  for v in range(100):
    q=prompt+f' Scenario variant {v+1}: include an explicit falsifiable hypothesis and a controlled experiment.'
    a=f'Answer should state assumptions, a falsifiable hypothesis, measurements, expected confounders, and rollback criteria. Minimum technical points: {keypoints}'
    cat='System Design' if v%3==0 else ('Troubleshooting' if v%3==1 else 'Performance Analysis')
    rows.append(item(i,cat,'design_or_diagnosis',q,a,'hard',['ai_infrastructure',category],'rubric')); i+=1

# Code/tool structured tasks.
code_specs=[
 ('KV cache estimator','Implement kv_cache_bytes(layers, seq_len, kv_heads, head_dim, bytes_per_value) using integer arithmetic and reject non-positive inputs.'),
 ('TP validator','Implement valid_tensor_parallel(world_size, tp) and require positive integers with world_size divisible by tp.'),
 ('tool-call parser','Parse a JSON object requiring a string tool name and object arguments; reject duplicate or unknown top-level fields.'),
 ('bounded retry','Implement retry with a maximum attempt count, bounded backoff, and a non-retryable exception predicate.'),
 ('latency percentiles','Compute p50 and p99 from request latencies, define the interpolation method, and handle an empty input explicitly.'),
 ('paged blocks','Return ceil(sequence_length/block_size) using integer arithmetic and validate both inputs.'),
 ('NCCL environment checker','Report missing required distributed-runtime variables without exposing secret values.'),
 ('prefill decode classifier','Classify a request using prompt and generated token counts with documented thresholds and tests.'),
 ('duplicate detector','Detect the first duplicate request ID in a stream while preserving input order.'),
 ('capacity planner','Implement a conservative capacity estimate that returns both the estimate and assumptions used.'),
]
for name,spec in code_specs:
  for v in range(50):
    q=f'Write a small, dependency-light Python implementation for {name}. Variant {v+1}. Contract: {spec} Include tests for boundary cases and a brief complexity analysis.'
    a=f'Provide typed or clearly documented code, input validation, deterministic tests, and explicit assumptions. Contract to satisfy: {spec}'
    rows.append(item(i,'Code/Tool Use','code',q,a,'medium' if v<25 else 'hard',['python','testing',name],'unit_test')); i+=1

assert len(rows)==1700, len(rows)
# Deterministic split by hash; benchmark/evaluation files are not included here.
rows.sort(key=lambda x:x['id'])
for split in ('train','validation'):
  with (OUT/f'{split}.jsonl').open('w') as f:
    for r in rows:
      h=int(hashlib.sha256(r['id'].encode()).hexdigest(),16)%100
      if (split=='train' and h<90) or (split=='validation' and 90<=h<100):
        f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
with (OUT/'manifest.json').open('w') as f:
  json.dump({'version':'aiinfra-sft-seed-v0.1','total':len(rows),'train':sum(int(int(hashlib.sha256(r['id'].encode()).hexdigest(),16)%100<90) for r in rows),'validation':sum(int(90<=int(hashlib.sha256(r['id'].encode()).hexdigest(),16)%100<100) for r in rows),'held_out_evaluation':'../benchmark.jsonl','review_status':'needs_domain_expert_review','provenance':'authored_synthetic_seed_v0.1'},f,indent=2)
print('total',len(rows))
print('train/validation written')
