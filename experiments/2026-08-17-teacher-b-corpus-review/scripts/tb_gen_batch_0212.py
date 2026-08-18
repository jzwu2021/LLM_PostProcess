import json, os
ROOT="/home/johnson/workspace/LLM_PostProcess"
CORPUS=f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP=f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT=f"{EXP}/results/train-batch-0212.jsonl"
START=2110  # 0-based positional offset
N=10

corpus=[json.loads(l) for l in open(CORPUS) if l.strip()]
sel=corpus[START:START+N]

FRAME = """Common frame (applies to every stance below).
Assumptions (must be restated by the answering engineer, not inherited silently):
A1. Single node, 8 GPUs, NVLink/NVSwitch intra-node; inter-node paths only appear where a stance says so explicitly.
A2. Decode-dominant, latency-sensitive serving: the SLO is TTFT p95 and TPOT (inter-token latency) p95, never a mean.
A3. Model weights fit in aggregate HBM with at least 20% KV-cache headroom at target concurrency.
A4. Exactly one variable moves per arm: no simultaneous change of quantization, batching policy, or speculative decoding.
Mechanism, stated plainly:
- Tensor parallelism (TP) shards every layer's GEMMs. Each transformer block needs two all-reduces (after attention out-projection and after the MLP down-projection), so decode carries L * 2 * allreduce_latency of synchronous cost, where L is the layer count. TP is latency-additive in collectives but capacity- and bandwidth-multiplying: per-GPU weight bytes and per-GPU KV bytes both fall by the TP degree.
- Pipeline parallelism (PP) shards layers into stages. Per token it adds only (PP-1) small point-to-point hidden-state sends, which are cheap, but a single request serializes through all stages and the bubble fraction is (PP-1)/(micro_batches + PP-1). At low concurrency there are too few in-flight micro-batches to fill the pipeline, so PP loses badly on single-request latency.
Boundary conditions that flip the answer:
- B1. On NVLink-class fabric, small-message all-reduce latency is in the single-digit microseconds and TP up to 8 is normally latency-viable. Over PCIe-only or across nodes on RoCE/IB the same collective's latency floor rises roughly an order of magnitude and TP stops paying past TP=2 (ESTIMATE; derivation: decode all-reduce payload is hidden_size * dtype_bytes per token per layer, which is small, so the collective is latency-bound rather than bandwidth-bound and the per-hop latency floor dominates).
- B2. If the model cannot fit on one GPU, sharding is mandatory and the question reduces to which axis, not whether.
- B3. Under high, steady concurrency the PP bubble amortizes and PP becomes competitive on throughput per GPU while still losing on single-request latency.
Default recommendation: use TP inside the node up to the point where collective cost stops being repaid by reduced per-GPU memory traffic; use PP only to cross a node boundary or to fit a model TP alone cannot fit. PP is not a latency optimization.
Measurement and evidence policy: every number below that was not produced by a run on this hardware is labelled ESTIMATE and carries its derivation. Only values read out of named benchmark artifacts may be labelled MEASURED. This review reports no MEASURED values, because no benchmark was executed for it."""

CRITIQUE = """Critique of the source item: the prompt is a legitimate infrastructure question and does ask for assumptions, a falsifiable hypothesis, measurements, confounders and rollback criteria, but the corpus pair is degenerate - the assistant turn contains only a rubric describing what an answer should contain, not an answer. There is therefore no substantive content to keep, and the item is rewritten into a complete response that supplies the mechanism, the boundary conditions that flip the recommendation, an explicit falsifiable hypothesis, a single-variable controlled experiment, the evidence artifacts required to adjudicate it, and a rollback gate. Every quantitative claim is labelled ESTIMATE and carries its derivation; no value here is MEASURED, because no benchmark run was performed for this review. This output is provisional teacher-B review material, not expert gold, and it is not evidence about any model's domain capability."""

import sys; sys.path.insert(0,'/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/scripts')
from stances_0212 import STANCES
assert len(STANCES)==N==len(sel)
rows=[]
for src,(sn,headline,body,hyp,exp,ev,risks,conf) in zip(sel,STANCES):
    m={x['role']:x['content'] for x in src['messages']}
    ca = (f"Analytical stance under test: Stance {sn} - {headline}\n\n"
          f"{FRAME}\n\n"
          f"{body} Falsifiable hypothesis {hyp} Controlled experiment: {exp} "
          f"Rollback gate: revert to the previously validated layout if the canary's TTFT p95, TPOT p95 or error rate breaches the pre-declared bound, "
          f"regardless of any aggregate throughput gain observed.\n\n"
          f"{CRITIQUE}")
    rows.append({
        "source_id": src["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": ca,
        "quality_dimensions": {"technical_correctness":3,"instruction_coverage":2,"operational_safety":2},
        "risks": risks,
        "evidence_required": ev,
        "confidence": conf,
    })

with open(OUT,"w") as f:
    for r in rows:
        f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("WROTE",OUT,len(rows))
print("IDS", rows[0]["source_id"], "->", rows[-1]["source_id"])
