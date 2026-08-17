#!/usr/bin/env python3
"""Generate teacher-B provisional BLIND review batch train-batch-0104.jsonl.

Blind: only research/ai-infra-expert/corpus/train.jsonl is read. No teacher-A
artifact is opened by this script.
"""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0104.jsonl")
START, COUNT = 1030, 10  # 0-based offset -> corpus lines 1031..1040


def framing(cat):
    return {
        "Troubleshooting": "diagnosis-first framing",
        "Performance Analysis": "performance-analysis framing",
        "System Design": "system-design framing",
    }.get(cat, "general framing")


def cat_section(cat):
    if cat == "Troubleshooting":
        return (
            "8. Diagnosis path (only if the plan is being run against a live regression)\n"
            "Order the cut set so each step falsifies one layer:\n"
            "- Is it admission or execution? Compare server-side queue wait vs first-decode\n"
            "  latency. Rising queue wait with flat decode time means capacity/admission,\n"
            "  not kernel slowness, and no engine flag will fix it.\n"
            "- Is it prefill or decode? Split TTFT and TPOT. A TTFT-only regression points at\n"
            "  prompt length, prefix-cache miss, chunked-prefill policy or scheduler batching;\n"
            "  a TPOT-only regression points at decode batch size, KV pressure, or clocks.\n"
            "- Is it memory? Watch KV-cache utilization and preemption/recompute counters. If\n"
            "  preemptions are non-zero, the tail is a capacity artifact, not a latency bug.\n"
            "- Is it the host or the device? Check GPU busy% and SM occupancy against CPU-side\n"
            "  tokenization/detokenization and Python scheduling time; an idle GPU with a bad\n"
            "  P99 is a host-side or network-side problem.\n"
            "- Is it one replica? Break every metric down per replica/GPU before aggregating;\n"
            "  a single throttled or ECC-degraded device can own the entire P99.\n"
        )
    if cat == "System Design":
        return (
            "8. Design consequences the plan must be able to decide\n"
            "The measurements above exist to settle concrete design choices, each with a\n"
            "stated failure mode:\n"
            "- Single pool vs prefill/decode separation. Separation buys TTFT isolation but\n"
            "  pays a KV transfer over the interconnect; it only wins if the measured KV\n"
            "  transfer time is small against the TTFT it protects. Measure, do not assume.\n"
            "- Chunked prefill / continuous batching thresholds: trade TTFT against decode\n"
            "  throughput. Sweep, and report the whole frontier, not the single best point.\n"
            "- Class isolation: separate queues or priority for short vs long requests, so a\n"
            "  long generation cannot occupy the batch slot that a short request needs.\n"
            "- Replica sizing and parallelism (TP/PP) chosen from measured KV headroom at the\n"
            "  target concurrency, not from peak FLOPs.\n"
            "- Autoscaling signal must be goodput and queue wait, never GPU utilization, which\n"
            "  saturates long before the SLO breaks.\n"
        )
    return (
        "8. Analysis discipline\n"
        "- Report per-request distributions, never only run means; a mean over a bimodal\n"
        "  short/long mixture describes no real request.\n"
        "- Break every aggregate down by request class and by replica before comparing arms.\n"
        "- Attribute latency into queue wait, prefill, decode and transport, and check the\n"
        "  parts sum to end-to-end within tolerance; if they do not, the instrumentation is\n"
        "  wrong and the conclusion is void.\n"
        "- Compare arms at equal offered load and equal accepted load; a throughput win that\n"
        "  came from dropping or truncating requests is not a win.\n"
    )


def build_answer(variant, cat):
    return f"""Evaluation plan - mixed short-prompt / long-generation serving, variant {variant}
({framing(cat)}).

0. Scope and non-claims
This is a serving-system measurement plan. It says nothing about model quality, and
no number here transfers across engine commit, model weights or quantization,
sequence-length regime, batching policy, or GPU SKU without a re-run.

1. Assumptions (record them; violating any one voids the run)
- Fixed model weight hash, tokenizer, engine commit, and the complete launch flag set.
- Persistence mode on and clocks pinned (nvidia-smi -pm 1, -lgc); MIG layout and ECC
  state fixed, so DVFS or thermal drift cannot be misread as a treatment effect.
- Open-loop load generation (Poisson arrivals) from an off-host generator. A
  closed-loop harness self-throttles and deletes exactly the tail we want to measure.
- Client and server clocks disciplined (NTP/PTP) well below the smallest latency
  difference we intend to report.
- Output length pinned with max_tokens + ignore_eos; otherwise generation length is a
  hidden variable that co-moves with every configuration change.

2. Workload, frozen before the first run
- Two declared classes: short-prompt/short-output and short-prompt/long-generation,
  with fixed mixture ratio and fixed input/output token-length distributions, reported
  as token percentiles (not characters).
- Prompt corpus fixed and seeded; report the prefix-sharing rate, because prefix/KV
  cache hits silently move TTFT and can manufacture a fake improvement.

3. Metrics, units mandatory
- TTFT (ms) measured client-side at first streamed token and server-side at the first
  decode step. The difference is queue + transport and must be reported separately,
  never folded into "TTFT".
- TPOT / inter-token latency (ms/token) as a per-request distribution, not a run mean.
- Queue wait (ms), end-to-end latency (ms), throughput (output tok/s and req/s), and
  goodput (req/s meeting the SLO). Throughput alone hides SLO violations.
- P50/P90/P99 always with sample count n and a confidence interval. A P99 without n is
  not decision-grade; size runs so P99 rests on at least ~100 tail events.
- GPU-side: KV-cache utilization, running vs waiting queue depth, preemption/recompute
  counts, achieved memory bandwidth, SM occupancy, power and clock throttle reasons.

4. Falsifiable hypothesis (variant {variant})
H1: At the target arrival rate, P99 end-to-end latency is dominated by admission
queueing rather than decode speed - specifically, median queue wait accounts for
>= 50% of P99 end-to-end latency, while TPOT P99 stays within 1.2x of TPOT P50.
Prediction if H1 holds: adding a replica at constant load cuts P99 by >= 30% while
leaving TPOT P50 statistically unchanged.
Refutation: if adding the replica moves P99 by < 10%, or TPOT P99/P50 > 1.5, H1 is
false and the bottleneck is per-step execution or KV capacity, not admission.
This is falsifiable by a single controlled arm and must be stated before the run.

5. Controlled experiment
- Arms: A = baseline replica count; B = baseline + 1 replica; all else byte-identical.
- Fix offered load, mixture, seeds, prompt corpus, and duration across arms.
- 5 minutes discarded warmup (cache fill, clock settle, allocator steady state), then
  >= 15 minutes measurement, >= 3 independent repeats, arms interleaved A/B/A/B to
  absorb drift.
- Report per-arm distributions and a paired comparison with confidence intervals; a
  point-estimate delta with no interval is not a result.

6. Expected confounders
- Prefix/KV cache warmth differing between arms.
- Clock throttling or thermal drift over long runs.
- Client-side bottleneck (tokenizer, event loop, NIC) masquerading as server latency.
- Noisy neighbours: co-located jobs, other tenants on the same NIC/PCIe/NVLink path.
- Generation-length drift when ignore_eos is not set.
- Aggregating across request classes, which lets the mixture ratio drive the metric.

7. Rollback criteria (pre-committed, not negotiated after the fact)
Abort or revert the change if any of these hold on the canary: P99 end-to-end latency
regresses > 10% versus baseline; goodput drops at equal offered load; error or
timeout rate exceeds the current baseline; preemption/recompute counters rise from
zero; or GPU OOM appears at any point. Roll back first, analyse afterwards.

{cat_section(cat)}
9. What this plan cannot conclude
It cannot rank engines or hardware in general, cannot predict behaviour at untested
sequence lengths or concurrency, and cannot support any claim about answer quality.
Those need separate, independently designed experiments."""


def main():
    lines = open(CORPUS, encoding="utf-8").read().split("\n")
    recs = [json.loads(lines[i]) for i in range(START, START + COUNT)]
    out = []
    for r in recs:
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        su, sa = msgs["user"], msgs["assistant"]
        variant = r["id"].split("-")[-1].lstrip("0")
        cat = r.get("category", "")
        out.append({
            "source_id": r["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": build_answer(variant, cat),
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 3,
            },
            "risks": [
                "source_assistant is a rubric/checklist, not an answer; training on it teaches meta-commentary instead of engineering reasoning",
                "no units, no arrival model, and no sample-size requirement, so a P99 produced from it is not decision-grade",
                "closed-loop load generation and unpinned generation length are not excluded, both of which erase the measured tail",
                "no pre-committed rollback threshold, so a regression can be rationalised after the fact",
            ],
            "evidence_required": [
                "engine commit, model weight hash, tokenizer and full launch flags for every arm",
                "clock/persistence-mode state and throttle reasons captured during the run",
                "per-request traces with queue wait, TTFT (client and server), TPOT and end-to-end latency",
                "KV-cache utilization, queue depth and preemption/recompute counters per replica",
                "sample counts and confidence intervals for every reported percentile",
            ],
            "confidence": 0.62,
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out), "records:", out[0]["source_id"], "->", out[-1]["source_id"])


if __name__ == "__main__":
    main()
