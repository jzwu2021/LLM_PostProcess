#!/usr/bin/env python3
"""Generate teacher-B provisional blind-review batch 0099 (train lines 981-990).

Blind review: teacher-A artifacts are NEVER read by this script.
Source text is copied verbatim from research/ai-infra-expert/corpus/train.jsonl.
The corrected_answer bodies below were authored independently by the reviewer model.
"""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0099.jsonl")
START, END = 981, 990  # 1-indexed inclusive

COMMON_HEAD = """Evaluation plan for a mixed short-prompt / long-generation LLM serving workload.

0. Scope and non-claims
This measures serving-system behavior only. It says nothing about model quality, and no
result transfers across model size, quantization, engine version, sequence-length regime
or GPU SKU without a re-run.

1. Assumptions (they bound every number below)
- Fixed model weights, quantization, engine build (record the commit) and launch flags.
- Fixed hardware and topology; MIG off, persistence mode on, clocks pinned with
  `nvidia-smi -lgc` so DVFS/thermal drift cannot masquerade as a treatment effect.
- Load generator on a separate host, its own latency floor measured against a null
  endpoint first, enough sockets/cores that the client is provably not the bottleneck.
- Streaming enabled (TTFT is otherwise unobservable) and token-level, not byte-level,
  accounting.

2. Metric definitions (ambiguity here invalidates every comparison)
- TTFT = first streamed token timestamp - client submit timestamp; it INCLUDES queue wait,
  so queue wait must also be exported separately from engine scheduler counters.
- TPOT = (last_token_t - first_token_t) / (output_tokens - 1), computed per request.
- E2E = TTFT + TPOT * (output_tokens - 1); reported per length stratum, never pooled.
- Throughput reported as three separate numbers: input tok/s, output tok/s, req/s.
  Output tok/s is the decode-capacity metric; req/s alone is meaningless under a mixed mix.
- Queueing delay = admission time - arrival time, taken from the engine, not inferred.
- P99 computed within a stratum with the sample count printed; require >=3000 requests per
  cell, otherwise the tail estimate is noise.

3. Workload control
- Freeze one replayable trace file (prompt_len, output_len, arrival offset) and record its
  hash; never resample the distribution per run.
- Stratify short/long prompt x short/long generation into four cells, report each plus a
  weighted total.
- Latency runs use ignore_eos with fixed max_tokens so decode work is deterministic; a
  separate realistic-EOS run supplies capacity numbers.
- Open-loop Poisson arrivals for latency (closed-loop-only harnesses suffer coordinated
  omission and understate P99); closed-loop fixed concurrency only for saturation curves.
- Warmup: discard until output tok/s and P50 TPOT are stationary by a rolling-window slope
  test; report the discarded window explicitly.

4. Prefill vs decode separation (mandatory mechanism)
Prefill is ~O(prompt_len) compute-bound and sets TTFT; decode is HBM-bandwidth and
KV-capacity bound and sets TPOT. Under continuous batching a long generation occupies a
slot for many steps, so its cost lands on other requests' TPOT, not its own TTFT. Any
average that mixes the two phases is uninterpretable.
"""

COMMON_TAIL = """
7. Expected confounders
- KV preemption / recompute silently inflating TPOT tails.
- Prefix-cache hit-rate divergence between arms when trace order changes.
- Per-request TPOT is NOT independent under continuous batching (neighbors couple it), so
  use paired per-run bootstrap statistics, never per-request t-tests.
- Coordinated omission from closed-loop clients; thermal/DVFS drift; co-tenancy on the
  node; NUMA and CPU pinning of the API server; detokenizer CPU cost appearing as TTFT.

8. Evidence required before any claim
- Engine name/version/commit and complete launch flags for both arms.
- Frozen trace file plus its hash, and the load-generator config.
- Per-request traces: request_id, arrival, admit, first_token, last_token, prompt_len,
  output_len.
- Scheduler counters: num_waiting, num_running, KV cache utilization, preemption/recompute
  counts, scheduler step time.
- DCGM / nvidia-smi dmon telemetry (SM occupancy, memory-BW util, power, clocks) aligned
  to the run window.
- >=3000 requests per stratum and >=5 interleaved (ABAB...) repetitions per arm with 95%
  bootstrap CIs on the paired per-run difference.

9. Rollback criteria (pre-registered, automated, not eyeballed)
Roll back if P99 E2E regresses >10% in ANY stratum, or output tok/s drops >5%, or the
preemption/recompute rate rises above baseline, or the error/timeout rate exceeds 0.1%.
Canary on <=5% of traffic for >=30 minutes covering a peak window before fleet rollout,
with the rollback wired to those SLO gates.
"""

# Per-variant distinct hypothesis + controlled experiment blocks (sections 5-6).
VARIANTS = [
    ("admission-control queue discipline",
     """5. Falsifiable hypothesis
H1: replacing FIFO admission with shortest-remaining-output-first (SROF) admission cuts
P99 TTFT of the short-prompt/short-gen stratum by >=40% while raising P99 E2E of the
long-generation stratum by <=15%. If the long-gen penalty exceeds 15%, H1 is rejected.

6. Controlled experiment
Two arms differing only in the scheduler policy flag, replaying the identical frozen trace
at a fixed 0.7x-saturation QPS, interleaved ABABAB, 5 repetitions, 3000+ requests per
stratum per repetition. Output length is known to the scheduler only via the client-declared
max_tokens, so a third arm with a deliberately mis-declared max_tokens (50% of requests
under-declaring by 4x) tests robustness: if SROF's gain disappears under mis-declaration,
the policy is not shippable because real clients lie. Accept only if the pre-registered
thresholds hold AND the 95% bootstrap CI of the paired difference excludes zero.""",
     ["technical_correctness", 3], 0.72,
     ["source_assistant is a grading rubric, not an answer; supervising on it teaches meta-commentary instead of engineering content",
      "no metric definitions, so TTFT/TPOT/P99 are not comparable across runs",
      "priority scheduling can starve long generations; without a per-stratum abort gate this ships a latency regression for the heaviest users",
      "client-declared output length is untrusted input; a policy that depends on it fails silently in production"],
     ["scheduler policy flag and engine commit for both arms",
      "per-stratum P99 TTFT and P99 E2E with sample counts",
      "starvation check: max queue wait and completion rate of the long-generation stratum",
      "mis-declared max_tokens robustness arm results"]),

    ("chunked prefill size sweep",
     """5. Falsifiable hypothesis
H1: enabling chunked prefill with a 2048-token chunk budget reduces P99 TPOT by >=20%
(because long prefills stop blocking decode steps) at a cost of <=10% increase in P50 TTFT
for the long-prompt stratum. If P50 TTFT for long prompts inflates more than 10%, H1 is
rejected and the chunk budget is re-tuned rather than shipped.

6. Controlled experiment
Arms: chunked prefill off, and on at chunk budgets {512, 2048, 8192}. Only that flag varies.
Identical frozen trace, open-loop Poisson at fixed QPS, 5 interleaved repetitions per arm.
Mechanism check, not just endpoint metrics: export per-step batched-token composition
(prefill tokens vs decode tokens per scheduler step) and show that the TPOT improvement
tracks the reduction in decode-step stalls. If TPOT improves but step composition is
unchanged, the effect is confounded (likely a cache or thermal artifact) and must be
re-investigated before any claim.""",
     ["technical_correctness", 3], 0.7,
     ["source_assistant is a grading rubric, not an answer",
      "chunk budget interacts with prefix caching and with max_num_batched_tokens; sweeping one flag while others drift produces an uninterpretable result",
      "no mechanism check, so an endpoint-metric win could be a thermal or cache artifact",
      "no rollback threshold for the long-prompt TTFT cost the change deliberately incurs"],
     ["per-scheduler-step prefill/decode token composition traces",
      "chunk budget, max_num_batched_tokens and max_num_seqs for every arm",
      "per-stratum P50/P99 TTFT and P99 TPOT with sample counts",
      "engine commit plus frozen trace hash"]),

    ("KV cache capacity vs fragmentation",
     """5. Falsifiable hypothesis
H1: the P99 TPOT tail is caused by KV-cache exhaustion triggering preemption/recompute, not
by raw compute saturation. Concretely: raising gpu_memory_utilization from 0.85 to 0.92
reduces the preemption rate by >=60% and reduces P99 TPOT by >=15%, while P50 TPOT moves
by <5%. If P99 TPOT improves without the preemption rate falling, H1 is rejected and the
tail has a different cause.

6. Controlled experiment
Two arms differing only in gpu_memory_utilization, same frozen trace, same QPS, 5
interleaved repetitions. Primary evidence is the causal chain, not the endpoint: log
preemption/recompute counts, KV utilization time series, and per-request TPOT, then show
the TPOT tail improvement is concentrated in exactly the requests that were preempted in
the baseline arm. Safety boundary: 0.92 shrinks the CUDA headroom, so the run must also
record peak allocated memory and any OOM/allocator-retry events; a single OOM in the
canary is an automatic abort regardless of latency wins.""",
     ["operational_safety", 3], 0.74,
     ["source_assistant is a grading rubric, not an answer",
      "raising gpu_memory_utilization trades tail latency for OOM risk; without an OOM abort gate this is an availability hazard",
      "endpoint-only evidence cannot distinguish preemption relief from unrelated variance",
      "no per-stratum reporting, so a win on short requests can hide a long-generation regression"],
     ["preemption/recompute counters and KV utilization time series for both arms",
      "per-request TPOT joined to preemption events (request-level attribution)",
      "peak allocated memory, allocator retries and any OOM events",
      "gpu_memory_utilization, max_num_seqs, block size and engine commit"]),

    ("tensor-parallel degree and NCCL cost",
     """5. Falsifiable hypothesis
H1: moving from TP=2 to TP=4 improves P99 TTFT for the long-prompt stratum by >=25%
(more FLOPs applied to compute-bound prefill) but degrades output tok/s per GPU by >=10%
because every decode step adds two all-reduces over the interconnect. If output tok/s per
GPU does NOT degrade, the NCCL cost model is wrong and must be re-derived before the result
is used for capacity planning.

6. Controlled experiment
Arms TP=2 and TP=4 on the same node, identical frozen trace, open-loop Poisson, 5
interleaved repetitions. Isolate the communication term directly: run nccl-tests
(all_reduce_perf) at the exact per-step message sizes for both degrees and record achieved
bus bandwidth, plus NCCL_DEBUG=INFO to confirm the chosen algorithm and that traffic stays
on NVLink rather than falling back to PCIe/SHM. A topology fallback would confound the
whole comparison, so the run is void if the ring/tree topology differs between arms for a
reason other than the TP degree itself.""",
     ["technical_correctness", 3], 0.71,
     ["source_assistant is a grading rubric, not an answer",
      "TP degree changes both compute and communication; without isolating the NCCL term the result cannot be attributed",
      "silent NVLink->PCIe fallback would invalidate the comparison and is invisible in endpoint metrics",
      "per-GPU efficiency, not aggregate throughput, is the capacity-planning number; reporting only aggregate hides the regression"],
     ["nccl-tests all_reduce_perf bus bandwidth at the per-step message sizes for both TP degrees",
      "NCCL_DEBUG=INFO transport/algorithm confirmation and topology dump",
      "output tok/s normalized per GPU, not only aggregate",
      "per-stratum P99 TTFT/TPOT with sample counts and engine commit"]),

    ("prefix caching under shared system prompts",
     """5. Falsifiable hypothesis
H1: with a workload where 40% of requests share a common system prefix, enabling automatic
prefix caching cuts P50 TTFT of the long-prompt stratum by >=35% and leaves P99 TPOT
unchanged within +/-5%. If P99 TPOT degrades beyond 5%, the cache is stealing KV blocks from
decode and H1 is rejected.

6. Controlled experiment
Arms: prefix caching off / on, identical frozen trace INCLUDING request order, since cache
hit rate is order-dependent and reshuffling the trace between arms is the single most likely
way to fake this result. Export the engine's prefix cache hit rate and require it to match
the trace's designed 40% sharing within a few points; if it does not, the measurement is
invalid and no latency claim may be made. Also verify correctness, not just speed: replay a
fixed prompt set with temperature 0 in both arms and assert token-identical outputs, because
a caching bug that corrupts KV blocks presents as a latency win.""",
     ["instruction_coverage", 3], 0.73,
     ["source_assistant is a grading rubric, not an answer",
      "prefix-cache hit rate is trace-order dependent; comparing arms with different orders is a common false-positive path",
      "caching bugs corrupt outputs while improving latency, so a correctness assertion is mandatory",
      "cache blocks compete with decode KV capacity, which can move the TPOT tail"],
     ["engine prefix-cache hit-rate counters per arm",
      "byte-identical trace file (same order) and its hash",
      "temperature-0 token-identical output assertion over a fixed prompt set",
      "KV utilization and preemption counters to check cache/decode contention"]),

    ("multi-replica routing policy",
     """5. Falsifiable hypothesis
H1: KV-aware / least-outstanding-decode-slots routing across replicas reduces cross-replica
P99 TTFT spread (max minus min replica P99) by >=50% versus round-robin, without lowering
fleet output tok/s by more than 3%. If the spread does not shrink, replica imbalance is not
the tail driver and H1 is rejected.

6. Controlled experiment
Fixed replica count and identical engine config on every replica; only the router policy
varies. Same frozen trace fanned out from one open-loop generator, 5 interleaved
repetitions. Per-replica metrics are mandatory: fleet-level averages structurally hide
imbalance, which is the very quantity under test. Watch the failure mode explicitly: a
KV-aware router can herd a burst onto whichever replica just freed capacity, so record the
per-replica arrival time series and flag any window where one replica takes >1.5x its fair
share. Boundary condition: results only hold while replicas are homogeneous; a mixed-SKU
fleet requires re-running.""",
     ["instruction_coverage", 3], 0.7,
     ["source_assistant is a grading rubric, not an answer",
      "fleet-level aggregation hides the replica imbalance being measured",
      "KV-aware routing can herd bursts onto a just-freed replica, converting an imbalance fix into an oscillation",
      "router health-check and drain behavior untested, so a routing change can amplify a partial outage"],
     ["per-replica P99 TTFT, output tok/s and arrival time series",
      "router policy configuration and version for both arms",
      "replica homogeneity evidence (same SKU, same engine commit, same flags)",
      "burst/herding check: per-replica share over sliding windows"]),

    ("speculative decoding acceptance rate",
     """5. Falsifiable hypothesis
H1: speculative decoding with a small draft model raises output tok/s per request by >=30%
at low load (<=0.3x saturation) but yields <5% gain, or a net loss, at >=0.8x saturation,
because at high load the extra draft+verify FLOPs compete with a compute-saturated batch.
If the high-load arm still shows a large win, the load level or the batching configuration
is not what we think it is and must be re-measured.

6. Controlled experiment
Arms: speculation off / on, swept at QPS = {0.3, 0.6, 0.8, 0.95} x measured saturation,
identical frozen trace, 5 interleaved repetitions per cell. Report the draft acceptance rate
per stratum, since acceptance is workload-dependent and a fleet-wide average will mispredict
per-tenant behavior. Correctness gate: with greedy decoding, speculative output must be
token-identical to non-speculative output on a fixed prompt set; any divergence is a
correctness bug and blocks the change regardless of throughput.""",
     ["technical_correctness", 3], 0.7,
     ["source_assistant is a grading rubric, not an answer",
      "speculative decoding gains are load-dependent; a single low-load benchmark will overstate production benefit",
      "acceptance rate varies by workload stratum, so a fleet average mispredicts per-tenant behavior",
      "draft model adds memory and a second failure domain, neither of which the rubric mentions"],
     ["per-stratum draft acceptance rate and rejected-token counts",
      "measured saturation QPS and the load levels used for the sweep",
      "greedy token-identity check between speculative and non-speculative arms",
      "additional GPU memory consumed by the draft model and its effect on KV capacity"]),

    ("autoscaling and cold-start behavior",
     """5. Falsifiable hypothesis
H1: the P99 TTFT spikes correlate with replica scale-up events, not with steady-state load:
specifically, >=70% of requests above the P99 TTFT threshold fall within 90 s of a scale-up,
and pre-warming replicas (weights resident, CUDA graphs captured before receiving traffic)
removes >=80% of those spikes. If spike requests are uniformly distributed in time, H1 is
rejected and the tail is a steady-state scheduling problem instead.

6. Controlled experiment
Replay a frozen trace containing a deliberate 3x step in arrival rate, arms = cold-start
routing vs warm-pool routing, only that variable changed, 5 interleaved repetitions. Join
per-request TTFT against replica lifecycle events (pod ready, first successful health probe,
first token served) to test the temporal correlation directly rather than inferring it from
aggregate curves. Boundary: a warm pool costs idle GPU-hours, so report the idle GPU-minutes
per arm; a latency win that costs more than the pre-registered budget is rejected on cost
grounds even if the latency hypothesis holds.""",
     ["operational_safety", 3], 0.68,
     ["source_assistant is a grading rubric, not an answer",
      "aggregate latency curves cannot attribute spikes to scale-up events; request-to-lifecycle joins are required",
      "readiness probes that pass before CUDA graphs are captured route traffic to a not-actually-ready replica",
      "warm pools consume idle GPU-hours; a latency win without a cost budget is not a shippable conclusion"],
     ["replica lifecycle event log joined to per-request TTFT timestamps",
      "readiness probe definition and time from pod-ready to first token",
      "idle GPU-minutes per arm and the pre-registered cost budget",
      "the frozen trace containing the arrival-rate step and its hash"]),

    ("disaggregated prefill/decode KV transfer",
     """5. Falsifiable hypothesis
H1: disaggregating prefill and decode onto separate GPU pools cuts P99 TTFT by >=30% while
adding <=15 ms of P99 KV-transfer latency over the interconnect, so P99 E2E improves. If the
measured KV-transfer P99 exceeds 15 ms, the interconnect (not the scheduler) is the binding
constraint and disaggregation must not ship on this fabric.

6. Controlled experiment
Arms: colocated vs disaggregated, same total GPU count so the comparison is capacity-neutral,
identical frozen trace, 5 interleaved repetitions. Instrument the transfer path explicitly:
per-request KV payload bytes, transfer start/end timestamps, and achieved bandwidth, plus
fabric counters (for RDMA paths, ibstat / perfquery port counters and any retransmit or
congestion-notification counts). A rising retransmit count invalidates the latency numbers.
Boundary conditions: the result depends on prompt length distribution (payload size scales
with prompt tokens times layers times 2 times head_dim times dtype bytes) and on fabric
class; it does not transfer to a TCP-only deployment.""",
     ["technical_correctness", 3], 0.66,
     ["source_assistant is a grading rubric, not an answer",
      "disaggregation moves the bottleneck to the KV transfer path, which the rubric never mentions measuring",
      "fabric retransmits or congestion events can invalidate latency numbers while endpoint metrics still look fine",
      "capacity-neutral comparison is required; otherwise the win is just extra hardware"],
     ["per-request KV payload bytes, transfer timestamps and achieved bandwidth",
      "fabric counters (RDMA port counters, retransmits, congestion notifications) over the run window",
      "identical total GPU count and topology documentation for both arms",
      "prompt-length distribution of the frozen trace, since payload size scales with it"]),

    ("SLO-based capacity headroom",
     """5. Falsifiable hypothesis
H1: the sustainable QPS at which P99 E2E stays under the SLO is <=65% of the saturation QPS
measured by a closed-loop throughput benchmark. That is, the closed-loop number overstates
deployable capacity by at least 35%. If the SLO-bound QPS is within 10% of the closed-loop
saturation number, the queueing model is wrong and capacity planning must be re-derived.

6. Controlled experiment
Sweep open-loop Poisson QPS in fine steps around the knee, 5 repetitions per step, and
locate the largest QPS whose P99 E2E is under the SLO in EVERY stratum (not on the weighted
average, which lets the cheap stratum mask the expensive one). Separately run the closed-loop
saturation benchmark and compare. Mechanism: as utilization approaches 1, queue delay grows
super-linearly, so P99 explodes well before throughput plateaus; the run must show the
queue-delay curve, not just latency, to demonstrate that this is queueing rather than a
compute cliff. Deploy at the SLO-bound QPS with headroom, and pre-register that exceeding it
triggers scale-out rather than degradation.""",
     ["instruction_coverage", 3], 0.72,
     ["source_assistant is a grading rubric, not an answer",
      "closed-loop saturation throughput is routinely mistaken for deployable capacity, which overloads the fleet in production",
      "weighted-average SLO checks let a cheap stratum mask an expensive one",
      "no headroom or scale-out trigger, so the system is planned to run at the point where P99 explodes"],
     ["open-loop QPS sweep with per-stratum P99 E2E and queue-delay curves",
      "the closed-loop saturation measurement for direct comparison",
      "explicit SLO definition (metric, percentile, stratum scope, window)",
      "pre-registered scale-out trigger and headroom factor"]),
]

def build_answer(idx):
    _topic, body, _dim, _conf, _r, _e = VARIANTS[idx]
    return COMMON_HEAD + "\n" + body + "\n" + COMMON_TAIL

def main():
    with open(CORPUS, encoding="utf-8") as f:
        lines = f.read().split("\n")
    rows = []
    for i, ln in enumerate(lines[START - 1:END], start=START):
        d = json.loads(ln)
        msgs = d["messages"]
        user = [m for m in msgs if m["role"] == "user"][0]["content"]
        asst = [m for m in msgs if m["role"] == "assistant"][0]["content"]
        k = i - START
        topic, body, weak_dim, conf, risks, evid = VARIANTS[k]
        qd = {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}
        qd[weak_dim[0]] = weak_dim[1]
        rows.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": user,
            "source_assistant": asst,
            "corrected_answer": build_answer(k),
            "quality_dimensions": qd,
            "risks": risks,
            "evidence_required": evid,
            "confidence": conf,
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("WROTE", OUT, len(rows), "rows;", "ids:", ",".join(r["source_id"] for r in rows))

if __name__ == "__main__":
    main()
