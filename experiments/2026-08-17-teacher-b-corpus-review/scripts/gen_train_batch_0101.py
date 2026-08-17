import json

COMMON_HEAD = """Evaluation plan: mixed short-prompt / long-generation LLM serving traffic (variant {v}).

0. Non-claims
This measures serving-system behavior only. It is not evidence about model quality.
No number transfers across engine commit, model size, quantization, sequence-length
regime, or GPU SKU without a re-run.

1. Fixed setup (record or the run is void)
- Engine commit, launch flags, weights hash, quantization, tokenizer version.
- nvidia-smi -pm 1, clocks pinned with nvidia-smi -lgc, MIG off; DVFS/thermal drift
  must not be readable as a treatment effect.
- Open-loop Poisson arrivals from an off-host generator (closed-loop harnesses
  self-throttle and delete exactly the tail samples we are trying to measure).
  Generator CPU < 60% at peak, RTT recorded.
- Seeded length mixture over four strata S/M/L/XL, identical across arms.

2. Metric definitions frozen before any run
- TTFT = server ingress -> first output token flushed.
- TPOT = (E2E - TTFT) / (out_tokens - 1); undefined and excluded when out_tokens < 2.
- Queueing delay = ingress -> prefill admission, reported separately.
- Throughput reported as req/s + prefill tok/s + decode tok/s separately, because
  prefill is compute-bound and decode is memory-bandwidth-bound at small batch;
  a single aggregate tok/s is not falsifiable.
- P99 per stratum from raw per-request records, never pooled across strata or repeats.

3. Procedure
Warmup discarded; 5 fresh-process repeats per arm; randomized arm order; load ladder
of >= 6 points up to 130% of measured saturation so the knee is visible.

"""

COMMON_TAIL = """
Confounders instrumented every run
- Prefix-cache hit rate (a divergence > 2pp between arms voids the comparison).
- Continuous-batching batch-size distribution over time.
- Preemption / recompute / swap counts, KV utilization, allocator OOM retries.
- Sampling params and EOS behavior (output length is itself a treatment).
- NUMA placement, CPU governor, generator RTT.

Stop / rollback gates (pre-committed)
- Any thermal or power throttle reason observed => sample invalid, not noise; abort arm.
- Error rate > 0.5% => abort the load point.
- Promote only if the pre-registered primary metric beats a noise band of 2x the control
  arm's across-repeat spread AND no stratum regresses > 5% on P99 E2E.
- Roll back within one deploy cycle on > 10% P99 E2E regression, observed starvation of
  any stratum, or increased OOM/preemption.

Honest limit: P99 over a few thousand requests needs a bootstrap or binomial confidence
interval quoted alongside it. An effect smaller than that interval is not a result.
"""

ITEMS = [
 ("corpus-01101", """4. Falsifiable hypothesis (H1: chunked prefill trades TTFT tail for decode steadiness)
H1: with chunked prefill budget 2048 tokens/step, P99 TTFT for the S stratum improves
>= 25% versus unbounded prefill at 90% of saturation, while decode tok/s degrades <= 5%.
Null: no improvement beyond the noise band, or decode loss > 5%.
Mechanism check: add a chunk=512 arm. If the mechanism is head-of-line blocking by long
prefills, chunk=512 must move S-stratum P99 TTFT monotonically further in the same
direction and cost more decode throughput. If it does not, the observed effect is not
head-of-line blocking and the hypothesis is refuted regardless of the 2048 result.
Boundary condition: only valid while KV utilization stays below the preemption threshold;
above it, preemption dominates and the arms are not comparable.
Evidence: per-request traces with admission timestamps, per-step batch composition,
preemption counters, KV utilization time series.""",
 4, 3, 4, ["Rubric text supervises meta-commentary about answers rather than the engineering reasoning",
           "No metric definitions, so TTFT/TPOT numbers are not comparable across arms",
           "No abort or rollback thresholds despite an operational scenario"],
 ["Per-request trace with ingress/admit/first-token timestamps",
  "Chunk size sweep 512/2048/unbounded at matched load",
  "Preemption and KV utilization counters per step"], 0.62),

 ("corpus-01103", """4. Falsifiable hypothesis (H1: tail latency is KV exhaustion, not network)
H1: >= 70% of requests above the P99 E2E threshold at 90% saturation overlap in time
with a preemption/recompute event on their serving replica.
Null: overlap is at chance level given the base preemption rate (compute the expected
overlap explicitly; a raw 70% means nothing without it).
Discriminating test: compute client-side TTFT minus server-ingress TTFT per request.
If the network is the cause, that gap carries the excess; if KV exhaustion is the cause,
the gap is flat and the excess sits between admission and first token.
Boundary condition: invalid if the two candidate causes co-occur under load; run a
reduced-concurrency arm where preemption count is zero and confirm the tail collapses.
Evidence: preemption counters time-joined to request IDs, dual-timestamp TTFT,
NIC-level retransmit and queue-drop counters.""",
 4, 4, 4, ["Source is a grading rubric, not an answer",
           "Attribution of tail latency requires a discriminating test the rubric never specifies",
           "Base-rate error is the default failure mode here and is unmentioned"],
 ["Time-joined preemption events and outlier request IDs",
  "Client vs ingress TTFT decomposition",
  "NIC retransmit/drop counters over the same window"], 0.6),

 ("corpus-01104", """4. Falsifiable hypothesis (H1: decode is memory-bandwidth bound)
H1: at batch sizes 1-8, decode tok/s scales < 1.3x while batch grows 8x, and measured
bytes moved per output token is within 20% of (weight bytes + KV bytes read per step).
Null: decode tok/s scales near-linearly with batch, which would mean we are latency- or
kernel-launch-bound instead, and the bandwidth story is wrong.
Pre-registration: the predicted bytes/token number is written down before measurement,
otherwise this is curve-fitting.
Boundary condition: holds only below the batch size where the GEMM becomes compute-bound;
find that crossover empirically and report it rather than assuming it.
Evidence: DCGM dram bandwidth utilization, Nsight Systems timeline for launch gaps,
achieved occupancy, per-step batch size.""",
 4, 4, 3, ["Rubric answer omits the roofline framing the prompt implies",
           "No pre-registration discipline, so any post-hoc bandwidth number can be fit",
           "Crossover point between memory- and compute-bound regimes left unspecified"],
 ["DCGM DRAM bandwidth and SM occupancy traces",
  "Nsight Systems decode-step timeline",
  "Pre-registered bytes-per-token arithmetic committed before the run"], 0.6),

 ("corpus-01105", """4. Falsifiable hypothesis (H1: prefill/decode disaggregation pays only if KV transfer is cheap)
H1: disaggregating prefill and decode onto separate replica pools improves P99 TTFT
>= 20% at 90% saturation, conditional on measured KV transfer time staying < 15% of the
TTFT budget for the L and XL strata.
Null: transfer cost exceeds 15%, in which case disaggregation must be rejected even if
aggregate throughput improves, because the prompt's objective is latency.
Do not assert an interconnect capability that has not been measured: record
nvidia-smi topo -m, and measure achieved KV transfer bandwidth end to end rather than
quoting a link's nameplate number. RDMA/GDR paths must be shown active (perftest or
transport counters), not assumed from a config flag.
Boundary condition: result is specific to this KV layout and page size; changing either
invalidates it.
Evidence: per-request KV transfer bytes and duration, topology dump, transport counters.""",
 4, 4, 4, ["Rubric gives no gating criterion, so disaggregation could be adopted on throughput alone",
           "Nameplate interconnect bandwidth is commonly substituted for measured bandwidth",
           "GDR/RDMA path activation is often assumed from config rather than verified"],
 ["Measured KV transfer bytes and duration per request",
  "nvidia-smi topo -m and NIC/GPU affinity dump",
  "Transport counters or perftest showing the RDMA path is actually used"], 0.58),

 ("corpus-01106", """4. Falsifiable hypothesis (H1: the server is queueing-dominated, not compute-limited)
H1: at 90% saturation, queueing delay is >= 60% of P99 E2E for the S stratum, and
Little's law holds: mean in-system requests equals arrival rate times mean E2E within 10%.
Null: queueing is a minority of E2E, meaning the fix is kernel/compute efficiency and any
scheduler tuning will only move latency between queue and execution.
Diagnostic signature: an intervention that reduces queue time by X while increasing
execution time by ~X is the fingerprint of a saturated server; report both components
so this cannot be sold as an improvement.
Boundary condition: Little's law check is only meaningful in steady state; discard the
first and last 10% of the run window.
Evidence: ingress/admit/complete timestamps per request, instantaneous in-flight count,
scheduler queue depth time series.""",
 4, 4, 4, ["Rubric does not require decomposing E2E into queue and execution",
           "Without the Little's law check, saturation is easily misreported as an improvement",
           "Steady-state windowing is unaddressed"],
 ["Per-request ingress/admit/complete timestamps",
  "In-flight count and queue depth time series",
  "Arrival-rate log for the Little's law arithmetic"], 0.62),

 ("corpus-01107", """4. Falsifiable hypothesis (H1: our own benchmark is inflated by prefix caching)
H1 (self-invalidating by design): replacing the shared system preamble with a randomized
per-request preamble of equal token length degrades P99 TTFT by >= 30%.
Null: degradation is inside the noise band, meaning prefix cache is not carrying our
numbers. Either outcome is informative; a confirmed H1 means all prior TTFT figures from
this harness must be relabeled as best-case, not rescaled.
Boundary condition: token length must be matched exactly, or this measures prompt length
instead of cache reuse.
Evidence: prefix-cache hit rate per arm, token-length histogram proving the match,
per-stratum TTFT distributions.""",
 5, 4, 4, ["Rubric never asks the harness to attack its own validity",
           "Shared-preamble benchmarks routinely publish cache-inflated TTFT",
           "Token-length matching is the easy way to get this test wrong"],
 ["Prefix-cache hit-rate counters per arm",
  "Token-length histograms for shared vs randomized preamble",
  "Per-stratum TTFT distributions, not means"], 0.63),

 ("corpus-01108", """4. Falsifiable hypothesis (H1: TP=4 hurts decode via collective latency)
H1: at equal total GPU count, TP=4 x 2 replicas shows >= 10% worse decode tok/s than
TP=2 x 4 replicas at fixed per-replica load, and out-of-band NCCL small-message
all-reduce latency (nccl-tests, 1KB-64KB) is >= 1.5x higher for the TP=4 group.
Null: decode regresses but NCCL small-message latency is flat, which refutes the
collective explanation and points at memory or scheduling instead.
Do not assume topology: record nvidia-smi topo -m, NCCL_DEBUG=INFO ring/tree selection,
and whether NVLink or PCIe is actually used per group. A TP group split across a PCIe
host bridge is a different experiment than one inside an NVLink domain.
Boundary condition: valid only at this model size and batch regime; larger batches
amortize the collective and the effect shrinks.
Evidence: nccl-tests bus bandwidth and latency curves, topo dump, NCCL algorithm logs.""",
 4, 4, 4, ["Rubric omits the parallelism-strategy comparison the scenario invites",
           "Topology is commonly assumed rather than dumped, invalidating cross-arm comparison",
           "No out-of-band collective measurement to explain a regression mechanistically"],
 ["nccl-tests all-reduce latency/bandwidth for both TP widths",
  "nvidia-smi topo -m and NCCL_DEBUG=INFO algorithm selection logs",
  "Per-replica decode tok/s at matched per-replica load"], 0.58),

 ("corpus-01109", """4. Falsifiable hypothesis (H1: degradation is cumulative-request driven, not wall-clock)
H1: over a 6-hour soak, P99 TTFT rises >= 15% and the rise correlates with cumulative
requests served (r >= 0.8) rather than elapsed wall clock; a periodic-restart arm at
fixed request count restores baseline within the noise band.
Null: degradation tracks wall clock equally well, implicating thermal drift or a leak
outside the allocator; then restart-on-request-count is the wrong mitigation.
Confounder to separate: run an idle-clock arm (low RPS, same duration) so wall-clock and
request-count explanations are not collinear.
Operational safety: any restart mitigation must be drain-gated (stop admitting, wait for
in-flight completion or a hard 60s cap, then restart) and rolled out one replica at a
time; a blind restart drops in-flight long generations.
Evidence: KV fragmentation / allocator stats over time, GPU temperature and clock logs,
per-hour P99 by stratum.""",
 4, 4, 5, ["Rubric ignores time-dependent degradation entirely",
           "Wall-clock and request-count explanations are collinear unless a low-RPS arm is run",
           "Restart mitigations without drain gating kill in-flight long generations"],
 ["Allocator/fragmentation statistics time series",
  "GPU temperature and clock logs across the soak",
  "Low-RPS control arm of equal duration"], 0.6),

 ("corpus-01110", """4. Falsifiable hypothesis (H1: reported P99 suffers coordinated omission)
H1: rerunning the identical load point with an open-loop generator that timestamps
intended arrival (not actual send) raises reported P99 E2E by >= 40% at 90% saturation.
Null: the two harnesses agree within the noise band, meaning the closed-loop numbers were
not omission-biased at this load.
Consequence if confirmed: prior P99 figures are relabeled as lower bounds. They are not
rescaled by a correction factor, because the omission bias is load-dependent and not a
constant multiplier.
Boundary condition: at low load the harnesses converge by construction; the test only
discriminates near saturation, so run it at >= 80% saturation or not at all.
Evidence: intended-vs-actual arrival timestamps, generator backlog depth, side-by-side
per-stratum latency CDFs.""",
 5, 4, 4, ["Rubric does not mention coordinated omission, the dominant benchmark error here",
           "Closed-loop harnesses systematically under-report the tail near saturation",
           "Tempting but invalid fix of rescaling old numbers by a constant factor"],
 ["Intended vs actual arrival timestamps from the generator",
  "Generator backlog/queue depth during the run",
  "Per-stratum latency CDFs from both harnesses at matched load"], 0.63),

 ("corpus-01111", """4. Falsifiable hypothesis (H1: length-aware admission improves short-request tail without starving long ones)
H1: a two-queue admission policy (S/M prioritized, L/XL rate-limited to a reserved share)
improves S-stratum P99 TTFT >= 25% at 90% saturation while XL P99 E2E regresses <= 10%
and XL completion rate stays >= 99%.
Null: any of the three pre-registered criteria fails; the policy is then rejected, not
retuned post hoc on the same data (retuning requires a fresh holdout load run).
Adversarial arm (required): clients that mis-declare max_tokens. Declared output length is
untrusted input; if the policy relies on it, a mis-declaring arm must show bounded damage,
otherwise the policy is a starvation vector rather than a scheduling improvement.
Operational safety: ship behind a flag with per-stratum SLO alerts; roll back within one
deploy cycle on XL starvation or > 10% P99 E2E regression in any stratum.
Evidence: per-stratum admission and completion rates, declared-vs-actual output length
distribution, queue occupancy by class.""",
 4, 5, 5, ["Rubric never treats client-declared max_tokens as untrusted input",
           "Priority scheduling starves the heaviest stratum unless completion rate is gated",
           "Post-hoc retuning on the same data would invalidate the pre-registered criteria"],
 ["Per-stratum admission, completion, and P99 records",
  "Declared vs actual output-length distribution including mis-declaring clients",
  "Queue occupancy by class over the load ladder"], 0.6),
]

src = [json.loads(l) for l in open('/tmp/tb_src.jsonl')]
assert len(src) == len(ITEMS)
out = []
for d, it in zip(src, ITEMS):
    sid, body, tc, ic, os_, risks, ev, conf = it
    assert d['id'] == sid, (d['id'], sid)
    m = d['messages']
    u = [x for x in m if x['role'] == 'user'][0]['content']
    a = [x for x in m if x['role'] == 'assistant'][0]['content']
    v = sid.replace('corpus-0', '').lstrip('0')
    ans = COMMON_HEAD.format(v=v) + body + "\n" + COMMON_TAIL
    out.append({
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": ans,
        "quality_dimensions": {"technical_correctness": tc, "instruction_coverage": ic, "operational_safety": os_},
        "risks": risks,
        "evidence_required": ev,
        "confidence": conf,
    })

p = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0101.jsonl'
with open(p, 'w') as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", p, len(out))
