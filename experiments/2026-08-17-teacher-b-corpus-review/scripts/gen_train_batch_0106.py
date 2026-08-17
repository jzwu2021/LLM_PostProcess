import json

COMMON_HEAD = """Evaluation plan: mixed short-prompt / long-generation LLM serving traffic (variant {v}).

0. Non-claims
This is a serving-system measurement protocol only. It is not evidence about model
quality, and no number here transfers across engine commit, weight hash, quantization,
sequence-length regime, tenancy, or GPU SKU without a full re-run.

1. Fixed setup (record it or the run is void)
- Engine commit + launch flags, weights hash, quantization scheme, tokenizer version,
  driver/CUDA version, container image digest.
- Persistence mode on, clocks pinned (nvidia-smi -pm 1, -lgc), MIG topology declared.
  Unpinned DVFS makes thermal drift readable as a treatment effect; that is the single
  most common way these studies lie.
- Open-loop Poisson arrivals from an off-host generator; generator CPU < 60% at peak
  and RTT recorded. Closed-loop harnesses self-throttle under overload and delete
  exactly the tail samples the P99 claim depends on.
- Seeded length mixture over four strata S/M/L/XL, byte-identical across arms.
- ignore_eos with fixed max_tokens per stratum, otherwise output length is an
  uncontrolled random variable and TPOT is uninterpretable.

2. Metric definitions frozen before the first run
- TTFT = server ingress -> first output token flushed to the socket. Also record
  client-side TTFT; the delta is the queue + network component.
- TPOT = (E2E - TTFT) / (out_tokens - 1), per request; undefined and excluded when
  out_tokens < 2. Never a global mean over pooled requests.
- Queueing delay = ingress -> prefill admission, reported as its own distribution.
- Throughput split into req/s, prefill tok/s, decode tok/s. Prefill is compute-bound,
  decode is memory-bandwidth-bound at small batch; one aggregate tok/s number hides
  the trade and is not falsifiable.
- Goodput = req/s meeting a pre-declared per-stratum SLO, not raw completions.
- P99 computed per stratum from raw per-request records, never pooled across strata
  or across repeats.

3. Procedure
Warmup discarded (first 2 min or 500 requests, whichever is larger); >= 5 fresh-process
repeats per arm; randomized arm order to break drift-vs-treatment aliasing; load ladder
of >= 6 points up to 130% of measured saturation so the knee and the post-knee behavior
are both visible.

"""

COMMON_TAIL = """
Confounders instrumented on every run
- Prefix/KV cache hit rate. A divergence > 2 percentage points between arms voids the
  comparison outright; it is not a covariate to adjust away.
- Continuous-batching batch-size distribution over time, not just its mean.
- Preemption / recompute / swap counts, KV block utilization, allocator OOM retries.
- Sampling parameters and EOS behavior (output length is itself a treatment).
- NUMA/CPU-affinity placement, CPU governor, tokenizer-thread saturation, generator RTT.
- Co-tenant activity on the same host and any NIC/PCIe contention.

Stop / rollback gates (pre-committed, before seeing data)
- Any throttle reason reported by nvidia-smi (thermal, power, reliability) => the sample
  is invalid, not noisy; abort and rerun the arm.
- Error rate > 0.5% at a load point => abort that load point, record it as saturation.
- Promote only if the pre-registered primary metric beats a noise band of 2x the control
  arm's across-repeat spread AND no stratum regresses > 5% on P99 end-to-end latency.
- Roll back within one deploy cycle on > 10% P99 E2E regression, observed starvation of
  any stratum, or an increase in preemption/OOM counters.

Honest limit: a P99 estimated from a few thousand requests carries a wide interval.
Quote a bootstrap or binomial CI next to it; an effect smaller than that interval is
not a result, and reporting it as one is the failure mode this plan exists to prevent.
"""

ITEMS = [
 ("corpus-01157", """4. Falsifiable hypothesis (H1: chunked prefill converts TTFT head-of-line blocking
into bounded, predictable decode jitter)
H1: with a chunked-prefill token budget of 2048 per scheduler step, P99 TTFT of the S
stratum improves >= 25% versus unbounded prefill at 90% of measured saturation, while
decode tok/s degrades <= 5%.
Null: improvement inside the noise band, or decode loss > 5%.
Mechanism check (this is what makes it a mechanism claim rather than a correlation):
add a chunk=512 arm. If head-of-line blocking by long prefills is really the cause,
S-stratum P99 TTFT must move monotonically further in the same direction, and the
per-step batch composition traces must show long prefills split across steps. If the
gain appears without that compositional change, the mechanism is wrong even if the
metric moved.
5. Controlled experiment
Arms: unbounded prefill (control), chunk=2048, chunk=512. Same seed, same mixture,
5 repeats, randomized order, single variable = chunk budget.
Boundary conditions where H1 is expected to fail and must be reported, not hidden:
prefill-dominated mixes (XL share > 60%) where chunking only adds scheduling overhead;
very small models where prefill is already short relative to a step; and any arm where
KV pressure forces preemption, which confounds the TTFT signal with recompute cost.""",
 4, 5, 5, ["Chunk budget tuned on the same data used to report the result, which invalidates the pre-registration",
           "Decode regression hidden by reporting only aggregate tok/s",
           "Clock or thermal drift misread as a chunking effect if clocks are unpinned"],
 ["Per-stratum TTFT/TPOT/E2E raw per-request records for all three arms",
  "Per-scheduler-step batch composition traces showing prefill chunk splitting",
  "Preemption, recompute, and KV utilization counters per arm",
  "nvidia-smi throttle-reason log covering every measurement window"], 0.62),

 ("corpus-01158", """4. Falsifiable hypothesis (H1: the observed P99 blowup is queueing, not per-request
slowdown)
Framing here is diagnostic: a service shows P99 E2E growing ~4x while median is flat.
H1: >= 70% of the P99 E2E excess over median is queueing delay (ingress -> admission),
not TTFT-after-admission and not TPOT.
Null: queueing accounts for < 70%, which would redirect the investigation to KV
pressure or per-token slowdown.
Differential diagnosis, each with a distinguishing signature:
(a) admission queueing -> queue-delay distribution carries the tail, batch size is at
    its cap, KV utilization high but stable;
(b) KV thrash / preemption -> preemption and recompute counters rise with the tail,
    decode tok/s sawtooths, TPOT tail inflates;
(c) long-prefill head-of-line blocking -> tail is in TTFT-after-admission and
    correlates with XL-stratum arrivals in the preceding 1-2 seconds;
(d) client/network -> client-side TTFT tail exists with no server-side counterpart.
5. Controlled experiment
Replay the captured trace at 60/80/100/120% of the incident rate against a pinned build.
Only (a) predicts that tail excess scales roughly with utilization/(1-utilization) while
per-request service time stays flat. Confirmatory perturbation: raise max concurrent
sequences and, separately, add one replica; if (a) holds, replica addition collapses the
tail and per-request service time is unchanged.""",
 4, 5, 5, ["Attributing the tail to a cause without separating queue delay from service time",
           "Trace replay that loses the original arrival burstiness and therefore cannot reproduce the tail",
           "Adding capacity as a fix while the real cause is KV thrash, which returns at the next traffic shape change"],
 ["Ingress, admission, first-token, and completion timestamps per request",
  "Preemption/recompute/swap counters and KV utilization time series",
  "Arrival-timestamp trace from the incident window with per-stratum labels",
  "Client-side vs server-side TTFT comparison for the same requests"], 0.6),

 ("corpus-01159", """4. Falsifiable hypothesis (H1: decode throughput at the operating point is limited by
HBM bandwidth for KV reads, not by GPU compute)
H1: in the decode-only regime, achieved decode tok/s scales within 15% of the ratio
predicted by KV bytes read per step / measured HBM bandwidth, and holds within 10%
when SM clocks are reduced 20% while memory clock is held fixed.
Null: decode tok/s drops roughly proportionally with the SM clock, which would mean
compute- or kernel-launch-bound, not bandwidth-bound.
Roofline arithmetic to state explicitly before measuring: per decode step the engine
reads the full attended KV for every active sequence, so bytes/step ~= 2 (K and V) *
layers * kv_heads * head_dim * dtype_bytes * sum(context_len). Divide by measured
(not datasheet) achievable bandwidth to get a floor on step time. Report datasheet vs
achieved bandwidth separately; assuming datasheet is the classic overstatement.
5. Controlled experiment
Arms: nominal clocks; SM clock -20% (memory fixed); memory clock -20% (SM fixed).
Sweep batch size across the regime where continuous batching keeps the batch full.
Boundary conditions: at very small batch the workload is launch-latency bound and H1
must fail; with FP8/INT8 KV or GQA the bytes/step term changes and the prediction must
be recomputed rather than reused; long contexts push attention toward compute-bound.""",
 5, 5, 4, ["Datasheet bandwidth substituted for measured achievable bandwidth, inflating the predicted ceiling",
           "Clock manipulation on a shared or production host affecting other tenants",
           "Batch size drifting between arms so the comparison is not single-variable"],
 ["Measured achievable HBM bandwidth from a bandwidth microbenchmark on the same GPU and clocks",
  "Model geometry: layers, kv_heads, head_dim, KV dtype, and per-request context lengths",
  "Per-step decode timing plus profiler counters (dram throughput, SM occupancy)",
  "Clock and throttle-reason logs for each arm"], 0.66),

 ("corpus-01160", """4. Falsifiable hypothesis (H1: prefill/decode disaggregation improves TTFT tail without
paying an unacceptable KV-transfer cost at this cluster's interconnect)
H1: splitting prefill and decode onto separate GPU pools, with KV handed over via
RDMA, improves S-stratum P99 TTFT >= 30% at 90% saturation while total cluster
output tok/s degrades <= 8%.
Null: TTFT gain inside the noise band, or throughput loss > 8%.
Mechanism and the boundary that decides it: handover cost is KV bytes per request
divided by achieved (not line-rate) fabric bandwidth. Compute this number first. If
per-request KV is on the order of hundreds of MB and the fabric delivers only a few
tens of GB/s achieved, transfer time lands in the same order as the prefill it was
meant to offload and H1 cannot hold; the experiment then becomes a falsification of
the design, which is a valid and cheap outcome.
5. Controlled experiment
Arms: colocated (control) vs disaggregated at matched total GPU count. Instrument the
handover path separately: transfer bytes, achieved bandwidth, completion latency, and
whether transfer overlaps decode of other sequences. Report interconnect type
explicitly (RoCEv2 vs InfiniBand vs PCIe/host-staged) because the conclusion is not
portable across them; a host-staged copy through system memory is a different
experiment with a different ceiling.""",
 4, 5, 5, ["Quoting fabric line rate instead of achieved bandwidth, which flatters the transfer budget",
           "Disaggregated arm given more total GPUs than the control, making the comparison meaningless",
           "RoCE deployment without verified PFC/ECN configuration, so congestion collapse is misread as a design result"],
 ["Per-request KV size and measured handover bandwidth and latency distribution",
  "Interconnect type, topology, and for RoCE the PFC/ECN and pause-frame counters",
  "Matched-GPU-count accounting for both arms",
  "Per-stratum TTFT/TPOT and cluster-level output tok/s per arm"], 0.58),

 ("corpus-01161", """4. Falsifiable hypothesis (H1: the intermittent latency spikes come from KV-cache
preemption under a shifted length mixture, not from a code regression)
Symptom: periodic multi-second E2E spikes on a build that previously looked stable.
H1: spike windows coincide (within 1 s) with nonzero preemption/recompute counters and
KV utilization above ~95%, and the incident-period length mixture has a heavier XL tail
than the pre-incident baseline.
Null: spikes occur with flat preemption counters and unchanged mixture, which points
instead at a code regression, a noisy neighbor, or host-level interference.
Competing hypotheses and their signatures:
- code regression -> spikes reproduce at fixed replayed mixture on the new build and
  vanish on the previous commit, with preemption flat;
- host interference -> spikes correlate with co-tenant activity or CPU steal, and the
  GPU-side step timing is clean;
- network/client -> client-side only, no server-side step-time excess.
5. Controlled experiment
Replay the pre-incident mixture and the incident mixture against both commits: a 2x2
that separates mixture from code. Only H1 predicts spikes follow the mixture regardless
of commit. Rollback gate: if H1 is confirmed, the mitigation is capacity or admission
control on context length, not a revert; revert only if the 2x2 shows a commit effect.""",
 4, 5, 5, ["Reverting a commit on correlation alone and declaring the incident fixed while the real driver is traffic shape",
           "Preemption counters not retained long enough to cover the incident window",
           "Replay that normalizes the length mixture and therefore destroys the very signal being tested"],
 ["Per-second KV utilization, preemption, recompute, and swap counters spanning the incident",
  "Length-mixture histograms for pre-incident and incident windows",
  "Both commits deployed on identical hardware for the 2x2 replay",
  "Host-level CPU steal and co-tenant activity logs"], 0.62),

 ("corpus-01162", """4. Falsifiable hypothesis (H1: speculative decoding gains are draft-acceptance-limited,
and the break-even acceptance rate is predictable in closed form)
H1: end-to-end decode speedup follows S = (expected accepted tokens per verify step) /
(1 + verify overhead ratio) within 15%, and falls below 1.0 when the mean acceptance
rate drops under the pre-computed break-even alpha for this draft/target pair.
Null: measured speedup deviates > 15% from the model, meaning the bottleneck is
elsewhere (batching interference, memory pressure from holding two models, or verify
kernel inefficiency).
Explicit mechanism: with draft length k and per-token acceptance alpha, expected
accepted tokens per step is (1 - alpha^(k+1))/(1 - alpha). The cost side is one target
forward over k+1 positions plus k draft forwards. Speculation is a latency optimization
that consumes throughput headroom; at high batch the target model is already
bandwidth-saturated and speculation can be net negative. That regime boundary must be
measured, not assumed.
5. Controlled experiment
Arms: no speculation (control), k in {2, 4, 8}, swept across batch 1, 8, 32, 64.
Per-arm record acceptance-rate distribution by stratum, verify-step time, and memory
headroom lost to the draft model. Report the batch size at which speculation crosses
from win to loss; that crossover is the deployable result, not the batch-1 headline.""",
 5, 5, 4, ["Reporting batch-1 speedup as if it were a serving result, when production runs at high batch",
           "Draft model memory reducing KV capacity and causing preemption that is then blamed on speculation",
           "Acceptance rate averaged across strata, hiding that long-context requests accept far less"],
 ["Per-stratum acceptance-rate distributions, not just the mean",
  "Verify-step and draft-step timing breakdown per arm",
  "KV capacity and preemption counters with and without the draft model resident",
  "Speedup vs batch-size curve identifying the crossover point"], 0.63),

 ("corpus-01164", """4. Falsifiable hypothesis (H1: tail latency during rolling deploys is caused by cold
KV/prefix cache and warmup on new replicas, not by load-balancer imbalance)
H1: for the first N minutes after a replica joins, its P99 TTFT exceeds the fleet median
by >= 2x, prefix-cache hit rate on that replica is < 30% versus > 70% fleet-wide, and
the excess decays with the cache hit rate rather than with request count alone.
Null: new replicas show fleet-comparable hit rates and the tail tracks a request-count
imbalance instead, which would point at the balancer's algorithm.
Distinguishing test: compare least-outstanding-requests balancing against cache-affinity
(prefix-aware) routing. If H1 is right, affinity routing shortens the warm-up transient;
if imbalance is right, only the balancer change matters and hit rate is unaffected.
5. Controlled experiment
Staged deploy on a canary slice: instrument per-replica hit rate, outstanding requests,
and P99 TTFT at 1-second granularity from join time. Pre-commit the gate: hold the
rollout if canary P99 E2E exceeds baseline by > 10% for more than 3 consecutive minutes,
and roll back if any replica fails to reach 50% of fleet hit rate within the declared
warmup window. Warmup traffic must be replayed real prefixes; synthetic random prompts
will not populate the prefix cache and will produce a falsely pessimistic result.""",
 4, 5, 5, ["Draining old replicas before new ones are warm, which concentrates cold-cache traffic",
           "Prefix-affinity routing creating hot spots that starve a replica; needs a load ceiling per replica",
           "Warmup validated with synthetic prompts that do not exercise the real prefix distribution"],
 ["Per-replica prefix-cache hit rate and P99 TTFT time series from join time",
  "Per-replica outstanding-request counts to test the imbalance alternative",
  "Deploy timeline with exact join and drain timestamps",
  "Canary vs baseline per-stratum latency with the pre-committed gate evaluated"], 0.6),

 ("corpus-01165", """4. Falsifiable hypothesis (H1: at this operating point the serving bottleneck is KV
capacity, so throughput scales with usable KV blocks rather than with FLOPs)
H1: raising usable KV capacity by X% (via KV quantization to FP8, or a lower
gpu_memory_utilization headroom, or shorter max context) raises sustained req/s by
0.6-1.0X% until a second bottleneck binds, and preemption counters fall correspondingly.
Null: req/s moves < 0.3X%, meaning compute or scheduling binds first and KV work is
misdirected effort.
Capacity arithmetic to state up front: KV bytes = 2 * layers * kv_heads * head_dim *
dtype_bytes * total_context_tokens_resident. Max concurrent sequences at a given mean
context follows directly; that number, not a batch-size flag, is the real concurrency
ceiling. Free VRAM after weights and activation workspace is what matters, and
fragmentation means usable is strictly less than free.
5. Controlled experiment
Arms: FP16 KV (control), FP8 KV, and a reduced-max-context arm sized to match FP8's
capacity gain. If capacity is the mechanism, the two treatment arms should land within
the noise band of each other despite completely different implementations; divergence
falsifies the pure-capacity story and implicates quantization overhead or accuracy-path
effects. KV quantization additionally requires an output-quality check on a held-out
set: throughput gained by silently degrading outputs is not a win, and that check is a
gate, not an afterthought.""",
 5, 5, 5, ["Treating KV quantization as a free win without an output-quality gate",
           "Raising gpu_memory_utilization until allocator fragmentation causes intermittent OOM under burst",
           "Reduced max context silently truncating real requests, converting a latency problem into a correctness problem"],
 ["Free-vs-usable VRAM accounting including fragmentation, per arm",
  "Preemption/recompute counters and achieved concurrent-sequence counts",
  "Held-out output-quality comparison for the FP8 KV arm",
  "Sustained req/s at matched SLO for all three arms with repeats"], 0.66),

 ("corpus-01168", """4. Falsifiable hypothesis (H1: tensor-parallel decode at TP=8 is communication-bound,
so measured decode tok/s falls short of linear scaling by an amount predicted by
all-reduce time on the measured NVLink/PCIe path)
H1: decode step time equals (compute estimate / TP) + 2 * per-layer all-reduce time
within 20%, and TP=8 delivers < 1.6x the decode tok/s of TP=4 on a topology where any
GPU pair crosses PCIe or a host bridge rather than NVLink.
Null: scaling is within 15% of linear, meaning communication is not the limiter and the
optimization target is elsewhere.
Mechanism: TP inserts two all-reduces per transformer layer on the decode path; at small
per-step token counts the messages are latency-dominated, so NCCL algorithm selection
and topology matter more than message size. Verify the topology with nvidia-smi topo -m
before claiming anything; a "TP=8 is slow" report on a box with mixed NVLink/PCIe links
is a topology finding, not an engine finding.
5. Controlled experiment
Arms: TP=1,2,4,8 at fixed model and fixed batch. Independently measure the all-reduce
in isolation at the exact message sizes and counts the decode path issues (nccl-tests or
an equivalent harness on the same communicator layout), then check additivity against
observed step time. Boundary conditions: at large batch the compute term grows and the
communication share falls, so H1 is a small-batch/decode claim and must not be
generalized to prefill or to offline batch scoring.""",
 5, 5, 4, ["Comparing TP arms across different physical topologies or NUMA placements",
           "Assuming NVLink everywhere without checking nvidia-smi topo -m",
           "Generalizing a decode-path communication result to prefill or to throughput-oriented offline batching"],
 ["nvidia-smi topo -m output and the NCCL communicator/rank-to-device mapping",
  "Isolated all-reduce latency at the decode path's message sizes on the same layout",
  "Per-step decode timing for TP=1/2/4/8 at fixed batch and model",
  "NCCL debug output confirming algorithm and transport actually selected"], 0.64),

 ("corpus-01169", """4. Falsifiable hypothesis (H1: admission control with per-stratum SLO targets raises
goodput under overload without starving long generations)
H1: at 120% of saturation, SLO-aware admission control (reject or shed early rather
than queue unboundedly) raises goodput >= 25% versus unbounded queueing, while the XL
stratum's completion rate stays within 10% of its share under the control arm.
Null: goodput gain inside the noise band, or XL completion share falls > 10%, i.e. the
policy bought aggregate numbers by starving the expensive class.
Mechanism: under open-loop overload an unbounded queue grows without bound and every
request eventually violates its SLO, so completed-but-useless work dominates. Shedding
early converts that into a bounded-latency, reduced-admission regime. The failure mode
of the fix is exactly the starvation the null tests for, which is why completion share
by stratum is a primary metric and not a diagnostic.
5. Controlled experiment
Arms: unbounded queue (control), queue-depth cap, and SLO-deadline-aware admission with
per-stratum reservation. Load ladder to 150% of saturation. Report goodput, rejection
rate, and per-stratum completion share at every point. Client-declared output length
must be treated as untrusted input: measure declared-vs-actual length and include an
adversarial arm where a fraction of clients under-declare, since a policy that only
works with honest declarations is not deployable.
Rollback gate: revert if rejection rate exceeds the declared budget at nominal (not
overload) load, or if any stratum's completion share drops below its pre-committed floor.""",
 4, 5, 5, ["Client-declared max_tokens trusted as scheduling input, which the adversarial arm must probe",
           "Priority or deadline scheduling starving the heaviest stratum unless completion share is gated",
           "Post-hoc retuning of thresholds on the same data, invalidating the pre-registered criteria"],
 ["Per-stratum admission, rejection, completion, and P99 records across the load ladder",
  "Declared vs actual output-length distribution including under-declaring clients",
  "Queue occupancy and wait-time distribution by class at each load point",
  "Goodput under the pre-declared SLO for every arm and load point"], 0.6),
]

src = [json.loads(l) for l in open('/tmp/tb_src.jsonl')]
assert len(src) == len(ITEMS), (len(src), len(ITEMS))
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

p = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0106.jsonl'
with open(p, 'w') as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", p, len(out))
